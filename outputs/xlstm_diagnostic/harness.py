"""Isolated CPU first-epoch diagnostic. Never creates official run artifacts.

No validation is permitted. Therefore stop at the first epoch boundary: continuing
would omit the frozen validation-driven controller and cease to be a faithful run.
"""
import argparse, hashlib, inspect, json, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

class FirstNonfinite(RuntimeError): pass

def stats(t):
    t=t.detach()
    finite=torch.isfinite(t)
    good=t[finite]
    return dict(shape=list(t.shape),dtype=str(t.dtype),nan=int(torch.isnan(t).sum()),
                inf=int(torch.isinf(t).sum()),min=float(good.min()) if good.numel() else None,
                max=float(good.max()) if good.numel() else None)

class Observer(TorchDispatchMode):
    def __init__(self,model,report):
        super().__init__();self.names={id(m):n for n,m in model.named_modules()}
        self.report=report;self.phase='forward';self.last_exp=None;self.previous={}
    def trace_forward(self,frame,event,arg):
        if not frame.f_code.co_filename.replace('\\','/').endswith('/models/xlstm.py'):
            return None
        if event=='line':
            previous=self.previous.get(id(frame))
            if previous is not None:
                v=frame.f_locals
                origin={'file':frame.f_code.co_filename,'line':previous,
                    'block':self.names.get(id(v.get('self'))),'time_step_zero_based':v.get('step')}
                # Attach source provenance to existing autograd nodes only. This
                # neither inserts tensor hooks nor changes any computation.
                for name,t in list(v.items()):
                    if not isinstance(t,torch.Tensor) or t.grad_fn is None:continue
                    stack=[t.grad_fn]
                    while stack:
                        node=stack.pop()
                        if 'diagnostic_origin' in node.metadata:continue
                        node.metadata['diagnostic_origin']={**origin,'forward_tensor':name}
                        if previous==145 and self.last_exp:
                            node.metadata['diagnostic_stabilizer']=self.last_exp
                        stack.extend(n for n,_ in node.next_functions if n is not None)
            self.previous[id(frame)]=frame.f_lineno
        elif event=='return':self.previous.pop(id(frame),None)
        return self.trace_forward
    def location(self):
        frame=sys._getframe()
        while frame:
            if frame.f_code.co_filename.endswith('models\\xlstm.py') or frame.f_code.co_filename.endswith('models/xlstm.py'):
                v=frame.f_locals
                return dict(file=frame.f_code.co_filename,line=frame.f_lineno,
                    block=self.names.get(id(v.get('self'))),time_step_zero_based=v.get('step')),
            frame=frame.f_back
        return ({'block':None,'time_step_zero_based':None},)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        values=[t for t in tree_flatten(out)[0] if isinstance(t,torch.Tensor) and t.is_floating_point()]
        # -inf in masked localization logits/log_softmax is intentional. Loss is
        # evaluated outside this mode; every model/backward/Adam output is checked.
        bad=next((t for t in values if not bool(torch.isfinite(t).all())),None)
        # The official loss deliberately masks bins outside [19,79] with -inf.
        # Autograd detaches that saved log-softmax tensor during backward.
        # Exempt only this exact support, never NaN/+inf or valid-bin nonfinites.
        if bad is not None and str(func) in ('aten.detach.default','aten.masked_fill.Scalar','aten._log_softmax.default'):
            if bad.ndim==2 and bad.shape[1]==99:
                mask=torch.arange(99).ge(19)&torch.arange(99).le(79)
                if bool(torch.isfinite(bad[:,mask]).all()) and bool(torch.isneginf(bad[:,~mask]).all()):
                    bad=None
                    self.report['intentional_mask_operations_ignored']=self.report.get('intentional_mask_operations_ignored',0)+1
        if str(func)=='aten.exp.default':
            loc=self.location()[0]
            if loc.get('line')==145:
                self.last_exp={'location':loc,'negative_m_state':stats(args[0]),
                    'm_state':stats(-args[0]),'exp_output':stats(out),
                    'clamp_output':None,'clamp_status':'not yet executed'}
                self.report['stabilizer']=self.last_exp
        if str(func)=='aten.clamp.default' and self.last_exp and self.location()[0].get('line')==145:
            self.last_exp['clamp_output']=stats(out);self.last_exp['clamp_status']='executed'
        if bad is not None:
            loc=self.location()[0]
            if self.phase=='backward':
                node=torch._C._current_autograd_node()
                if node is not None:
                    loc={**loc,**node.metadata.get('diagnostic_origin',{}),'autograd_node':node.name()}
                    if 'diagnostic_stabilizer' in node.metadata:
                        self.report['stabilizer_at_failing_node']=node.metadata['diagnostic_stabilizer']
            self.report['first_nonfinite']={'phase':self.phase,'operation':str(func),
                **loc,'tensor':'operation output','output':stats(bad),
                'operands':[stats(t) for t in tree_flatten((args,kwargs))[0] if isinstance(t,torch.Tensor) and t.is_floating_point()],
                'stack':traceback.format_stack(limit=12)}
            if loc.get('line')==145 and str(func)=='aten.exp.default':
                self.report['first_nonfinite']['tensor']='exp(-m_state), before clamp'
            raise FirstNonfinite(str(func))
        return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--train',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-seconds',type=float,default=600);ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();project=a.project.resolve();output=a.output.resolve()
    if project in output.parents or 'campaign_runpod_confirmatory_fresh' in output.parts:
        raise ValueError('Output must be outside original project and campaign')
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(project))
    from runpod.core import verify_code
    from confirmatory.frozen import get_row,parameters
    from confirmatory.access_data import checked_path,read_rows,TRAIN_NAME
    from confirmatory.training_engine import seed_run
    from confirmatory.model_factory import build_model_from_hyperparameters
    from confirmatory.optim import build_common_adam
    from confirmatory.losses import joint_loss
    from confirmatory.contracts import make_valid_position_mask
    import h5py
    seal=verify_code();s4,s6,row=get_row('xlstm','cfg_01');p=parameters(row)
    spec=json.loads((project/'runpod/manifests/frozen_inputs.json').read_text())['files']['train']
    train=checked_path(a.train.parent,TRAIN_NAME,spec['sha256'])
    if a.train.resolve()!=train:raise ValueError('Train path mismatch')
    real_h5=h5py.File;opened=[]
    def only_train(name,mode='r',*args,**kwargs):
        if Path(name).resolve()!=train or mode!='r':raise PermissionError('Only train read access permitted')
        opened.append(str(train));return real_h5(name,mode,*args,**kwargs)
    report={'identity':{'architecture':'xlstm','configuration_slot':'cfg_01','seed':11},
        'hyperparameters':p,'source_code_manifest_sha256':seal,'train_sha256':spec['sha256'],
        'device':'cpu','torch':str(torch.__version__),'numpy':np.__version__,'threads':a.threads,
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'first epoch only, no validation or scheduler decisions; not an official attempt',
        'indexing':'epoch and batch zero-based and one-based explicitly recorded',
        'loss_total':None,'detection_loss':None,'localization_loss':None,
        'logits_det':None,'logits_loc':None,'first_nonfinite':None,'completed_batches':0}
    torch.set_num_threads(a.threads)
    # Exact factory/seed sequence; data loading uses no torch RNG.
    seed_run(11);model=build_model_from_hyperparameters('xlstm',p).cpu()
    optimizer=build_common_adam(model,p['learning_rate']);generator=torch.Generator().manual_seed(11)
    assert model.count_parameters()==row['trainable_parameters']==173266
    with patch.object(h5py,'File',only_train):batch=read_rows(train,np.arange(200000,dtype=np.int64))
    order=torch.randperm(len(batch['x']),generator=generator).numpy()
    report['order_sha256']=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
    np.save(output/'epoch_000_order.npy',order,allow_pickle=False)
    valid=make_valid_position_mask(99,device='cpu');model.train();obs=Observer(model,report)
    started=time.monotonic()
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        report['hdf5_opened']=opened;report['source_code_manifest_after']=verify_code()
        (output/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for b,start in enumerate(range(0,len(order),p['batch_size'])):
            if time.monotonic()-started>=a.max_seconds:
                report['status']='TIME_BUDGET_NO_NONFINITE';break
            report.update(epoch_zero_based=0,epoch_one_based=1,batch_zero_based=b,batch_one_based=b+1,
                loss_total=None,detection_loss=None,localization_loss=None,logits_det=None,logits_loc=None)
            ids=order[start:start+p['batch_size']];report['batch_row_indices']=ids.tolist()
            x=torch.from_numpy(batch['x'][ids]);c=torch.from_numpy(batch['has_changepoint'][ids]);k=torch.from_numpy(batch['cp_dx'][ids].astype(np.int64))
            report['inputs']=stats(x);optimizer.zero_grad(set_to_none=True)
            report['adam_before_step']={n:{key:stats(v) for key,v in optimizer.state.get(t,{}).items() if isinstance(v,torch.Tensor)} for n,t in model.named_parameters()}
            obs.phase='forward'
            prior_trace=sys.gettrace();sys.settrace(obs.trace_forward)
            try:
                with obs:det,loc=model(x)
            finally:sys.settrace(prior_trace)
            report['logits_det']=stats(det);report['logits_loc']=stats(loc)
            obs.phase='loss'
            with obs:losses=joint_loss(det,loc,c,k,valid,negative_weight=p['negative_class_weight'],lambda_detection=p['lambda_detection'],lambda_localization=p['lambda_localization'])
            for name,t in zip(['loss_total','detection_loss','localization_loss'],losses):
                report[name]=stats(t)
                if not bool(torch.isfinite(t).all()):
                    report['first_nonfinite']={'phase':'loss','tensor':name,'operation':'joint_loss','output':stats(t)};raise FirstNonfinite(name)
            obs.phase='backward'
            with obs:losses[0].backward()
            obs.phase='optimizer.step'
            with obs:optimizer.step()
            report['completed_batches']=b+1
            if b%5==0:save();print(json.dumps({'batch_completed':b+1,'seconds':time.monotonic()-started}),flush=True)
        else:report['status']='FIRST_EPOCH_FINITE_STOP_BEFORE_VALIDATION'
    except FirstNonfinite:
        report['status']='FIRST_NONFINITE_DETECTED'
    except BaseException as e:
        report['status']='DIAGNOSTIC_ERROR';report['error']=repr(e);report['traceback']=traceback.format_exc()
    finally:
        report['nonfinite_parameter_gradients']={n:stats(t.grad) for n,t in model.named_parameters() if t.grad is not None and not bool(torch.isfinite(t.grad).all())}
        report['gradient_note']='Backward may not have started or may be interrupted before leaf gradients are populated.'
        save();print(json.dumps({'status':report['status'],'completed_batches':report['completed_batches'],'output':str(output)}),flush=True)

if __name__=='__main__':main()
