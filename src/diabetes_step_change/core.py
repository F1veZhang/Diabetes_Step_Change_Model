"""Joint interval reconstruction with profiled nuisance effects.

The solver applies nuclear-norm or sparse penalties to a seven-by-marker
excess-rate matrix. Inputs are arrays prepared by the caller. The estimator
does not load cohort records or export participant observations.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np

EXCESS = np.arange(7,14)
NUISANCE = np.r_[np.arange(7),np.arange(14,22)]
GRIDS = {
    'independent': [0.0],
    'smooth': [0.0,.001,.01,.1,1.0],
    'posthoc': [1,2,3,5,7],
    'nuclear': [.01,.03,.1,.3],
    'lowrank_sparse': [{'fraction': f, 'ratio': r} for f in [.01,.03,.1,.3] for r in [.2,.5]],
}
# Pure-sparse extension, evaluated separately from the original simulation grid.
SPARSE_GRID = [.002,.005,.006,.015,.02,.05,.06,.15]

def build_problem(data):
    """Profile nuisance terms while retaining each marker's observation rows."""
    blocks=[]
    for a in data:
        X=np.asarray(a['X'],float);y=np.asarray(a['y'],float)
        w=np.asarray(a.get('w',np.ones(len(y))),float)
        if X.shape!=(len(y),22) or len(w)!=len(y): raise ValueError('Expected X n by 22 and matching y, w')
        if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w>=0).all() and w.sum()>0):
            raise ValueError('Nonfinite/negative input')
        w=w/w.sum();Z=X[:,NUISANCE];E=X[:,EXCESS]
        ZZ=Z.T@(w[:,None]*Z);ZE=Z.T@(w[:,None]*E);zy=Z.T@(w*y)
        inv=np.linalg.pinv(ZZ,rcond=1e-10)
        H=E.T@(w[:,None]*E)-ZE.T@inv@ZE
        H=(H+H.T)/2
        ev,Q=np.linalg.eigh(H);H=(Q*np.maximum(ev,0))@Q.T
        b=E.T@(w*y)-ZE.T@inv@zy
        blocks.append({'H':H,'b':b,'z_inv':inv,'ze':ZE,'zy':zy,'constant':float(y@(w*y)-zy@inv@zy),'n':len(y)})
    H=np.stack([x['H'] for x in blocks]);b=np.column_stack([x['b'] for x in blocks])
    return {'H':H,'b':b,'blocks':blocks,'n_markers':len(blocks),
            'lipschitz':float(max(np.linalg.eigvalsh(x).max() for x in H)),
            'lambda_max':float(np.linalg.norm(b,ord=2)),
            'mean_objective':False}

def _grad(problem,theta):
    return np.einsum('mkl,lm->km',problem['H'],theta)-problem['b']

def _loss(problem,theta):
    return float(.5*np.sum(theta*np.einsum('mkl,lm->km',problem['H'],theta))-np.sum(theta*problem['b']))

def _svt(A,lam):
    U,s,V=np.linalg.svd(A,full_matrices=False)
    return (U*np.maximum(s-lam,0))@V

def _nuclear(A):return float(np.linalg.svd(A,compute_uv=False).sum())

def _finish(problem,theta,lowrank,sparse,method,param,diagnostics):
    beta=np.zeros((22,problem['n_markers']));beta[EXCESS]=theta
    for m,b in enumerate(problem['blocks']): beta[NUISANCE,m]=b['z_inv']@(b['zy']-b['ze']@theta[:,m])
    s=np.linalg.svd(theta,compute_uv=False)
    diagnostics.update({'effective_rank':int(np.sum(s>max(s[0]*1e-6,1e-10))) if len(s) else 0,
                        'singular_values':s.tolist(),'profiled_data_loss':_loss(problem,theta),
                        'equal_marker_objective':True,'number_of_markers':problem['n_markers']})
    return {'theta':theta,'beta':beta,'lowrank':lowrank,'sparse':sparse,
            'method':method,'param':param,'diagnostics':diagnostics}

def _fit_sparse(problem,param,max_iter,tol,lambda_reference):
    """Fit an elementwise L1 penalty for the sparse-component comparison."""
    lam=float(param)*(problem['lambda_max'] if lambda_reference is None else float(lambda_reference))
    lip=max(problem['lipschitz'],1e-12);zero=np.zeros_like(problem['b'])
    S=zero.copy();Y=S.copy();alpha=1.;previous=np.inf;converged=False;relative=np.inf
    for iteration in range(1,max_iter+1):
        a=Y-_grad(problem,Y)/lip;new=np.sign(a)*np.maximum(np.abs(a)-lam/lip,0)
        obj=_loss(problem,new)+lam*np.abs(new).sum()
        if obj>previous+1e-12:
            Y=S;alpha=1.;a=S-_grad(problem,S)/lip
            new=np.sign(a)*np.maximum(np.abs(a)-lam/lip,0);obj=_loss(problem,new)+lam*np.abs(new).sum()
        relative=float(np.linalg.norm(new-S)/max(1.,np.linalg.norm(S)))
        newalpha=(1+np.sqrt(1+4*alpha**2))/2;Y=new+(alpha-1)/newalpha*(new-S)
        S=new;alpha=newalpha;previous=obj
        if iteration>20 and relative<tol:converged=True;break
    return _finish(problem,S,zero,S,'sparse',param,{'iterations':iteration,'converged':converged,
        'relative_change':relative,'penalized_objective':float(previous),'lambda_nuclear':0.,'lambda_sparse':lam})

def fit(problem,method='nuclear',param=.03,max_iter=4000,tol=2e-7,lambda_reference=None):
    """Fit a profiled objective by closed form or restarted proximal steps."""
    M=problem['n_markers'];H=problem['H'];b=problem['b']
    zero=np.zeros((7,M));ols=np.column_stack([np.linalg.pinv(H[j],rcond=1e-10)@b[:,j] for j in range(M)])
    if method=='sparse':return _fit_sparse(problem,param,max_iter,tol,lambda_reference)
    if method=='independent' or (method in ('nuclear','smooth') and float(param)==0):
        return _finish(problem,ols,ols,zero,method,param,{'iterations':0,'converged':True,'relative_change':0.})
    if method=='smooth':
        D=np.diff(np.eye(7),n=2,axis=0);P=D.T@D
        theta=np.column_stack([np.linalg.pinv(H[j]+float(param)*np.trace(H[j])/7*P,rcond=1e-10)@b[:,j] for j in range(M)])
        return _finish(problem,theta,theta,zero,method,param,{'iterations':1,'converged':True,'relative_change':0.})
    if method=='posthoc':
        U,s,V=np.linalg.svd(ols,full_matrices=False);r=int(param)
        theta=(U[:,:r]*s[:r])@V[:r]
        return _finish(problem,theta,theta,zero,method,param,{'iterations':1,'converged':True,'relative_change':0.})
    if method not in ('nuclear','lowrank_sparse'):raise ValueError(method)
    fraction=float(param if method=='nuclear' else param['fraction']);lam=fraction*(problem['lambda_max'] if lambda_reference is None else float(lambda_reference))
    if fraction==0:return _finish(problem,ols,ols,zero,method,param,{'iterations':0,'converged':True,'relative_change':0.})
    ratio=0 if method=='nuclear' else float(param['ratio']);lams=lam*ratio
    lip=max(problem['lipschitz'],1e-12)*(1 if method=='nuclear' else 2)
    L=zero.copy();S=zero.copy();YL=L.copy();YS=S.copy();alpha=1.;previous=np.inf;relative=np.inf
    converged=False
    def objective(l,s):return _loss(problem,l+s)+lam*_nuclear(l)+(lams*np.abs(s).sum() if method=='lowrank_sparse' else 0)
    for iteration in range(1,max_iter+1):
        g=_grad(problem,YL+YS)
        nl=_svt(YL-g/lip,lam/lip)
        ns=np.sign(YS-g/lip)*np.maximum(np.abs(YS-g/lip)-lams/lip,0) if method=='lowrank_sparse' else zero
        obj=objective(nl,ns)
        # Restart from the current feasible point if the objective increases.
        if obj>previous+1e-12:
            YL=L;YS=S;alpha=1.;g=_grad(problem,L+S)
            nl=_svt(L-g/lip,lam/lip)
            ns=np.sign(S-g/lip)*np.maximum(np.abs(S-g/lip)-lams/lip,0) if method=='lowrank_sparse' else zero
            obj=objective(nl,ns)
        relative=float(np.sqrt(np.sum((nl-L)**2)+np.sum((ns-S)**2))/max(1.,np.sqrt(np.sum(L**2)+np.sum(S**2))))
        newalpha=(1+np.sqrt(1+4*alpha**2))/2
        YL=nl+(alpha-1)/newalpha*(nl-L);YS=ns+(alpha-1)/newalpha*(ns-S)
        L,S=nl,ns;alpha=newalpha;previous=obj
        if iteration>20 and relative<tol:
            converged=True;break
    return _finish(problem,L+S,L,S,method,param,{'iterations':iteration,'converged':converged,
        'relative_change':relative,'penalized_objective':float(previous),'lambda_nuclear':lam,'lambda_sparse':lams})

def predict(fitted,X,marker_index):return np.asarray(X)@fitted['beta'][:,marker_index]

def score(fitted,data):
    losses=[]
    for j,d in enumerate(data):
        resid=np.asarray(d['y'])-predict(fitted,d['X'],j)
        losses.append(float(np.average(resid**2,weights=d.get('w',None))))
    return {'mean_mse':float(np.mean(losses)),'marker_mse':losses}

def self_check():
    rng=np.random.default_rng(131030);data=[]
    for _ in range(4):
        X=rng.normal(size=(200,22));y=rng.normal(size=200);w=rng.uniform(.5,2,200)
        data.append({'X':X,'y':y,'w':w})
    p=build_problem(data);a=fit(p,'independent',0);c=fit(p,'nuclear',0)
    direct=[]
    for d in data:
        X,y,w=d['X'],d['y'],d['w'];direct.append(np.linalg.pinv(X.T@(w[:,None]*X))@(X.T@(w*y)))
    err=float(np.max(abs(a['beta']-np.column_stack(direct))));assert err<1e-9
    assert np.max(abs(a['theta']-c['theta']))==0
    # Under identity observation, the nuclear objective has a closed-form SVT solution.
    p2=dict(p);p2['H']=np.repeat(np.eye(7)[None],4,axis=0);p2['b']=rng.normal(size=(7,4));p2['lipschitz']=1.;p2['lambda_max']=np.linalg.norm(p2['b'],2)
    f=fit(p2,'nuclear',.1);truth=_svt(p2['b'],.1*p2['lambda_max']);e2=float(np.max(abs(f['theta']-truth)));assert e2<1e-9
    return {'weighted_FWL_vs_direct_WLS_max_error':err,'nuclear_identity_vs_closed_SVT_max_error':e2,'passed':True}

if __name__=='__main__':
    import json
    print(json.dumps(self_check()))
