"""Independent synthetic calibration and evaluation; no cohort data are read."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib,time,datetime
import numpy as np,pandas as pd
from numpy.polynomial.legendre import leggauss
from . import core as jc

O=None
SCENES=['smooth_only','shared_with_exception','unshared_processes']
N_CAL=12;N_EVAL=24;NSET=240;M=20;BASE=131030000
EDGES=np.arange(-8,6.1,2.);QN,QW=leggauss(24);QW=QW/2

def basis(t):
    t=np.asarray(t);b=np.clip(t[...,None]-EDGES[:-1],0,2.)
    b[...,0]+=np.minimum(t+8,0);b[...,-1]+=np.maximum(t-6,0)
    return b

def kernel_basis(t):return np.einsum('nqj,q->nj',basis(np.asarray(t)[:,None]+QN),QW)

def phase_matrix():
    out=[]
    for a,b in [(-6,-4),(-4,0),(0,4)]:out.append((kernel_basis([b])-kernel_basis([a]))[0])
    return np.array(out)

def scene_truth(scene):
    rng=np.random.default_rng(131030)
    if scene=='smooth_only':return np.ones((7,1))@np.linspace(-.035,.05,M)[None,:]
    if scene=='shared_with_exception':
        u=np.array([0,.025,.12,.10,.015,-.01,-.005])
        v=np.array([0,0,0,.005,.08,.12,.02])
        a=np.r_[np.linspace(.5,1.3,8),np.linspace(-.8,-.3,4),np.zeros(8)]
        b=np.r_[np.zeros(8),np.linspace(.5,1.2,8),[-.7,-.9,0,0]]
        theta=np.outer(u,a)+np.outer(v,b)
        theta[:,19]=[0,0,0,0,-.17,-.12,0]
        return theta
    if scene=='unshared_processes':
        # Retain seven temporal directions in the unshared-process control.
        return rng.normal(0,.075,(7,M))
    raise ValueError(scene)

def simulate(scene,seed):
    rng=np.random.default_rng(seed);N=NSET*3
    setid=np.repeat(np.arange(NSET),3);case=np.tile([1,0,0],NSET)
    age=np.repeat(rng.uniform(55,80,NSET),3)+rng.normal(0,1.2,N);sex=rng.integers(0,2,N)
    major=rng.binomial(1,.2,N)
    t0=rng.uniform(-7.5,-2.5,N)
    times=t0[:,None]+np.array([0,4,8])+rng.normal(0,.2,(N,3))
    shift=np.repeat(rng.uniform(-1,1,NSET),3)
    theta=scene_truth(scene)
    bg=np.outer(np.linspace(.7,1.3,7),np.linspace(-.018,.025,M))
    nuis=rng.normal(0,.018,(8,M));nuis[3:6]*=1.5;nuis[6:]=0
    intercept=rng.normal(0,.5,(N,M)) + rng.normal(0,.4,(N,1))@np.linspace(.4,1,M)[None,:]
    vals=np.zeros((N,3,M))
    load=rng.normal(size=(3,M));load/=np.sqrt((load**2).sum(axis=0))
    for k in range(3):
        t=times[:,k];B=basis(t+shift)-basis(-8+shift)
        elapsed=t+8
        Z=np.column_stack([elapsed*(age-70)/10,elapsed*sex,elapsed*major,
                           np.ones(N)*(k==1),np.ones(N)*(k==2),np.zeros(N),np.zeros(N),np.zeros(N)])
        noise=.15*rng.normal(size=(N,3))@load+.23*rng.normal(size=(N,M))
        vals[:,k]=intercept+B@bg+case[:,None]*(B@theta)+Z@nuis+noise
    obs=rng.random((N,3,M))>.15
    rows=[]
    # Retain observed successive spans; bridge a missed middle visit only once.
    for m in range(M):
        ii=[];aa=[];bb=[]
        for a,b in [(0,1),(1,2),(0,2)]:
            ok=obs[:,a,m]&obs[:,b,m]
            if a==0 and b==2:ok &= ~obs[:,1,m]
            ids=np.where(ok)[0];ii.extend(ids);aa.extend([a]*len(ids));bb.extend([b]*len(ids))
        ii=np.asarray(ii);aa=np.asarray(aa);bb=np.asarray(bb)
        valid_sets=np.intersect1d(np.unique(setid[ii[case[ii]==1]]),np.unique(setid[ii[case[ii]==0]]))
        keep=np.isin(setid[ii],valid_sets);ii=ii[keep];aa=aa[keep];bb=bb[keep]
        ta=times[ii,aa];tb=times[ii,bb];dt=tb-ta;E=kernel_basis(tb)-kernel_basis(ta)
        Z=np.column_stack([dt*(age[ii]-70)/10,dt*sex[ii],dt*major[ii],
            (bb==1).astype(float)-(aa==1),(bb==2).astype(float)-(aa==2),np.zeros(len(ii)),np.zeros(len(ii)),np.zeros(len(ii))])
        X=np.column_stack([E,E*case[ii,None],Z]);y=vals[ii,bb,m]-vals[ii,aa,m]
        cnt=np.bincount(ii,minlength=N);nctrl={s:len(np.unique(ii[(setid[ii]==s)&(case[ii]==0)])) for s in valid_sets}
        w=np.array([1 if case[i] else 1/nctrl[setid[i]] for i in ii])/cnt[ii]
        rows.append({'X':X,'y':y,'w':w,'set':setid[ii]})
    perm=rng.permutation(NSET);testsets=perm[:80]
    train=[];test=[]
    for d in rows:
        mask=np.isin(d['set'],testsets)
        train.append({k:v[~mask] for k,v in d.items() if k!='set'})
        test.append({k:v[mask] for k,v in d.items() if k!='set'})
    return train,test,theta

def main(argv=None):
    import argparse
    global O,N_CAL,N_EVAL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=['quick','manuscript'],default='quick')
    args=parser.parse_args(argv)
    O=args.output.resolve()
    if O.exists() and any(O.iterdir()):raise RuntimeError('Choose a new or empty output directory')
    O.mkdir(parents=True,exist_ok=True)
    N_CAL,N_EVAL=(12,24) if args.profile=='manuscript' else (1,2)
    start=time.time()
    freeze={'frozen_at':datetime.datetime.now().astimezone().isoformat(),'seed_base':BASE,'scenes':SCENES,
        'calibration_replicates_per_scene':N_CAL,'evaluation_replicates_per_scene':N_EVAL,'matched_sets_per_dataset':NSET,
        'markers':M,'people_per_set':3,'visits':3,'visit_spacing_years':4,'visit_jitter_sd':.2,
        'report_error_uniform_years':[-1,1],'independent_marker_visit_missing_probability':.15,
        'train_sets':160,'test_sets':80,'grids':jc.GRIDS,'parameter_selection':'Lowest mean held-out observed interval MSE pooled equally over the 3 calibration scenes; one fixed parameter per method in independent evaluation',
        'primary_target':'Broad-phase case-minus-reference amplitudes from actual integrated observations',
        'exception_target':'Prespecified marker 20 post-report amplitude in shared_with_exception',
        'failure_rule':'A joint route that erases the marker exception or worsens unshared-process recovery substantially is not a general replacement; report all scenes without tuning after evaluation',
        'source':'Independent Gaussian synthetic cohort; no real participant data read',
        'core_sha256':hashlib.sha256(Path(jc.__file__).read_bytes()).hexdigest(),
        'run_profile':args.profile,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fp=O/'EXPERIMENT_FREEZE.json'
    if fp.exists():raise RuntimeError('Experiment already frozen; preserve it')
    fp.write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf8')
    (O/'solver_self_check.json').write_text(json.dumps(jc.self_check(),indent=2),encoding='utf8')
    cal=[]
    for scene_index,scene in enumerate(SCENES):
        for rep in range(N_CAL):
            seed=BASE+scene_index*1000000+rep;train,test,true=simulate(scene,seed);problem=jc.build_problem(train)
            for method,grid in jc.GRIDS.items():
                for param in grid:
                    fit=jc.fit(problem,method,param);score=jc.score(fit,test)
                    cal.append({'scene':scene,'replicate':rep,'seed':seed,'method':method,'param':json.dumps(param,sort_keys=True),
                                'test_interval_mse':score['mean_mse'],'iterations':fit['diagnostics']['iterations'],'converged':fit['diagnostics']['converged']})
            print('calibration',scene,rep+1,flush=True)
    cd=pd.DataFrame(cal);cd.to_csv(O/'calibration_results.csv',index=False)
    mean=cd.groupby(['method','param'],as_index=False).test_interval_mse.mean()
    choices={}
    for method,g in mean.groupby('method'):
        best=g.sort_values(['test_interval_mse','param']).iloc[0];choices[method]=json.loads(best['param'])
    (O/'selected_parameters.json').write_text(json.dumps({'selected_at':datetime.datetime.now().astimezone().isoformat(),'parameters':choices,'evaluation_not_yet_generated':True},indent=2),encoding='utf8')
    results=[];representative=[];PH=phase_matrix()
    for scene_index,scene in enumerate(SCENES):
        for rep in range(N_EVAL):
            seed=BASE+scene_index*1000000+100+rep;train,test,truth=simulate(scene,seed);problem=jc.build_problem(train);truephase=PH@truth
            for method,param in choices.items():
                fit=jc.fit(problem,method,param);score=jc.score(fit,test);est=PH@fit['theta'];error=est-truephase
                base={'scene':scene,'replicate':rep,'seed':seed,'method':method,'param':json.dumps(param,sort_keys=True),
                      'test_interval_mse':score['mean_mse'],'phase_rmse':float(np.sqrt(np.mean(error**2))),
                      'exception_post_estimate':float(est[2,19]),'exception_post_truth':float(truephase[2,19]),
                      'exception_post_abs_error':float(abs(error[2,19])),
                      'exception_direction_preserved':bool(est[2,19]*truephase[2,19]>0),
                      'mean_profile_rate_error':float(np.sqrt(np.mean((fit['theta']-truth)**2))),
                      'iterations':fit['diagnostics']['iterations'],'converged':fit['diagnostics']['converged'],
                      'matrix_rank':fit['diagnostics']['effective_rank'],
                      'pre_post_rate_contrast_rms':float(np.sqrt(np.mean(((est[1]-est[2])/4)**2)))}
                for k,name in enumerate(['early','pre','post']):base[name+'_amplitude_rmse']=float(np.sqrt(np.mean(error[k]**2)))
                results.append(base)
                if rep==0:
                    for j in range(M):
                        for k in range(7):representative.append({'scene':scene,'seed':seed,'method':method,'marker':j+1,'bin_left':EDGES[k],'bin_right':EDGES[k+1],
                            'true_rate':truth[k,j],'estimated_rate':fit['theta'][k,j],'lowrank_component':fit['lowrank'][k,j],'sparse_component':fit['sparse'][k,j]})
            print('evaluation',scene,rep+1,flush=True)
    d=pd.DataFrame(results);d.to_csv(O/'evaluation_results.csv',index=False);pd.DataFrame(representative).to_csv(O/'representative_rates.csv',index=False)
    cols=['test_interval_mse','phase_rmse','exception_post_abs_error','exception_direction_preserved','mean_profile_rate_error','pre_post_rate_contrast_rms','early_amplitude_rmse','pre_amplitude_rmse','post_amplitude_rmse','matrix_rank']
    summaries=[]
    for (scene,method),g in d.groupby(['scene','method']):
        r={'scene':scene,'method':method,'n':len(g),'converged_count':int(g.converged.sum())}
        for c in cols:r[c+'_mean']=float(g[c].mean());r[c+'_sd']=float(g[c].std())
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(O/'evaluation_summary.csv',index=False)
    pairs=[]
    for scene,g in d.groupby('scene'):
        base=g[g.method.eq('independent')].set_index('replicate')
        for method,h in g.groupby('method'):
            h=h.set_index('replicate')
            for metric in ['phase_rmse','test_interval_mse','exception_post_abs_error']:
                delta=h[metric]-base[metric]
                pairs.append({'scene':scene,'method':method,'metric':metric,'mean_paired_difference':delta.mean(),'paired_se':delta.std()/np.sqrt(len(delta)),
                              'relative_mean_change':h[metric].mean()/base[metric].mean()-1})
    pd.DataFrame(pairs).to_csv(O/'paired_method_differences.csv',index=False)
    completion={'status':'complete','at':datetime.datetime.now().astimezone().isoformat(),'synthetic_datasets':3*(N_CAL+N_EVAL),
        'calibration_fits':len(cd),'evaluation_fits':len(d),'all_evaluation_converged':bool(d.converged.all()),'evaluation_nonconverged':int((~d.converged).sum()),
        'elapsed_seconds':time.time()-start,'real_participant_records_read':False,'person_records_exported':False,'selected_parameters':choices,
        'methods_are_distinct_estimators_not_distinct_original_algorithms':True}
    (O/'COMPLETED.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(completion),flush=True)

if __name__=='__main__':main()
