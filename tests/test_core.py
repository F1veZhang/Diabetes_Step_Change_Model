import numpy as np
import pytest
from diabetes_step_change import core
from diabetes_step_change.benchmark import basis,simulate,phase_matrix

def sample(seed=193):
    rng=np.random.default_rng(seed)
    return [{"X":rng.normal(size=(80+m*9,22)),"y":rng.normal(size=80+m*9),"w":rng.uniform(.2,2,size=80+m*9)} for m in range(3)]

def test_profiled_estimate_matches_direct_weighted_fit():
    data=sample();f=core.fit(core.build_problem(data),"independent",0)
    for m,d in enumerate(data):
        X,y,w=d["X"],d["y"],d["w"]
        direct=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        np.testing.assert_allclose(f["beta"][:,m],direct,atol=1e-9,rtol=1e-9)

def identity_problem():
    p=core.build_problem(sample());rng=np.random.default_rng(28)
    p["H"]=np.repeat(np.eye(7)[None],3,axis=0)
    p["b"]=rng.normal(size=(7,3));p["lipschitz"]=1.
    p["lambda_max"]=np.linalg.norm(p["b"],2)
    return p

def test_nuclear_identity_matches_closed_singular_value_threshold():
    p=identity_problem();lam=.12*p["lambda_max"]
    U,s,V=np.linalg.svd(p["b"],full_matrices=False)
    expected=(U*np.maximum(s-lam,0))@V
    got=core.fit(p,"nuclear",.12)
    assert got["diagnostics"]["converged"]
    np.testing.assert_allclose(got["theta"],expected,atol=1e-9)

def test_sparse_identity_matches_closed_soft_threshold():
    p=identity_problem();lam=.12*p["lambda_max"]
    expected=np.sign(p["b"])*np.maximum(abs(p["b"])-lam,0)
    got=core.fit(p,"sparse",.12)
    np.testing.assert_allclose(got["theta"],expected,atol=1e-9)

def test_weight_rescaling_leaves_estimate_invariant():
    d=sample();p=core.build_problem(d)
    for x in d:x["w"]=x["w"]*17.2
    p2=core.build_problem(d)
    np.testing.assert_allclose(p["H"],p2["H"],atol=1e-12)
    np.testing.assert_allclose(p["b"],p2["b"],atol=1e-12)

def test_invalid_observations_rejected():
    d=sample();d[0]["y"][0]=np.nan
    with pytest.raises(ValueError):core.build_problem(d)
    d=sample();d[0]["w"][0]=-1
    with pytest.raises(ValueError):core.build_problem(d)
    d=sample();d[0]["X"]=d[0]["X"][:,:21]
    with pytest.raises(ValueError):core.build_problem(d)

def test_synthetic_seed_and_phase_units():
    a,b,truth=simulate("shared_with_exception",131030100)
    a2,b2,t2=simulate("shared_with_exception",131030100)
    for x,y in zip(a,a2):np.testing.assert_array_equal(x["y"],y["y"])
    np.testing.assert_array_equal(truth,t2)
    np.testing.assert_allclose(phase_matrix().sum(axis=1),[2,4,4],atol=1e-12)
    model=core.fit(core.build_problem(a),"nuclear",.03)
    assert model["diagnostics"]["converged"]
    assert len(core.predict(model,b[0]["X"],0))==len(b[0]["y"])
    assert np.isfinite(core.score(model,b)["mean_mse"])
