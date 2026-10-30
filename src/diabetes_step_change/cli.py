"""Command-line access to synthetic experiments and array-based fitting."""
from pathlib import Path
import argparse,json
import numpy as np
from . import core

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["demo","benchmark"]:
        p=sub.add_parser(name)
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--profile",choices=["quick","manuscript"],default="quick")
    p=sub.add_parser("fit")
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--method",choices=["independent","smooth","nuclear","sparse","posthoc"],default="nuclear")
    p.add_argument("--parameter",type=float,default=.03)
    a=parser.parse_args(argv)
    if a.command in ["demo","benchmark"]:
        from .benchmark import main as run
        run(["--output",str(a.output),"--profile",a.profile]);return
    if a.output.exists() and any(a.output.iterdir()):raise ValueError("Choose a new or empty output directory")
    with np.load(a.input,allow_pickle=False) as z:
        n=int(z["n_markers"]);data=[]
        for m in range(n):
            row={"X":z[f"X_{m}"],"y":z[f"y_{m}"]}
            if f"w_{m}" in z:row["w"]=z[f"w_{m}"]
            data.append(row)
    result=core.fit(core.build_problem(data),a.method,a.parameter)
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/"coefficients.npz",theta=result["theta"],beta=result["beta"],lowrank=result["lowrank"],sparse=result["sparse"])
    report={"method":a.method,"parameter":a.parameter,"diagnostics":result["diagnostics"]}
    (a.output/"fit.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))
