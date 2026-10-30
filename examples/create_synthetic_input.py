"""Create an explicitly synthetic NPZ input for the fit command."""
from pathlib import Path
import argparse,numpy as np
from diabetes_step_change.benchmark import simulate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
data,_,_=simulate("shared_with_exception",131030100)
arrays={"n_markers":np.array(len(data))}
for m,d in enumerate(data):
    for key in ["X","y","w"]:arrays[f"{key}_{m}"]=d[key]
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,**arrays)
print("Synthetic input written")
