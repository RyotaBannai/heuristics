from os import listdir
from os.path import isfile, join
from pathlib import Path
from subprocess import run

import ipdb as pdb

base = Path("/Users/ryotabannai/Documents/dev/heuristics/heuristics/atcoder/topological_map")
ind = base / "tools/in"
outd = base / "tools/out"
onlyfiles = sorted([f for f in listdir(ind) if isfile(join(ind, f))])


for path in onlyfiles:
    result = run(
        f"cd tools && cargo run -r --bin vis {str(ind / path)} {str(outd / path)}",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)
