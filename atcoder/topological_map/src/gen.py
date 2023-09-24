import sys
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
    with open(ind / path, "rb") as inf, open(outd / path, "a+") as outf:
        data = inf.read()
        run("python src/main.py", input=data, stdout=outf, shell=True)
