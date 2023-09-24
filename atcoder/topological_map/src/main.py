from collections import defaultdict as dd
from collections import deque
from itertools import product
from typing import Literal


def readline() -> list[int]:
    return list(map(int, input().strip().split(" ")))


def readlines(n: int) -> list[list[int]]:
    return [readline() for _ in range(n)]


n, m = readline()
dat = readlines(n)


checked: dd[tuple[int, int], Literal[True]] = dd(lambda: True)


def out(ni: int, nj: int) -> bool:
    return ni < 0 or ni >= n or nj < 0 or nj >= n


adj: dd[int, set[int]] = dd(set)

for i in range(n):
    for j in range(n):
        for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            ni, nj = i + di, j + dj
            if out(ni, nj):
                continue

            if dat[i][j] != dat[ni][nj]:
                adj[dat[i][j]].add(dat[ni][nj])

forbits: set[int] = set()


def rec(par: int, S: set[int]):
    tmp = adj.get(par)
    if tmp is None:
        return

    v = tmp.copy()
    if len(v - S) == 1:
        k = v.pop()
        forbits.add(k)
        rec(k, S=set([par]) | S)


for i in range(1, m + 1):
    rec(i, set())

# pdb.set_trace()


def solve(I: int, J: int):
    q: deque[tuple[int, int, int]] = deque()
    C = dat[I][J]
    if C == 0 or C in forbits:
        return
    q.append((C, I, J))

    while len(q) > 0:
        C, i, j = q.pop()
        if checked.get((i, j)) or C in forbits:
            continue
        checked[(i, j)]

        # 全方位同じいろor 0
        flag = True
        for di, dj in product(range(-1, 2), range(-1, 2)):
            ni, nj = i + di, j + dj
            if out(ni, nj):
                continue
            if dat[ni][nj] != 0 and dat[ni][nj] != C:
                flag = False
                break
        if not flag:
            continue
        dat[i][j] = 0

        # マスが斜めに飛ばないように上下左右のみ移動
        for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            ni, nj = i + di, j + dj
            if checked.get((ni, nj)):
                continue
            if out(ni, nj):
                continue
            if C != dat[ni][nj]:
                continue
            if C in forbits:
                continue
            q.append((C, ni, nj))


for i in [0, n - 1]:
    for j in range(n):
        solve(I=i, J=j)


for j in [0, n - 1]:
    for i in range(n):
        solve(I=i, J=j)


for i in range(n):
    print(" ".join(map(str, dat[i])))
