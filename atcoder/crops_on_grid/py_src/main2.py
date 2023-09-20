from __future__ import annotations

import copy
from collections import defaultdict as dd
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

import ipdb as pdb


@dataclass
class Input:
    T: int
    H: int
    W: int
    i0: tuple[int, int]
    h: list[list[bool]]
    v: list[list[bool]]
    K: int
    S: list[int]
    D: list[int]


@dataclass
class Work:
    k: int
    s: int
    d: int
    i: int | None = None
    j: int | None = None


class Graph:
    graph: ClassVar[dd[tuple[int, int], set[tuple[int, int]]] | None] = None


def set_graph(dat: Input) -> None:
    if Graph.graph is not None:
        return

    ad: dd = dd(set)
    for i in range(dat.H):
        for j in range(dat.W):
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if ni >= 0 and ni < dat.H and nj >= 0 and nj < dat.W:
                    ad[(i, j)].add((ni, nj))

    # 南側
    for i, row in enumerate(dat.h):
        for j, x in enumerate(row):
            if x:
                ad[(i, j)].remove((i + 1, j))
                ad[(i + 1, j)].remove((i, j))

    # 東側
    for i, row in enumerate(dat.v):
        for j, x in enumerate(row):
            if x:
                ad[(i, j)].remove((i, j + 1))
                ad[(i, j + 1)].remove((i, j))

    Graph.graph = ad


def solve(dat: Input, works: list[Work], n: int) -> list[Work]:
    set_graph(dat)
    ad = Graph.graph
    if ad is None:
        raise Exception("Graph has a problem.")

    # BFS
    q: deque[tuple[int, tuple[int, int]]] = deque([(-1, dat.i0)])
    dist: dd[tuple[int, int], int] = dd(int)
    while len(q) > 0:
        (d, u) = q.pop()
        if dist.get(u) is not None:
            continue
        dist[u] = d + 1
        for y in ad.get(u, []):
            q.append((dist[u], y))

    grids = sorted([(v, k) for k, v in dist.items()], reverse=True)

    bins = [int((dat.T / n) * x) for x in range(0, n)]
    bins.append(dat.T)

    slots: list[list[Work]] = [[] for _ in range(n)]
    for w in works:
        for i in range(n):
            if w.s > bins[i] and w.d <= bins[i + 1]:
                slots[i].append(w)

    ans: list[Work] = []
    for i in range(n):
        slot = slots[i]
        slot.sort(key=lambda w: (w.d, -w.s))
        qu = deque(slot)
        for _, (ni, nj) in grids:
            if len(qu) == 0:
                break
            p = qu.pop()
            p.i = ni
            p.j = nj
            p.s = bins[i] + 1  # works の参照だと class で開始位置が変わってしまう？
            ans.append(p)

    return ans


def readline() -> list[int]:
    return list(map(int, input().strip().split(" ")))


def read_input() -> Input:
    T, H, W, i0 = readline()
    h = [[c == "1" for c in input().strip()] for _ in range(H - 1)]
    v = [[c == "1" for c in input().strip()] for _ in range(H)]
    K = readline()[0]
    SD = [readline() for _ in range(K)]
    S, D = [row[0] for row in SD], [row[1] for row in SD]
    return Input(T, H, W, (i0, 0), h, v, K, S, D)


def main():
    dat = read_input()
    works = [Work(k=k, s=dat.S[k], d=dat.D[k]) for k in range(dat.K)]
    best = []
    best_score = 0
    for n in range(3, 11):
        plan = solve(dat, copy.deepcopy(works), n=n)
        score = sum(dat.D[w.k] - dat.S[w.k] + 1 for w in plan)
        if best_score < score:
            best = plan
            best_score = score

    print(len(best))
    for w in best:
        print(w.k + 1, w.i, w.j, w.s)


if __name__ == "__main__":
    main()
