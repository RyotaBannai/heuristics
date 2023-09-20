import copy
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import ClassVar, Final

import ipdb as pdb
from scipy.cluster.hierarchy import DisjointSet


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
    i: int
    j: int
    s: int


def create_random_work(dat: Input, pool: set[int], n: int) -> list[Work]:
    picked: list[Work] = []
    for _ in range(min(len(pool), n)):
        k: int = random.sample(list(pool), 1)[0]
        pool.remove(k)
        i = random.randint(0, dat.H - 1)
        j = random.randint(0, dat.W - 1)
        # s = 0 if dat.S[k] == 0 else random.randint(0, dat.S[k] - 1)
        s = dat.S[k]

        picked.append(Work(k, i, j, s))

    return picked


def compute_score(works: list[Work], dat: Input) -> int:
    if validate_day(works, dat) and validate_path(works, dat):
        return sum(dat.D[w.k] - dat.S[w.k] + 1 for w in works)
    else:
        return 0


def validate_day(works: list[Work], dat: Input) -> bool:
    """全計画の日程が期限を守れてるかチェック."""
    return all(w.s <= dat.S[w.k] for w in works)


class Graph:
    graph: ClassVar[defaultdict[tuple[int, int], set[tuple[int, int]]] | None] = None


def set_graph(dat: Input) -> None:
    if Graph.graph is not None:
        return

    ad: defaultdict = defaultdict(set)
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


def validate_path(works: list[Work], dat: Input) -> bool:
    """計画通り植えられて収穫もできるかチェック.

    TODO: 判定と共にエラーメッセージを返す.
    """
    # 無向グラフの隣接リスト
    set_graph(dat=dat)
    ad = Graph.graph
    if ad is None:
        raise Exception("Graph has a problem.")

    # `植える日`をkey にした計画をリストとた辞書
    ws: defaultdict[int, list[Work]] = defaultdict(list[Work])
    for w in works:
        ws[w.s].append(w)
    # `植えた日`をkey にした計画をリストとた辞書
    planted: defaultdict[int, list[Work]] = defaultdict(list[Work])
    # 区画を使っているか
    used: defaultdict[tuple[int, int], int] = defaultdict(int)

    disjoint_set = DisjointSet([(i, j) for j in range(dat.W) for i in range(dat.H)])

    for t in range(dat.T):
        ds = copy.deepcopy(disjoint_set)
        q = deque([dat.i0])
        checked: defaultdict[tuple[int, int], bool] = defaultdict(bool)
        while len(q) > 0:
            u = q.pop()
            if checked.get(u, False):
                continue
            checked[u] = True
            for y in ad.get(u, []):
                ds.merge(y, u)
                # 使われているなら、繋がってるが通れない
                if used.get(y) is None:
                    q.append(y)
                else:
                    checked[y] = True

        # 先に植える
        for w in ws.get(t, []):
            grid = (w.i, w.j)
            # 出入口にあっても、その日に収穫できるなら問題ない
            if grid == dat.i0 and dat.D[w.k] != t:
                return False
            if not ds.connected(grid, dat.i0):
                return False

            # すでに使ってる
            if used.get(grid) is not None:
                return False

            used[grid] = w.k
            planted[dat.D[w.k]].append(w)

        ds = copy.deepcopy(disjoint_set)
        q = deque([dat.i0])
        checked.clear()

        while len(q) > 0:
            u = q.pop()
            if checked.get(u, False):
                continue
            checked[u] = True
            for y in ad.get(u, []):
                ds.merge(y, u)
                k = used.get(y)
                # 使われていなら通れる
                if k is None:
                    q.append(y)
                # 使われててもその日に収穫するなら、順序よく収穫して通れるようになる
                elif dat.D[k] == t:
                    q.append(y)
                else:
                    checked[u] = True
        # 収穫
        for w in planted.get(t, []):
            grid = (w.i, w.j)
            if not ds.connected(grid, dat.i0):
                return False
            used.pop(grid)

    return True


def simulated_annealing(dat: Input) -> list[Work]:
    """焼きなまし法."""

    T0: Final[int] = 2 * 10**3
    T1: Final[int] = 6 * 10**6
    TL: Final[float] = 10  # 制限実行時間
    now = time.time()

    def get_time() -> float:
        return time.time() - now

    cnt = 0
    T = T0
    best_plan: list[Work] = []
    pool = set()
    for k in range(0, dat.K + 1):
        pool.add(k)
    best_score: float = 1

    while True:
        cnt += 1
        t = get_time() / TL
        if t >= 1.0:
            break
        T = pow(T0, 1.0 - t) * pow(T1, t)

        tmp_plan = copy.deepcopy(best_plan)
        np = create_random_work(dat, pool, min(math.ceil(TL / get_time()) * 3, 30))
        tmp_plan.extend(np)

        tmp_score = compute_score(works=tmp_plan, dat=dat)
        p = math.exp(tmp_score - best_score) / T
        r = random.uniform(0, 1)
        if best_score < tmp_score or r < p:
            best_score = tmp_score
            best_plan = tmp_plan
        else:
            for w in np:
                pool.add(w.k)

    return best_plan


def readline() -> list[int]:
    return list(map(int, input().strip().split(" ")))


def read_input() -> Input:
    T, H, W, i0 = readline()
    h = [[c == "1" for c in input().strip()] for _ in range(H - 1)]
    v = [[c == "1" for c in input().strip()] for _ in range(H)]
    K = readline()[0]
    SD = [readline() for _ in range(K)]
    K -= 1
    S, D = [row[0] - 1 for row in SD], [row[1] - 1 for row in SD]

    # pdb.set_trace()
    return Input(T, H, W, (i0, 0), h, v, K, S, D)


def main():
    dat = read_input()
    # works = [
    #     Work(1, 0, 0, 2),
    #     Work(2, 0, 1, 1),
    #     # Work(3, 1, 0, 4),
    #     Work(4, 0, 2, 1),
    #     # Work(5, 3, 1, 2),
    #     Work(6, 2, 0, 4),
    #     # Work(7, 4, 0, 2),
    #     Work(10, 3, 2, 1),
    #     # Work(15, 2, 0, 8),
    #     # Work(18, 5, 0, 2),
    #     Work(19, 0, 3, 1),
    #     # Work(20, 4, 1, 2),
    # ]
    # works = [Work(w.k - 1, w.i, w.j, w.s - 1) for w in works]
    # a = validate_day(works=works, dat=dat)
    # b = validate_path(works=works, dat=dat)

    works = simulated_annealing(dat)
    print(len(works))
    for w in works:
        print(w.k + 1, w.i, w.j, w.s + 1)


if __name__ == "__main__":
    main()
