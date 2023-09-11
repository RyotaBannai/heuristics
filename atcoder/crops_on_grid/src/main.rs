use heuristics::utils::test_all::test_all;
use itertools::Itertools;
use proconio::{input, marker::Chars};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::time;

struct Input {
    T: usize,
    H: usize,
    W: usize,
    i0: usize,
    h: Vec<Vec<usize>>,
    v: Vec<Vec<usize>>,
    K: usize,
    S: Vec<usize>,
    D: Vec<usize>,
}

#[derive(Clone, Debug)]
struct Work {
    k: usize,
    i: usize,
    j: usize,
    s: usize,
}
impl Work {
    fn new(k: usize, i: usize, j: usize, s: usize) -> Self {
        Self { k, i, j, s }
    }
    fn rnd_create(input: &Input) -> Self {
        let mut rng = rand::thread_rng(); // こっちを使う

        // let mut rng = rand_pcg::Pcg64Mcg::new(890482);
        let k = rng.gen_range(0..input.K);
        let i = rng.gen_range(0..input.H);
        let j = rng.gen_range(0..input.W);
        let s = rng.gen_range(0..=input.S[k]);
        Self::new(k, i, j, s)
    }
}

fn numbering(i: usize, j: usize, w: usize) -> usize {
    i * w + j
}

fn validate(works: &[Work], input: &Input) -> bool {
    let vs = input.H * input.W;
    let mut es = vec![BTreeSet::<usize>::new(); vs]; // 隣接リスト
    let num = |i: usize, j: usize| numbering(i, j, input.W);

    // 土地の区画について、無向グラフの隣接リスト
    for i in 0..input.H {
        for j in 0..input.W {
            let v = num(i, j); // 頂点番号

            // 右左上下
            for (di, dj) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let ni = i as isize + di;
                let nj = j as isize + dj;

                // 範囲内
                if 0 <= nj && (nj as usize) < input.W && 0 <= ni && (ni as usize) < input.H {
                    let neighbor = num(ni as usize, nj as usize);
                    es[v].insert(neighbor);
                }
            }
        }
    }

    // 水路があれば隣接する区画から辺を取り除く.
    for (i, row) in input.h.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x == 1 {
                // もし水路があれば、`南側`の隣接する頂点間の経路を両隣接リストから取り除く.
                let up = num(i, j);
                let low = num(i + 1, j);
                es[up].remove(&low);
                es[low].remove(&up);
            }
        }
    }

    for (i, row) in input.v.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x == 1 {
                // もし水路があれば、`東側`の隣接する頂点間の経路を両隣接リストから取り除く.
                let left = num(i, j);
                let right = num(i, j + 1);
                es[left].remove(&right);
                es[right].remove(&left);
            }
        }
    }

    // 植える日s をkey にした計画をリストとした辞書を用意
    let mut ps = BTreeMap::<usize, Vec<Work>>::new();
    works
        .iter()
        .for_each(|p| ps.entry(p.s).or_insert(vec![]).push(p.clone()));

    // 収穫日のd をkey にした計画をリストとした辞書を用意
    let mut planted = BTreeMap::<usize, Vec<Work>>::new();

    // NOTE: used: その区画に作物を植えているか否か.
    // 値:-1 は使っていない. 値: k はタイプk の作物を植えている
    let mut used = std::iter::repeat(-1).take(vs).collect::<Vec<_>>();

    for t in 0..input.T {
        // 連結成分 出入口から探索開始

        let mut par = (0..vs).collect_vec();
        let mut q = VecDeque::new();
        q.push_back(input.i0);
        while let Some(u) = q.pop_back() {
            for &y in &es[u] {
                // 未訪問の頂点（マーカーが自分自身）and 区画に作物が植えられていない
                if par[y] == y && used[y] == -1 {
                    q.push_back(y);
                    par[y] = input.i0;
                }
            }
        }

        // 先に植える
        if let Some(xs) = ps.get(&t) {
            for p in xs {
                // 出入口は塞げない.
                if p.i == input.i0 && p.j == 0 {
                    return false;
                }

                // 連結してない（道が塞がっている）or S が決められた日付より後に植える予定になっている, or すでに区画に植えられてる
                let v = num(p.i, p.j);
                if input.i0 != par[v] || p.s > input.S[p.k] || used[v] != -1 {
                    return false;
                }
                used[v] = p.k as isize; // 植える

                // 収穫する日程をkey
                planted
                    .entry(input.D[used[v] as usize])
                    .or_insert(vec![])
                    .push(p.clone());
            }
        }

        // 作物を植えた後の連結区画の具合で、収穫ができたりできなかったり...
        let mut par = (0..vs).collect_vec();
        let mut q = VecDeque::new();
        q.push_back(input.i0);
        while let Some(u) = q.pop_back() {
            for &y in &es[u] {
                // 未訪問の頂点（マーカーが自分自身）and (区画に作物が植えられていない or 植えられている作物が収穫日と一致)
                if par[y] == y && (used[y] == -1 || input.D[used[y] as usize] == t) {
                    q.push_back(y);
                    par[y] = input.i0;
                }
            }
        }

        // 収穫
        if let Some(xs) = planted.get(&t) {
            for x in xs {
                // 連結してない
                let v = num(x.i, x.j);
                if input.i0 != par[v] {
                    return false;
                }
                used[v] = -1; // 収穫
            }
        }
    }

    true
}

// 計画に整合性があるかを先にチェックして問題がなければスコア計算する, 問題があれば0
fn compute_score(works: &Vec<Work>, input: &Input) -> isize {
    if validate(works, input) {
        let mut score = 0;
        for p in works {
            score += input.D[p.k] - input.S[p.k] + 1;
        }
        score as isize
    } else {
        0
    }
}
#[allow(clippy::suspicious_else_formatting)]
fn simulated_annealing(input: &Input) -> Vec<Work> {
    const T0: f64 = 2e3; // 2000
    const T1: f64 = 6e2; // 600
    const TL: f64 = 0.2;
    let now = time::Instant::now(); // 今の時刻を取得
    let get_time = || -> f64 { now.elapsed().as_secs_f64() }; // 最初の時刻からの経過時間を計算
    let mut cnt = 0;
    let mut T = T0;
    // 初期解はランダム
    let mut rng = rand_pcg::Pcg64Mcg::new(890482);
    let mut best_plan = vec![Work::rnd_create(input)];
    let mut used = vec![0; input.K];
    used[best_plan[0].k] = 1;
    // let mut to_vec = |m: &BTreeMap<usize, Work>| m.into_iter().map(|(k, v)| *v).collect_vec();
    let mut best_score = compute_score(&best_plan, input);

    loop {
        // 冷却スケジュール
        cnt += 1;
        if cnt % 100 == 0 {
            let t = get_time() / TL;
            if t >= 1.0 {
                break;
            }
            T = T0.powf(1.0 - t) * T1.powf(t);
        }

        let mut tmp_plan = best_plan.clone();
        if rng.gen_bool(0.2) {
            // 一点変換
            // 1. 期間をS に近づける
            let i = rng.gen_range(0..tmp_plan.len());
            let p = &mut tmp_plan[i];
            let ran = p.s..=input.S[p.k];
            if !ran.is_empty() {
                let ns = rng.gen_range(ran);
                p.s = ns;
            }
        }
        // else if rng.gen_bool(0.07) {
        //     // 一点変換
        //     // 2.k を変える.
        //     let i = rng.gen_range(0..tmp_plan.len());
        //     let p = &mut tmp_plan[i];
        //     let nk = rng.gen_range(0..input.K);
        //     // すでにk を使ってたら、２つ目は入れられない
        //     if used[nk] != 1 {
        //         let ran = 0..=input.S[p.k];
        //         if !ran.is_empty() {
        //             let ns = rng.gen_range(ran);
        //             p.s = ns;
        //             used[p.k] = 0;
        //             used[nk] = 1;
        //             p.k = nk;
        //         }
        //     }
        // } else if rng.gen_bool(0.07) {
        //     // 一点変換
        //     // 3.区画を変える.
        //     let i = rng.gen_range(0..tmp_plan.len());
        //     let p = &mut tmp_plan[i];
        //     p.i = rng.gen_range(0..input.H);
        //     p.j = rng.gen_range(0..input.W);
        // }
        else {
            // １計画を追加
            let np = Work::rnd_create(input);
            if used[np.k] != 1 {
                used[np.k] = 1;
                tmp_plan.push(np);
            }
        }
        let tmp_score = compute_score(&tmp_plan, input);
        let p: f64 = f64::exp((tmp_score - best_score) as f64) / T;
        if best_score < tmp_score || rng.gen_bool(p) {
            best_score = tmp_score;
            best_plan = tmp_plan;
        }
    }
    best_plan
}

fn solve(input: &Input) -> Vec<Work> {
    simulated_annealing(input)
}

fn main() {
    input! {
        T: usize,
        H: usize,
        W: usize,
        i0: usize,
        h: [Chars;H-1],// 南側の水路
        v: [Chars;H], // 東側の水路
        K: usize,
        SD: [(usize,usize); K],
    }

    let mut S: Vec<usize> = vec![];
    let mut D: Vec<usize> = vec![];
    for (s, d) in SD {
        S.push(s - 1);
        D.push(d - 1);
    }
    // h,v は文字列で入力されるから、'0'->0, '1'->1 に変換.
    let (nh, nv) = vec![h, v]
        .into_iter()
        .map(|xs| {
            xs.into_iter()
                .map(|cs| {
                    cs.into_iter()
                        .map(|c| if c == '0' { 0 } else { 1 })
                        .collect::<Vec<usize>>()
                })
                .collect_vec()
        })
        .collect_tuple()
        .unwrap();

    let input: Input = Input {
        T,
        H,
        W,
        i0,
        h: nh,
        v: nv,
        K,
        S,
        D,
    };

    // 入力例1 を無事判定できることを確認.
    // 一番下の開始日時を2 にするとfalse 判定.

    // true pattern
    // let works = vec![
    //     Work::new(0, 0, 0, 1),
    //     Work::new(1, 0, 1, 0),
    //     Work::new(2, 1, 0, 3),
    //     Work::new(3, 0, 2, 0),
    //     Work::new(4, 3, 1, 1),
    //     Work::new(5, 2, 0, 3),
    //     Work::new(6, 4, 0, 1),
    //     Work::new(9, 3, 2, 0),
    //     Work::new(14, 2, 0, 7),
    //     Work::new(17, 5, 0, 1),
    //     Work::new(18, 0, 3, 0),
    //     Work::new(19, 4, 1, 1),
    // ];

    // false pattern
    // let works = vec![
    //     Work::new(425, 7, 16, 0),
    //     Work::new(4491, 17, 16, 0),
    //     Work::new(4513, 10, 3, 0),
    //     Work::new(5126, 2, 7, 0),
    //     Work::new(2450, 8, 15, 0),
    //     Work::new(5273, 17, 16, 1),
    // ];

    // println!("{}", validate(&works, &input));

    let works = solve(&input);
    println!("{}", works.len());
    for p in works {
        println!("{} {} {} {}", p.k + 1, p.i, p.j, p.s + 1);
    }
    test_all();
}

#[cfg(test)]
mod tests {
    use crate::numbering;

    #[test]
    fn test_numbearing() {
        assert_eq!(numbering(0, 0, 4), 0);
        assert_eq!(numbering(1, 0, 4), 4);
        assert_eq!(numbering(2, 3, 4), 11);
    }
}
