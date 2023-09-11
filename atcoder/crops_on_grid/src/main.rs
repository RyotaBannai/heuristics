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
    h: Vec<Vec<bool>>,
    v: Vec<Vec<bool>>,
    K: usize,
    S: Vec<usize>,
    D: Vec<usize>,
}
impl Input {
    pub fn is_valid_point(&self, x: usize, y: usize) -> bool {
        x < self.H && y < self.W
    }
}

#[derive(Clone, Debug, Copy)]
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
        let k = rng.gen_range(1..=input.K);
        let i = rng.gen_range(0..input.H);
        let j = rng.gen_range(0..input.W);
        let ran = 1..=input.S[k - 1];
        let s = if ran.is_empty() {
            1
        } else {
            rng.gen_range(ran)
        };
        Self::new(k, i, j, s)
    }
}

pub struct Output {
    M: usize,
    works: Vec<Work>,
}

impl Output {
    fn validate(&self, input: &Input) -> Result<(), String> {
        // check range
        for Work {
            k, i: _, j: _, s, ..
        } in self.works.iter().cloned()
        {
            let ub = input.S[k];
            if s > ub {
                return Err(format!("Cannot plant crop {} after month {}", k + 1, ub));
            }
        }

        // check duplicates
        {
            let mut items = BTreeSet::new();
            for Work { k, .. } in self.works.iter() {
                if !items.insert(*k) {
                    return Err(format!("Crop {} is planted more than once", k + 1));
                }
            }
        }
        Ok(())
    }
}

fn compute_score(input: &Input, out: &Output) -> (i64, String) {
    if let Err(msg) = out.validate(input) {
        return (0, msg);
    }

    let mut scheduled_works: Vec<Vec<Work>> = vec![vec![]; input.T + 1];
    for w in out.works.iter().cloned() {
        scheduled_works[w.s].push(w)
    }

    let mut workspace = vec![vec![None; input.W]; input.H];
    let adj = {
        let mut adj = vec![vec![Vec::new(); input.W]; input.H];
        for i in 0..input.H {
            for j in 0..input.W {
                if i + 1 < input.H && !input.h[i][j] {
                    adj[i + 1][j].push((i, j));
                    adj[i][j].push((i + 1, j));
                }
                if j + 1 < input.W && !input.v[i][j] {
                    adj[i][j + 1].push((i, j));
                    adj[i][j].push((i, j + 1))
                }
            }
        }
        adj
    };

    let si = input.i0;
    let sj = 0;
    let start = (si, sj);
    let mut score = 0;

    for t in 1..=input.T {
        // beginning of month t
        {
            // check reachability
            if !scheduled_works[t].is_empty() {
                let mut visited = vec![vec![false; input.W]; input.H];

                if workspace[si][sj].is_none() {
                    let mut q = VecDeque::new();
                    q.push_back(start);
                    visited[si][sj] = true;

                    while !q.is_empty() {
                        let Some((x, y)) = q.pop_front() else { unreachable!() };
                        assert!(workspace[x][y].is_none());
                        for (x1, y1) in adj[x][y].iter().cloned() {
                            if input.is_valid_point(x1, y1)
                                && workspace[x1][y1].is_none()
                                && !visited[x1][y1]
                            {
                                q.push_back((x1, y1));
                                visited[x1][y1] = true;
                            }
                        }
                    }
                }

                for &Work { k, i, j, .. } in &scheduled_works[t] {
                    if !visited[i][j] {
                        return (
                            0,
                            format!(
                                "{} is scheduled at unreachable position {}, {}",
                                k + 1,
                                i,
                                j
                            ),
                        );
                    }
                }
            }

            // update workspace
            for &Work { k, i, j, s, .. } in &scheduled_works[t] {
                if let Some((k1, _)) = workspace[i][j] {
                    return (
                        0,
                        format!("Block ({}, {}) is occupied by crop {}", i, j, k1 + 1),
                    );
                } else {
                    workspace[i][j] = Some((k, s))
                }
            }
        }

        // end of month t; harvest crops
        let can_start = {
            if let Some((k, _s)) = workspace[si][sj] {
                input.D[k] == t
            } else {
                true
            }
        };

        if can_start {
            let mut q = VecDeque::new();
            q.push_back(start);
            let mut visited = vec![vec![false; input.W]; input.H];
            visited[si][sj] = true;

            while !q.is_empty() {
                let Some((i, j)) = q.pop_front() else { unreachable!() };
                if let Some((k, s)) = workspace[i][j] {
                    if input.D[k] == t {
                        workspace[i][j] = None;
                        let span = t - s + 1;
                        // this should hold because we do not
                        // allow planting crop k after month S[k]
                        assert!(span >= input.D[k] - input.S[k] + 1);
                        score += input.D[k] - input.S[k] + 1;
                    } else if input.D[k] < t {
                        return (
                            0,
                            format!("Cannot harvest crop {} in month {}", k + 1, input.D[k]),
                        );
                    }
                }

                for &(i1, j1) in &adj[i][j] {
                    assert!(input.is_valid_point(i1, j1));
                    let is_blocked = {
                        if let Some((k, _s)) = workspace[i1][j1] {
                            input.D[k] > t
                        } else {
                            false
                        }
                    };
                    if !is_blocked && !visited[i1][j1] {
                        q.push_back((i1, j1));
                        visited[i1][j1] = true;
                    }
                }
            }
        }
    }

    (
        ((score as u64 * 1_000_000) as f64 / (input.H * input.W * input.T) as f64).round() as i64,
        String::new(),
    )
}

#[allow(clippy::suspicious_else_formatting)]
fn simulated_annealing(input: &Input) -> Output {
    const T0: f64 = 2e3; // 2000
    const T1: f64 = 6e2; // 600
    const TL: f64 = 0.2;
    let now = time::Instant::now(); // 今の時刻を取得
    let get_time = || -> f64 { now.elapsed().as_secs_f64() }; // 最初の時刻からの経過時間を計算
    let mut cnt = 0;
    let mut T = T0;
    // 初期解はランダム
    let mut rng = rand_pcg::Pcg64Mcg::new(890482);
    let mut best_plan: Vec<Work> = vec![];
    let mut best_score = 0;

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

        if tmp_plan.is_empty() || rng.gen_bool(0.8) {
            // １計画を追加
            let np = Work::rnd_create(input);
            tmp_plan.push(np);
        } else if rng.gen_bool(0.1) {
            // 一点変換: k を変える.
            let i = rng.gen_range(0..tmp_plan.len());
            let p = &mut tmp_plan[i];
            let nk = rng.gen_range(1..=input.K);
            let ran = 1..=input.S[nk - 1];
            let ns = if ran.is_empty() {
                1
            } else {
                rng.gen_range(ran)
            };
            p.s = ns;
            p.k = nk;
        } else {
            // 一点変換: 区画を変える.
            let i = rng.gen_range(0..tmp_plan.len());
            let p = &mut tmp_plan[i];
            p.i = rng.gen_range(0..input.H);
            p.j = rng.gen_range(0..input.W);
        }

        let (tmp_score, _) = compute_score(
            input,
            &Output {
                M: tmp_plan.len(),
                works: tmp_plan.clone(),
            },
        );
        if tmp_score == 0 {
            continue;
        }
        let p: f64 = f64::exp((tmp_score - best_score) as f64) / T;
        if best_score < tmp_score || rng.gen_bool(p) {
            best_score = tmp_score;
            best_plan = tmp_plan;
        }
    }
    Output {
        M: best_plan.len(),
        works: best_plan,
    }
}

fn solve(input: &Input) -> Output {
    simulated_annealing(input)
}

fn main() {
    input! {
        T: usize,
        H: usize,
        W: usize,
        i0: usize,
        h1: [Chars; H - 1],
        v1: [Chars; H],
        K: usize,
        SD: [(usize, usize); K],
    }
    let h = h1
        .iter()
        .map(|i| i.iter().map(|&b| b == '1').collect())
        .collect();
    let v = v1
        .iter()
        .map(|i| i.iter().map(|&b| b == '1').collect())
        .collect();
    let S = SD.iter().map(|x| x.0).collect();
    let D = SD.iter().map(|x| x.1).collect();
    let input = Input {
        T,
        H,
        W,
        i0,
        h,
        v,
        K,
        S,
        D,
    };

    let output = solve(&input);
    println!("{}", output.M);
    for w in output.works {
        println!("{} {} {} {}", w.k, w.i, w.j, w.s);
    }
}
