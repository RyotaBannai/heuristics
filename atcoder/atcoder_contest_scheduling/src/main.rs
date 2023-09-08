use proconio::input;
use rand::Rng;
use std::time;

struct Input {
    D: usize,
    s: Vec<Vec<i64>>,
    c: Vec<i64>,
}

// コンテストi に決定した時のd における満足度を計算
fn compute_score(input: &Input, out: &Vec<usize>) -> i64 {
    let mut score = 0;
    let mut last = vec![0; 26]; // コンテスト1~26 の最後の開催日を記録

    // 問題文より、d 日目終わるごとに c * (d-last(d,i)) だけ満足度が下がる
    for d in 0..out.len() {
        let j: usize = out[d] - 1; // input.c は0-index だから１引いておく.
        last[j] = d + 1; // j 番目のコンテスト
        for i in 0..26 {
            score -= (d + 1 - last[i]) as i64 * input.c[i];
        }
        score += input.s[d][j];
    }
    score
}

// compute_score はd 日目における満足度を i 日目のコンテストを決定する際に計算して、ベストになるものを選択するようにしていたが
// evaluate では、初回の満足度に加えてk 日後までに低下する量も考慮した時にベストなコンテストを選択する
// kは任意の変数だけど、k=0~26 まで全部試したときのスコアの平均をベストなコンテストとして毎回選択するようにしてもよい
fn evaluate(input: &Input, out: &Vec<usize>, k: usize) -> i64 {
    let mut score = 0;
    let mut last = vec![0; 26];

    for d in 0..out.len() {
        let j: usize = out[d] - 1;
        last[j] = d + 1;
        for i in 0..26 {
            score -= (d + 1 - last[i]) as i64 * input.c[i];
        }
        score += input.s[d][j];
    }
    // ここまでは compute_score と同じ

    // 最大はD 日目とする.
    // i 日目に加え、それ以降のk 日目(<=D)までの満足度まで考慮
    for d in out.len()..(out.len() + k).min(input.D) {
        for i in 0..26 {
            score -= (d + 1 - last[i]) as i64 * input.c[i];
        }
    }
    score
}

// 答を生成するheuristics アルゴリズム.
// その日に一番最大価値になるコンテストを選択することをD 日分決める.
fn greedy(input: &Input) -> Vec<usize> {
    let mut out = vec![];
    // D 日分のコンテストを順に決める
    for _ in 0..input.D {
        let mut max_score = i64::min_value();
        let mut best_i = 0;
        // 1~26 のうちどれにした時にベストスコアとなるか
        for i in 1..=26 {
            out.push(i);
            let score = compute_score(input, &out);
            if max_score < score {
                max_score = score;
                best_i = i;
            }
            out.pop();
        }
        out.push(best_i);
    }
    out
}

fn greedy2(input: &Input) -> Vec<usize> {
    let mut best = vec![];
    let mut global_best_score = i64::MIN;
    // search global max
    // k 日後までを考慮したときの満足度がベストになる決定の仕方を順に試して、ベストなコンテスト開催予定を選ぶ
    for k in 1..=26 {
        // search local max
        let mut out = vec![];
        for _ in 0..input.D {
            let mut max_score = i64::MIN;
            let mut best_i = 0;
            for i in 1..=26 {
                out.push(i);
                let score = evaluate(input, &out, k);
                if max_score < score {
                    max_score = score;
                    best_i = i;
                }
                out.pop();
            }
            out.push(best_i);
        }
        let local_best_score = evaluate(input, &out, k);
        if local_best_score > global_best_score {
            global_best_score = local_best_score;
            best = out;
        }
    }

    best
}

// 局所探索 - 山登り法
fn hill_climbing(input: &Input) -> Vec<usize> {
    const TL: f64 = 1.9; // 制限時間が来るまで探索
    let now = time::Instant::now(); // 今の時刻を取得
    let get_time = || -> f64 { now.elapsed().as_secs_f64() }; // 最初の時刻からの経過時間を計算
    let mut rnd = rand_pcg::Pcg64Mcg::new(890482);
    let mut out = (0..input.D)
        .map(|_| rnd.gen_range(1..=26))
        .collect::<Vec<_>>();
    let mut score = compute_score(input, &out);
    while get_time() < TL {
        let d = rnd.gen_range(0..input.D); // ランダムに更新したい日を選択
        let q = rnd.gen_range(1..=26); // ランダムにコンテストを選択
        let old = out[d];
        out[d] = q;
        let new_score = compute_score(input, &out);
        // スコアが更新されればキープ
        if score > new_score {
            out[d] = old;
        } else {
            score = new_score;
        }
    }
    out
}

fn cost(a: usize, b: usize) -> i64 {
    let d = b - a;
    if d == 0 {
        return 0;
    }
    (d * (d - 1) / 2) as i64
}
struct State {
    out: Vec<usize>,
    score: i64,
    ds: Vec<Vec<usize>>,
}

impl State {
    // 入ってくるd は0-index, 内部では1-indexとして扱う
    fn new(input: &Input, out: Vec<usize>) -> State {
        let mut ds = vec![vec![]; 26];
        for d in 0..input.D {
            ds[out[d] - 1].push(d + 1);
        }
        let score = compute_score(input, &out);
        State { out, score, ds }
    }
    fn change(&mut self, input: &Input, d: usize, mut new_i: usize) {
        // 今までのコンテストを消す操作
        let mut old_i = self.out[d]; // d 日目のコンテストの種類
        old_i -= 1; // 0-index で管理してる

        // old_i タイプのコンテストが old_i タイプの開催日リストの何番目にあるか
        let p = self.ds[old_i].iter().position(|x| *x == d + 1).unwrap();
        let prev = if p == 0 {
            0 // 0 日目　の意味.
        } else {
            self.ds[old_i].get(p - 1).cloned().unwrap_or(0)
        };
        let next = self.ds[old_i].get(p + 1).cloned().unwrap_or(input.D + 1);
        self.ds[old_i].remove(p);
        // perv について、p が存在していた時の p までの満足度の減少と、pからnext までの満足度の減少を消してから、
        // p がない場合の prev からnext までの満足度の減少を計算.
        // このとき、p が p+1 からnext-1 まで満足度が減少する分も考慮すべきだが、交換する日を近くのものから選択するから、
        // 大体同じコストになると推測して、その分は考えない.
        // ここで、本来は、p が消えることにより、p によるnext 以降の満足度の減少も考慮しないといけないが、近くの日のコンテストと交換すると言う条件があるため、その繰り返された後の減少も、大体同じである、と考えている.
        // この点において単に消すだけなら、スコアに大きな影響がある

        // 本来は以下の計算が正しい(ちゃんと区間を見て +-1 している)が、
        // prev, next の加減と上限を設定している都合上, -1 を入れると、cost 計算でa よりb の方が小さい値になってしまうことがある.これは、局所探索でランダムに１点をswap するため、整合性の取れないコンテストタイプが入ってくることがあるため
        // そのため、計算の厳密性は多少損なわれるが、-1 はせずに大体の値を使って計算する.
        // またd は0-index の日付だから、1-index する

        // self.score += (cost(prev + 1, d -1 + 1) + cost(d + 1 +1, next - 1) - cost(prev + 1, next - 1)) * input.c[old_i];
        self.score += (cost(prev, d + 1) + cost(d + 1, next) - cost(prev, next)) * input.c[old_i];

        new_i -= 1; // 0-index で管理してる

        // 新しいコンテストを追加する操作
        // new_i をd の位置に入れたいから、d よりも後に開催されるnew_i の位置を探す.
        // もしなかったら、d の満足度の減少は最終日まで続くから、ds[new_i].len として存在しないindex を求める.
        // 存在しないindex により、next のpos の計算で、最終日の後の日を選択できるため、最終日までの満足度の減少を計算できる.
        let p = self.ds[new_i]
            .iter()
            .position(|x| *x > d + 1)
            .unwrap_or(self.ds[new_i].len());
        let prev = if p == 0 {
            0
        } else {
            self.ds[new_i].get(p - 1).cloned().unwrap_or(0)
        };
        let next = self.ds[new_i].get(p).cloned().unwrap_or(input.D + 1);
        self.ds[new_i].insert(p, d + 1);
        self.score -= (cost(prev, d + 1) + cost(d + 1, next) - cost(prev, next)) * input.c[new_i];
        self.score += input.s[d][new_i] - input.s[d][old_i]; // コンテスト開催することによる満足度の変化について計算

        self.out[d] = new_i + 1;
    }
}

fn simulated_annealing(input: &Input) -> Vec<usize> {
    const T0: f64 = 2e3; // 2000
    const T1: f64 = 6e2; // 600
    const TL: f64 = 1.9;
    let mut rng = rand_pcg::Pcg64Mcg::new(890482);
    // 初期解はランダム
    let out = (0..input.D).map(|_| rng.gen_range(1..=26)).collect();
    let mut state = State::new(input, out);
    let mut cnt = 0;
    let mut T = T0;
    let mut best = state.score;
    let mut best_out = state.out.clone();

    let now = time::Instant::now(); // 今の時刻を取得
    let get_time = || -> f64 { now.elapsed().as_secs_f64() }; // 最初の時刻からの経過時間を計算

    loop {
        cnt += 1;
        if cnt % 100 == 0 {
            let t = get_time() / TL;
            if t >= 1.0 {
                break;
            }
            T = T0.powf(1.0 - t) * T1.powf(t);
        }
        let old_score = state.score;
        if rng.gen_bool(0.5) {
            let d = rng.gen_range(0..input.D);
            let old = state.out[d];
            state.change(input, d, rng.gen_range(1..=26));
            let p: f64 = f64::exp((state.score - old_score) as f64) / T;
            // スコアが悪化、かつ、悪化する時に採択する確率p を元に採択するかしないか判定の結果、false になった場合.
            if old_score > state.score && !rng.gen_bool(p) {
                // 新しい組合せを採択しないから、元に戻す.
                state.change(input, d, old);
            }
        } else {
            // 2点swap
            let d1 = rng.gen_range(0..input.D - 1); // 一つ目は最後以外にしておく. 二つ目で選ばれれば良い
            let d2 = rng.gen_range((d1 + 1)..(d1 + 16).min(input.D)); // 1つ目の日程と近い日程を選択
            let (a, b) = (state.out[d1], state.out[d2]);
            // 交換
            state.change(input, d1, b);
            state.change(input, d2, a);
            let p = f64::exp((state.score - old_score) as f64) / T;
            if old_score > state.score && !rng.gen_bool(p) {
                state.change(input, d1, a);
                state.change(input, d2, b);
            }
        }
        // println!("{}, {}", best, state.score);

        if best < state.score {
            // 更新
            best = state.score;
            best_out = state.out.clone();
        }
    }
    best_out
}

fn solve(input: &Input) -> Vec<usize> {
    // greedy(input) // score: 62634806
    // greedy2(input) // score: 104195466
    // hill_climbing(input) // score: 78879391
    simulated_annealing(input)
}

fn main() {
    input! {
        d: usize,
        c: [i64; 26],
        s: [[i64; 26]; d]
    }

    let input = Input { D: d, s, c };
    let out = solve(&input);
    for x in out {
        println!("{}", x);
    }
}
