use proconio::input;

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

fn solve(input: &Input) -> Vec<usize> {
    // greedy(input) // score: 62634806
    greedy2(input) // score: 104195466
}

fn main() {
    input! {
        d: usize,
        c: [i64; 26],
        s: [[i64; 26]; d]
    }

    let input = Input { D: d, s, c };
    // println!("{}", compute_score(&input, &greedy(&input)));
    for x in &solve(&input) {
        println!("{}", x);
    }
}
