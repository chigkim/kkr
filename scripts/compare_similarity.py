#!/usr/bin/env python3
'''Paired comparison of two speaker_similarity.py evaluations.

Takes the results.json files of a baseline and a candidate, one per seed, and
compares per-speaker margins (candidate minus baseline). Each speaker's seeds
are averaged first, and the 95% interval comes from resampling speakers, so
it describes evaluation variation for these two checkpoints and this
enrollment set, not training reproducibility.

The baseline and candidate must use the same seeds, speakers and real files
(run the candidate with --real-files-from on the baseline's results.json).

Verdict (is the candidate better than the baseline):
- improved: mean paired margin difference > --threshold, its interval
  excludes zero, and mean top-1 is not lower
- worse: the same in the other direction
- inconclusive: otherwise

Tier (what the candidate's own level means for the next step; thresholds
are judgment, fixed in the journal before the X4 data, J-29):
- target: top-1 >= 20 at every seed and mean margin >= 0.025
- substantial: mean margin >= 0.015 and mean top-1 >= 16
- below: otherwise

Usage:
    uv run scripts/compare_similarity.py \
        --baseline /tmp/kd_x3/sim_seed{0,1,2}/results.json \
        --candidate /tmp/kd_x4/sim_seed{0,1,2}/results.json
'''

import argparse
import json
from pathlib import Path

import numpy as np


def load(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    return {r['seed']: r for r in runs}


def main():
    parser = argparse.ArgumentParser(description='Paired comparison of speaker_similarity.py results')
    parser.add_argument('--baseline', nargs='+', required=True, help='Baseline results.json, one per seed')
    parser.add_argument('--candidate', nargs='+', required=True, help='Candidate results.json, one per seed')
    parser.add_argument('--threshold', type=float, default=0.005, help='Smallest mean margin difference that counts')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--out', default=None, help='Optional path for a JSON summary')
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    if sorted(base) != sorted(cand):
        raise SystemExit(f'Seeds differ: baseline {sorted(base)}, candidate {sorted(cand)}')
    seeds = sorted(base)
    speakers = sorted(base[seeds[0]]['margins'])
    for s in seeds:
        if sorted(base[s]['margins']) != speakers or sorted(cand[s]['margins']) != speakers:
            raise SystemExit(f'Speakers differ at seed {s}')
        if base[s]['real_files'] != cand[s]['real_files']:
            raise SystemExit(f'Real files differ at seed {s}; run the candidate with --real-files-from')
        if base[s]['texts'] != cand[s]['texts'] or base[s]['sha256']['script'] != cand[s]['sha256']['script']:
            raise SystemExit(f'Evaluation script or texts differ at seed {s}')

    # Per speaker, averaged over seeds: shape [speakers].
    diff = np.array([np.mean([cand[s]['margins'][k] - base[s]['margins'][k] for s in seeds]) for k in speakers])
    rng = np.random.default_rng(0)
    boot = diff[rng.integers(0, len(diff), size=(args.bootstrap, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean = float(diff.mean())
    top1_base = float(np.mean([base[s]['top1'] for s in seeds]))
    top1_cand = float(np.mean([cand[s]['top1'] for s in seeds]))

    if mean > args.threshold and lo > 0 and top1_cand >= top1_base:
        verdict = 'improved'
    elif mean < -args.threshold and hi < 0 and top1_cand <= top1_base:
        verdict = 'worse'
    else:
        verdict = 'inconclusive'

    def avg(runs, key):
        return float(np.mean([runs[s][key] for s in seeds]))

    def median_margin(runs):
        return float(np.median([np.mean([runs[s]['margins'][k] for s in seeds]) for k in speakers]))

    cand_margin = avg(cand, 'mean_margin')
    if min(cand[s]['top1'] for s in seeds) >= 20 and cand_margin >= 0.025:
        tier = 'target'
    elif cand_margin >= 0.015 and top1_cand >= 16:
        tier = 'substantial'
    else:
        tier = 'below'

    summary = {
        'seeds': seeds,
        'mean_margin_diff': mean,
        'ci95': [float(lo), float(hi)],
        'threshold': args.threshold,
        'top1': [top1_base, top1_cand],
        'mean_margin': [avg(base, 'mean_margin'), avg(cand, 'mean_margin')],
        'median_margin': [median_margin(base), median_margin(cand)],
        'mean_margin_real': [avg(base, 'mean_margin_real'), avg(cand, 'mean_margin_real')],
        'between_generated_centered': [avg(base, 'between_generated_centered'), avg(cand, 'between_generated_centered')],
        'generated_self_closest': [avg(base, 'generated_self_closest'), avg(cand, 'generated_self_closest')],
        'per_speaker_diff': dict(zip(speakers, diff.round(4).tolist())),
        'verdict': verdict,
        'tier': tier,
    }
    print(f'Seeds {seeds}, {len(speakers)} speakers')
    print(f'Mean paired margin difference {mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], threshold {args.threshold}')
    print(f'Top-1 {top1_base:.2f} -> {top1_cand:.2f}; mean margin {summary["mean_margin"][0]:.4f} -> {summary["mean_margin"][1]:.4f} (real {summary["mean_margin_real"][0]:.4f}); median {summary["median_margin"][0]:.4f} -> {summary["median_margin"][1]:.4f}')
    print(f'Centered between-speaker cosine {summary["between_generated_centered"][0]:.3f} -> {summary["between_generated_centered"][1]:.3f}')
    print(f'Verdict: {verdict}; candidate tier: {tier}')
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
