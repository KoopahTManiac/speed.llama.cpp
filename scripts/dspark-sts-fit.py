#!/usr/bin/env python3
"""Fit DSpark Sequential Temperature Scaling (paper 3.2.1) from serving data.

Input: a --spec-conf-log file - lines of "<accepted> <c1> <c2> ... <ck>",
one per drafted round: the block's raw per-position confidences and the
round's realized accepted draft length (bonus token excluded).

Position k's survival label is 1 iff accepted > k (the chain reached and
accepted position k). STS calibrates the CUMULATIVE product left to right:
at each position, a 1D grid search picks the temperature minimizing the
Expected Calibration Error of prod_{i<=k} sigmoid(logit(c_i)/T_i) against
the empirical survival rate, keeping earlier positions' temperatures fixed
(order-preserving, per the paper).

Usage: dspark-sts-fit.py <conf.log> [--bins 15]
Prints the fitted temperatures as a --spec-sts argument.
"""
import math
import sys


def ece(pred, label, bins):
    lo = sorted(zip(pred, label))
    n = len(lo)
    total = 0.0
    for b in range(bins):
        seg = lo[b * n // bins:(b + 1) * n // bins]
        if not seg:
            continue
        p = sum(x for x, _ in seg) / len(seg)
        y = sum(y for _, y in seg) / len(seg)
        total += len(seg) / n * abs(p - y)
    return total


def main():
    path = sys.argv[1]
    bins = int(sys.argv[sys.argv.index("--bins") + 1]) if "--bins" in sys.argv else 15

    rounds = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 2:
            continue
        rounds.append((int(parts[0]), [float(x) for x in parts[1:]]))
    if not rounds:
        sys.exit("no usable rounds in " + path)

    k_max = max(len(c) for _, c in rounds)
    print(f"# {len(rounds)} rounds, block depth {k_max}", file=sys.stderr)

    grid = [0.05 * 1.15 ** i for i in range(48)]  # 0.05 .. ~40, log-spaced
    eps = 1e-6

    def z(c):
        c = min(max(c, eps), 1.0 - eps)
        return math.log(c / (1.0 - c))

    temps = []
    # cumulative calibrated product per round, positions fitted so far
    cum = [1.0] * len(rounds)
    for k in range(k_max):
        usable = [(ri, acc, conf[k]) for ri, (acc, conf) in enumerate(rounds) if len(conf) > k]
        if len(usable) < 50:
            print(f"# position {k}: only {len(usable)} samples - broadcasting last temp", file=sys.stderr)
            break
        labels = [1.0 if acc > k else 0.0 for _, acc, _ in usable]
        best_T, best_e = 1.0, float("inf")
        for T in grid:
            pred = [cum[ri] * (1.0 / (1.0 + math.exp(-z(c) / T))) for ri, _, c in usable]
            e = ece(pred, labels, bins)
            if e < best_e:
                best_e, best_T = e, T
        temps.append(best_T)
        for ri, _, c in usable:
            cum[ri] *= 1.0 / (1.0 + math.exp(-z(c) / best_T))
        rate = sum(labels) / len(labels)
        print(f"# position {k}: T={best_T:.3f} ece={best_e:.4f} survival={rate:.3f} n={len(usable)}", file=sys.stderr)

    print("--spec-sts " + ",".join(f"{t:.3f}" for t in temps))


if __name__ == "__main__":
    main()
