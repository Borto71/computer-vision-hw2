| approach | test acc [95% CI] | macro F1 [95% CI] |
|---|---|---|
| full | 0.810 [0.797, 0.823] | 0.807 [0.794, 0.819] |
| head | 0.864 [0.853, 0.875] | 0.863 [0.851, 0.873] |
| features_logreg | 0.857 [0.846, 0.868] | 0.855 [0.844, 0.866] |
| features_linsvm | 0.838 [0.827, 0.850] | 0.834 [0.821, 0.845] |

| A vs B | acc A - B [95% CI] | only A right | only B right | McNemar p | Holm p |
|---|---|---|---|---|---|
| full vs head | -5.4 pp [-6.5, -4.3] | 132 | 330 | 1.2e-20 | 7.5e-20 |
| full vs features_logreg | -4.7 pp [-5.9, -3.6] | 144 | 317 | 5.2e-16 | 2.6e-15 |
| full vs features_linsvm | -2.8 pp [-4.0, -1.7] | 199 | 303 | 4e-06 | 1.2e-05 |
| head vs features_logreg | +0.7 pp [-0.1, +1.5] | 126 | 101 | 0.11 | 0.11 |
| head vs features_linsvm | +2.6 pp [+1.5, +3.6] | 231 | 137 | 1.1e-06 | 4.4e-06 |
| features_logreg vs features_linsvm | +1.9 pp [+1.0, +2.8] | 181 | 112 | 6.6e-05 | 0.00013 |
