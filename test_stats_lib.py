"""Unit tests for stats_lib.py.   Run:  python -m unittest test_stats_lib -v   (or: pytest test_stats_lib.py)"""
import math
import unittest

import numpy as np
from scipy import stats

import stats_lib as S


class TestMultiplicity(unittest.TestCase):
    def test_holm_hand_example(self):
        # sorted p: .005,.01,.03,.04 -> x4,x3,x2,x1 = .02,.03,.06,.04 -> monotone: .02,.03,.06,.06
        out = S.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(out, [0.03, 0.06, 0.06, 0.02])

    def test_holm_caps_at_one_and_ignores_nan(self):
        out = S.holm([0.6, np.nan, 0.7])
        self.assertTrue(np.isnan(out[1]))
        np.testing.assert_allclose([out[0], out[2]], [1.0, 1.0])

    def test_bh_hand_example(self):
        np.testing.assert_allclose(S.bh([0.01, 0.02, 0.03, 0.04]), [0.04] * 4)
        np.testing.assert_allclose(S.bh([0.001, 0.2]), [0.002, 0.2])

    def test_add_holm_family_size(self):
        import pandas as pd
        df = pd.DataFrame({"fam": ["a", "a", "b"], "p_boot": [0.01, 0.02, 0.03]})
        r = S.add_holm(df, ["fam"])
        self.assertEqual(list(r.family_size), [2, 2, 1])
        np.testing.assert_allclose(r.p_holm, [0.02, 0.02, 0.03])


class TestBootstrap(unittest.TestCase):
    def test_known_mean(self):
        rng = np.random.default_rng(1)
        d = rng.normal(0.1, 1.0, 2000)
        m, lo, hi, p = S.paired_bootstrap(d, B=4000, seed=0)
        self.assertAlmostEqual(m, d.mean())
        self.assertLess(lo, 0.1)
        self.assertGreater(hi, 0.1)
        self.assertAlmostEqual(hi - lo, 2 * 1.96 / math.sqrt(2000), delta=0.01)
        self.assertLess(p, 0.05)

    def test_p_floor(self):
        d = np.full(200, 0.5) + np.random.default_rng(0).normal(0, 0.01, 200)
        self.assertEqual(S.paired_bootstrap(d, B=1000, seed=0)[3], 1 / 1000)

    def test_reproducible(self):
        d = np.random.default_rng(2).normal(0.02, 0.2, 300)
        self.assertEqual(S.paired_bootstrap(d, 500, 7), S.paired_bootstrap(d, 500, 7))
        self.assertEqual(S.contrast(d, 500, 7), S.contrast(d, 500, 7))

    def test_too_small(self):
        self.assertTrue(np.isnan(S.paired_bootstrap([1.0])[0]))


class TestWilcoxonEffects(unittest.TestCase):
    def test_matches_scipy_no_zeros(self):
        x = np.array([0.5, -0.2, 0.7, 0.1, 0.9, -0.05, 0.3, 0.45, 0.62, -0.31])
        self.assertAlmostEqual(S.wilcoxon_p(x), stats.wilcoxon(x, zero_method="pratt").pvalue)

    def test_zeros_and_all_zero(self):
        x = np.array([0, 0, 0, 0.2, 0.3, -0.1, 0.4, 0.5, 0.25, 0.6, 0.15, 0.35])
        p = S.wilcoxon_p(x)
        self.assertTrue(0 <= p <= 1)
        self.assertEqual(S.wilcoxon_p(np.zeros(20)), 1.0)

    def test_effects(self):
        d = np.array([0.1, 0.0, -0.1, 0.3, 0.0, 0.2])
        self.assertEqual(S.win_tie_loss(d), (3, 2, 1))
        self.assertAlmostEqual(S.cohens_d(d), d.mean() / d.std(ddof=1))
        self.assertEqual(S.helped_harmed(d, eps=0.15), (2 / 6, 0.0))


class TestPooling(unittest.TestCase):
    def test_random_effects_hand_case(self):
        r = S.random_effects([0.1, 0.3], [0.01, 0.01], hksj=False)
        self.assertAlmostEqual(r["tau2"], 0.01)
        self.assertAlmostEqual(r["mu"], 0.2)
        self.assertAlmostEqual(r["se"], 0.1)
        self.assertAlmostEqual(r["I2"], 0.5)

    def test_random_effects_homogeneous(self):
        r = S.random_effects([0.2, 0.2, 0.2], [0.01, 0.02, 0.03])
        self.assertEqual(r["tau2"], 0.0)
        self.assertAlmostEqual(r["mu"], 0.2)

    def test_macro_is_mean_of_means(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0.1, 0.1, 1000), rng.normal(0.5, 0.1, 10)
        df = S.pooled_estimates({"big": a, "small": b}, B=500, seed=0).set_index("estimator")
        self.assertAlmostEqual(df.loc["macro_average", "mean_delta"], (a.mean() + b.mean()) / 2)
        self.assertAlmostEqual(df.loc["all_query_weighted", "mean_delta"], np.concatenate([a, b]).mean())
        self.assertTrue(df.loc["macro_average", "primary"])

    def test_leave_one_out_shape(self):
        rng = np.random.default_rng(0)
        d = {k: rng.normal(0.05, 0.1, 50) for k in "abc"}
        df = S.leave_one_dataset_out(d, B=200)
        self.assertEqual(set(df.left_out), set("abc"))
        self.assertEqual(len(df), 9)


class TestSeedsAndMetrics(unittest.TestCase):
    def test_seed_variability(self):
        rng = np.random.default_rng(0)
        same = np.repeat(rng.random((100, 1)), 3, axis=1)
        r = S.seed_variability({"x": same}, B=200).iloc[0]
        self.assertAlmostEqual(r.mean_per_query_sd, 0.0)
        self.assertAlmostEqual(r.sd_of_seed_means, 0.0)
        noisy = same + rng.normal(0, 0.2, same.shape)
        r2 = S.seed_variability({"x": noisy}, base=same[:, 0], B=200).iloc[0]
        self.assertGreater(r2.mean_per_query_sd, 0.1)
        self.assertGreaterEqual(r2.ci_qs_hi - r2.ci_qs_lo, 0)

    def test_ndcg_judged_only_hand_case(self):
        qrels = {"a": 1, "b": 0, "c": 2}
        run = ["x", "a", "b", "c"]  # x unjudged
        ideal = 2 / math.log2(2) + 1 / math.log2(3)
        plain = (1 / math.log2(3) + 2 / math.log2(5)) / ideal
        cond = (1 / math.log2(2) + 0 + 2 / math.log2(4)) / ideal
        self.assertAlmostEqual(S.ndcg_at_k(run, qrels), plain)
        self.assertAlmostEqual(S.ndcg_at_k(run, qrels, judged_only=True), cond)
        self.assertAlmostEqual(S.judged_at_k(run, qrels, k=4), 0.75)
        self.assertEqual(S.ndcg_at_k(run, {}), 0.0)


class TestPowerNull(unittest.TestCase):
    def test_mde_formula(self):
        self.assertAlmostEqual(S.mde(1.0, 100), (1.959964 + 0.841621) / 10, places=5)

    def test_power_row(self):
        d = np.random.default_rng(0).normal(0.0, 0.1, 50)
        r = S.power_row(d)
        self.assertAlmostEqual(r["mde_80"], S.mde(r["sd_delta"], 50))

    def test_null_rejection_rate_about_alpha(self):
        d = np.random.default_rng(3).normal(0.0, 0.2, 100)
        r = S.sign_flip_null(d, reps=300, B=300, seed=1)
        self.assertTrue(0.01 <= r["reject_rate_wilcoxon"] <= 0.10, r)
        self.assertTrue(0.01 <= r["reject_rate_boot"] <= 0.10, r)


if __name__ == "__main__":
    unittest.main()