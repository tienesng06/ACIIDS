import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from analysis import acceptance_probability, bootstrap_maps_fast, GameSet, match_candidates_one_to_one, rerank_score_multiset, risk_coverage, risk_coverage_tie_bounds
from common import NUM_CLASSES
from train_branch import History, rank_loss
from SoccerNet.Evaluation.ActionSpotting import compute_class_scores as official_compute_class_scores
import torch


class MatchingTests(unittest.TestCase):
    def test_one_detection_cannot_match_two_events(self):
        detections = np.full((80, NUM_CLASSES), -1.0, dtype=float)
        detections[30, 0] = 0.9
        events = [(25, 0, 1), (35, 0, 0)]
        matched = match_candidates_one_to_one(detections, events, 0.1, delta=10)
        self.assertEqual(matched, {(30, 0): 1})

    def test_highest_unmatched_detection_is_selected(self):
        detections = np.full((80, NUM_CLASSES), -1.0, dtype=float)
        detections[20, 0] = 0.7
        detections[30, 0] = 0.9
        events = [(25, 0, 1)]
        matched = match_candidates_one_to_one(detections, events, 0.1, delta=10)
        self.assertEqual(matched, {(30, 0): 1})

    def test_class_and_tolerance_are_respected(self):
        detections = np.full((80, NUM_CLASSES), -1.0, dtype=float)
        detections[20, 1] = 0.9
        detections[40, 0] = 0.9
        events = [(20, 0, 1)]
        self.assertEqual(match_candidates_one_to_one(detections, events, 0.1, delta=10), {})

    def test_candidate_labels_match_official_greedy_convention(self):
        detections = np.full((80, NUM_CLASSES), -1.0, dtype=float)
        detections[[18, 25, 31, 60], 0] = [.7, .9, .8, .6]
        events = [(20, 0, 1), (30, 0, 1)]
        ours = match_candidates_one_to_one(detections, events, 0.1, delta=10)
        target = np.zeros(80, int); target[[20, 30]] = 1
        closest = GameSet._closest(target[:, None])[:, 0]
        official, _, _ = official_compute_class_scores(target, closest, detections[:, 0], delta=20)
        pred_frames = np.where(detections[:, 0] >= 0)[0]
        expected = {(int(f), 0) for f, row in zip(pred_frames, official) if row[1] == 1}
        self.assertEqual(set(ours), expected)


class TieAwareRiskTests(unittest.TestCase):
    def test_equal_scores_are_input_order_invariant(self):
        scores = np.array([0.9, 0.9, 0.5, 0.5])
        labels_a = np.array([1, 0, 1, 0])
        labels_b = np.array([0, 1, 0, 1])
        _, risk_a, aurc_a = risk_coverage(scores, labels_a)
        _, risk_b, aurc_b = risk_coverage(scores, labels_b)
        np.testing.assert_allclose(risk_a, risk_b)
        self.assertAlmostEqual(aurc_a, aurc_b)

    def test_expected_curve_lies_within_tie_extrema(self):
        scores = np.array([0.9, 0.9, 0.9, 0.4])
        labels = np.array([1, 0, 1, 0])
        _, _, expected = risk_coverage(scores, labels)
        lower, upper = risk_coverage_tie_bounds(scores, labels)
        self.assertLessEqual(lower, expected)
        self.assertLessEqual(expected, upper)

    def test_target_coverage_randomizes_boundary_tie(self):
        scores = np.array([.9, .5, .5, .5])
        keep = acceptance_probability(scores, .5)
        np.testing.assert_allclose(keep, [1, 1 / 3, 1 / 3, 1 / 3])
        self.assertAlmostEqual(float(keep.sum()), 2.0)


class TransductiveRerankingTests(unittest.TestCase):
    def test_testwide_reassignment_preserves_class_score_multiset(self):
        gs = SimpleNamespace(games=["g0"])
        probs = [np.zeros((10, NUM_CLASSES), float) for _ in range(2)]
        probs[0][2, 0] = .8; probs[1][5, 0] = .3
        rows = [{"game": "g0", "half": 1, "frame": 2, "cls": 0, "p_f": .8},
                {"game": "g0", "half": 2, "frame": 5, "cls": 0, "p_f": .3}]
        out = rerank_score_multiset(gs, probs, rows, np.array([.1, .9]))
        reassigned = [out[0][2, 0], out[1][5, 0]]
        np.testing.assert_allclose(sorted(reassigned), [.3, .8])
        np.testing.assert_allclose(reassigned, [.3, .8])


class BootstrapMapTests(unittest.TestCase):
    def test_cached_bootstrap_matches_direct_official_metric(self):
        targets, detections = [], []
        for game in range(2):
            for half in range(2):
                t = np.zeros((50, NUM_CLASSES), dtype=int)
                if half == 0: t[20 + game, 0] = 1
                d = np.full((50, NUM_CLASSES), -1.0)
                d[20 + game, 0] = .9
                d[40, 0] = .2
                targets.append(t); detections.append(d)
        gs = SimpleNamespace(games=["g0", "g1"], targets=targets, closests=[GameSet._closest(t) for t in targets])
        picks = np.array([[0, 1], [0, 0]])
        fast = bootstrap_maps_fast(gs, detections, picks)
        for j, pick in enumerate(picks):
            subset = [2 * i + h for i in pick for h in (0, 1)]
            direct = GameSet.mAP(gs, detections, "loose", subset)
            np.testing.assert_allclose([fast[0][j], fast[1][j], fast[2][j]], [direct["mAP"], direct["visible"], direct["unshown"]], atol=3e-4)


class QMFPortTests(unittest.TestCase):
    def test_equal_history_pair_has_zero_crl_loss(self):
        history = History(4)
        confidence = torch.tensor([0.1, 0.9, 0.2, 0.8])
        loss = rank_loss(confidence, np.arange(4), history, torch.device("cpu"))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
