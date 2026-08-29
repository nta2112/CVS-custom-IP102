import numpy as np

from metrics.lifelong import LifelongMetric
from metrics.openworld import OpenWorldMetric


def test_lifelong_forgetting_ignores_nan_and_empty_scores():
    metric = LifelongMetric(2)
    metric.update(0, [0.9, 0.8, np.nan, 0.7])
    metric.update(1, [0.8, 0.7, 0.6, 0.5])

    result = metric.compute_forgetting()

    assert np.isfinite(result)
    assert result >= 0.0


def test_openworld_auroc_handles_nan_scores_and_constant_labels():
    metric = OpenWorldMetric()
    is_unseen = np.array([False, False, True, True], dtype=bool)
    ood_scores = np.array([np.nan, 0.9, 0.8, np.nan], dtype=float)

    result = metric.compute_auroc(is_unseen, ood_scores)

    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


def test_openworld_no_unseen_returns_safe_values():
    metric = OpenWorldMetric()
    query_labels = np.array([[0], [1], [0]])
    k_closest = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [0, 2, 1],
    ])

    result = metric(query_labels, k_closest, {0, 1}, all_seen=True)

    assert result['R@1_U'] == 0.0
    assert result['AUROC'] == 0.5
    assert result['FPR@TPR95'] == 1.0
