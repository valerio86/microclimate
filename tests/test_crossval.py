import numpy as np
import pandas as pd
import pytest

from microclimate.analysis import backtest as bt
from tests.test_backtest import synthetic


def test_folds_never_train_on_the_future():
    data = synthetic(days=400)
    folds = list(bt.walk_forward_folds(data, n_folds=4))

    assert len(folds) == 4
    for train, test in folds:
        assert train["valid_time"].max() < test["valid_time"].min()


def test_training_window_expands_and_tests_move_forward():
    data = synthetic(days=400)
    folds = list(bt.walk_forward_folds(data, n_folds=4))

    train_sizes = [len(train) for train, _ in folds]
    test_starts = [test["valid_time"].min() for _, test in folds]

    assert train_sizes == sorted(train_sizes), "training window should expand"
    assert test_starts == sorted(test_starts), "test windows should advance"


def test_test_windows_do_not_overlap():
    data = synthetic(days=400)
    folds = list(bt.walk_forward_folds(data, n_folds=4))
    for (_, earlier), (_, later) in zip(folds, folds[1:]):
        assert earlier["valid_time"].max() < later["valid_time"].min()


def test_sealed_holdout_is_the_final_period_and_disjoint():
    data = synthetic(days=400)
    development, holdout = bt.seal_holdout(data, holdout_fraction=0.2)

    assert development["valid_time"].max() < holdout["valid_time"].min()
    assert len(development) + len(holdout) == len(data)
    assert 0.15 < len(holdout) / len(data) < 0.25


def test_holdout_is_excluded_from_every_fold():
    data = synthetic(days=400)
    development, holdout = bt.seal_holdout(data, holdout_fraction=0.2)
    boundary = holdout["valid_time"].min()

    for train, test in bt.walk_forward_folds(development, n_folds=4):
        assert train["valid_time"].max() < boundary
        assert test["valid_time"].max() < boundary


def test_cross_validate_reports_a_row_per_method_and_fold():
    data = synthetic(days=500, bias_fn=lambda f: np.full(len(f), 3.0))
    results, summary = bt.cross_validate(
        data, n_folds=3, correctors_factory=lambda: [bt.ConstantCorrector()]
    )

    assert len(results) == 3
    assert set(results["fold"]) == {0, 1, 2}
    assert summary.loc[summary.method == "constant", "skill_mean"].iloc[0] > 0.9


def test_cross_validate_exposes_fold_to_fold_variability():
    # A bias that only exists in the first half: a method fitted on it will do
    # well early and badly later. The spread must make that visible.
    def half_bias(frame):
        return np.where(frame["valid_time"] < pd.Timestamp("2024-09-01", tz="UTC"), 5.0, 0.0)

    data = synthetic(days=500, bias_fn=half_bias)
    _, summary = bt.cross_validate(
        data, n_folds=5, correctors_factory=lambda: [bt.ConstantCorrector()]
    )
    row = summary.iloc[0]

    assert row["skill_worst"] < row["skill_best"] - 0.2, "spread should be visible"


def test_correctors_are_rebuilt_each_fold():
    made = []

    def factory():
        corrector = bt.ConstantCorrector()
        made.append(corrector)
        return [corrector]

    data = synthetic(days=400, bias_fn=lambda f: np.full(len(f), 2.0))
    bt.cross_validate(data, n_folds=3, correctors_factory=factory)

    assert len(made) == 3
    assert len({id(corrector) for corrector in made}) == 3, "must not reuse fitted state"


def test_pure_noise_earns_no_skill_across_folds():
    data = synthetic(days=500, bias_fn=None, noise=3.0)
    _, summary = bt.cross_validate(
        data, n_folds=4, correctors_factory=lambda: [bt.BucketCorrector()]
    )
    assert summary["skill_mean"].iloc[0] < 0.05
