import numpy as np
import pytest
from scipy.stats import ranksums

from pycistarget.motif_enrichment_dem import (
    mean_axis1,
    get_log2_fc,
    ranksums_numba_multiple,
    p_adjust_bh,
    get_optimal_threshold_roc,
)


def test_mean_axis1_matches_numpy():
    rng = np.random.RandomState(0)
    arr = rng.rand(20, 50)
    np.testing.assert_allclose(mean_axis1(arr), arr.mean(axis=1))


def test_get_log2_fc_matches_numpy_formula():
    rng = np.random.RandomState(1)
    fg = rng.rand(15, 40)
    bg = rng.rand(15, 30)
    expected = np.log2(
        (fg.mean(axis=1) + 1e-12) / (bg.mean(axis=1) + 1e-12)
    )
    np.testing.assert_allclose(get_log2_fc(fg, bg), expected)


def test_get_log2_fc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        get_log2_fc(np.zeros((3, 4)), np.zeros((2, 4)))


def test_ranksums_numba_matches_scipy():
    rng = np.random.RandomState(2)
    X = rng.rand(30, 200)
    Y = rng.rand(30, 180)
    z, p = ranksums_numba_multiple(X, Y)
    for i in range(X.shape[0]):
        z_ref, p_ref = ranksums(X[i], Y[i])
        np.testing.assert_allclose(z[i], z_ref, atol=1e-9)
        np.testing.assert_allclose(p[i], p_ref, atol=1e-9)


def test_p_adjust_bh_matches_scipy():
    fdc = pytest.importorskip("scipy.stats").false_discovery_control
    rng = np.random.RandomState(3)
    p = rng.rand(100)
    np.testing.assert_allclose(p_adjust_bh(p), fdc(p, method="bh"))


def test_p_adjust_bh_bounds_and_monotonicity():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    q = p_adjust_bh(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    # BH-adjusted values are monotone non-decreasing in p-value rank order.
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_get_optimal_threshold_roc_separates_classes():
    fg = np.array([0.8, 0.9, 1.0, 0.95])
    bg = np.array([0.1, 0.2, 0.05, 0.15])
    thr = get_optimal_threshold_roc(fg, bg)
    assert bg.max() < thr <= fg.min()
