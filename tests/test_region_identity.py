import numpy as np
from probes.region_identity import region_identity_from_hashes


def test_cell_stays_together_scores_one():
    base = np.array([0, 0, 0, 1, 1, 1])
    # one ensemble member, patterns change but cells stay intact
    pert = [np.array([7, 7, 7, 9, 9, 9])]
    scores, cell_ids = region_identity_from_hashes(base, pert, min_mass=3)
    assert np.allclose(scores, 1.0)
    assert set(cell_ids.tolist()) == {0, 1}


def test_point_splitting_off_scores_zero():
    base = np.array([0, 0, 0, 1, 1, 1])
    # point 0 splits away from its cell; points 1,2 keep each other
    pert = [np.array([5, 7, 7, 9, 9, 9])]
    scores, _ = region_identity_from_hashes(base, pert, min_mass=3)
    assert scores[0] == 0.0
    assert np.isclose(scores[1], 0.5)  # keeps 1 of 2 co-members
    assert np.isclose(scores[2], 0.5)
    assert np.allclose(scores[3:], 1.0)


def test_below_min_mass_is_nan():
    base = np.array([0, 0, 0, 1, 1])
    pert = [np.array([0, 0, 0, 1, 1])]
    scores, cell_ids = region_identity_from_hashes(base, pert, min_mass=3)
    assert np.isnan(scores[3]) and np.isnan(scores[4])
    assert cell_ids[3] == -1 and cell_ids[4] == -1


def test_mean_over_ensemble():
    base = np.array([0, 0, 0])
    pert = [np.array([1, 1, 1]),   # together -> 1.0 each
            np.array([1, 2, 2])]   # point 0 alone -> 0.0; 1,2 keep 1 of 2 co-members -> 0.5
    scores, _ = region_identity_from_hashes(base, pert, min_mass=3)
    # pt0: mean(1.0, 0.0) = 0.5
    # pt1: mean(1.0, 0.5) = 0.75  (pert[1]: co=2 of |C|=3 → (2-1)/(3-1)=0.5)
    # pt2: same as pt1 = 0.75
    assert np.isclose(scores[0], 0.5)
    assert np.isclose(scores[1], 0.75)
    assert np.isclose(scores[2], 0.75)


def test_object_hashes_supported():
    # hash_patterns returns object arrays of tuples for >64 gates
    base = np.empty(4, dtype=object)
    base[:] = [(1, 2), (1, 2), (3, 4), (3, 4)]
    pert_arr = np.empty(4, dtype=object)
    pert_arr[:] = [(9, 9), (9, 9), (8, 8), (7, 7)]
    scores, _ = region_identity_from_hashes(base, [pert_arr], min_mass=2)
    assert np.allclose(scores[:2], 1.0)
    assert np.allclose(scores[2:], 0.0)
