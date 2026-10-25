"""Fast deterministic checks; these are not evidence of hardware validation."""
import numpy as np
import torch
from physym.paths import CHECKPOINTS_DIR, REFERENCE_DIR
from physym.run_experiments import make_case, symp
from physym.physnet import PhysicsGNN, q_from_pc, predict
from physym.eval_transfer import edge_index, init_noise, errors, qtrue
from physym.eval_physics import batched_policy


def test_symplectic_antisymmetry_and_shapes():
    rng = np.random.default_rng(17)
    v = rng.integers(0, 2, size=(12, 8), dtype=np.uint8)
    w = symp(v, v, 4)
    assert np.array_equal(w, w.T)
    assert not np.any(np.diag(w))
    faults, _, tests, synd, *_ = make_case(4, 11, 0.06)
    assert synd.shape == (len(faults), len(tests))
    assert np.array_equal(synd, symp(faults, tests, 4))


def test_character_map_matches_independent_edge_bernoulli_process():
    rng = np.random.default_rng(42)
    n = 4
    p = np.array([.021, .07, .045, .08], dtype='float32')
    ei = edge_index(n)
    c = np.linspace(.01, .10, len(ei), dtype='float32')
    masks = np.array([[(a >> bit) & 1 for bit in range(n)] for a in range(1, 2**n)], dtype='uint8')
    qt = qtrue(p, c, ei, masks)
    tq = q_from_pc(torch.from_numpy(p)[None], torch.from_numpy(c)[None], torch.from_numpy(masks.astype('float32')), ei)
    assert np.allclose(qt, tq.numpy().flatten(), atol=2e-6)
    x = errors(rng, p, c, ei, 50000)
    qb = ((x.astype('int16') @ masks.T.astype('int16')) % 2).mean(0)
    assert np.max(np.abs(qb-qt)) < .013


def test_graph_checkpoint_on_larger_qubit_graph():
    ck = torch.load(CHECKPOINTS_DIR / 'physics_gnn.pt', map_location='cpu', weights_only=True)
    model = PhysicsGNN()
    model.load_state_dict(ck['state'])
    rng = np.random.default_rng(192)
    p, c, ei = init_noise(10, rng, 'correlated')
    err = errors(rng, p, c, ei, 48)
    masks = rng.integers(0, 2, size=(28, 10), dtype='uint8')
    pred = predict(model, err, masks, ei)
    assert pred.shape == (28,)
    assert np.isfinite(pred).all()
    assert np.logical_and(pred > 0, pred < .5).all()


def test_controller_matched_episode():
    rng = np.random.default_rng(11)
    S = rng.integers(0, 2, size=(13, 19), dtype='uint8')
    q = np.full(19, .11)
    labels = rng.integers(0, 13, size=20)
    shots = rng.random((20, 8))
    loc = rng.integers(0, 19, size=(20, 8))
    a, e = batched_policy(S, q, q, shots, labels, loc)
    assert len(a) == 2
    assert 0 <= a[-1] <= 1 and 0 <= e[-1] <= np.log(13)


def test_reference_datasets_present_and_nonempty():
    import pandas as pd
    for name in ['physics_results.csv', 'ensemble_results.csv', 'transfer_results.csv']:
        df = pd.read_csv(REFERENCE_DIR/name)
        assert not df.empty
        assert {'method','budget','acc'}.issubset(df.columns)
