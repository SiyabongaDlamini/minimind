import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from neuronet import NeuralNetwork  # noqa: E402
from neuronet.viz import TrainingRecorder, animate_training  # noqa: E402


def test_animation_is_written_and_weights_restored(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((60, 16))
    y = (X[:, :8].sum(axis=1) > X[:, 8:].sum(axis=1)).astype(int)
    net = NeuralNetwork([16, 6, 2], seed=0)
    recorder = TrainingRecorder(every=2)
    history = net.fit(X, y, epochs=6, callbacks=[recorder])
    before = net.get_weights()

    out = tmp_path / "anim.gif"
    animate_training(net, recorder, history, X[0], image_shape=(4, 4), save_path=out, fps=4, dpi=40)

    assert out.exists() and out.stat().st_size > 1000
    assert recorder.epochs == [0, 2, 4, 6]
    for (W0, b0), (W1, b1) in zip(before, net.get_weights(), strict=True):
        assert np.array_equal(W0, W1) and np.array_equal(b0, b1)


def test_empty_recorder_is_rejected():
    net = NeuralNetwork([4, 3, 2])
    with pytest.raises(ValueError):
        animate_training(net, TrainingRecorder(), {"loss": [1.0], "accuracy": [0.5]}, np.zeros(4))
