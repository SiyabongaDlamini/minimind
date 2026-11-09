import numpy as np
import pytest

from neuronet import NeuralNetwork
from neuronet.losses import SoftmaxCrossEntropy, softmax


def test_softmax_rows_sum_to_one_and_are_stable():
    p = softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(p).all()


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_backprop_matches_numerical_gradient(activation):
    rng = np.random.default_rng(1)
    net = NeuralNetwork([5, 7, 4, 3], activation=activation, seed=2)
    X = rng.standard_normal((6, 5))
    y = rng.integers(0, 3, size=6)

    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(net.forward(X, training=True), y)
    grad = loss_fn.backward()
    for layer in reversed(net.layers):
        grad = layer.backward(grad)

    eps = 1e-6
    for layer in net._dense_layers():
        for param, analytic in layer.parameters():
            flat, flat_grad = param.reshape(-1), analytic.reshape(-1)
            for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
                original = flat[i]
                flat[i] = original + eps
                up = SoftmaxCrossEntropy().forward(net.forward(X), y)
                flat[i] = original - eps
                down = SoftmaxCrossEntropy().forward(net.forward(X), y)
                flat[i] = original
                numeric = (up - down) / (2 * eps)
                assert abs(numeric - flat_grad[i]) < 1e-5


def test_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    y = np.array([0, 1, 1, 0] * 25)
    net = NeuralNetwork([2, 8, 2], activation="tanh", seed=0)
    net.fit(X, y, epochs=200, batch_size=16, lr=0.05)
    assert net.evaluate(X, y)[1] == 1.0


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_loss_decreases(optimizer):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    net = NeuralNetwork([4, 8, 2], seed=0)
    history = net.fit(X, y, epochs=30, optimizer=optimizer, lr=0.05 if optimizer == "adam" else 0.2)
    assert history["loss"][-1] < history["loss"][0] * 0.6


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    net = NeuralNetwork([4, 6, 3], activation="tanh", seed=3)
    net.save(tmp_path / "m.npz", meta={"class_names": ["a", "b", "c"]})
    loaded = NeuralNetwork.load(tmp_path / "m.npz")
    assert loaded.layer_sizes == (4, 6, 3)
    assert loaded.activation == "tanh"
    assert loaded.meta["class_names"] == ["a", "b", "c"]
    assert np.allclose(net.predict_proba(X), loaded.predict_proba(X))


def test_callback_can_stop_training_and_sees_epoch_zero():
    seen = []

    def stop_at_three(epoch, total, net, history):
        seen.append(epoch)
        return epoch >= 3

    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((20, 3)), rng.integers(0, 2, 20)
    history = NeuralNetwork([3, 4, 2]).fit(X, y, epochs=50, callbacks=[stop_at_three])
    assert seen == [0, 1, 2, 3]
    assert len(history["loss"]) == 3


def test_input_validation():
    net = NeuralNetwork([3, 4, 2])
    with pytest.raises(ValueError):
        net.predict(np.zeros((2, 5)))
    with pytest.raises(ValueError):
        net.fit(np.zeros((4, 3)), np.array([0, 1, 2, 0]), epochs=1)
    with pytest.raises(ValueError):
        NeuralNetwork([3])
    with pytest.raises(ValueError):
        NeuralNetwork([3, 2], activation="nope")
