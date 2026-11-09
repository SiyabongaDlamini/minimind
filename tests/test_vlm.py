from pathlib import Path

import numpy as np
import pytest

from minimind import (
    GELU,
    InfoNCELoss,
    LayerNorm,
    MultiHeadAttention,
    VisionLanguageModel,
)


def test_layernorm_and_gelu_numerical_gradients():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((5, 8))
    ln = LayerNorm(8)
    gelu = GELU()

    out = gelu.forward(ln.forward(x))
    upstream = rng.standard_normal(out.shape)
    dx = ln.backward(gelu.backward(upstream))

    eps = 1e-6
    for idx in range(5):
        orig = x.flat[idx]
        x.flat[idx] = orig + eps
        up = float(np.sum(GELU().forward(ln.forward(x)) * upstream))
        x.flat[idx] = orig - eps
        down = float(np.sum(GELU().forward(ln.forward(x)) * upstream))
        x.flat[idx] = orig
        assert abs((up - down) / (2 * eps) - dx.flat[idx]) < 1e-5


def test_multihead_attention_numerical_gradients():
    rng = np.random.default_rng(11)
    mha = MultiHeadAttention(d_model=8, n_heads=2, rng=rng)
    x = rng.standard_normal((2, 4, 8))
    upstream = rng.standard_normal((2, 4, 8))

    out = mha.forward(x)
    dx = mha.backward(upstream)
    assert out.shape == (2, 4, 8)
    assert dx.shape == (2, 4, 8)

    eps = 1e-6
    for i in range(4):
        orig = x.flat[i]
        x.flat[i] = orig + eps
        up = float(np.sum(mha.forward(x) * upstream))
        x.flat[i] = orig - eps
        down = float(np.sum(mha.forward(x) * upstream))
        x.flat[i] = orig
        assert abs((up - down) / (2 * eps) - dx.flat[i]) < 1e-4


def test_infonce_loss_numerical_gradients():
    rng = np.random.default_rng(42)
    img_f = rng.standard_normal((6, 10))
    txt_f = rng.standard_normal((6, 10))
    labels = np.array([0, 1, 0, 1, 2, 2])

    loss_fn = InfoNCELoss(temperature=0.1)
    loss_fn.forward(img_f, txt_f, labels=labels)
    d_img, d_txt = loss_fn.backward()

    eps = 1e-6
    for i in range(6):
        orig = img_f.flat[i]
        img_f.flat[i] = orig + eps
        up = InfoNCELoss(0.1).forward(img_f, txt_f, labels=labels)
        img_f.flat[i] = orig - eps
        down = InfoNCELoss(0.1).forward(img_f, txt_f, labels=labels)
        img_f.flat[i] = orig
        assert abs((up - down) / (2 * eps) - d_img.flat[i]) < 1e-5

        orig_t = txt_f.flat[i]
        txt_f.flat[i] = orig_t + eps
        up_t = InfoNCELoss(0.1).forward(img_f, txt_f, labels=labels)
        txt_f.flat[i] = orig_t - eps
        down_t = InfoNCELoss(0.1).forward(img_f, txt_f, labels=labels)
        txt_f.flat[i] = orig_t
        assert abs((up_t - down_t) / (2 * eps) - d_txt.flat[i]) < 1e-5


def test_vlm_training_zero_shot_and_persistence(tmp_path):
    pytest.importorskip("PIL")
    data_dir = Path(__file__).resolve().parent.parent / "data" / "images"
    test_img = Path(__file__).resolve().parent.parent / "data" / "test_square.png"

    vlm = VisionLanguageModel(image_size=(16, 16), hidden_dim=64, embed_dim=32, seed=0)
    history = vlm.fit_folder(data_dir, epochs=35, batch_size=12, lr=0.008, seed=0)
    assert history["loss"][-1] < history["loss"][0]
    assert history["alignment"][-1] >= 0.90

    preds = vlm.zero_shot_classify(test_img, ["circle", "square"])
    assert preds[0][0] == "square"
    assert preds[0][1] > 0.75

    circle_img = next((data_dir / "circle").glob("*.png"))
    matches = vlm.search_images("a photo of a circle", [test_img, circle_img], top_k=2)
    assert matches[0].item == str(circle_img)

    save_file = tmp_path / "vlm_test.npz"
    vlm.save(save_file)
    loaded = VisionLanguageModel.load(save_file)
    assert np.allclose(vlm.encode_image(test_img), loaded.encode_image(test_img))
    assert np.allclose(vlm.encode_text("square shape"), loaded.encode_text("square shape"))
