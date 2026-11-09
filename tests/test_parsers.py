import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from minimind import ImageParser, load_csv, load_image_folder  # noqa: E402


def make_image(path, shape="square", ink=0, paper=255, size=64):
    img = Image.new("L", (size, size), paper)
    draw = ImageDraw.Draw(img)
    if shape == "square":
        draw.rectangle([20, 20, 44, 44], fill=ink)
    else:
        draw.ellipse([12, 20, 52, 44], fill=ink)
    img.save(path)


def test_dark_ink_on_paper_becomes_bright_ink(tmp_path):
    make_image(tmp_path / "a.png", ink=0, paper=255)
    vec = ImageParser(size=(8, 8)).parse(tmp_path / "a.png")
    assert vec.shape == (64,)
    assert vec.min() >= 0.0 and vec.max() <= 1.0
    centre = vec.reshape(8, 8)[3:5, 3:5].mean()
    corner = vec.reshape(8, 8)[0, 0]
    assert centre > 0.8
    assert corner < 0.5 < centre


def test_light_on_dark_gives_same_vector(tmp_path):
    make_image(tmp_path / "dark_on_light.png", ink=0, paper=255)
    make_image(tmp_path / "light_on_dark.png", ink=255, paper=0)
    parser = ImageParser(size=(8, 8))
    a = parser.parse(tmp_path / "dark_on_light.png")
    b = parser.parse(tmp_path / "light_on_dark.png")
    assert np.allclose(a, b, atol=0.05)


def test_accepts_arrays_and_rgba(tmp_path):
    parser = ImageParser(size=(4, 6))
    arr = np.zeros((20, 30), dtype=np.uint8)
    arr[5:15, 8:22] = 255
    assert parser.parse(arr).shape == (24,)
    rgba = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).rectangle([5, 5, 15, 15], fill=(0, 0, 0, 255))
    assert parser.parse(rgba).shape == (24,)


def test_blank_image_does_not_crash(tmp_path):
    Image.new("L", (10, 10), 255).save(tmp_path / "blank.png")
    vec = ImageParser(size=(8, 8)).parse(tmp_path / "blank.png")
    assert np.allclose(vec, 0.0)


def test_image_folder_trains_a_classifier(tmp_path):
    for i in range(12):
        for name in ("square", "ellipse"):
            folder = tmp_path / name
            folder.mkdir(exist_ok=True)
            make_image(folder / f"{i}.png", shape=name, ink=(i * 7) % 60)
    data = load_image_folder(tmp_path, ImageParser(size=(12, 12)))
    assert data.class_names == ("ellipse", "square")
    assert len(data) == 24 and data.image_shape == (12, 12)

    from minimind import NeuralNetwork

    train, test = data.split(test_size=0.25)
    net = NeuralNetwork([144, 16, 2], seed=0)
    net.fit(train.X, train.y, epochs=60, lr=0.02)
    assert net.evaluate(test.X, test.y)[1] >= 0.9


def test_image_folder_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_folder(tmp_path / "missing")
    (tmp_path / "only_one").mkdir()
    with pytest.raises(ValueError):
        load_image_folder(tmp_path)


def test_load_csv_maps_text_labels(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n")
    data = load_csv(path, target="label")
    assert data.class_names == ("no", "yes")
    assert data.X.shape == (3, 2)
    assert data.y.tolist() == [1, 0, 1]
    assert load_csv(path).class_names == ("no", "yes")  # default: last column


def test_load_csv_rejects_non_numeric_features(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,label\nx,1\ny,0\n")
    with pytest.raises(ValueError, match="non-numeric"):
        load_csv(path)
