from pathlib import Path
from neuronet import NeuralNetwork, load_csv, load_image_folder, ImageParser, KnowledgeBase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def test_real_csv_pipeline():
    csv_file = DATA_DIR / "iris.csv"
    assert csv_file.exists(), "real CSV file should exist"
    data = load_csv(csv_file, target="species")
    assert len(data) == 150
    assert data.n_features == 4
    assert len(data.class_names) == 3

    train, test = data.split(test_size=0.2, seed=42)
    net = NeuralNetwork([4, 16, 3], activation="relu", seed=42)
    net.fit(train.X, train.y, epochs=30, batch_size=16)
    _, acc = net.evaluate(test.X, test.y)
    assert acc > 0.85

def test_real_images_pipeline():
    images_dir = DATA_DIR / "images"
    assert images_dir.exists(), "real image folder should exist"
    data = load_image_folder(images_dir, ImageParser(size=(16, 16)))
    assert len(data) == 50
    assert data.class_names == ("circle", "square")

    train, test = data.split(test_size=0.2, seed=42)
    net = NeuralNetwork([256, 16, 2], activation="relu", seed=42)
    net.fit(train.X, train.y, epochs=25, batch_size=8)
    _, acc = net.evaluate(test.X, test.y)
    assert acc >= 0.90

def test_real_docs_knowledge_base():
    doc_path = DATA_DIR / "docs" / "handbook.md"
    assert doc_path.exists(), "real handbook document should exist"
    kb = KnowledgeBase()
    kb.add_paths([doc_path])
    
    ans = kb.ask("When are deployments scheduled?")
    assert ans.found
    assert "Tuesday" in ans.text

