# Minimind

A small, readable neural network in **pure NumPy** — with parsers that turn images and
CSV files into training data, a live **matplotlib animation** of the network learning, and
a document question-answering assistant that answers from *your own files*.

![Training animation](docs/demo.gif)

No deep-learning framework is involved: forward pass, backpropagation, softmax
cross-entropy, SGD and Adam are all implemented in a few hundred lines you can read in an
afternoon.

## Features

- **Neural network** — fully connected layers, ReLU / tanh / sigmoid, softmax cross-entropy,
  SGD (with momentum) and Adam, mini-batches, validation, early stopping via callbacks,
  save / load to a single `.npz` file (no pickle).
- **Parsers** — turn scans and photos into feature vectors (`ImageParser`), load a folder of
  images with one subfolder per class (`load_image_folder`), or load a CSV (`load_csv`).
- **Animation** — watch weights, neuron activations, output probabilities and the loss curve
  evolve, and export to GIF.
- **Ask your documents** — index `.txt`, `.md`, `.docx`, `.pdf`, `.csv` files and get answers
  with the sources they came from (`KnowledgeBase`, `minimind ask`).
- **Tested** — numerical gradient checks, learning tests, parser and Q&A tests, CI on
  Python 3.10–3.13.

## Install

```bash
git clone <your-repo-url> && cd minimind
pip install -e ".[all]"        # everything
# or pick extras:  .[image]  .[viz]  .[docs]  .[demo]  .[dev]
```

The core library only needs NumPy.

## Quick start

```python
from minimind import NeuralNetwork
from minimind.datasets import load_digits

data = load_digits()                      # 8x8 handwritten digits
train, test = data.split(test_size=0.2)

net = NeuralNetwork([64, 32, 10], activation="relu")
net.fit(train.X, train.y, epochs=60, batch_size=32, validation=(test.X, test.y), verbose=10)

loss, accuracy = net.evaluate(test.X, test.y)
print(f"test accuracy: {accuracy:.1%}")
net.save("digits.npz")
```

Or from the command line:

```bash
minimind demo --epochs 60 --animate docs/demo.gif
```

## Train on your own images

Put images in one folder per class:

```
my_data/
├── cat/   1.png 2.jpg ...
└── dog/   1.png 2.jpg ...
```

```bash
minimind train --images my_data --size 16x16 --hidden 64,32 --epochs 80 \
               --out model.npz --animate training.gif
minimind predict model.npz some_new_photo.jpg
```

`ImageParser` handles the messy parts of real scans: transparency, colour → grayscale,
dark-ink-on-white-paper → bright-ink convention, contrast stretching, cropping to the
content and resizing to a fixed shape.

```python
from minimind import ImageParser
vector = ImageParser(size=(28, 28)).parse("scan.png")   # flat float vector in [0, 1]
```

CSV works the same way (`minimind train --csv data.csv --target label`).

## Ask your documents

```bash
minimind ask reports/ notes.md contract.pdf -q "What is the notice period?"
minimind ask reports/            # interactive prompt
```

```python
from minimind import KnowledgeBase

kb = KnowledgeBase()
kb.add_paths(["reports/", "notes.md"])
print(kb.ask("What is the notice period?"))
```

**What this is, and what it is not.** `KnowledgeBase` is *retrieval*: it splits your
documents into passages, scores them against the question with TF-IDF vectors (NumPy), and
returns the most relevant sentences plus their sources. It never invents text; if nothing
matches it tells you so. It does not reason across documents, do arithmetic, or write new
prose — that needs a large language model. Scanned images and image-only PDFs have no text
layer, so run OCR (for example Tesseract) first and add the resulting text.

## Project layout

```
src/minimind/
├── layers.py      Dense, ReLU, Tanh, Sigmoid (forward + backward)
├── losses.py      softmax, SoftmaxCrossEntropy
├── optim.py       SGD, Adam
├── network.py     NeuralNetwork: fit / predict / evaluate / save / load
├── data.py        Dataset (stratified split), Standardizer
├── parsers.py     ImageParser, load_image_folder, load_csv
├── knowledge.py   KnowledgeBase: document ingestion and question answering
├── viz.py         TrainingRecorder, animate_training
├── datasets.py    built-in digits dataset
└── cli.py         minimind demo | train | predict | ask
tests/             pytest suite (includes numerical gradient checks)
```

## How it learns

For a batch with inputs `X` and one-hot labels `Y`:

1. forward: `h = relu(X @ W1 + b1)`, `p = softmax(h @ W2 + b2)`
2. loss: `-mean(log p[correct class])`
3. backward: output gradient is `(p - Y) / n`; it flows back through each layer with the chain
   rule (`dW = xᵀ · grad`, `grad_in = grad · Wᵀ`, masked by the ReLU)
4. update: the optimizer moves every weight against its gradient

`tests/test_network.py` checks step 3 against finite differences.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## Contributing

Issues and pull requests are welcome. Please keep the core dependency list to NumPy, add a
test for every change, and make sure `ruff check .` and `pytest` pass.

## License

MIT — see [LICENSE](LICENSE).
