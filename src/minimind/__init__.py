"""Minimind: a small, readable neural network in pure NumPy."""

from .data import Dataset, Standardizer
from .knowledge import Answer, KnowledgeBase
from .network import NeuralNetwork
from .optim import SGD, Adam
from .parsers import ImageParser, load_csv, load_image_folder

__version__ = "0.1.0"

__all__ = [
    "SGD",
    "Adam",
    "Answer",
    "Dataset",
    "ImageParser",
    "KnowledgeBase",
    "NeuralNetwork",
    "Standardizer",
    "__version__",
    "load_csv",
    "load_image_folder",
]
