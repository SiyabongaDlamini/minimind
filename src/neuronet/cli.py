"""Command line interface: ``neuronet train | predict | ask | ui``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .data import Dataset, Standardizer
from .network import NeuralNetwork


def _size(text: str) -> tuple[int, int]:
    try:
        h, w = (int(p) for p in text.lower().split("x"))
        if h < 1 or w < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected HxW such as 28x28, got {text!r}") from None
    return h, w


def _invert(text: str) -> bool | None:
    return {"auto": None, "yes": True, "no": False}[text]


def _train_and_report(
    data: Dataset,
    hidden: Sequence[int],
    args: argparse.Namespace,
    meta: dict,
    animate: str | None = None,
) -> NeuralNetwork:
    train, test = data.split(test_size=args.test_size, seed=args.seed)
    net = NeuralNetwork([data.n_features, *hidden, data.n_classes], args.activation, args.seed)
    callbacks = []
    recorder = None
    if animate:
        import matplotlib

        matplotlib.use("Agg")
        from .viz import TrainingRecorder

        recorder = TrainingRecorder(every=args.animate_every)
        callbacks.append(recorder)
    print(f"{net}  |  {len(train)} train / {len(test)} test samples")
    history = net.fit(
        train.X,
        train.y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        validation=(test.X, test.y),
        seed=args.seed,
        callbacks=callbacks,
        verbose=max(1, args.epochs // 10),
    )
    _, test_acc = net.evaluate(test.X, test.y)
    print(f"test accuracy: {test_acc:.1%}")
    net.meta = {**meta, "class_names": list(data.class_names), "test_accuracy": test_acc}
    if recorder is not None:
        from .viz import animate_training

        # Probe with a sample the trained network gets right, so the animation shows learning.
        correct = np.flatnonzero(net.predict(test.X) == test.y)
        probe = int(correct[0]) if correct.size else 0
        animate_training(
            net,
            recorder,
            history,
            test.X[probe],
            probe_label=data.class_names[int(test.y[probe])],
            image_shape=data.image_shape,
            class_names=data.class_names,
            save_path=animate,
        )
        print(f"animation saved to {animate}")
    return net


def _cmd_train(args: argparse.Namespace) -> int:
    from .parsers import ImageParser, load_csv, load_image_folder

    meta: dict = {}
    if args.images:
        parser = ImageParser(size=args.size, invert=_invert(args.invert))
        data = load_image_folder(args.images, parser)
        meta.update(image_size=list(args.size), invert=args.invert)
    else:
        data = load_csv(args.csv, target=args.target)
        scaler = Standardizer().fit(data.X)
        data = Dataset(scaler.transform(data.X), data.y, data.class_names)
        meta.update(standardizer=scaler.to_dict())
    hidden = [int(h) for h in args.hidden.split(",") if h]
    net = _train_and_report(data, hidden, args, meta, animate=args.animate)
    net.save(args.out, net.meta)
    print(f"model saved to {args.out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from .parsers import ImageParser

    net = NeuralNetwork.load(args.model)
    if "image_size" not in net.meta:
        print("this model was not trained on images", file=sys.stderr)
        return 2
    parser = ImageParser(size=tuple(net.meta["image_size"]), invert=_invert(net.meta["invert"]))
    names = net.meta["class_names"]
    for path in args.images:
        probs = net.predict_proba(parser.parse(path)[None])[0]
        top = np.argsort(-probs)[:3]
        ranked = ", ".join(f"{names[i]} {probs[i]:.0%}" for i in top)
        print(f"{path}: {names[top[0]]}  ({ranked})")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .knowledge import KnowledgeBase

    kb = KnowledgeBase()
    report = kb.add_paths(args.documents)
    skipped = [name for name, n in report.items() if n < 0]
    if skipped:
        print(f"skipped unsupported files: {', '.join(skipped)}", file=sys.stderr)
    if not len(kb):
        print("no readable documents were found", file=sys.stderr)
        return 2
    print(f"indexed {len(kb)} passages from {len(kb.sources)} document(s)")

    def respond(question: str) -> None:
        print(f"\n{kb.ask(question)}\n")

    if args.question:
        respond(args.question)
        return 0
    print("Ask a question about your documents (empty line to quit).")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        respond(question)
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("UI requires uvicorn and fastapi: pip install uvicorn fastapi python-multipart", file=sys.stderr)
        return 1
    print(f"Starting Neuronet Control Console on http://{args.host}:{args.port}")
    uvicorn.run("neuronet.ui:app", host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuronet", description="A small NumPy neural network with parsers and animations."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_training_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--lr", type=float, default=0.01)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
        p.add_argument("--test-size", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--animate", metavar="FILE.gif", help="save a training animation")
        p.add_argument("--animate-every", type=int, default=2, help="epochs per animation frame")

    train = sub.add_parser("train", help="train on your own images or CSV")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", help="folder with one subfolder of images per class")
    source.add_argument("--csv", help="CSV file with a header row")
    train.add_argument("--target", default="-1", help="CSV target column name or index")
    train.add_argument("--size", type=_size, default=(16, 16), help="image size HxW")
    train.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    train.add_argument("--hidden", default="32", help="comma-separated hidden sizes")
    train.add_argument("--out", default="model.npz")
    add_training_options(train)
    train.set_defaults(func=_cmd_train)

    predict = sub.add_parser("predict", help="classify images with a trained model")
    predict.add_argument("model")
    predict.add_argument("images", nargs="+")
    predict.set_defaults(func=_cmd_predict)

    ask = sub.add_parser("ask", help="answer questions from your documents")
    ask.add_argument("documents", nargs="+", help="files or folders (txt, md, docx, pdf, ...)")
    ask.add_argument("-q", "--question", help="ask once instead of starting a prompt")
    ask.set_defaults(func=_cmd_ask)

    ui = sub.add_parser("ui", help="launch black visualizer console")
    ui.add_argument("--host", default="127.0.0.1", help="host to bind (default: 127.0.0.1)")
    ui.add_argument("--port", type=int, default=8000, help="port to bind (default: 8000)")
    ui.set_defaults(func=_cmd_ui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "target", None) is not None and args.command == "train":
        args.target = int(args.target) if args.target.lstrip("-").isdigit() else args.target
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
