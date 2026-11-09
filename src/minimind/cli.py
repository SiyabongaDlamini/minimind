from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .data import Dataset, Standardizer
from .network import NeuralNetwork
from .vlm import VisionLanguageModel


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


def _cmd_demo(args: argparse.Namespace) -> int:
    from .datasets import load_digits

    net = _train_and_report(load_digits(), [32], args, {}, animate=args.animate)
    if args.out:
        net.save(args.out)
        print(f"model saved to {args.out}")
    return 0


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


def _cmd_vlm_train(args: argparse.Namespace) -> int:
    vlm = VisionLanguageModel(
        image_size=args.size,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        seed=args.seed,
    )
    print(f"{vlm}  |  training on {args.images}")
    history = vlm.fit_folder(
        args.images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=max(1, args.epochs // 10),
    )
    final_align = history["alignment"][-1] if history["alignment"] else 0.0
    print(f"final multimodal alignment: {final_align:.1%}")
    vlm.save(args.out)
    print(f"vlm saved to {args.out}")
    return 0


def _cmd_zero_shot(args: argparse.Namespace) -> int:
    vlm = VisionLanguageModel.load(args.model)
    labels = args.labels or vlm.meta.get("class_names", [])
    if not labels:
        print("provide --labels or use a model trained with class_names", file=sys.stderr)
        return 2
    for path in args.images:
        scores = vlm.zero_shot_classify(path, labels, prompt_template=args.template)
        top_str = ", ".join(f"{lbl} {prob:.0%}" for lbl, prob in scores[:3])
        print(f"{path}: {scores[0][0]}  ({top_str})")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimind",
        description="Lightweight Vision-Language Model & Neural Network Engine in pure NumPy.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_training_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--lr", type=float, default=0.01)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument(
            "--activation",
            choices=["relu", "leaky_relu", "gelu", "silu", "tanh", "sigmoid"],
            default="relu",
        )
        p.add_argument("--test-size", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--animate", metavar="FILE.gif", help="save a training animation")
        p.add_argument("--animate-every", type=int, default=2, help="epochs per animation frame")

    demo = sub.add_parser("demo", help="train on the built-in digits dataset")
    add_training_options(demo)
    demo.add_argument("--out", help="save the trained model (.npz)")
    demo.set_defaults(func=_cmd_demo)

    train = sub.add_parser("train", help="train a classifier on images or CSV")
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

    vlm_train = sub.add_parser("vlm-train", help="train a contrastive vision-language model")
    vlm_train.add_argument("--images", required=True, help="folder with class subfolders")
    vlm_train.add_argument("--size", type=_size, default=(16, 16), help="image size HxW")
    vlm_train.add_argument("--hidden-dim", type=int, default=128)
    vlm_train.add_argument("--embed-dim", type=int, default=64)
    vlm_train.add_argument("--epochs", type=int, default=40)
    vlm_train.add_argument("--batch-size", type=int, default=16)
    vlm_train.add_argument("--lr", type=float, default=0.005)
    vlm_train.add_argument("--seed", type=int, default=0)
    vlm_train.add_argument("--out", default="vlm.npz")
    vlm_train.set_defaults(func=_cmd_vlm_train)

    zero_shot = sub.add_parser("zero-shot", help="zero-shot classify images via text prompts")
    zero_shot.add_argument("model", help="trained VLM (.npz)")
    zero_shot.add_argument("images", nargs="+", help="image file paths")
    zero_shot.add_argument("--labels", nargs="+", help="candidate text labels")
    zero_shot.add_argument("--template", default="a photo of a {}")
    zero_shot.set_defaults(func=_cmd_zero_shot)

    ask = sub.add_parser("ask", help="answer questions from your documents")
    ask.add_argument("documents", nargs="+", help="files or folders (txt, md, docx, pdf, ...)")
    ask.add_argument("-q", "--question", help="ask once instead of starting a prompt")
    ask.set_defaults(func=_cmd_ask)
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
