"""Matplotlib animation of a network learning (requires the ``viz`` extra).

Record training with :class:`TrainingRecorder`, then render it with
:func:`animate_training`. The animation shows the network's weights changing (orange =
positive, blue = negative), the activations for one probe sample, the output
probabilities and the loss / accuracy curves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .network import History, NeuralNetwork

Array = np.ndarray

BG, PANEL = "#0b0b0d", "#121216"
TEXT, DIM = "#d4d4d8", "#6b6b73"
ACCENT, NEGATIVE, GOOD = "#e8944a", "#5aa9e6", "#4ade80"


@dataclass
class TrainingRecorder:
    """Training callback that snapshots the weights for later animation.

    Args:
        every: Record one frame every ``every`` epochs (epoch 0 and the last epoch are
            always recorded).
    """

    every: int = 1
    epochs: list[int] = field(default_factory=list)
    snapshots: list[list[tuple[Array, Array]]] = field(default_factory=list)

    def __call__(self, epoch: int, total: int, net: NeuralNetwork, history: History) -> None:
        if epoch % self.every == 0 or epoch == total:
            self.epochs.append(epoch)
            self.snapshots.append(net.get_weights())


def _mix(t: Array, low: str, high: str) -> Array:
    from matplotlib.colors import to_rgba

    lo, hi = np.array(to_rgba(low)), np.array(to_rgba(high))
    return lo + (hi - lo) * np.clip(t, 0, 1)[:, None]


def animate_training(
    net: NeuralNetwork,
    recorder: TrainingRecorder,
    history: History,
    probe: Array,
    *,
    probe_label: str | None = None,
    image_shape: tuple[int, int] | None = None,
    class_names: Sequence[str] | None = None,
    title: str = "Minimind",
    save_path: str | Path | None = None,
    fps: int = 12,
    max_nodes: int = 16,
    dpi: int = 80,
):
    """Build (and optionally save) the training animation.

    Args:
        net: The trained network; used as scratch space. Its weights are restored after
            saving; if you only show the animation, snapshot with ``net.get_weights()`` first.
        recorder: A recorder that was passed to ``fit(callbacks=[recorder])``.
        history: The history returned by ``fit``.
        probe: One feature vector to push through the network in every frame.
        probe_label: Optional true label shown above the probe.
        image_shape: ``(h, w)`` to display the probe as an image.
        class_names: Names for the output neurons.
        save_path: ``.gif`` (uses Pillow) or any format matplotlib can write.
        fps: Frames per second.
        max_nodes: Maximum neurons drawn per layer (evenly sampled for wider layers).
        dpi: Output resolution.

    Returns:
        The ``matplotlib.animation.FuncAnimation``. Keep a reference to it if you show it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    if not recorder.snapshots:
        raise ValueError("the recorder is empty; pass it to fit(callbacks=[recorder])")
    from matplotlib.colors import to_rgba

    POS_RGBA, NEG_RGBA = np.array(to_rgba(ACCENT)), np.array(to_rgba(NEGATIVE))
    probe = np.asarray(probe, dtype=float).ravel()
    sizes = net.layer_sizes
    names = list(class_names) if class_names else [str(i) for i in range(sizes[-1])]
    total_epochs = recorder.epochs[-1]
    original = net.get_weights()

    fig = plt.figure(figsize=(11, 6), facecolor=BG)
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1, 3, 1.5], height_ratios=[3, 1.5], wspace=0.12, hspace=0.3,
        left=0.04, right=0.97, top=0.9, bottom=0.08,
    )
    ax_img, ax_net, ax_out = (fig.add_subplot(grid[0, i]) for i in range(3))
    ax_curve = fig.add_subplot(grid[1, :])
    for ax in (ax_img, ax_net, ax_out, ax_curve):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color("#26262c")
    fig.suptitle(title, color=TEXT, fontsize=14, fontweight="bold", x=0.04, ha="left")

    # --- probe image
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    if image_shape and probe.size == image_shape[0] * image_shape[1]:
        ax_img.imshow(probe.reshape(image_shape), cmap="gray", vmin=0, vmax=1)
    else:
        ax_img.text(0.5, 0.5, "probe\ninput", ha="center", va="center", color=DIM)
    ax_img.set_title(
        f"probe · label {probe_label}" if probe_label else "probe", color=DIM, fontsize=9
    )

    # --- network geometry (fixed); colours are updated every frame
    shown = [np.unique(np.linspace(0, s - 1, min(s, max_nodes)).round().astype(int)) for s in sizes]
    xs = np.linspace(0.05, 0.95, len(sizes))
    node_xy, layer_slices, start = [], [], 0
    for x, idx in zip(xs, shown, strict=True):
        ys = np.linspace(0.04, 0.96, len(idx)) if len(idx) > 1 else np.array([0.5])
        node_xy.extend(zip([x] * len(idx), ys, strict=True))
        layer_slices.append(slice(start, start + len(idx)))
        start += len(idx)
    node_xy = np.array(node_xy)
    segments, edge_slices, e = [], [], 0
    for li in range(len(sizes) - 1):
        for i in range(len(shown[li])):
            for j in range(len(shown[li + 1])):
                segments.append([node_xy[layer_slices[li]][i], node_xy[layer_slices[li + 1]][j]])
        count = len(shown[li]) * len(shown[li + 1])
        edge_slices.append(slice(e, e + count))
        e += count
    edges = LineCollection(segments, linewidths=0.6)
    ax_net.add_collection(edges)
    nodes = ax_net.scatter(node_xy[:, 0], node_xy[:, 1], s=34, edgecolors="#3a3a42", linewidths=0.8)
    ax_net.set_xlim(0, 1)
    ax_net.set_ylim(-0.04, 1.08)
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    for x, size in zip(xs, sizes, strict=True):
        ax_net.text(x, -0.02, str(size), ha="center", va="top", color=DIM, fontsize=8)
    hud = ax_net.text(0.01, 1.02, "", color=ACCENT, fontsize=9, family="monospace")

    # --- output probabilities
    bars = ax_out.barh(range(len(names)), np.zeros(len(names)), color=DIM)
    ax_out.set_xlim(0, 1)
    ax_out.set_ylim(len(names) - 0.4, -0.6)
    ax_out.set_yticks(range(len(names)))
    ax_out.set_yticklabels(names, color=TEXT, fontsize=8)
    ax_out.tick_params(axis="x", colors=DIM, labelsize=7)
    ax_out.tick_params(axis="y", length=0)
    ax_out.set_title("output probability", color=DIM, fontsize=9)

    # --- curves
    epochs_axis = np.arange(1, len(history["loss"]) + 1)
    (loss_line,) = ax_curve.plot([], [], color=ACCENT, lw=1.6, label="loss")
    ax_acc = ax_curve.twinx()
    (acc_line,) = ax_acc.plot([], [], color=GOOD, lw=1.6, label="accuracy")
    ax_curve.set_xlim(0, total_epochs)
    ax_curve.set_ylim(0, max(history["loss"]) * 1.05)
    ax_acc.set_ylim(0, 1.02)
    ax_curve.tick_params(colors=ACCENT, labelsize=7)
    ax_acc.tick_params(colors=GOOD, labelsize=7)
    ax_acc.set_facecolor("none")
    for spine in ax_acc.spines.values():
        spine.set_color("#26262c")
    ax_curve.set_xlabel("epoch", color=DIM, fontsize=8)

    def draw(frame: int):
        epoch = recorder.epochs[frame]
        net.set_weights(recorder.snapshots[frame])
        weights = recorder.snapshots[frame]

        edge_colors = np.zeros((len(segments), 4))
        for li, (W, _) in enumerate(weights):
            sub = W[np.ix_(shown[li], shown[li + 1])].ravel()
            strength = np.abs(sub) / (np.abs(sub).max() + 1e-12)
            rgba = np.where((sub >= 0)[:, None], POS_RGBA, NEG_RGBA)
            rgba[:, 3] = 0.04 + 0.6 * strength
            edge_colors[edge_slices[li]] = rgba
        edges.set_color(edge_colors)

        trace = net.forward_trace(probe)
        face = np.zeros((len(node_xy), 4))
        for li, act in enumerate(trace):
            vals = act[0][shown[li]]
            scale = vals.max() if li < len(trace) - 1 else 1.0
            face[layer_slices[li]] = _mix(vals / (scale + 1e-12), "#2a2a30", ACCENT)
        nodes.set_facecolor(face)

        probs = trace[-1][0]
        best = int(probs.argmax())
        for i, (bar, p) in enumerate(zip(bars, probs, strict=True)):
            bar.set_width(p)
            bar.set_color(GOOD if i == best else DIM)

        k = epoch
        loss_line.set_data(epochs_axis[:k], history["loss"][:k])
        acc_line.set_data(epochs_axis[:k], history["accuracy"][:k])
        loss = history["loss"][k - 1] if k else float("nan")
        acc = history["accuracy"][k - 1] if k else float("nan")
        hud.set_text(f"epoch {epoch}/{total_epochs}   loss {loss:.3f}   acc {acc:.1%}")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(recorder.epochs), interval=1000 / fps, blit=False)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".gif":
            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
        else:
            anim.save(path, fps=fps, dpi=dpi, savefig_kwargs={"facecolor": BG})
        net.set_weights(original)
        plt.close(fig)
    return anim
