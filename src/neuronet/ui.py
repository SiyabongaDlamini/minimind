"""Web visualizer and control center for Neuronet."""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .data import Dataset, Standardizer
from .knowledge import KnowledgeBase
from .network import NeuralNetwork
from .parsers import ImageParser, load_csv, load_image_folder

app = FastAPI(title="Neuronet Console", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
MODELS_DIR = Path(".")


def _scan_environment() -> dict[str, Any]:
    models = []
    for p in sorted(Path(".").glob("*.npz")):
        try:
            net = NeuralNetwork.load(p)
            meta = getattr(net, "meta", {})
            models.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "layer_sizes": list(net.layer_sizes),
                    "activation": net.activation,
                    "class_names": meta.get("class_names", []),
                    "test_accuracy": meta.get("test_accuracy"),
                    "is_image_model": "image_size" in meta,
                    "image_size": meta.get("image_size"),
                }
            )
        except Exception:
            continue

    csv_files = [str(p) for p in sorted(Path("data").rglob("*.csv"))]
    image_dirs = []
    for d in sorted(Path("data").iterdir()):
        if d.is_dir() and d.name != "docs" and d.name != "uploads":
            subdirs = [s.name for s in d.iterdir() if s.is_dir() and not s.name.startswith(".")]
            if len(subdirs) >= 2:
                image_dirs.append({"path": str(d), "classes": subdirs})

    docs = [str(p) for p in sorted(Path("data").rglob("*.md"))] + [
        str(p) for p in sorted(Path("data").rglob("*.txt"))
    ]
    if Path("README.md").exists():
        docs.insert(0, "README.md")

    return {
        "models": models,
        "csv_files": csv_files,
        "image_dirs": image_dirs,
        "documents": docs,
    }


@app.get("/api/environment")
def get_environment() -> JSONResponse:
    return JSONResponse(_scan_environment())


@app.post("/api/upload/file")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("csv"),
    class_label: str = Form(""),
) -> JSONResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "uploaded_file").name

    if category == "csv":
        target = UPLOAD_DIR / filename
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            data = load_csv(target)
            return JSONResponse(
                {
                    "status": "success",
                    "file_path": str(target),
                    "type": "csv",
                    "samples": len(data),
                    "features": data.n_features,
                    "classes": list(data.class_names),
                }
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV structure: {e}") from e

    elif category == "image":
        label_dir = UPLOAD_DIR / "images" / (class_label or "unlabeled")
        label_dir.mkdir(parents=True, exist_ok=True)
        target = label_dir / filename
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(
            {
                "status": "success",
                "file_path": str(target),
                "type": "image",
                "class_label": class_label,
            }
        )

    else:
        target = UPLOAD_DIR / filename
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(
            {
                "status": "success",
                "file_path": str(target),
                "type": "document",
            }
        )


@app.post("/api/upload/bulk_images")
async def upload_bulk_images(
    files: list[UploadFile] = File(...),
    class_label: str = Form(...),
) -> JSONResponse:
    if not class_label:
        raise HTTPException(status_code=400, detail="Class label is required for image grouping.")
    label_dir = UPLOAD_DIR / "images" / class_label
    label_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        name = Path(f.filename or "img").name
        target = label_dir / name
        with target.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_paths.append(str(target))

    return JSONResponse(
        {
            "status": "success",
            "class_label": class_label,
            "count": len(saved_paths),
            "folder": str(label_dir),
        }
    )


@app.post("/api/train")
async def train_model(payload: dict[str, Any]) -> JSONResponse:
    dataset_type = payload.get("dataset_type", "csv")
    dataset_path = payload.get("dataset_path", "")
    target_column = payload.get("target_column", "-1")
    hidden_str = str(payload.get("hidden_layers", "32"))
    epochs = int(payload.get("epochs", 40))
    lr = float(payload.get("lr", 0.01))
    batch_size = int(payload.get("batch_size", 16))
    activation = str(payload.get("activation", "relu"))
    test_size = float(payload.get("test_size", 0.2))
    model_name = str(payload.get("model_name", "custom_model.npz")).strip()
    if not model_name.endswith(".npz"):
        model_name += ".npz"

    try:
        hidden = [int(h.strip()) for h in hidden_str.split(",") if h.strip()]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid hidden layer configuration.") from None

    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"Dataset path not found: {dataset_path}")

    meta: dict[str, Any] = {}

    try:
        if dataset_type == "csv":
            tgt = int(target_column) if target_column.lstrip("-").isdigit() else target_column
            raw_data = load_csv(dataset_path, target=tgt)
            scaler = Standardizer().fit(raw_data.X)
            data = Dataset(scaler.transform(raw_data.X), raw_data.y, raw_data.class_names)
            meta["standardizer"] = scaler.to_dict()
        else:
            size_h = int(payload.get("image_height", 16))
            size_w = int(payload.get("image_width", 16))
            parser = ImageParser(size=(size_h, size_w))
            data = load_image_folder(dataset_path, parser)
            meta["image_size"] = [size_h, size_w]
            meta["invert"] = "auto"

        train, test = data.split(test_size=test_size, seed=42)
        net = NeuralNetwork([data.n_features, *hidden, data.n_classes], activation, seed=42)

        epoch_logs: list[dict[str, Any]] = []

        def logger_callback(epoch: int, total: int, network: NeuralNetwork, hist: dict[str, list[float]]) -> None:
            if epoch > 0:
                epoch_logs.append(
                    {
                        "epoch": epoch,
                        "loss": round(hist["loss"][-1], 4),
                        "accuracy": round(hist["accuracy"][-1], 4),
                        "val_loss": round(hist["val_loss"][-1], 4) if hist["val_loss"] else None,
                        "val_accuracy": round(hist["val_accuracy"][-1], 4) if hist["val_accuracy"] else None,
                    }
                )

        history = net.fit(
            train.X,
            train.y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            validation=(test.X, test.y),
            callbacks=[logger_callback],
            seed=42,
            verbose=0,
        )

        test_loss, test_acc = net.evaluate(test.X, test.y)
        meta["class_names"] = list(data.class_names)
        meta["test_accuracy"] = round(test_acc, 4)
        net.save(model_name, meta)

        return JSONResponse(
            {
                "status": "completed",
                "model_name": model_name,
                "layer_sizes": list(net.layer_sizes),
                "activation": net.activation,
                "train_samples": len(train),
                "test_samples": len(test),
                "classes": list(data.class_names),
                "final_loss": round(float(history["loss"][-1]), 4),
                "final_accuracy": round(float(history["accuracy"][-1]), 4),
                "test_loss": round(float(test_loss), 4),
                "test_accuracy": round(float(test_acc), 4),
                "history": {
                    "loss": [round(float(v), 4) for v in history["loss"]],
                    "accuracy": [round(float(v), 4) for v in history["accuracy"]],
                    "val_loss": [round(float(v), 4) for v in history["val_loss"]],
                    "val_accuracy": [round(float(v), 4) for v in history["val_accuracy"]],
                },
                "epochs_log": epoch_logs,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
async def run_prediction(
    model_name: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    existing_file: str = Form(""),
    tabular_features: str = Form(""),
) -> JSONResponse:
    model_path = Path(model_name)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    try:
        net = NeuralNetwork.load(model_path)
        meta = getattr(net, "meta", {})
        classes = meta.get("class_names", [f"Class {i}" for i in range(net.layer_sizes[-1])])

        results: list[dict[str, Any]] = []

        if "image_size" in meta:
            size = tuple(meta["image_size"])
            parser = ImageParser(size=size)

            targets: list[tuple[str, Any]] = []
            if files:
                for f in files:
                    content = await f.read()
                    from PIL import Image

                    img = Image.open(io.BytesIO(content))
                    targets.append((f.filename or "image", img))
            elif existing_file and Path(existing_file).exists():
                targets.append((existing_file, existing_file))

            if not targets:
                raise HTTPException(status_code=400, detail="No image file provided for image model.")

            for name, item in targets:
                feat = parser.parse(item)
                probs = net.predict_proba(feat[None])[0]
                pred_idx = int(np.argmax(probs))
                dist = [{"class": classes[i], "probability": round(float(probs[i]), 4)} for i in range(len(classes))]
                dist.sort(key=lambda x: x["probability"], reverse=True)
                results.append(
                    {
                        "source": name,
                        "predicted_class": classes[pred_idx],
                        "confidence": round(float(probs[pred_idx]), 4),
                        "distribution": dist,
                        "vector_preview": [round(float(v), 2) for v in feat[:16]],
                    }
                )

        else:
            if tabular_features:
                raw_floats = [float(x.strip()) for x in tabular_features.split(",") if x.strip()]
                X = np.array([raw_floats], dtype=float)
            elif existing_file and Path(existing_file).exists():
                d = load_csv(existing_file)
                X = d.X[:10]
            else:
                raise HTTPException(status_code=400, detail="No features provided for tabular model.")

            if "standardizer" in meta:
                scaler = Standardizer.from_dict(meta["standardizer"])
                X_norm = scaler.transform(X)
            else:
                X_norm = X

            probs_all = net.predict_proba(X_norm)
            for idx, probs in enumerate(probs_all):
                pred_idx = int(np.argmax(probs))
                dist = [{"class": classes[i], "probability": round(float(probs[i]), 4)} for i in range(len(classes))]
                dist.sort(key=lambda x: x["probability"], reverse=True)
                results.append(
                    {
                        "source": f"Sample #{idx + 1}",
                        "predicted_class": classes[pred_idx],
                        "confidence": round(float(probs[pred_idx]), 4),
                        "distribution": dist,
                        "raw_features": [round(float(v), 3) for v in X[idx]],
                    }
                )

        return JSONResponse(
            {
                "status": "success",
                "model_name": model_name,
                "total_tested": len(results),
                "predictions": results,
            }
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ask")
async def ask_documents(payload: dict[str, Any]) -> JSONResponse:
    doc_paths = payload.get("documents", [])
    query = str(payload.get("query", "")).strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not doc_paths:
        raise HTTPException(status_code=400, detail="No documents specified.")

    kb = KnowledgeBase()
    report = kb.add_paths(doc_paths)
    ans = kb.ask(query)

    return JSONResponse(
        {
            "query": query,
            "found": ans.found,
            "answer": ans.text,
            "sources": list(dict.fromkeys(p.source for p in ans.passages)),
            "confidence": round(float(ans.confidence), 4),
            "indexed_passages": len(kb),
            "report": report,
        }
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Neuronet // Neural System Console</title>
  <style>
    :root {
      --bg: #09090b;
      --card: #121215;
      --panel: #18181b;
      --border: #27272a;
      --border-focus: #3f3f46;
      --text: #fafafa;
      --muted: #a1a1aa;
      --dim: #71717a;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 13px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }
    header {
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #000000;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-title {
      font-family: var(--mono);
      font-weight: 700;
      letter-spacing: 0.08em;
      font-size: 14px;
      color: #ffffff;
    }
    .status-badge {
      font-family: var(--mono);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 2px;
      background: #18181b;
      border: 1px solid var(--border);
      color: var(--success);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success);
    }
    .nav-tabs {
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--border);
      background: #0d0d10;
      padding: 0 24px;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--muted);
      padding: 10px 16px;
      font-family: var(--mono);
      font-size: 12px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      transition: all 0.15s ease;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active {
      color: var(--text);
      border-bottom-color: var(--accent);
      background: rgba(59, 130, 246, 0.05);
    }
    main {
      padding: 24px;
      max-width: 1400px;
      margin: 0 auto;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
    }
    .grid-full {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 16px;
    }
    .card-title {
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .field {
      margin-bottom: 12px;
    }
    label {
      display: block;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    input, select, textarea {
      width: 100%;
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 10px;
      font-family: var(--mono);
      font-size: 12px;
      border-radius: 2px;
      outline: none;
      transition: border-color 0.15s;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
    }
    .btn {
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 14px;
      font-family: var(--mono);
      font-size: 12px;
      border-radius: 2px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s;
    }
    .btn:hover {
      background: #27272a;
      border-color: var(--border-focus);
    }
    .btn-primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      font-weight: 500;
    }
    .btn-primary:hover {
      background: var(--accent-hover);
      border-color: var(--accent-hover);
    }
    .btn-block {
      width: 100%;
    }
    .metrics-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric-box {
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 2px;
    }
    .metric-lbl {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .metric-val {
      font-family: var(--mono);
      font-size: 20px;
      font-weight: 700;
      color: #ffffff;
      margin-top: 4px;
    }
    .chart-container {
      position: relative;
      height: 240px;
      background: #09090b;
      border: 1px solid var(--border);
      border-radius: 2px;
      margin-bottom: 16px;
      padding: 8px;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .table-container {
      overflow-x: auto;
      max-height: 260px;
      border: 1px solid var(--border);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--mono);
      font-size: 11px;
    }
    th, td {
      padding: 6px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: var(--panel);
      color: var(--muted);
      position: sticky;
      top: 0;
      font-weight: 600;
    }
    tr:hover td {
      background: #18181b;
    }
    .dropzone {
      border: 1px dashed var(--border);
      padding: 20px;
      text-align: center;
      background: var(--panel);
      border-radius: 2px;
      cursor: pointer;
      margin-bottom: 12px;
      transition: border-color 0.15s;
    }
    .dropzone:hover {
      border-color: var(--accent);
    }
    .prob-bar-container {
      margin-top: 4px;
      background: #27272a;
      height: 6px;
      border-radius: 2px;
      overflow: hidden;
    }
    .prob-bar {
      height: 100%;
      background: var(--accent);
    }
    .tag {
      font-family: var(--mono);
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 2px;
      background: #27272a;
      color: var(--text);
    }
    .hidden { display: none !important; }
    .flex-between { display: flex; justify-content: space-between; align-items: center; }
  </style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-title">NEURONET // NEURAL SYSTEM CONSOLE</div>
    <span class="tag">REAL NUMPY CORE</span>
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>
    <span id="system-status-indicator">SYSTEM READY</span>
  </div>
</header>

<nav class="nav-tabs">
  <button class="tab-btn active" onclick="switchTab('train')">Model Training</button>
  <button class="tab-btn" onclick="switchTab('predict')">Testing & Inference</button>
  <button class="tab-btn" onclick="switchTab('upload')">Dataset Manager</button>
  <button class="tab-btn" onclick="switchTab('ask')">Document Retrieval</button>
</nav>

<main>
  <!-- TAB: TRAIN -->
  <section id="tab-train">
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Training Configuration</div>

        <div class="field">
          <label>Dataset Type</label>
          <select id="train-type" onchange="toggleDatasetType()">
            <option value="csv">Structured CSV Dataset</option>
            <option value="images">Categorized Image Folders</option>
          </select>
        </div>

        <div class="field">
          <label>Dataset Source</label>
          <select id="train-dataset-path"></select>
        </div>

        <div class="field" id="target-col-group">
          <label>Target Column (Name or -1 for last)</label>
          <input type="text" id="train-target" value="species">
        </div>

        <div class="field hidden" id="image-dim-group">
          <label>Image Resize Shape (HxW)</label>
          <div style="display:flex; gap:8px;">
            <input type="number" id="train-img-h" value="16" placeholder="H">
            <input type="number" id="train-img-w" value="16" placeholder="W">
          </div>
        </div>

        <div class="field">
          <label>Hidden Layers (comma separated)</label>
          <input type="text" id="train-hidden" value="32">
        </div>

        <div class="field">
          <label>Activation Function</label>
          <select id="train-activation">
            <option value="relu">ReLU</option>
            <option value="tanh">Tanh</option>
            <option value="sigmoid">Sigmoid</option>
          </select>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <div class="field">
            <label>Epochs</label>
            <input type="number" id="train-epochs" value="50">
          </div>
          <div class="field">
            <label>Batch Size</label>
            <input type="number" id="train-batch" value="16">
          </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 8px;">
          <div class="field">
            <label>Learning Rate</label>
            <input type="number" id="train-lr" step="0.001" value="0.01">
          </div>
          <div class="field">
            <label>Test Split Ratio</label>
            <input type="number" id="train-test-size" step="0.05" value="0.2">
          </div>
        </div>

        <div class="field">
          <label>Output Model (.npz)</label>
          <input type="text" id="train-model-out" value="trained_model.npz">
        </div>

        <button class="btn btn-primary btn-block" id="btn-start-train" onclick="executeTraining()">
          Execute Network Training
        </button>
      </div>

      <div>
        <div class="metrics-row">
          <div class="metric-box">
            <div class="metric-lbl">Architecture</div>
            <div class="metric-val" id="metric-arch">-</div>
          </div>
          <div class="metric-box">
            <div class="metric-lbl">Validation Loss</div>
            <div class="metric-val" id="metric-loss">-</div>
          </div>
          <div class="metric-box">
            <div class="metric-lbl">Validation Accuracy</div>
            <div class="metric-val" id="metric-acc">-</div>
          </div>
          <div class="metric-box">
            <div class="metric-lbl">Total Samples</div>
            <div class="metric-val" id="metric-samples">-</div>
          </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
          <div class="card-title">
            <span>Training Dynamics // Loss & Accuracy Curves</span>
            <span class="tag" id="curve-status">IDLE</span>
          </div>
          <div class="chart-container">
            <canvas id="loss-chart"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Epoch Progression Telemetry</div>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Epoch</th>
                  <th>Train Loss</th>
                  <th>Train Accuracy</th>
                  <th>Val Loss</th>
                  <th>Val Accuracy</th>
                </tr>
              </thead>
              <tbody id="epoch-table-body">
                <tr><td colspan="5" style="color:var(--dim); text-align:center;">No training executed yet.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB: PREDICT -->
  <section id="tab-predict" class="hidden">
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Inference Engine</div>

        <div class="field">
          <label>Select Trained Model</label>
          <select id="pred-model-select" onchange="inspectModelDetails()"></select>
        </div>

        <div class="field">
          <label>Model Architecture Metadata</label>
          <div id="model-meta-info" style="font-family:var(--mono); font-size:11px; color:var(--muted); padding:8px; background:var(--panel); border:1px solid var(--border);">
            Select a model to view parameters.
          </div>
        </div>

        <div id="pred-image-input-section" class="hidden">
          <div class="field">
            <label>Upload Image(s) For Testing (Single or Multiple)</label>
            <input type="file" id="pred-image-files" multiple accept="image/*">
          </div>
          <div class="field">
            <label>Or Test Pre-Existing Real File</label>
            <input type="text" id="pred-existing-file" value="data/test_square.png">
          </div>
        </div>

        <div id="pred-tabular-input-section">
          <div class="field">
            <label>Tabular Features (Comma-separated values)</label>
            <input type="text" id="pred-tabular-features" value="5.1, 3.5, 1.4, 0.2" placeholder="e.g. 5.1, 3.5, 1.4, 0.2">
          </div>
          <div class="field">
            <label>Or Path to CSV to evaluate</label>
            <input type="text" id="pred-tabular-csv" value="data/iris.csv">
          </div>
        </div>

        <button class="btn btn-primary btn-block" onclick="executePrediction()">
          Run Inference Check
        </button>
      </div>

      <div class="card">
        <div class="card-title">Prediction Results & Class Probabilities</div>
        <div id="prediction-results" style="display:flex; flex-direction:column; gap:12px;">
          <div style="color:var(--dim); font-family:var(--mono);">Run inference to view classification confidence.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB: UPLOAD -->
  <section id="tab-upload" class="hidden">
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Bulk Image Ingestion</div>
        <div class="field">
          <label>Assign Target Class Label (e.g. "triangle", "defect")</label>
          <input type="text" id="bulk-class-label" placeholder="class_name">
        </div>
        <div class="field">
          <label>Select Real Images (Bulk Multi-Select)</label>
          <input type="file" id="bulk-image-files" multiple accept="image/*">
        </div>
        <button class="btn btn-primary btn-block" onclick="uploadBulkImages()">
          Upload Bulk Class Images
        </button>
        <div id="bulk-upload-msg" style="margin-top:8px; font-family:var(--mono); font-size:11px;"></div>
      </div>

      <div class="card">
        <div class="card-title">CSV Dataset Ingestion</div>
        <div class="field">
          <label>Select Real CSV File</label>
          <input type="file" id="csv-file-input" accept=".csv">
        </div>
        <button class="btn btn-primary btn-block" onclick="uploadCsvFile()">
          Upload Structured CSV
        </button>
        <div id="csv-upload-msg" style="margin-top:8px; font-family:var(--mono); font-size:11px;"></div>

        <div style="margin-top: 24px; border-top: 1px solid var(--border); padding-top: 16px;">
          <div class="card-title">Discovered Workspace Assets</div>
          <div id="workspace-inventory" style="font-family:var(--mono); font-size:11px; color:var(--muted);">
            Scanning...
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB: ASK -->
  <section id="tab-ask" class="hidden">
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Document Q&A Retrieval Engine</div>
        <div class="field">
          <label>Document Paths (One per line)</label>
          <textarea id="ask-doc-paths" rows="4">data/docs/handbook.md
README.md</textarea>
        </div>
        <div class="field">
          <label>Question</label>
          <input type="text" id="ask-query" value="When are production deployments scheduled?">
        </div>
        <button class="btn btn-primary btn-block" onclick="executeDocumentAsk()">
          Query Knowledge Base
        </button>
      </div>

      <div class="card">
        <div class="card-title">Verbatim Retrieved Knowledge</div>
        <div id="ask-results-box" style="font-family:var(--mono); font-size:12px; line-height:1.6; color:var(--text); white-space: pre-wrap;">
          Submit a query against documents to retrieve verified excerpts.
        </div>
      </div>
    </div>
  </section>
</main>

<script>
let envData = { models: [], csv_files: [], image_dirs: [], documents: [] };

async function refreshEnvironment() {
  try {
    const res = await fetch('/api/environment');
    envData = await res.json();
    populateDatasetSelects();
    populateModelSelects();
    renderInventory();
  } catch (err) {
    console.error("Environment refresh error:", err);
  }
}

function switchTab(tabId) {
  document.querySelectorAll('nav .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('main section').forEach(s => s.classList.add('hidden'));
  
  event.target.classList.add('active');
  document.getElementById('tab-' + tabId).classList.remove('hidden');
  if (tabId === 'predict') inspectModelDetails();
}

function toggleDatasetType() {
  const type = document.getElementById('train-type').value;
  const isCsv = type === 'csv';
  document.getElementById('target-col-group').classList.toggle('hidden', !isCsv);
  document.getElementById('image-dim-group').classList.toggle('hidden', isCsv);
  populateDatasetSelects();
}

function populateDatasetSelects() {
  const type = document.getElementById('train-type').value;
  const sel = document.getElementById('train-dataset-path');
  sel.innerHTML = '';
  
  if (type === 'csv') {
    envData.csv_files.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    });
  } else {
    envData.image_dirs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.path;
      opt.textContent = `${d.path} (${d.classes.join(', ')})`;
      sel.appendChild(opt);
    });
  }
}

function populateModelSelects() {
  const sel = document.getElementById('pred-model-select');
  sel.innerHTML = '';
  envData.models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = `${m.name} [${m.layer_sizes.join('-')}] (${m.is_image_model ? 'IMAGE' : 'CSV'})`;
    sel.appendChild(opt);
  });
  inspectModelDetails();
}

function inspectModelDetails() {
  const name = document.getElementById('pred-model-select').value;
  const model = envData.models.find(m => m.name === name);
  const info = document.getElementById('model-meta-info');
  const imgSection = document.getElementById('pred-image-input-section');
  const tabSection = document.getElementById('pred-tabular-input-section');

  if (!model) {
    info.textContent = "No model selected.";
    return;
  }

  const isImg = model.is_image_model;
  imgSection.classList.toggle('hidden', !isImg);
  tabSection.classList.toggle('hidden', isImg);

  info.innerHTML = `
    <strong>Architecture:</strong> ${model.layer_sizes.join(' → ')}<br>
    <strong>Activation:</strong> ${model.activation}<br>
    <strong>Classes:</strong> ${model.class_names.join(', ')}<br>
    <strong>Validation Accuracy:</strong> ${model.test_accuracy ? (model.test_accuracy * 100).toFixed(1) + '%' : 'N/A'}<br>
    <strong>Input Type:</strong> ${isImg ? 'Image Vector ' + (model.image_size || []).join('x') : 'Tabular Feature Vector'}
  `;
}

function renderInventory() {
  const div = document.getElementById('workspace-inventory');
  div.innerHTML = `
    <div style="margin-bottom:6px;"><strong>Models:</strong> ${envData.models.map(m => m.name).join(', ') || 'None'}</div>
    <div style="margin-bottom:6px;"><strong>CSV Files:</strong> ${envData.csv_files.join(', ') || 'None'}</div>
    <div style="margin-bottom:6px;"><strong>Image Folders:</strong> ${envData.image_dirs.map(d => d.path).join(', ') || 'None'}</div>
    <div><strong>Docs:</strong> ${envData.documents.join(', ') || 'None'}</div>
  `;
}

async function executeTraining() {
  const btn = document.getElementById('btn-start-train');
  btn.disabled = true;
  btn.textContent = "Training in progress...";
  document.getElementById('curve-status').textContent = "RUNNING";

  const payload = {
    dataset_type: document.getElementById('train-type').value,
    dataset_path: document.getElementById('train-dataset-path').value,
    target_column: document.getElementById('train-target').value,
    image_height: parseInt(document.getElementById('train-img-h').value),
    image_width: parseInt(document.getElementById('train-img-w').value),
    hidden_layers: document.getElementById('train-hidden').value,
    activation: document.getElementById('train-activation').value,
    epochs: parseInt(document.getElementById('train-epochs').value),
    batch_size: parseInt(document.getElementById('train-batch').value),
    lr: parseFloat(document.getElementById('train-lr').value),
    test_size: parseFloat(document.getElementById('train-test-size').value),
    model_name: document.getElementById('train-model-out').value
  };

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Training failed");

    document.getElementById('metric-arch').textContent = data.layer_sizes.join('-');
    document.getElementById('metric-loss').textContent = data.test_loss.toFixed(4);
    document.getElementById('metric-acc').textContent = (data.test_accuracy * 100).toFixed(1) + '%';
    document.getElementById('metric-samples').textContent = `${data.train_samples} / ${data.test_samples}`;
    document.getElementById('curve-status').textContent = "COMPLETED";

    renderEpochTable(data.epochs_log);
    renderCurves(data.history);
    await refreshEnvironment();
  } catch (err) {
    alert("Error: " + err.message);
    document.getElementById('curve-status').textContent = "FAILED";
  } finally {
    btn.disabled = false;
    btn.textContent = "Execute Network Training";
  }
}

function renderEpochTable(logs) {
  const tbody = document.getElementById('epoch-table-body');
  tbody.innerHTML = '';
  logs.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.epoch}</td>
      <td>${row.loss.toFixed(4)}</td>
      <td>${(row.accuracy * 100).toFixed(1)}%</td>
      <td>${row.val_loss !== null ? row.val_loss.toFixed(4) : '-'}</td>
      <td>${row.val_accuracy !== null ? (row.val_accuracy * 100).toFixed(1) + '%' : '-'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCurves(history) {
  const canvas = document.getElementById('loss-chart');
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  ctx.scale(2, 2);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = '#18181b';
  ctx.lineWidth = 1;
  for (let y = 20; y < h - 20; y += 30) {
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  const loss = history.loss || [];
  const valLoss = history.val_loss || [];
  const acc = history.accuracy || [];
  const valAcc = history.val_accuracy || [];
  const n = loss.length;
  if (n < 2) return;

  function plotLine(series, color, isDashed = false) {
    const maxVal = Math.max(...loss, ...valLoss, 1.0);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (isDashed) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
    series.forEach((val, idx) => {
      const x = 35 + (idx / (n - 1)) * (w - 60);
      const y = (h - 25) - (val / maxVal) * (h - 50);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  plotLine(loss, '#3b82f6', false);
  plotLine(valLoss, '#ef4444', true);

  // Legend
  ctx.setLineDash([]);
  ctx.font = '10px monospace';
  ctx.fillStyle = '#3b82f6';
  ctx.fillText('Train Loss', 40, 16);
  ctx.fillStyle = '#ef4444';
  ctx.fillText('Val Loss (Dashed)', 120, 16);
}

async function executePrediction() {
  const modelName = document.getElementById('pred-model-select').value;
  const model = envData.models.find(m => m.name === modelName);
  const formData = new FormData();
  formData.append('model_name', modelName);

  if (model && model.is_image_model) {
    const fileInput = document.getElementById('pred-image-files');
    if (fileInput.files.length > 0) {
      for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('files', fileInput.files[i]);
      }
    } else {
      formData.append('existing_file', document.getElementById('pred-existing-file').value);
    }
  } else {
    formData.append('tabular_features', document.getElementById('pred-tabular-features').value);
    formData.append('existing_file', document.getElementById('pred-tabular-csv').value);
  }

  const resultsDiv = document.getElementById('prediction-results');
  resultsDiv.innerHTML = '<div style="color:var(--dim); font-family:var(--mono);">Running inference...</div>';

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Inference failed");

    resultsDiv.innerHTML = '';
    data.predictions.forEach(item => {
      const card = document.createElement('div');
      card.style.background = 'var(--panel)';
      card.style.border = '1px solid var(--border)';
      card.style.padding = '12px';
      card.style.borderRadius = '2px';
      
      let barsHtml = item.distribution.map(d => `
        <div style="margin-top:6px; font-family:var(--mono); font-size:11px;">
          <div class="flex-between">
            <span>${d.class}</span>
            <span>${(d.probability * 100).toFixed(1)}%</span>
          </div>
          <div class="prob-bar-container">
            <div class="prob-bar" style="width: ${d.probability * 100}%;"></div>
          </div>
        </div>
      `).join('');

      card.innerHTML = `
        <div class="flex-between" style="border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 6px;">
          <strong style="font-family:var(--mono);">${item.source}</strong>
          <span class="tag" style="background:var(--accent); color:#fff;">PREDICTED: ${item.predicted_class}</span>
        </div>
        ${barsHtml}
      `;
      resultsDiv.appendChild(card);
    });
  } catch (err) {
    resultsDiv.innerHTML = `<div style="color:var(--danger); font-family:var(--mono);">${err.message}</div>`;
  }
}

async function uploadBulkImages() {
  const label = document.getElementById('bulk-class-label').value.trim();
  const fileInput = document.getElementById('bulk-image-files');
  const msg = document.getElementById('bulk-upload-msg');

  if (!label || fileInput.files.length === 0) {
    alert("Specify class label and choose images.");
    return;
  }

  const form = new FormData();
  form.append('class_label', label);
  for (let i = 0; i < fileInput.files.length; i++) {
    form.append('files', fileInput.files[i]);
  }

  msg.textContent = "Uploading images...";
  try {
    const res = await fetch('/api/upload/bulk_images', { method: 'POST', body: form });
    const data = await res.json();
    msg.style.color = "var(--success)";
    msg.textContent = `Uploaded ${data.count} images for class '${data.class_label}' to ${data.folder}`;
    await refreshEnvironment();
  } catch (err) {
    msg.style.color = "var(--danger)";
    msg.textContent = "Upload failed: " + err.message;
  }
}

async function uploadCsvFile() {
  const fileInput = document.getElementById('csv-file-input');
  const msg = document.getElementById('csv-upload-msg');
  if (fileInput.files.length === 0) {
    alert("Select a CSV file.");
    return;
  }

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  form.append('category', 'csv');

  msg.textContent = "Uploading CSV...";
  try {
    const res = await fetch('/api/upload/file', { method: 'POST', body: form });
    const data = await res.json();
    msg.style.color = "var(--success)";
    msg.textContent = `Uploaded ${data.file_path} (${data.samples} samples, ${data.features} features)`;
    await refreshEnvironment();
  } catch (err) {
    msg.style.color = "var(--danger)";
    msg.textContent = "CSV upload error: " + err.message;
  }
}

async function executeDocumentAsk() {
  const docsText = document.getElementById('ask-doc-paths').value;
  const docs = docsText.split('\\n').map(s => s.trim()).filter(Boolean);
  const query = document.getElementById('ask-query').value.trim();
  const out = document.getElementById('ask-results-box');

  out.textContent = "Searching document vectors...";
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ documents: docs, query: query })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Query failed");

    out.innerHTML = `<strong>QUERY:</strong> ${data.query}\\n<strong>MATCH FOUND:</strong> ${data.found}\\n<strong>PASSAGES INDEXED:</strong> ${data.indexed_passages}\\n\\n${data.answer}`;
  } catch (err) {
    out.textContent = "Error: " + err.message;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  refreshEnvironment();
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_TEMPLATE

