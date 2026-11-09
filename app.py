#!/usr/bin/env python3
"""Neuronet Visualizer & Control Console.

Run:
    python3 app.py
"""

import sys
import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    host = "127.0.0.1"
    print(f"Starting Neuronet Control Console on http://{host}:{port}")
    uvicorn.run("neuronet.ui:app", host=host, port=port, log_level="info")

