#!/usr/bin/env python3
"""
Serve the Cesium 3D globe via a local HTTP server.

Usage:
    python cesium/serve.py          # from the project root
    python serve.py                 # from inside cesium/

Opens http://localhost:8080/index.html in the default browser automatically.
Run weather_map.py first to generate the map PNGs.

Why a server is needed:
    When index.html is opened directly via file://, browsers block Cesium from
    drawing local images to its WebGL canvas (cross-origin canvas restriction).
    Serving via HTTP removes this restriction.
"""
import http.server
import os
import socket
import threading
import webbrowser

CESIUM_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CESIUM_DIR)


def _free_port(start: int = 8080) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise OSError(f"No free port found in range {start}–{start + 19}")


PORT = _free_port()
threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{PORT}/index.html")).start()

print(f"Globe: http://localhost:{PORT}/index.html")
print("Press Ctrl-C to stop.\n")

with http.server.HTTPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    httpd.serve_forever()
