"""Flask backend for LiDAR tree species classification (Stage 3).

Endpoints:
  GET /api/health              -> Simple health check
  GET /api/pointcloud/meta     -> Basic metadata (point count, bounds, species classes)
  GET /api/pointcloud/points   -> (Paginated) raw point data (x,y,z,species/semantic)
  WS /ws/classify (Socket.IO)  -> Streams per-tree classification results incrementally

This is a lightweight implementation designed for local development / demo.
Not optimized for production or huge point clouds (streams in batches).
"""

from __future__ import annotations
import os
import math
import time
import json
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import laspy
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from stage_2 import (
    normalize_heights,
    calculate_tree_metrics,
    _ensure_normalized_dimension,
    _get_tree_ids,
    _get_species_id_array,
)

def _to_float_list(view):
    """Safely convert laspy ScaledArrayView or numpy array slice to a Python float list."""
    return np.asarray(view, dtype=float).tolist()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

DEFAULT_FILE = os.environ.get("LIDAR_FILE", "D:/Boreal3D/EasyPlot34/ULS.laz")

_LAS_CACHE = None  # type: Optional[laspy.LasData]
_LAS_PATH = None
_STREAM_THREAD = None
_STREAM_STOP = threading.Event()


def load_las(path: str) -> laspy.LasData:
    global _LAS_CACHE, _LAS_PATH
    if _LAS_CACHE is not None and _LAS_PATH == path:
        return _LAS_CACHE
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LAS/LAZ file not found: {path}")
    with laspy.open(path) as f:
        las = f.read()
    _ensure_normalized_dimension(las)
    las.z_normalized = normalize_heights(las)
    _LAS_CACHE = las
    _LAS_PATH = path
    return las


def _bounds(las) -> Dict[str, float]:
    return {
        'min_x': float(np.min(las.x)), 'max_x': float(np.max(las.x)),
        'min_y': float(np.min(las.y)), 'max_y': float(np.max(las.y)),
        'min_z': float(np.min(las.z)), 'max_z': float(np.max(las.z)),
    }


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/pointcloud/meta')
def pointcloud_meta():
    path = request.args.get('file', DEFAULT_FILE)
    try:
        las = load_las(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    species_arr = _get_species_id_array(las)
    unique_species, species_counts = np.unique(species_arr, return_counts=True)
    tree_ids_unique, _ = _get_tree_ids(las)
    return jsonify({
        'file': path,
        'point_count': int(las.header.point_count),
        'tree_count_estimate': int(len(tree_ids_unique)),
        'bounds': _bounds(las),
        'species_distribution': [
            {'species_id': int(s), 'count': int(c)} for s, c in zip(unique_species, species_counts)
        ]
    })


@app.route('/api/pointcloud/points')
def pointcloud_points():
    path = request.args.get('file', DEFAULT_FILE)
    limit = int(request.args.get('limit', 50000))  # basic throttle
    offset = int(request.args.get('offset', 0))
    debug = request.args.get('debug') == '1'
    try:
        las = load_las(path)
        total = las.header.point_count
        if debug:
            print(f"[points] path={path} total={total} limit={limit} offset={offset}")
        if total == 0:
            return jsonify({'error': 'Point cloud is empty'}), 500
        if offset >= total:
            return jsonify({'error': 'Offset beyond total points', 'total': int(total)}), 400
        end = min(offset + limit, total)
        idx = slice(offset, end)
        # Acquire arrays
        species_arr = _get_species_id_array(las)
        z_norm_arr = getattr(las, 'z_normalized', None)
        if z_norm_arr is None or len(z_norm_arr) != total:
            if debug:
                print(f"[points] Regenerating normalization: z_norm_len={0 if z_norm_arr is None else len(z_norm_arr)} expected={total}")
            las.z_normalized = normalize_heights(las)
            z_norm_arr = las.z_normalized
        # Sanity log lengths
        if debug:
            print(f"[points] slice end={end} species_len={len(species_arr)} z_norm_len={len(z_norm_arr)} x_len={len(las.x)}")
        # Guard against length mismatch
        min_len = min(len(las.x), len(species_arr), len(z_norm_arr))
        if end > min_len:
            end = min_len
            idx = slice(offset, end)
            if debug:
                print(f"[points] Adjusted end to {end} due to length mismatch")
        payload = {
            'offset': offset,
            'returned': end - offset,
            'total': int(total),
            'points': {
                'x': _to_float_list(las.x[idx]),
                'y': _to_float_list(las.y[idx]),
                'z': _to_float_list(las.z[idx]),
                'z_norm': _to_float_list(z_norm_arr[idx]),
                'species_id': np.asarray(species_arr[idx], dtype=int).tolist(),
            }
        }
        if debug:
            payload['debug'] = {
                'path_used': path,
                'limit': limit,
                'offset': offset,
                'end': end,
                'has_species_id': hasattr(las, 'species_id'),
                'point_format_id': las.header.point_format_id,
            }
        return jsonify(payload)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[points][ERROR] {e}\n{tb}")
        body = {'error': str(e)}
        if debug:
            body['trace'] = tb
        return jsonify(body), 500


def _stream_classification(las):
    species_arr = _get_species_id_array(las)
    tree_ids_unique, tree_id_array = _get_tree_ids(las)

    for i, tree_id in enumerate(tree_ids_unique):
        if _STREAM_STOP.is_set():
            break
        mask = (tree_id_array == tree_id)
        pts = np.column_stack([las.x[mask], las.y[mask], las.z_normalized[mask]])
        metrics = calculate_tree_metrics(pts)
        if not metrics:
            continue
        sp_vals = species_arr[mask]
        species_label = int(np.bincount(sp_vals).argmax())
        metrics['tree_id'] = int(tree_id)
        metrics['species_id'] = species_label
        socketio.emit('tree_classification', metrics, namespace='/ws')
        time.sleep(0.05)  # small delay to simulate streaming
    socketio.emit('stream_complete', {'status': 'done'}, namespace='/ws')


@socketio.on('start_stream', namespace='/ws')
def start_stream(message):  # message may contain {'file': path}
    global _STREAM_THREAD
    _STREAM_STOP.clear()
    path = message.get('file') if isinstance(message, dict) else None
    path = path or DEFAULT_FILE
    try:
        las = load_las(path)
    except Exception as e:
        emit('error', {'error': str(e)})
        return
    if _STREAM_THREAD and _STREAM_THREAD.is_alive():
        emit('info', {'message': 'Stream already running'})
        return
    _STREAM_THREAD = threading.Thread(target=_stream_classification, args=(las,), daemon=True)
    _STREAM_THREAD.start()
    emit('info', {'message': f'Stream started for {path}'})


@socketio.on('stop_stream', namespace='/ws')
def stop_stream():
    _STREAM_STOP.set()
    emit('info', {'message': 'Stopping stream...'})


def create_app():  # For external usage/testing
    return app, socketio


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LiDAR Flask backend (Stage 3)')
    parser.add_argument('--file', '-f', default=DEFAULT_FILE, help='Path to LAS/LAZ file')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    DEFAULT_FILE = args.file
    print(f"Starting backend with file: {DEFAULT_FILE}")
    socketio.run(app, host=args.host, port=args.port)
