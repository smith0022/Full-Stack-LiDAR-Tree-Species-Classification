import React, { useEffect, useRef, useState } from 'react';
import './App.css';
import * as THREE from 'three';

const BACKEND_BASE = process.env.REACT_APP_BACKEND || 'http://localhost:5000';
const WS_BASE = BACKEND_BASE.replace('http', 'ws');

const SPECIES_COLORS = {
  1: new THREE.Color('#2e8b57'), // Spruce
  2: new THREE.Color('#228b22'), // Pine
  3: new THREE.Color('#deb887'), // Birch
  4: new THREE.Color('#808080'), // Other
  0: new THREE.Color('#cccccc'), // Unknown
};

function PointCloudViewer({ points, classified, onAutoFit }) {
  const mountRef = useRef();
  const threeRef = useRef({});

  useEffect(() => {
  const width = mountRef.current.clientWidth;
  const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000);
  camera.position.set(0, -50, 50); // temporary; will be reset after data load
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);

  const controlsModulePromise = import('three/examples/jsm/controls/OrbitControls.js');

    threeRef.current = { scene, camera, renderer };
    let frameId;

    function animate() {
      frameId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();

    controlsModulePromise.then(mod => {
      const OrbitControls = mod.OrbitControls;
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(0, 0, 0);
      controls.update();
      threeRef.current.controls = controls;
    });

  return () => {
      // Cleanup with defensive checks (React StrictMode double-invokes effects in dev)
      try {
        cancelAnimationFrame(frameId);
        if (threeRef.current.scene) {
          threeRef.current.scene.traverse(obj => {
            if (obj.isMesh) {
              obj.geometry?.dispose?.();
              if (Array.isArray(obj.material)) {
                obj.material.forEach(m => m.dispose?.());
              } else {
                obj.material?.dispose?.();
              }
            }
          });
        }
        renderer.dispose();
        if (renderer.domElement && renderer.domElement.parentNode) {
          renderer.domElement.parentNode.removeChild(renderer.domElement);
        }
      } catch (e) {
        // swallow cleanup errors in dev
        // console.debug('Cleanup error', e);
      }
    };
  }, []);

  useEffect(() => {
    const { scene } = threeRef.current;
    if (!scene) return;
    // Remove previous point cloud
    const old = scene.getObjectByName('pointcloud');
    if (old) {
      if (old.geometry) old.geometry.dispose();
      if (old.material) {
        if (Array.isArray(old.material)) old.material.forEach(m => m.dispose?.());
        else old.material.dispose?.();
      }
      scene.remove(old);
    }
    if (!points || points.x.length === 0) return;

    const n = points.x.length;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    // Compute center to recenter coordinates around origin (better precision & view)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      if (points.x[i] < minX) minX = points.x[i];
      if (points.x[i] > maxX) maxX = points.x[i];
      if (points.y[i] < minY) minY = points.y[i];
      if (points.y[i] > maxY) maxY = points.y[i];
      if (points.z[i] < minZ) minZ = points.z[i];
      if (points.z[i] > maxZ) maxZ = points.z[i];
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const spanZ = maxZ - minZ;
    const radius = Math.max(spanX, spanY, spanZ) * 0.6 || 10;

    for (let i = 0; i < n; i++) {
      positions[3 * i] = points.x[i] - cx;
      positions[3 * i + 1] = points.y[i] - cy;
      positions[3 * i + 2] = points.z[i] - cz;
      let color;
      if (classified) {
        color = SPECIES_COLORS[points.species_id[i]] || SPECIES_COLORS[0];
      } else {
        // Raw: encode height as color gradient
        const h = points.z_norm[i];
        const t = Math.min(1, Math.max(0, h / 30));
        color = new THREE.Color().setHSL(0.6 - 0.6 * t, 1.0, 0.5);
      }
      colors[3 * i] = color.r;
      colors[3 * i + 1] = color.g;
      colors[3 * i + 2] = color.b;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({ size: 0.3, vertexColors: true });
    const cloud = new THREE.Points(geometry, material);
    cloud.name = 'pointcloud';
    scene.add(cloud);

    // Auto-fit camera
    const { camera, controls } = threeRef.current;
    if (camera) {
      camera.position.set(0, -radius * 2, radius * 1.2);
      camera.near = radius / 1000 || 0.01;
      camera.far = radius * 20 + 100;
      camera.updateProjectionMatrix();
    }
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
    onAutoFit && onAutoFit({ center: { cx, cy, cz }, radius });
  }, [points, classified]);

  return <div style={{ width: '100%', height: '100%', background: '#111' }} ref={mountRef} />;
}

function App() {
  const [points, setPoints] = useState(null);
  const [classified, setClassified] = useState(false);
  const [loading, setLoading] = useState(false);
  const [meta, setMeta] = useState(null);
  const [streamStatus, setStreamStatus] = useState('idle');
  const socketRef = useRef(null);
  const [error, setError] = useState(null);
  const [fitInfo, setFitInfo] = useState(null);

  // Fetch metadata & initial points
  useEffect(() => {
    async function fetchMeta() {
      const res = await fetch(`${BACKEND_BASE}/api/pointcloud/meta`);
      const data = await res.json();
      setMeta(data);
    }
    fetchMeta();
    async function fetchPoints() {
      setLoading(true);
      try {
        const res = await fetch(`${BACKEND_BASE}/api/pointcloud/points?limit=25000`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        setPoints(data.points);
      } catch (e) {
        console.error('Fetch points failed:', e);
        setError(`Failed to load points: ${e.message}`);
      } finally {
        setLoading(false);
      }
    }
    fetchPoints();
  }, []);

  // WebSocket (Socket.IO) for streaming classification updates
  useEffect(() => {
    // dynamic import socket.io-client to avoid bundling issues
    import('socket.io-client').then(({ io }) => {
      const socket = io(BACKEND_BASE + '/ws');
      socketRef.current = socket;
      socket.on('connect', () => setStreamStatus('connected'));
      socket.on('tree_classification', (msg) => {
        // Could be used to update a side panel or highlight a tree
        // For now just log
        console.debug('Tree metrics', msg);
      });
      socket.on('stream_complete', () => setStreamStatus('complete'));
      socket.on('info', (m) => console.log(m));
      socket.on('error', (e) => console.error(e));
    });
    return () => {
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  const startStream = () => {
    if (socketRef.current) {
      setStreamStatus('streaming');
      socketRef.current.emit('start_stream', {});
    }
  };

  const stopStream = () => {
    if (socketRef.current) {
      socketRef.current.emit('stop_stream');
      setStreamStatus('stopping');
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', color: '#eee', fontFamily: 'sans-serif' }}>
      <div style={{ width: '250px', background: '#1e1e1e', padding: '12px', boxSizing: 'border-box', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0 }}>LiDAR Viewer</h2>
        {meta && (
          <div style={{ fontSize: '0.8rem' }}>
            <div><strong>File:</strong> {meta.file}</div>
            <div><strong>Points:</strong> {meta.point_count}</div>
            <div><strong>Trees:</strong> {meta.tree_count_estimate}</div>
          </div>
        )}
        <hr />
        <label style={{ display: 'block', marginBottom: '8px' }}>
          <input type="checkbox" checked={classified} onChange={() => setClassified(c => !c)} /> Show classified colors
        </label>
        <button onClick={startStream} disabled={streamStatus === 'streaming'} style={{ width: '100%', marginBottom: '6px' }}>Start Stream</button>
        <button onClick={stopStream} style={{ width: '100%' }}>Stop Stream</button>
        <div style={{ marginTop: '10px', fontSize: '0.75rem' }}>Status: {streamStatus}</div>
        <hr />
        <div style={{ fontSize: '0.7rem' }}>
          <p>Toggle to switch between raw height-based coloring and species classification coloring.</p>
          <p>Streaming simulates per-tree classification metrics arriving from backend.</p>
        </div>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        {loading && <div style={{ position: 'absolute', top: 10, left: 10 }}>Loading point cloud...</div>}
        {error && <div style={{ position: 'absolute', top: 30, left: 10, color: 'red' }}>{error}</div>}
        {!loading && points && points.x.length === 0 && !error && <div style={{ position: 'absolute', top: 10, left: 10 }}>No points returned.</div>}
        <PointCloudViewer
          points={points || { x: [], y: [], z: [], z_norm: [], species_id: [] }}
          classified={classified}
          onAutoFit={(info) => setFitInfo(info)}
        />
        {fitInfo && <div style={{ position: 'absolute', bottom: 10, left: 10, fontSize: '0.6rem', opacity: 0.6 }}>Radius: {fitInfo.radius.toFixed(1)}</div>}
      </div>
    </div>
  );
}

export default App;
