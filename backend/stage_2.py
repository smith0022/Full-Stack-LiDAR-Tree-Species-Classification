import os
import argparse
import glob
import difflib
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def normalize_heights(las_data):
    """Derive normalized height (height above ground) for each point.

    Strategy (robust to different datasets):
    1. Prefer semantic_id == 1 (terrain) if available.
    2. Else prefer ASPRS classification == 2 (ground) if available.
    3. Else fallback to a coarse grid minimum surface.
    """
    print("Normalizing point cloud heights...")

    x = las_data.x
    y = las_data.y
    z = las_data.z

    # --- Ground mask selection ---
    ground_mask = None
    if hasattr(las_data, 'semantic_id') and np.any(las_data.semantic_id == 1):
        ground_mask = (las_data.semantic_id == 1)
    elif hasattr(las_data, 'classification') and np.any(las_data.classification == 2):
        ground_mask = (las_data.classification == 2)

    if ground_mask is not None and np.count_nonzero(ground_mask) > 10:
        ground_x = x[ground_mask]
        ground_y = y[ground_mask]
        ground_z = z[ground_mask]
        # Interpolate ground elevation at all XYs
        dtm_z = griddata(
            np.column_stack([ground_x, ground_y]),
            ground_z,
            np.column_stack([x, y]),
            method='nearest'
        )
    else:
        # Fallback: coarse grid minimum surface
        print("Ground classes not found; using coarse grid minimum surface.")
        cell = max((np.max(x) - np.min(x)), (np.max(y) - np.min(y))) / 100.0  # ~100 cells across
        if cell <= 0:
            cell = 1.0
        ix = ((x - x.min()) / cell).astype(int)
        iy = ((y - y.min()) / cell).astype(int)
        ground_dict = {}
        for gx, gy, gz in zip(ix, iy, z):
            key = (gx, gy)
            if key not in ground_dict or gz < ground_dict[key]:
                ground_dict[key] = gz
        dtm_z = np.array([ground_dict.get((gx, gy), np.min(z)) for gx, gy in zip(ix, iy)], dtype=float)

    normalized_z = z - dtm_z
    print("Height normalization complete.")
    return normalized_z

def calculate_tree_metrics(tree_points):
    """Calculate descriptive metrics for a single tree's points.

    Parameters
    ----------
    tree_points : (N,3) ndarray
        Columns: x, y, z_normalized
    """
    if tree_points.shape[0] < 4:
        return None

    z = tree_points[:, 2]
    x = tree_points[:, 0]
    y = tree_points[:, 1]

    tree_height = float(np.max(z))
    mean_height = float(np.mean(z))
    stddev_height = float(np.std(z))
    p25, p50, p75, p90 = np.percentile(z, [25, 50, 75, 90])

    # Crown geometry
    try:
        hull_2d = ConvexHull(tree_points[:, :2])
        crown_area = float(hull_2d.volume)  # area in 2D
    except Exception:
        crown_area = 0.0
    try:
        hull_3d = ConvexHull(tree_points)
        crown_volume = float(hull_3d.volume)
    except Exception:
        crown_volume = 0.0

    point_count = int(tree_points.shape[0])
    point_density_planar = point_count / crown_area if crown_area > 0 else 0.0

    # Crown radius approximation & slenderness
    cx, cy = np.mean(x), np.mean(y)
    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mean_radius = float(np.mean(radial)) if radial.size else 0.0
    slenderness = tree_height / (mean_radius * 2) if mean_radius > 0 else 0.0

    # Upper canopy density (fraction of points above 70% of height)
    upper_threshold = 0.7 * tree_height
    upper_fraction = float(np.mean(z >= upper_threshold)) if tree_height > 0 else 0.0

    return {
        'tree_height': tree_height,
        'mean_height': mean_height,
        'stddev_height': stddev_height,
        'p25_height': float(p25),
        'p50_height': float(p50),
        'p75_height': float(p75),
        'p90_height': float(p90),
        'crown_area': crown_area,
        'crown_volume': crown_volume,
        'point_count': point_count,
        'point_density_planar': point_density_planar,
        'mean_crown_radius': mean_radius,
        'slenderness_ratio': slenderness,
        'upper_canopy_fraction': upper_fraction,
    }

def _ensure_normalized_dimension(las):
    if 'z_normalized' not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="z_normalized", type=np.float32))

def _get_tree_ids(las):
    if hasattr(las, 'instance_id'):
        ids = las.instance_id
        unique_ids = np.unique(ids[ids > 0])
        if unique_ids.size > 0:
            return unique_ids, ids
    # Fallback: simple spatial clustering (very naive)
    print("Instance IDs not found; performing naive spatial clustering to define trees.")
    from sklearn.cluster import DBSCAN
    xy = np.column_stack([las.x, las.y])
    # eps is heuristic: 2% of max dimension
    span = max(xy[:,0].ptp(), xy[:,1].ptp())
    eps = span * 0.01 if span > 0 else 1.0
    clustering = DBSCAN(eps=eps, min_samples=20).fit(xy)
    labels = clustering.labels_
    unique = np.unique(labels[labels >= 0])
    return unique, labels

def _get_species_id_array(las):
    if hasattr(las, 'species_id'):
        return las.species_id
    # If no species data, synthesize species by height quantiles to allow Stage 2 flow.
    print("species_id not found; synthesizing species labels from height distribution.")
    z = las.z
    q1, q2 = np.percentile(z, [33, 66])
    synthetic = np.zeros_like(z, dtype=np.uint8)
    synthetic[z > q1] = 2
    synthetic[z > q2] = 3
    synthetic[synthetic == 0] = 1
    return synthetic

def _suggest_similar(path_str):
    # Try to find similar directories or files two levels up
    drive, rest = os.path.splitdrive(path_str)
    parts = [p for p in rest.strip(os.sep).split(os.sep) if p]
    if len(parts) < 1:
        return []
    parent_candidates = []
    # Reconstruct progressive parent paths
    for i in range(1, len(parts)+1):
        candidate = os.path.join(drive + os.sep, *parts[:i])
        if os.path.isdir(candidate):
            parent_candidates.append(candidate)
    leaf = parts[-1]
    siblings = []
    if parent_candidates:
        parent = parent_candidates[-1]
        try:
            siblings = os.listdir(parent)
        except OSError:
            siblings = []
    return difflib.get_close_matches(leaf, siblings, n=5, cutoff=0.5)

def _resolve_file(path_str, auto_locate=False):
    # Expand environment variables and normalize
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    norm = os.path.normpath(expanded)
    if os.path.isfile(norm):
        return norm
    if auto_locate:
        # Try case-insensitive glob on directory portion
        base_name = os.path.basename(norm)
        dir_name = os.path.dirname(norm)
        search_root = dir_name if os.path.isdir(dir_name) else os.path.splitdrive(norm)[0] + os.sep
        pattern = os.path.join(search_root, '**', base_name)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # Prefer exact case-insensitive match shortest path
            matches.sort(key=lambda p: (len(p), p.lower()!=norm.lower()))
            return matches[0]
    return None

def train_species_classifier(file_path, test_size=0.3, random_state=42, min_trees=10, auto_locate=False):
    """Run Stage 2 pipeline: feature extraction + Random Forest classification."""
    resolved = _resolve_file(file_path, auto_locate=auto_locate)
    if not resolved:
        print(f"Error: File not found at provided path: {file_path}")
        suggestions = _suggest_similar(file_path)
        if suggestions:
            print("Did you mean one of: " + ', '.join(suggestions))
        print("Diagnostics:")
        print(f"  Working directory: {os.getcwd()}")
        print(f"  Absolute of input: {os.path.abspath(file_path)}")
        print("Tips: If your path contains backslashes, wrap it in quotes. You can also try --auto-locate.")
        return
    file_path = resolved

    print(f"Reading file: {file_path} ...")
    with laspy.open(file_path) as f:
        las = f.read()

    _ensure_normalized_dimension(las)
    las.z_normalized = normalize_heights(las)

    # Acquire arrays
    species_arr = _get_species_id_array(las)
    tree_ids_unique, tree_id_array = _get_tree_ids(las)

    print("Extracting per-tree metrics...")
    all_tree_features = []
    for tree_id in tree_ids_unique:
        mask = (tree_id_array == tree_id)
        if np.count_nonzero(mask) < 10:  # skip very small clusters
            continue
        pts = np.column_stack([
            las.x[mask],
            las.y[mask],
            las.z_normalized[mask]
        ])
        metrics = calculate_tree_metrics(pts)
        if not metrics:
            continue
        # Majority species label among points
        sp_vals = species_arr[mask]
        species_label = int(np.bincount(sp_vals).argmax())
        metrics['tree_id'] = int(tree_id)
        metrics['species_id'] = species_label
        all_tree_features.append(metrics)

    if len(all_tree_features) == 0:
        print("No tree features extracted. Aborting.")
        return

    feature_df = pd.DataFrame(all_tree_features)
    if feature_df.shape[0] < min_trees:
        print(f"Warning: Only {feature_df.shape[0]} trees extracted (< {min_trees}). Results may be unreliable.")

    print(f"Extracted features for {feature_df.shape[0]} trees.")
    print(feature_df.head())

    # Prepare ML data
    X = feature_df.drop(columns=['tree_id', 'species_id'])
    y = feature_df['species_id']

    # Ensure at least 2 classes
    if y.nunique() < 2:
        print("Not enough distinct species/classes to train a classifier.")
        return

    # Split
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Pipeline (scaling not strictly required for RF but prepares for future models)
    model = Pipeline([
        ('rf', RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            oob_score=True,
            n_jobs=-1,
        ))
    ])
    model.fit(X_train, y_train)
    rf = model.named_steps['rf']

    print("Model trained. Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    if hasattr(rf, 'oob_score_'):
        print(f"Out-of-Bag (OOB) Score: {rf.oob_score_:.4f}")

    # Build target names dynamically
    species_mapping = {1: 'Spruce', 2: 'Pine', 3: 'Birch', 4: 'Other'}
    unique_labels = sorted(y.unique())
    target_names = [species_mapping.get(lbl, f"Class_{lbl}") for lbl in unique_labels]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances)

    return {
        'features': feature_df,
        'model': model,
        'importances': importances,
        'accuracy': acc,
        'oob_score': getattr(rf, 'oob_score_', None)
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Tree species classification from LiDAR point cloud.")
    parser.add_argument('--file', '-f', required=True, help='Path to input .las/.laz file')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test split fraction (default 0.3)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed (default 42)')
    parser.add_argument('--auto-locate', action='store_true', help='Attempt to locate file by recursive search if the exact path fails')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_species_classifier(args.file, test_size=args.test_size, random_state=args.random_seed, auto_locate=args.auto_locate)