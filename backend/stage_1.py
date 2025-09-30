import laspy
import numpy as np
import os

def analyze_boreal3d_uls_data(file_path):
    """
    Reads a Boreal3D.laz file, parses its 3D point cloud data,
    and generates a summary as required by the assignment.

    Args:
        file_path (str): The full path to the.laz file.
    """
    # Verify the file exists before proceeding
    if not os.path.exists(file_path):
        print(f"Error: The file was not found at the specified path: {file_path}")
        return

    try:
        # 1. Read the.laz file
        # Using a 'with' statement ensures the file is properly closed
        with laspy.open(file_path) as f:
            # It's efficient to read the header first for metadata
            header = f.header
            # Then read all the point data into memory
            las_data = f.read()

        print(f"--- LiDAR Data Summary for: {os.path.basename(file_path)} ---")

        # 2. Parse 3D point cloud data and generate the summary

        # --- Total number of points ---
        total_points = header.point_count
        print(f"\nTotal Number of Points: {total_points:,}")

        # --- Coordinate ranges (min/max for x, y, z) ---
        # Access the scaled x, y, and z coordinates
        x_coords = las_data.x
        y_coords = las_data.y
        z_coords = las_data.z

        # Calculate the min and max for each dimension
        min_coords = (x_coords.min(), y_coords.min(), z_coords.min())
        max_coords = (x_coords.max(), y_coords.max(), z_coords.max())

        print("\nCoordinate Ranges:")
        print(f"  X-axis: Min = {min_coords[0]:.3f}, Max = {max_coords[0]:.3f}")
        print(f"  Y-axis: Min = {min_coords[1]:.3f}, Max = {max_coords[1]:.3f}")
        print(f"  Z-axis: Min = {min_coords[2]:.3f}, Max = {max_coords[2]:.3f}")

        # --- Number of points for each classification code ---
        # The Boreal3D dataset uses the 'semantic_id' field for classification [3]
        classification_labels = {
            0: "Understory",
            1: "Terrain",
            2: "Leaf",
            3: "Wood"
        }

        # Determine which attribute to use for classification. Some LAS/LAZ files
        # may not contain 'semantic_id'. Fallback to standard 'classification'.
        if hasattr(las_data, 'semantic_id'):
            class_attr = las_data.semantic_id
            attr_name = 'semantic_id'
        else:
            # las_data.classification always exists in standard LAS specs.
            class_attr = las_data.classification
            attr_name = 'classification'

        # Use numpy.unique to efficiently get the codes and their counts
        class_codes, counts = np.unique(class_attr, return_counts=True)

        print(f"\nPoint Count by Classification Code ({attr_name}):")
        # Create a dictionary for easy lookup
        point_counts = dict(zip(class_codes, counts))

        # Print the count for each known label, handling cases where a code might be absent
        for code, label in classification_labels.items():
            count = point_counts.get(code, 0)
            print(f"  - Code {code} ({label}): {count:,} points")

        print("\n--- End of Summary ---")

    except Exception as e:
        msg = str(e)
        if 'No LazBackend selected' in msg:
            print("Error: No LAZ decompression backend is available.\n"
                  "Install one of the optional backends for laspy. For example:\n"
                  "  pip install \"laspy[lazrs,laszip]\"\n"
                  "or (conda):\n"
                  "  conda install -c conda-forge laspy lazrs laszip\n"
                  "After installation, re-run this script.")
        else:
            print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # IMPORTANT: Replace the placeholder below with the actual path to your
    # Boreal3D ULS.laz file.
    #
    # Example for Windows: "C:\\Users\\YourUser\\Downloads\\Boreal3D\\EasyPlot1\\ULS.laz"
    # Example for macOS/Linux: "/home/youruser/Downloads/Boreal3D/EasyPlot1/ULS.laz"
    laz_file_path = "D:\\Boreal3D\\EasyPlot34\\ULS.laz"

    # Run the analysis function
    analyze_boreal3d_uls_data(laz_file_path)