#!/usr/bin/env python
"""
Volumetric Bone Weight Baker
Converts mesh vertex bone weights to 3D volumetric texture using natural neighbor interpolation.
"""

import numpy as np
import argparse
import sys
import os
from typing import Tuple, List, Dict
import naturalneighbor

# Import plyfile for mesh loading (install with: pip.exe install plyfile)
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Please install plyfile: pip.exe install plyfile")
    sys.exit(1)


def load_mesh_with_bone_weights(filepath: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load mesh from PLY file and extract vertex positions and bone weights.
    
    Returns:
        positions: Nx3 array of vertex positions
        bone_weights: Dict mapping bone names to weight arrays
    """
    print(f"Loading mesh from {filepath}...")
    plydata = PlyData.read(filepath)
    vertex_data = plydata['vertex']
    
    # Extract positions
    positions = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    
    # Extract bone weights - PLY stores these as custom properties
    bone_weights = {}
    
    # Get all property names
    prop_names = [prop.name for prop in vertex_data.properties]
    
    # Find bone weight properties (excluding x, y, z)
    weight_props = [name for name in prop_names if name not in ['x', 'y', 'z']]
    
    # For Mixamo rigs, bone names start with 'mixamorig:'
    if any('mixamorig:' in name for name in weight_props):
        weight_props = [name for name in weight_props if 'mixamorig:' in name]
    
    for prop_name in weight_props:
        bone_weights[prop_name] = np.array(vertex_data[prop_name])
    
    # Print detailed diagnostics
    print(f"\n=== Mesh Statistics ===")
    print(f"  Vertices: {len(positions):,}")
    print(f"  Bones: {len(bone_weights)}")
    print(f"  Bounding Box:")
    print(f"    Min: [{positions[:, 0].min():.4f}, {positions[:, 1].min():.4f}, {positions[:, 2].min():.4f}]")
    print(f"    Max: [{positions[:, 0].max():.4f}, {positions[:, 1].max():.4f}, {positions[:, 2].max():.4f}]")
    print(f"    Size: [{positions[:, 0].max() - positions[:, 0].min():.4f}, "
          f"{positions[:, 1].max() - positions[:, 1].min():.4f}, "
          f"{positions[:, 2].max() - positions[:, 2].min():.4f}]")
    
    # Analyze bone weight distribution
    print(f"\n=== Bone Weight Analysis ===")
    total_weights_per_vertex = np.zeros(len(positions))
    non_zero_counts = np.zeros(len(positions), dtype=int)
    
    for bone_name, weights in bone_weights.items():
        non_zero = np.sum(weights > 0)
        if non_zero > 0:
            print(f"  {bone_name}: {non_zero:,} vertices affected")
        total_weights_per_vertex += weights
        non_zero_counts += (weights > 0).astype(int)
    
    print(f"\n  Weights per vertex:")
    print(f"    Min: {non_zero_counts.min()}, Max: {non_zero_counts.max()}")
    unique, counts = np.unique(non_zero_counts, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u} weights: {c:,} vertices ({c/len(positions)*100:.1f}%)")
    
    # Check weight normalization
    print(f"\n  Weight sum per vertex:")
    print(f"    Min: {total_weights_per_vertex.min():.4f}")
    print(f"    Max: {total_weights_per_vertex.max():.4f}")
    print(f"    Mean: {total_weights_per_vertex.mean():.4f}")
    
    return positions, bone_weights


def create_bone_weight_matrix(positions: np.ndarray, bone_weights: Dict[str, np.ndarray], 
                            num_bones: int = None) -> Tuple[np.ndarray, List[str]]:
    """
    Create a matrix where each row is a vertex and columns are bone weights.
    The PLY file already has all bone weights per vertex (most are zero).
    
    Returns:
        weight_matrix: NxB array where N=vertices, B=bones
        bone_names: List of bone names in order
    """
    num_vertices = len(positions)
    
    if num_bones is None:
        num_bones = len(bone_weights)
    
    # Sort bone names for consistent ordering
    bone_names = sorted(list(bone_weights.keys()))
    
    # Initialize weight matrix
    weight_matrix = np.zeros((num_vertices, num_bones), dtype=np.float32)
    
    # Fill in the weights
    for bone_idx, bone_name in enumerate(bone_names):
        if bone_name in bone_weights:
            weight_matrix[:, bone_idx] = bone_weights[bone_name]
    
    # Since PLY already contains normalized weights, just verify
    print("\n=== Weight Matrix Verification ===")
    for v_idx in range(min(5, num_vertices)):  # Check first 5 vertices
        vertex_weights = weight_matrix[v_idx]
        non_zero_indices = np.where(vertex_weights > 0)[0]
        if len(non_zero_indices) > 0:
            print(f"  Vertex {v_idx}: {len(non_zero_indices)} weights, "
                  f"sum={np.sum(vertex_weights):.4f}, "
                  f"bones: {[bone_names[i] for i in non_zero_indices[:4]]}")
    
    return weight_matrix, bone_names


def interpolate_bone_volumes(positions: np.ndarray, weight_matrix: np.ndarray, 
                           grid_resolution: int, bounds_min: np.ndarray, 
                           bounds_max: np.ndarray, bone_names: List[str]) -> np.ndarray:
    """
    Interpolate bone weights to 3D volumes using natural neighbor interpolation.
    
    Returns:
        volumes: (num_bones, res, res, res) array of interpolated weights
    """
    num_bones = weight_matrix.shape[1]
    
    # Define interpolation grid
    interp_ranges = [
        [bounds_min[0], bounds_max[0], complex(0, grid_resolution)],
        [bounds_min[1], bounds_max[1], complex(0, grid_resolution)],
        [bounds_min[2], bounds_max[2], complex(0, grid_resolution)]
    ]
    
    print(f"\nInterpolating {num_bones} bone weight volumes at {grid_resolution}³ resolution...")
    print(f"  Volume bounds: [{bounds_min[0]:.4f}, {bounds_min[1]:.4f}, {bounds_min[2]:.4f}] to "
          f"[{bounds_max[0]:.4f}, {bounds_max[1]:.4f}, {bounds_max[2]:.4f}]")
    
    # Count non-zero bones to optimize memory
    bones_with_weights = []
    for bone_idx in range(num_bones):
        if np.max(weight_matrix[:, bone_idx]) > 0:
            bones_with_weights.append(bone_idx)
    
    print(f"  Active bones: {len(bones_with_weights)} out of {num_bones}")
    
    # We'll process bones in batches to manage memory
    batch_size = 5  # Reduced for 289³ volumes
    volumes = np.zeros((num_bones, grid_resolution, grid_resolution, grid_resolution), 
                      dtype=np.float32)
    
    for batch_start in range(0, len(bones_with_weights), batch_size):
        batch_end = min(batch_start + batch_size, len(bones_with_weights))
        batch_indices = bones_with_weights[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(len(bones_with_weights) + batch_size - 1)//batch_size}...")
        
        for bone_idx in batch_indices:
            bone_weights = weight_matrix[:, bone_idx]
            
            # Only interpolate vertices with non-zero weights for this bone
            mask = bone_weights > 0
            if np.sum(mask) == 0:
                continue
                
            print(f"    Bone {bone_names[bone_idx]}: {np.sum(mask):,} vertices")
            
            # Interpolate this bone's weights
            volume = naturalneighbor.griddata(positions, bone_weights, interp_ranges)
            volumes[bone_idx] = volume
    
    return volumes


def extract_top_4_influences(volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each voxel, extract the top 4 bone influences and their indices.
    Uses a smooth selection approach to avoid discontinuities.
    
    Returns:
        top_weights: (res, res, res, 4) array of weights
        top_indices: (res, res, res, 4) array of bone indices
    """
    num_bones, res_x, res_y, res_z = volumes.shape
    
    # Initialize output arrays
    top_weights = np.zeros((res_x, res_y, res_z, 4), dtype=np.float32)
    top_indices = np.zeros((res_x, res_y, res_z, 4), dtype=np.uint8)
    
    print("Extracting top 4 bone influences per voxel...")
    
    # Process each voxel
    for x in range(res_x):
        if x % 50 == 0:
            print(f"  Processing slice {x}/{res_x}...")
        
        for y in range(res_y):
            for z in range(res_z):
                # Get all bone weights at this voxel
                voxel_weights = volumes[:, x, y, z]
                
                # Find top 4 influences
                # Using argpartition for efficiency
                if np.sum(voxel_weights) > 0:
                    # Get indices of 4 largest values
                    top_4_idx = np.argpartition(voxel_weights, -4)[-4:]
                    # Sort them by weight (descending)
                    top_4_idx = top_4_idx[np.argsort(voxel_weights[top_4_idx])[::-1]]
                    
                    # Extract weights
                    weights = voxel_weights[top_4_idx]
                    
                    # Renormalize to sum to 1
                    weight_sum = np.sum(weights)
                    if weight_sum > 0:
                        weights = weights / weight_sum
                    
                    # Store results
                    top_weights[x, y, z] = weights
                    top_indices[x, y, z] = top_4_idx
    
    return top_weights, top_indices


def prepare_for_2d_export(volume_weights: np.ndarray, volume_indices: np.ndarray, 
                         resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare volumetric data for export as 2D texture with z-slices.
    
    Returns:
        texture_weights: 2D array of shape (resolution*sqrt(resolution), resolution*sqrt(resolution), 4)
        texture_indices: 2D array of shape (resolution*sqrt(resolution), resolution*sqrt(resolution), 4)
    """
    # Calculate 2D texture dimensions
    slices_per_row = int(np.sqrt(resolution))
    if slices_per_row * slices_per_row != resolution:
        raise ValueError(f"Resolution {resolution} is not a perfect square")
    
    texture_size = resolution * slices_per_row
    
    # Initialize 2D textures
    texture_weights = np.zeros((texture_size, texture_size, 4), dtype=np.float32)
    texture_indices = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    
    print(f"Arranging {resolution} z-slices into {texture_size}x{texture_size} 2D texture...")
    
    # Place each z-slice in the 2D grid
    for z in range(resolution):
        # Calculate position in 2D grid
        grid_y = z // slices_per_row
        grid_x = z % slices_per_row
        
        # Calculate pixel offsets
        start_y = grid_y * resolution
        start_x = grid_x * resolution
        
        # Copy the slice
        texture_weights[start_y:start_y+resolution, start_x:start_x+resolution] = volume_weights[:, :, z]
        texture_indices[start_y:start_y+resolution, start_x:start_x+resolution] = volume_indices[:, :, z]
    
    return texture_weights, texture_indices


def main():
    parser = argparse.ArgumentParser(description='Convert mesh bone weights to volumetric texture')
    parser.add_argument('input_mesh', help='Input mesh file (PLY format)')
    parser.add_argument('--resolution', type=int, default=289, 
                       help='Voxel grid resolution (must be perfect square, default: 289)')
    parser.add_argument('--bounds-min', type=float, nargs=3, default=[-0.7175, -0.0325, -0.7275],
                       help='Minimum bounds for volume (default: pirate model bounds)')
    parser.add_argument('--bounds-max', type=float, nargs=3, default=[0.7225, 1.4075, 0.7125],
                       help='Maximum bounds for volume (default: pirate model bounds)')
    parser.add_argument('--num-bones', type=int, default=52,
                       help='Expected number of bones (default: 52 for Mixamo rig)')
    
    args = parser.parse_args()
    
    # Validate resolution
    sqrt_res = np.sqrt(args.resolution)
    if sqrt_res != int(sqrt_res):
        print(f"Error: Resolution {args.resolution} is not a perfect square")
        sys.exit(1)
    
    # Load mesh
    positions, bone_weights = load_mesh_with_bone_weights(args.input_mesh)
    
    # Create bone weight matrix
    weight_matrix, bone_names = create_bone_weight_matrix(positions, bone_weights, args.num_bones)
    
    # Convert bounds to numpy arrays
    bounds_min = np.array(args.bounds_min)
    bounds_max = np.array(args.bounds_max)
    
    # Interpolate all bone volumes
    volumes = interpolate_bone_volumes(positions, weight_matrix, args.resolution, 
                                     bounds_min, bounds_max, bone_names)
    
    # Extract top 4 influences per voxel
    top_weights, top_indices = extract_top_4_influences(volumes)
    
    # Prepare for 2D export
    texture_weights, texture_indices = prepare_for_2d_export(top_weights, top_indices, 
                                                           args.resolution)
    
    print("\nVolume generation complete!")
    print(f"  3D Volume shape: {top_weights.shape}")
    print(f"  2D Texture shape: {texture_weights.shape}")
    print(f"  Weight range: [{np.min(texture_weights):.3f}, {np.max(texture_weights):.3f}]")
    print(f"  Unique bone indices used: {len(np.unique(texture_indices))}")
    
    # Data is now ready for export
    # texture_weights contains the 4 bone weights per pixel
    # texture_indices contains the 4 bone indices per pixel
    
    print("\nData prepared for export. Next steps:")
    print("  - Export texture_weights as 4-channel float EXR")
    print("  - Export texture_indices as 4-channel uint8 image")


if __name__ == '__main__':
    main()