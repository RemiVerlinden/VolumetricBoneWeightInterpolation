# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python script that converts 3D mesh vertex attributes (specifically bone weights for skinned meshes) into 3D volumetric textures using natural neighbor interpolation. The main goal is to transform irregular vertex data into a dense voxel grid suitable for raymarching applications.

**Key Dependencies:**
- `naturalneighbor` package from https://github.com/innolitics/natural-neighbor-interpolation
- numpy (already installed)

**Core Workflow:**
1. Input: 3D model with vertex bone weight attributes (4 separate bone weights per vertex)
2. Process: Use natural neighbor interpolation to generate volumetric data
3. Output: 3D volumetric texture containing interpolated bone weight values

**Purpose:** Enable skinned raymarching by providing volumetric bone weight data instead of mesh-based data.

## Python Environment

**CRITICAL:** ALWAYS use Windows Python installation via WSL interop:
- Use `python.exe` instead of `python` or `python3`
- Use `pip.exe` instead of `pip`
- This ensures commands run against Windows Python 3.11.9, not WSL Python 3.12.3

## Development Commands

```bash
# Test Python environment
python.exe test_basic.py

# Install numpy dependency
pip.exe install numpy

# Clone and build naturalneighbor dependency
git clone https://github.com/innolitics/natural-neighbor-interpolation.git
cd natural-neighbor-interpolation
python.exe setup.py build_ext --inplace
python.exe setup.py install --user
```

## Dependencies Setup

**STATUS: COMPLETED** - naturalneighbor package successfully built and installed.

Since `naturalneighbor` cannot be installed via pip, it was built manually:

1. ✅ Clone the repository: `git clone https://github.com/innolitics/natural-neighbor-interpolation.git`
2. ✅ Build from source using the repository's setup.py
3. ✅ The package provides 3D natural neighbor interpolation using discrete Sibson method
4. ✅ Compatible with Python 3.11.9 and numpy 1.26.4

**Build Results:**
- Successfully compiled with Windows Python 3.11.9
- Compatible with numpy 1.26.4 (no compatibility issues found)
- Package installed to user directory and import verified
- Only deprecation warnings about numpy.distutils (non-breaking)

## Architecture Notes

- Single-purpose Python script for vertex attribute interpolation
- Main computational challenge: efficiently interpolating sparse vertex data to dense 3D grid
- Natural neighbor interpolation chosen for smooth transitions between bone weight regions
- Output format compatible with graphics pipeline volumetric texture requirements
- Uses discrete Sibson interpolation method for fast 3D natural neighbor interpolation

## Implementation Status

**Completed Features:**
- ✅ PLY file loading with bone weight extraction
- ✅ Diagnostic output showing mesh statistics and bone weight distribution
- ✅ 52-bone volume interpolation with memory-efficient batch processing
- ✅ Top-4 bone influence extraction per voxel with proper renormalization
- ✅ 2D texture layout preparation (z-slices arranged in square grid)
- ✅ Fixed numpy compatibility issues in naturalneighbor package

**Usage:**
```bash
# Install required package
pip.exe install plyfile

# Run with small test resolution
python.exe volumetric_bone_weight_baker.py pirate.ply --resolution 16

# Run with full resolution (289³ = 24,188,169 voxels)
python.exe volumetric_bone_weight_baker.py pirate.ply --resolution 289
```

**Model Specifications (pirate.ply):**
- Vertices: 14,241
- Bones: 52 (Mixamo rig)
- Bounds: [-0.7175, -0.0325, -0.7275] to [0.7225, 1.4075, 0.7125]
- Bone weight distribution: 55.7% vertices have 1 weight, 28.3% have 2, 13.5% have 3, 2.5% have 4