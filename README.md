# VolumetricBoneWeightInterpolation

## What

- Python script: bakes vertex attributes from 3D model to 3D volumetric texture  
- Focus: skinned mesh, vertex bone weights  
- Uses natural neighbor interpolation  
- Populates dense voxel grid from irregular vertex data  
- Outputs: 3D volume with interpolated bone weights  
- For: skinned raymarching project (requires 3D volumetric textures for raymarching sampling, not mesh data)
