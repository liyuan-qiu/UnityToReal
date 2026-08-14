# Depth-Anything pair pack (20260813 NoTagUnity10)

## Contents
- 
gb/Photo{i}.png : real NoTag, undistorted (20260813 K), color-masked (both-mask)
- depth/Photo{i}.png : Unity depth after same warp as RGB (rot180, sx/sy=0.75/0.85, dx/dy=+32/+78), masked
- mask/Photo{i}.png : intersection color mask
- depth_color/ : turbo preview of warped depth
- preview/ : side-by-side compare panels
- manifest.csv

## Important
1. Depth PNG is **Unity visualization (uint8)**, NOT confirmed metric meters.
2. Only 3 frames — for pipeline smoke / small fine-tune, not full training.
3. Intrinsics: 	rainingData20260813/capsule_intrinsics.npz
4. Use mask when computing loss; ignore black pixels.

## Suggested Depth-Anything use
- Metric fine-tune: first convert Unity depth to **linear meters** (near/far + depth shader), then replace depth/.
- Relative fine-tune: can use current depth as relative target inside mask (scale/shift invariant losses).
