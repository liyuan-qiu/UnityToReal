# TagTraining6 / StomachTraining6 对比实验记录

本文记录 `trainingData` 下 **tag 对齐（TagTraining\*）** 与 **stomach 对齐（StomachTraining6）** 的多次实验结果。  
内参统一使用：`trainingData/trainingData/capsule_intrinsics.npz`。

---

## 0. 内参（新标定）

文件：`trainingData/trainingData/capsule_intrinsics.npz`

| 项 | 值 |
|----|-----|
| `fx, fy` | ≈ 755.48, 754.85 |
| `cx, cy` | ≈ 648.49, 371.44（按 **1080×720**） |
| `dist` | `k1,k2,p1,p2,k3` ≈ −0.39645, 0.16694, −0.00245, −0.00174, −0.04199 |
| 理论 HFOV | ≈ **71.11°**（`2·atan(W/(2·fx))`, W=1080） |
| 理论 `lensShift`（1080×720） | \(X=-(c_x-W/2)/W\approx\mathbf{-0.1005}\)，\(Y=(c_y-H/2)/H\approx\mathbf{+0.0159}\) |

对比脚本对 **real** 做 `cv2.undistort` 时会用到上述 `K` + `dist`（Unity 侧无 Brown–Conrady）。

---

## 1. 数据路径

| 角色 | 路径 |
|------|------|
| Real tag | `trainingData/trainingData/Photo{1..7}_tag.jpg` |
| Real stomach（NoTag） | `trainingData/trainingData/Photo{1..7}_NoTag.jpg` |
| Unity tag 采集（最终） | `trainingData/TagTraining6/`（1080×720） |
| Unity stomach 采集 | `trainingData/StomachTraining6/`（1080×720） |
| Unity 姿态 CSV | `trainingData/trainingData/camera_pose_unity_cam2tag_face.csv` |
| Pose 导出校验 | `*/unity_camera_quat_export.csv`（与输入 quat 一致，\|q·q\|≈1） |

说明：早期 `StomachTraing` 为 **640×480**，勿与 `StomachTraining6` 混淆。

---

## 2. Tag 侧：lensShift.X 迭代 → TagTraining6

对比脚本：`compare_testphoto_undist_vs_unity.py`  
流程：real **undistort**（无后处理 FOV scale）→ 搜 Unity 的 `rot / sx,sy / dx,dy`。  
输出目录：`compare_out/training_tag_vs_TagTraining*_newK/`。

### 2.1 各版汇总（median）

| 版本 | 分辨率 | 假定 `lensShift.X` | dx median | dy median | Unity sx,sy mean | edgeNCC mean | 备注 |
|------|--------|-------------------|-----------|-----------|------------------|--------------|------|
| TagTraining2 | 640 时代设定 | −0.1005 | **+180** | +22 | 0.77, 0.80 | ~0.33 | |
| TagTraining3 | 同上 | +0.1005 | **−40** | +22 | 0.76, 0.80 | ~0.33 | X 过冲 |
| TagTraining4 | 同上 | ~**0.064**（插值） | **0** | +22 | 0.77, 0.80 | ~0.34 | X 对齐 |
| TagTraining5 | 改 **1080×720** | −0.1005（理论） | **+162** | +22 | 0.75, 0.80 | **~0.50** | 分辨率修正后 NCC 升 |
| **TagTraining6** | 1080×720 | 从 −0.1005 回调 | **+114** | **+22** | **0.75, 0.80** | **~0.50** | 当前 tag 基准 |

全部最佳匹配均为 **rot = 180**。

### 2.2 TagTraining6 逐图（`training_tag_vs_TagTraining6_newK`）

| id | rot | sx | sy | dx | dy | edgeNCC |
|----|-----|----|----|----|----|---------|
| 1 | 180 | 0.75 | 0.80 | +118 | +20 | 0.464 |
| 2 | 180 | 0.75 | 0.80 | +114 | +24 | 0.441 |
| 3 | 180 | 0.75 | 0.80 | +108 | +22 | 0.527 |
| 4 | 180 | 0.75 | 0.80 | +122 | +18 | 0.617 |
| 5 | 180 | 0.75 | 0.80 | +116 | +24 | 0.388 |
| 6 | 180 | 0.75 | 0.80 | +104 | +22 | 0.469 |
| 7 | 180 | 0.75 | 0.80 | +110 | +24 | 0.572 |

**采用的全局 Tag6 几何残差（用于后续 stomach）：**

| 量 | 值 | 含义 |
|----|-----|------|
| Unity `sx, sy` | **0.75, 0.80** | 对齐时 Unity 相对去畸变 real 需缩小 |
| Unity `dx, dy` | **+114, +22**（median） | 对齐时 Unity 平移（px，1080×720） |
| rot | **180** | 固定 |
| 反变换到 real | scale **(1.3333, 1.2500)**，shift **(−114, −22)** | 把 real 烤进与 Unity 同一图像帧 |

线性插值（Tag2/Tag3，错误分辨率时期）：`lensShift.X≈0.064` 使 dx→0；换成真 1080 后需重新调（见 Tag5→Tag6）。

### 2.3 重跑命令（tag）

```powershell
python compare_testphoto_undist_vs_unity.py `
  --intrinsics "trainingData\trainingData\capsule_intrinsics.npz" `
  --real-dir "trainingData\trainingData" `
  --unity-dir "trainingData\TagTraining6" `
  --ids 1-7 `
  --real-name "Photo{id}_tag.jpg" `
  --unity-name "Photo{id}_tag_Unity.jpg" `
  --out-name "training_tag_vs_TagTraining6_newK"
```

---

## 3. Stomach 侧：多次调整实验（重点）

Real：`Photo*_NoTag.jpg`  
Unity：`StomachTraining6`（**1080×720**）  
脚本：`compare_stomach_overlap_from_tag6.py`、`compare_stomach_angle_correct.py`、`compare_stomach_xy_refine.py`、`compare_stomach_raw_dxdy_scale.py`

共同结论：**tag 上稳定的 2D 残差不能直接让 stomach 全局重合**——tag 对齐的是平面标记；stomach 是 3D 腔体 + 不同外观/光照，edge-NCC 低且参数跨图发散。

---

### 3.1 实验 A — Tag6 warp 烤进 real（推荐基线流程）

**目录：** `compare_out/stomach_NoTag_vs_StomachTraining6_tag6warp/`  
**脚本：** `compare_stomach_overlap_from_tag6.py`

| 步骤 | 处理 |
|------|------|
| Real | `undistort(K,dist)` → `center_scale(1.3333, 1.2500)` → `shift(−114, −22)` |
| Unity | **rot180**（无额外 scale/shift） |
| 输出 | 重叠裁剪 **964×696**，yellow/cyan + panels |

| 指标 | 值 |
|------|-----|
| mean \|real−unity\| | mean **34.4**，median **34.5** |
| Unity 分辨率 | **1080×720**（确认） |

说明：首次跑时 `StomachTraining6` 为空，曾 fallback 到 640×480 的 `StomachTraing`；补齐 1080 数据后已重跑覆盖。

```powershell
python compare_stomach_overlap_from_tag6.py `
  --unity-dir "trainingData\StomachTraining6" `
  --out-name "stomach_NoTag_vs_StomachTraining6_tag6warp"
```

---

### 3.2 实验 B — 在 A 之上做残余角度

**目录：** `compare_out/stomach_NoTag_vs_StomachTraining6_angle/`  
**脚本：** `compare_stomach_angle_correct.py`  
在 A 的 real bake + Unity rot180 后，搜残余角 ∈ [−20°, +20°]。

| id | 残余角 (°) | 额外 dx,dy | edgeNCC | mean\|d\| |
|----|-----------|------------|---------|----------|
| 1 | −0.50 | −26, +404 | 0.215 | 35.1 |
| 2 | +0.75 | −16, −110 | 0.245 | 32.4 |
| 3 | +1.00 | −14, −110 | 0.271 | 28.6 |
| 4 | +0.75 | −18, −110 | 0.173 | 43.8 |
| 5 | −1.00 | +158, −110 | 0.198 | 35.7 |
| 6 | −1.00 | +258, −110 | 0.229 | 36.0 |
| 7 | +0.75 | −18, −110 | 0.207 | 47.2 |

| 结论 | |
|------|--|
| 中位残余角 | **+0.75°**（几乎可忽略） |
| mean \|d\| | ~37（未优于 A） |
| 含义 | stomach 主问题**不是**固定转角 |

---

### 3.3 实验 C — 先 undistort+Tag6 scale，再单独搜 XY

**目录：**  
- 逐图：`compare_out/stomach_NoTag_vs_StomachTraining6_xy/`  
- 统一中位数：`compare_out/stomach_NoTag_vs_StomachTraining6_xy_median/`  

**脚本：** `compare_stomach_xy_refine.py`  
**不**再烤入 Tag6 的 (±114, ±22)；Unity rot180；平移加在 Unity 上。

| id | dx | dy | edgeNCC |
|----|----|----|---------|
| 1 | −4 | +304 | 0.136 |
| 2 | −44 | +88 | 0.152 |
| 3 | +174 | +54 | 0.089 |
| 4 | +122 | +290 | 0.093 |
| 5 | +162 | +104 | 0.143 |
| 6 | −80 | +128 | 0.107 |
| 7 | +190 | +62 | 0.073 |

| 汇总 | |
|------|--|
| median dx,dy | **(+122, +104)** |
| mean edgeNCC | ~**0.11**（远低于 tag 的 ~0.50） |
| 相对 Tag6 (+114,+22) | **dx 接近，dy 明显更大且跨图不稳** |

统一用中位数时 mean\|d\|≈34.8，与实验 A 接近，但几何一致性仍差。

---

### 3.4 实验 D — 完全不做 undistort：先 dx/dy 再 scale（曾试 rot0/180）

**目录：** `compare_out/stomach_raw_dxdy_scale/`  
**脚本：** `compare_stomach_raw_dxdy_scale.py --also-try-0`

结果：各图最优 rot 在 0/180 间跳变，sx/sy/dx/dy 发散；mean NCC≈0.18。  
**不推荐**（与「始终 rot180」约定冲突）。

---

### 3.5 实验 E — 固定 rot180 后，再 sx/sy + dx/dy（无 undistort）

**目录：** `compare_out/stomach_raw_rot180_scale_xy/`  
**约定：** Unity **必须先 rot180**，再搜 scale 与平移。

| id | sx | sy | dx | dy | edgeNCC |
|----|----|----|----|----|---------|
| 1 | 1.05 | 0.70 | +84 | +464 | 0.150 |
| 2 | 0.85 | 1.40 | +156 | +382 | 0.209 |
| 3 | 0.95 | 0.70 | −162 | +30 | 0.200 |
| 4 | 1.00 | 1.35 | −340 | +324 | 0.143 |
| 5 | 0.70 | 1.30 | +62 | +368 | 0.167 |
| 6 | 0.85 | 0.75 | +222 | +92 | 0.146 |
| 7 | 1.40 | 1.30 | +282 | +84 | 0.171 |

| 汇总 | |
|------|--|
| median sx,sy | **(0.95, 1.30)** |
| median dx,dy | **(+84, +324)** |
| mean edgeNCC | ~**0.17** |
| mean \|d\| | ~45.9 |

相对 tag 的稳定 `(0.75, 0.80)/(+114,+22)`，stomach raw 解仍**高度发散**。

```powershell
python compare_stomach_raw_dxdy_scale.py --out-name "stomach_raw_rot180_scale_xy"
```

---

## 4. 实验对照总表（Stomach）

| 实验 | undistort | FOV scale | 固定平移 | 角度 | rot180 | 典型 NCC / \|d\| | 输出目录 |
|------|-----------|-----------|----------|------|--------|------------------|----------|
| A Tag6warp | ✓ | Tag6 1/s | Tag6 反变换 | — | ✓ | \|d\|≈34 | `..._tag6warp` |
| B +angle | ✓ | Tag6 | Tag6 + 搜角 | ~±1° | ✓ | NCC≈0.22 | `..._angle` |
| C +XY refine | ✓ | Tag6 | **重搜** | — | ✓ | NCC≈0.11 | `..._xy` / `_xy_median` |
| D raw 自由 rot | ✗ | 搜 | 搜 | — | 可选 | NCC≈0.18 | `stomach_raw_dxdy_scale` |
| E raw 固定 180 | ✗ | 搜 | 搜 | — | **强制** | NCC≈0.17 | `stomach_raw_rot180_scale_xy` |

---

## 5. 为何 tag 对齐好、stomach 仍差

1. **Tag**：实拍与 Unity 是同一平面标记；edge-NCC 锁定标记边缘 → blend 易「重合」。  
2. **Stomach**：真实/假体胃壁 ≠ Unity 模型纹理与高光；即使相机正确也会 yellow/cyan 重影。  
3. **Tag6 的 0.75×0.80** 是平面、特定距离下的 2D 残差；胃为弯曲 3D，全局 scale/shift 无法同时对齐所有深度（视差）。  
4. Stomach 上 NCC 低、dx/dy 跨图乱跳 → 更像 **内容/3D/场景误差**，不是再调一个全局 `lensShift` 就能齐。

---

## 6. 当前建议参数快查

| 用途 | 建议 |
|------|------|
| Unity 渲染分辨率 | **1080×720** |
| Unity HFOV | ≈ **71.11°** |
| Unity `lensShift` 初值 | X≈−0.1005，Y≈+0.0159；再按 tag 对比微调 X（Tag6 时 dx 仍约 +114） |
| Tag 对比固定 | **rot180**；残差 Unity shrink ≈ **0.75×0.80**，dx,dy≈**(+114,+22)** |
| Stomach 基线对比 | 实验 A：real und+scale(1.333,1.25)+shift(−114,−22)，Unity rot180 |
| Stomach 再拧 XY/角度 | 改善有限；见实验 B/C/E |

---

## 7. 相关脚本一览

| 脚本 | 作用 |
|------|------|
| `regen_unity_cam2tag_face.py` | tag CSV → Unity 姿态 CSV |
| `compare_testphoto_undist_vs_unity.py` | tag：undistort real vs Unity |
| `compare_stomach_overlap_from_tag6.py` | stomach：Tag6 warp + overlap |
| `compare_stomach_angle_correct.py` | stomach：残余角 |
| `compare_stomach_xy_refine.py` | stomach：und 后搜 XY |
| `compare_stomach_raw_dxdy_scale.py` | stomach：无 und，rot180 后 sx/sy/dx/dy |
| `PoseCsvAutoCapture.cs` | Unity 按 CSV 采 RGB/Depth |

---

*记录对应会话内 TagTraining2–6 与 StomachTraining6 多次迭代；数值以各 `compare_out/*/compare_summary.csv` 为准。*
