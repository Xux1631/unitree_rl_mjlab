# 训练与复现手册（`quadruped_3320`）

本文档给出从环境安装、模型检查、训练到 `play` 演示的完整可复制命令。

- 上游：Unitree 官方 `unitree_rl_mjlab`，commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- 本机：Ubuntu 22.04，内核 6.8.0-138，NVIDIA GeForce RTX 4060 Laptop（**8 GB 显存**），
  24 核 CPU，31 GB 内存
- Conda 环境：`unitree_rl_mjlab`（Python 3.11）

---

## 0. 前置：驱动一致性（本机踩过的坑）

本机曾同时安装了 550 与 580 两套驱动，**内核态加载 550、用户态是 580**，
导致 `nvidia-smi` 报 `couldn't communicate with the NVIDIA driver`，且
`torch.cuda.is_available()` 为 `False`。torch 为 `2.14.1+cu130`，CUDA 13 要求驱动 ≥ 580。

判定与修复：

```bash
cat /sys/module/nvidia/version      # 内核态版本
ls -l /usr/lib/x86_64-linux-gnu/libcuda.so.1   # 用户态版本
# 若两者不一致：
sudo reboot
# 重启后确认：
cat /sys/module/nvidia/version      # 期望 580.178.04
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**在 `torch.cuda.is_available()` 为 `False` 之前不要启动训练**：训练脚本会在
`mjlab/utils/gpu.py:select_gpus()` 处报 `IndexError: list index out of range`。

---

## 1. 安装

```bash
cd ~/RC/mjlab/unitree_rl_mjlab
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab
pip install -e .
# 版本冲突处理：mujoco_warp 需要 mujoco<3.8（否则报 mjENBL_MULTICCD 缺失）
pip install "mujoco<3.8.0" scipy
```

---

## 2. 构建与检查自建模型

原始 STL 包（约 18 MB 压缩 / 51 MB 解压）需放在：

```
src/assets/robots/quadruped_3320/source/meshes_stl.zip
src/assets/robots/quadruped_3320/source/robot_solidworks_export.urdf   # 已入库
```

然后：

```bash
cd ~/RC/mjlab/unitree_rl_mjlab
export MPLCONFIGDIR=/tmp/mpl

# (a) 重建 A2 兼容 URDF + MJCF（幂等；会抽稀网格并简化碰撞体）
python scripts/build_a2_urdf.py

# (b) A2 接口一致性 + 站立姿态 + 关节运动方向
python scripts/verify_a2_urdf.py
#   -> RESULT: PASS

# (c) 四级台阶地形几何（ray-casting 逐毫米验证）
python scripts/check_stairs_terrain.py
#   -> RESULT: PASS - 4 steps x 0.13 m rise x 0.30 m tread, total rise 0.52 m

# (d) 物理稳定性（CPU MuJoCo，模拟 4 s）
python scripts/check_model_physics.py
#   -> RESULT: PASS - 无 NaN、无穿地、仅四足接地、机身偏移竖直 0.37 deg

# (e) 确认任务已注册
python scripts/list_envs.py | grep 3320
```

四个任务：

| task_id | 用途 |
|---|---|
| `Quadruped-3320-Stairs` | 训练任务；`play` 时恒定 0.4 m/s（达标档） |
| `Quadruped-3320-Stairs-0.6` | `play` 恒定 0.6 m/s（良好档） |
| `Quadruped-3320-Stairs-0.8` | `play` 恒定 0.8 m/s（优秀档） |
| `Quadruped-3320-Flat` | 平地任务（"平地连续稳定行走 30 s"） |

### 2.1 环境构建时已验证与踩到的坑

本机软件版本：`mjlab 1.2.0`、`mujoco-warp 3.5.0`、`warp-lang 1.18.0`、`mujoco 3.7.0`。

**（重要）`nconmax` / `njmax` 必须放大。** 上游 `SimulationCfg` 默认
`nconmax=35, njmax=1500`，是按参考 A2 的约 10 个碰撞体定的。本模型有 17 个
A2 风格碰撞体 + 整段台阶地形，用默认值会在环境构造阶段直接报错：

```
ValueError: nconmax overflow (nconmax must be >= 44)
```

因此 `env_cfgs.py` 中显式设置 `cfg.sim.nconmax = 128`、`cfg.sim.njmax = 3000`
（`nconmax` 是**每个 world** 的接触容量，与并行环境数无关，128 留足余量）。

**（已知怪癖）首次构建环境可能出现 warp 代码生成报错。** 第一次构造环境时
warp 需要并行编译约 40 个 kernel，偶发出现：

```
warp._src.codegen.WarpCodegenKeyError:
  ... function "_frame_axis" ... Referencing undefined symbol: xmat
```

这是 `mujoco_warp 3.5.0` + `warp-lang 1.18.0` 的 CPU 代码生成路径缺陷
（`_frame_axis` 的 `else: # UNKNOWN` 分支引用了未赋值的 `xmat`），与模型无关
（官方 `Unitree-A2-Rough` 同样会偶发）。**重试即可**：一旦某个 kernel 编译成功
并进入 warp 缓存（本机为 `WARP_CACHE_PATH`），后续构建稳定。
GPU 后端使用的是另一条 CUDA 代码生成路径。若在 GPU 上遇到同样的报错，直接重跑。

---

## 3. 训练

### 3.1 显存与并行环境数（8 GB 是本机硬约束）

`Unitree-A2-Rough` 默认 4096 个环境，在 8 GB 显存上**很可能放不下**。
建议从小到大实测：

```bash
# 冒烟测试：确认链路可用（约 1 分钟）
python scripts/train.py Quadruped-3320-Stairs \
  --env.scene.num-envs=256 --agent.max-iterations=5 --enable-nan-guard

# 逐步放大，找到不 OOM 的最大值（每档跑 10 次迭代即可）
for N in 512 1024 2048 3072; do
  echo "=== num-envs=$N ==="
  python scripts/train.py Quadruped-3320-Stairs \
    --env.scene.num-envs=$N --agent.max-iterations=10 --enable-nan-guard || break
done
```

> 注意：CLI 只支持**已声明的 dataclass 字段**，形如
> `--env.scene.terrain.terrain_generator.sub_terrains...` 的深层嵌套参数**无法**通过
> 命令行覆盖，地形与奖励必须在
> `src/tasks/velocity/config/quadruped_3320/env_cfgs.py` 中修改。

### 3.2 正式训练

```bash
python scripts/train.py Quadruped-3320-Stairs \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=30000 \
  --agent.seed=42 \
  --enable-nan-guard
```

日志与 checkpoint 由 runner 写入 `logs/<task_id>/<timestamp>_<run_name>/`，
其中 `model_<iter>.pt` 为策略权重，`params/` 保存本次的完整配置快照
（训练配置与 checkpoint 一一对应，满足 §7 的可复现性要求）。

### 3.3 训练过程中需要盯的曲线

| TensorBoard 标量 | 含义 | 期望 |
|---|---|---|
| `Curriculum/terrain_levels` | 地形课程难度（0~9） | 持续上升；停在低档说明学不动 |
| `Episode/rew_track_lin_vel_xy` | 速度跟踪 | 上升并稳定 |
| `Episode/rew_feet_slip` | 足端滑动惩罚 | 趋近 0（越接近 0 越好） |
| `Episode/rew_flat_orientation_l2` | 机身姿态 | 上升，说明姿态变稳 |
| `Loss/...`、`Policy/mean_std` | PPO 健康度 | 无 NaN、std 不塌缩 |

---

## 4. `play` 演示（四级台阶测试）

```bash
# 达标档：恒定 0.4 m/s
python scripts/play.py Quadruped-3320-Stairs \
  --checkpoint logs/Quadruped-3320-Stairs/<run>/model_29999.pt

# 良好 / 优秀档
python scripts/play.py Quadruped-3320-Stairs-0.6 --checkpoint <ckpt>
python scripts/play.py Quadruped-3320-Stairs-0.8 --checkpoint <ckpt>
```

`play` 模式下地形栅格被重置为 5 x 5 且关闭课程；由于只注册了一个
`straight_stairs` 子地形，每块 patch 都是同一段台阶，几何是**确定**的。
`reset_base` 的 yaw 被固定为 0，机器人始终朝 +x 面向台阶。

### 自动化测试与 `results.csv`

```bash
python scripts/eval_stairs.py Quadruped-3320-Stairs \
  --checkpoint logs/Quadruped-3320-Stairs/<run>/model_29999.pt \
  --trials 5 --out results.csv --seed 0
```

输出 `results.csv`，列为
`trial_id, seed, command_speed, success, elapsed_time, max_abs_roll, max_abs_pitch, failure_reason, notes`。

成功判定（§4.3）四条同时满足；任一失败条件触发即记为失败。
详见 `docs/stair_evaluation.md`。

---

## 5. 单腿足端轨迹 + 笛卡尔 PD 演示

```bash
cd ~/RC/leg_pd_control
python -m leg_pd_control.demo --leg RL --cycles 6
# 曲线输出到 plots/，逐控制步日志输出到 log_foot_trajectory.csv
python -m pytest tests/ -q
```

---

## 6. 四级台阶地形规格（供核对）

| 项 | 值 |
|---|---|
| 平级数 | **4** |
| 每级高度 | **0.13 m** |
| 每级沿前进方向宽度（踏面） | **0.30 m** |
| 总高 | **0.52 m** |
| 平地接近区 | 1.6 m |
| 顶部平台 | 1.7 m |
| 单块 patch | 4.5 m × 4.5 m（1.6 + 4×0.30 + 1.7 = 4.5） |
| 前进方向 | **+x** |
| 出生点 | 第一级踢面之前 0.9 m 的接近区，yaw = 0 |

碰撞：每级台阶为一个从 z = −0.5 m 到该级顶面的实心 box，`contype=1 / conaffinity=1`，
相邻台阶严格贴合、无缝隙无重叠。**未添加任何隐藏斜坡，未删除台阶碰撞，未缩小实际高度。**

---

## 7. 已知问题与对策

| 现象 | 原因 | 对策 |
|---|---|---|
| `nvidia-smi` 失败 / `torch.cuda.is_available()` False | 内核态 550 与用户态 580 不一致 | 重启加载 580（见 §0） |
| `AttributeError: mjENBL_MULTICCD` | mujoco 与 mujoco_warp 版本冲突 | `pip install "mujoco<3.8.0"` |
| `ModuleNotFoundError: scipy` | 依赖缺失 | `pip install scipy` |
| CLI 报 `Unrecognized options` | mjlab 的 CLI 无法覆盖深层嵌套字段 | 直接改 `env_cfgs.py` |
| 训练 OOM | 8 GB 显存 | 降低 `--env.scene.num-envs` |
| 训练发散（QACC 爆炸） | 该机型仅 9.2 kg，A2 参考增益偏硬 | 降低 `stiffness`/提高 `damping`，或用 `--enable-nan-guard` 定位 |
