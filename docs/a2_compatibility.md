# A2 接口兼容性对照表（`quadruped_3320`）

本文档逐项说明从 SolidWorks 装配体导出的模型如何映射到考核方
`assessment/reference/A2_urdf/INTERFACE.md` 规定的 A2 接口。

- 源数据：考核方提供的 SolidWorks 装配体（STEP 文件名 `四足3.3.20.STEP`，
  导出器 `coordinate_schema_version = 3`），转换为 URDF 后为
  `src/assets/robots/quadruped_3320/source/robot_solidworks_export.urdf`。
- 转换工具：`scripts/build_a2_urdf.py`（可重复执行，幂等）。
- 检查工具：`scripts/verify_a2_urdf.py`（当前 **PASS**）。
- 产物：
  - `src/assets/robots/quadruped_3320/urdf/quadruped_3320.urdf`
  - `src/assets/robots/quadruped_3320/xmls/quadruped_3320.xml`（训练实际加载）
  - `src/assets/robots/quadruped_3320/meshes/`（510 个 STL，包内相对路径）

装配体共 510 个零件、14 个 link、13 个 joint（1 个浮动 + 12 个转动）。
`base_link` 质量 5.5446 kg，整机 **9.2106 kg**。

---

## 1. 坐标系映射：Y-up → Z-up（ROS）

导出的 `export_report.json` 声称坐标系为 `z_up_x_forward`，**实测不符**：

| 证据 | 结论 |
|---|---|
| `IntelRealsense_D435_Multibody`、`mid-360-asm`、`imu`、`电池`、`雷达连接件` 的 bbox 中心 **z 分量全为 0** | **z 是左右对称轴（横向）** |
| `身体` bbox = 420 (x) × 188 (y) × 271 (z) mm；420 是前后长度、271 是横向宽度 | x = 前后，z = 横向 |
| 四条腿的足端位于 **min y**（y = −0.3955 m），髋关节 y ≈ −0.0279 m | **y 是竖直方向** |

因此导出坐标系实为 **Y-up**：x 前后（前方为 −x）、y 竖直、z 横向（左为 +z）。
必须整体旋转到 ROS 约定的 Z-up（x 前、y 左、z 上）：

```
M = [[-1, 0, 0],        v_new = M · v_old
     [ 0, 0, 1],
     [ 0, 1, 0]]
```

`base_link` 的帧取 `Mᵀ`，因此旋转后 `base_link` 与世界帧对齐，前腿落在 +x、左腿落在 +y。

**验证**：旋转后参考 A2 的前髋在 +x（`a2.xml` 中 FL/FR hip x = +0.25944），本模型
`FL_hip_joint` 的 origin x = **+0.2029671**，方向一致。

---

## 2. 关节轴：重定帧而非改物理

导出器把 **全部 12 个关节的 axis 都写成 `0 0 1`**，与各自 child link 帧的朝向
组合后，物理效果恰好是合理的（髋 = 绕前后轴外展，大腿/小腿 = 绕横向轴俯仰），
但 axis 字面量并不等于参考值。

**判定依据**——「关节轴是否垂直于它驱动的连杆段」：

| 模型 | 髋 | 大腿 | 小腿 |
|---|---:|---:|---:|
| 参考 A2 | 0.0000 | 0.0000 | 0.0000 |
| 导出模型（Y-up 原帧） | 0.9780 | 0.4356 | — |

即导出器没有携带真实配合轴信息，`0 0 1` 是占位值。

**处理方式：保持几何完全不动，只重新定义每个 link 的帧。**

对每个 link L 引入额外局部旋转 `E_L`，使新朝向 `A'_L = M · A_L · E_L`。对每个
转动关节（child = L）要求其物理轴等于参考轴 `a_ref`：

```
A'_L · a_ref = a_ref      ⇒      E_L = align(a_ref → A_Lᵀ Mᵀ a_ref)
```

- link 几何：`v_new = E_Lᵀ v_old`（visual / collision / inertial 的 origin 与惯量张量
  `I' = E_Lᵀ I E_L`）
- 关节 origin：`R' = E_Pᵀ R E_L`，`p' = E_Pᵀ p`
- 关节 axis 字面量：直接写为参考值

这**严格保持每个 link 的物理位姿与惯量不变**，只改变了坐标表达，因此
`axis` 字面量、物理轴向与正负号同时与参考一致。

---

## 3. 关节顺序

INTERFACE §1 规定 `RL → FL → RR → FR`，每条腿内 `hip → thigh → calf`。
导出器原始顺序是 `FR, FL, RL, RR`，已在重建时改写。

**注意：顺序在三个层面都显式固定，不依赖任何隐式排序。**

| 层面 | 顺序 |
|---|---|
| URDF `<joint>` 声明顺序 | `RL_hip, RL_thigh, RL_calf, FL_…, RR_…, FR_…` |
| MJCF 关节数组 | 同上 |
| MJCF actuator（= 动作向量） | 同上 |

官方 A2 配置用 `.*hip_.*` / `.*thigh_.*` / `.*calf_.*` 三条族级正则，导致动作向量
按关节族分组（hip×4, thigh×4, calf×4），并不满足 INTERFACE §1。
本模型改为**每个关节一个 actuator 并按规定顺序声明**，同时在
`quadruped_3320_constants.py` 中用 `assert_policy_order()` 在 import 期断言，
顺序漂移会直接报错。

---

## 4. 关节映射总表

`零点偏置` 见第 5 节。角度单位 rad，effort 单位 N·m，velocity 单位 rad/s。

| 关节 | parent | child | origin xyz | origin rpy | axis | lower | upper | effort | velocity | 零点偏置 (rad) |
|---|---|---|---|---|---|---|---|---|---|---|
| `RL_hip_joint` | `base_link` | `RL_hip` | `-0.1935329 0.075 -0.02790032` | `2.85477545 0 0` | `1 0 0` | -1.01 | 1.01 | 120 | 22 | +0.286817 |
| `RL_thigh_joint` | `RL_hip` | `RL_thigh` | `-0.0739711434 -0.015789998 2.4e-09` | `3.4e-08 -0.661649653 3.14159263` | `0 1 0` | -1.56 | 3.94 | 120 | 22 | +0.661650 |
| `RL_calf_joint` | `RL_thigh` | `RL_calf` | `0.13998634 0.106460005 0.169716887` | `0 1.18071106 0` | `0 1 0` | -2.77 | -0.54 | 180 | 14.6667 | -1.180711 |
| `FL_hip_joint` | `base_link` | `FL_hip` | `0.2029671 0.075 -0.02790032` | `-0.272727272 0 0` | `1 0 0` | -1.01 | 1.01 | 120 | 22 | +0.272727 |
| `FL_thigh_joint` | `FL_hip` | `FL_thigh` | `0.0564711435 0.0169999985 2.0e-09` | `3.1415926 1.04222474 3.14159261` | `0 1 0` | -2.34 | 3.15 | 120 | 22 | +1.042225 |
| `FL_calf_joint` | `FL_thigh` | `FL_calf` | `0.19190645 0.0880000132 0.107572828` | `3.14159265 1.45705068 3.14159265` | `0 1 0` | -2.77 | -0.54 | 180 | 14.6667 | -1.684542 |
| `RR_hip_joint` | `base_link` | `RR_hip` | `-0.2110329 -0.075 -0.02790032` | `-2.85930632 0 0` | `1 0 0` | -1.01 | 1.01 | 120 | 22 | -0.282286 |
| `RR_thigh_joint` | `RR_hip` | `RR_thigh` | `-0.0564711425 0.0170000015 1.1e-09` | `3.4e-08 -0.661884285 3.14159263` | `0 1 0` | -1.56 | 3.94 | 120 | 22 | +0.661884 |
| `RR_calf_joint` | `RR_thigh` | `RR_calf` | `0.13998634 -0.104999995 0.169716893` | `0 1.18121571 0` | `0 1 0` | -2.77 | -0.54 | 180 | 14.6667 | -1.181216 |
| `FR_hip_joint` | `base_link` | `FR_hip` | `0.1854671 -0.075 -0.02790032` | `0.272605911 0 0` | `1 0 0` | -1.01 | 1.01 | 120 | 22 | -0.272606 |
| `FR_thigh_joint` | `FR_hip` | `FR_thigh` | `0.0739711425 -0.017000002 1.5e-09` | `3.14159262 0.632795646 3.14159263` | `0 1 0` | -2.34 | 3.15 | 120 | 22 | +0.632796 |
| `FR_calf_joint` | `FR_thigh` | `FR_calf` | `0.13422766 -0.0877500047 0.174307018` | `0 1.18467312 0` | `0 1 0` | -2.77 | -0.54 | 180 | 14.6667 | -1.184673 |

固定关节：`{RL,FL,RR,FR}_foot_joint`（`*_calf → *_foot`）、
`camera_joint`（`base_link → camera_link`）、
`front_lidar_joint`（`base_link → front_lidar_link`）、
`rear_lidar_joint`（`front_lidar_link → rear_lidar_link`）。
与参考包的父子关系一致；三个 sensor link 在参考包中也是空 link，此处同样为空。

---

## 5. 零点偏置（零位定义与参考包一致）

**问题**：SolidWorks 装配体并非按对称站姿建模——

- `FL` 腿的姿态与其他三条不同：在导出姿态下 FL 足底比其他腿高 **81.9 mm**；
- 前后腿**长度不同**：后腿 thigh 244.4 mm，前腿 236.9 mm。

若直接以建模姿态为关节零位，同一组关节角在四条腿上会得到不同姿态，且
`FL` 腿明显异常；同时 calf 的建模角落在参考限位 `[-2.77, -0.54]` 之外。

**处理方式**：对每条腿解一次 IK，求出「足端位于大腿关节正下方、离地高度等于
A2 默认站姿伸展比」的原始关节角，然后平移该腿的零点，使这组角恰好等于
**参考 A2 的默认站姿**（hip ∓0.1、thigh 0.9、calf −1.8）。
IK 残差对四条腿均为 **0.00 mm**。

这样规定之后：

- 相同关节指令 → 四条腿姿态一致、四足共面；
- `q = 0` 对应 A2 的「腿伸直」中性位，因此 calf 上限 −0.54 的物理含义与
  参考完全一致（A2 的腿同样不能完全伸直）；
- 建模姿态（`q = 0` 之外）是否在限位内不重要，训练与 `play` 都使用默认站姿。

偏置数值写入 `src/assets/robots/quadruped_3320/joint_zero_offsets.json`。

---

## 6. 站立几何与默认站姿

默认站姿：hip ∓0.1（RL/FL 为 −0.1，RR/FR 为 +0.1）、thigh 0.9、calf −1.8。

| 腿 | 髋→大腿关节 (m) | 大腿关节→膝 (m) | 膝→足端球心 (m) | 足端高度 (m) |
|---|---:|---:|---:|---:|
| RL | 0.0756 | 0.2444 | 0.2082 | 0.0340 |
| FL | 0.0590 | 0.2369 | 0.2067 | 0.0340 |
| RR | 0.0590 | 0.2438 | 0.2082 | 0.0340 |
| FR | 0.0759 | 0.2369 | 0.2079 | 0.0340 |

（上表为 `base_link` 原点在 z = 0.33 m 时的高度；四足高度差 **0.000 m**。）

- 足端碰撞球半径 0.032 m（取自参考 A2），因此接地点在 z = 0.002 m，留 2 mm 余隙。
- 膝到足底（小腿网格最低点）的长度约为 0.2402 m，膝到球心为 0.2082 m。

---

## 7. 补充的 link 与碰撞几何

### 7.1 足端与 sensor link

导出模型**没有独立足端 link**（足部几何包含在小腿零件 `小腿改` 内），
参考接口要求 `RL_foot` 等 4 个固定关节。处理方式：

- 新建 `{LEG}_foot` 固定 link，携带半径 0.032 m 的接触球（与参考 A2 相同），
  球心布置在足底沿小腿轴线方向外移一个半径处，使球与足底相切；
- 质量取参考 A2 的足端质量 0.046 kg 与相应球惯量（`2/5 m r²`）；
- 足端**不重复添加可视网格**（足部外观已由小腿网格体现），这在小腿质量中已包含，
  因此整机质量比「小腿 + 独立足」略高 4 × 0.046 = 0.184 kg（约 2%），此处如实记录。

同时按参考包添加空的 `camera_link` / `front_lidar_link` / `rear_lidar_link`，
位置取自装配体中 `IntelRealsense_D435_Multibody` 与 `mid-360-asm` 的 bbox 中心。

### 7.2 碰撞几何简化

导出器把**每一个可视零件同时导出为碰撞体**，导致 `base_link` 一个 link 上就有
**158 个碰撞 geom**，整机 190 个。参考 A2 全机仅约 10 个。直接使用会让碰撞检测
与自碰撞对数量爆炸，训练不可用。

按 A2 的做法替换为简单图元（整机 **17 个碰撞体**）：

| link | 碰撞体 |
|---|---|
| `base_link` | 1 个 box |
| `{LEG}_hip` | 1 个 box |
| `{LEG}_thigh` | 1 个圆柱（沿大腿轴，长度取连杆段的 70%，r ≤ 0.05 m） |
| `{LEG}_calf` | 1 个圆柱（沿小腿轴，止于足端球心前方 72% 处，r ≈ 0.041 m） |
| `{LEG}_foot` | 1 个球（r = 0.032 m） |

`base_link` 的 box 刻意取 3% 分位 bbox 的 60%，与参考 A2 一致地保持紧凑
（A2 可视机身约 0.7 m 长，其碰撞盒仅 0.24 m），避免机身过早刮碰台阶。
所有碰撞体命名为 `*_collision`，以匹配训练配置中的
`geom_names_expr=(".*_collision",)` 与 `^[FR][LR]_foot_collision$`。

两处刻意的缩短，均由 `scripts/check_model_physics.py` 的仿真结果驱动：

- **大腿圆柱从连杆段 30% 处开始**：前腿（FL/FR）的髋关节横向只有 59 mm，
  位于机身包围盒之内，若圆柱覆盖整段则必然与机身盒重叠。缩短后仅剩约 6~9 mm
  的代理体重叠（见下）。
- **小腿圆柱止于足端球心前方 72% 处**：默认站姿下小腿轴相对竖直方向倾斜
  约 50~60°，若圆柱延伸到足端球心，其远端边缘会低于足端球，导致**小腿而非足端**
  接地（实测穿地 1.4~2.0 mm）。缩短后四个足端球成为唯一接地点。

机身与大腿代理体在髋部仍有约 6~9 mm 重叠，但**不影响仿真**：训练使用
`FULL_COLLISION`，它给所有 `*_collision` 设 `contype=1, conaffinity=0`，
机器人自身之间不产生接触（参考 A2 对足端也做了同样的处理）。
`scripts/check_model_physics.py` 会如实打印该重叠并标注为 INFO。

### 7.3 网格抽稀

`part_001_part.stl`（机身外壳）有 **548,986 个三角面**，超过 MuJoCo 的 200,000
上限，直接加载报错。用顶点聚类抽稀到 **148,505 面**（取满足上限的最细网格）。
其余 509 个网格均在上限内，未改动。

---

## 8. 导出过程中发现的问题与修正

| # | 现象 | 定位过程 | 修正 |
|---|---|---|---|
| 1 | `export_report.json` 声称 `z_up_x_forward`，但模型在仿真中侧躺 | 统计相机/雷达/IMU/电池 bbox 中心，发现 z 恒为 0；足端位于 min y | 全局施加 Y-up→Z-up 旋转 `M` |
| 2 | 12 个关节 axis 全为 `0 0 1` | 检查「轴是否垂直于连杆段」：参考 A2 为 0.0000，本模型髋 0.9780、大腿 0.4356 | 逐 link 重定帧，使 axis 字面量为 `1 0 0` / `0 1 0`，几何不动 |
| 3 | 同一组关节角下四条腿姿态不一致 | 在建模姿态下计算四足高度，FL 高出 81.9 mm | 逐腿解 IK 求零点偏置 |
| 4 | calf 建模角落在参考限位之外 | 把建模角与 INTERFACE 限位表对比 | 零点偏置使 `q=0` 对应 A2 中性位 |
| 5 | 缺 4 个 `*_foot` 与 3 个 sensor link | 与 INTERFACE §3 的名称清单逐项比对 | 按 7.1 补齐 |
| 6 | `base_link` 有 158 个碰撞体 | 统计 URDF 中各 link 的 collision 数量 | 按 7.2 简化为图元 |
| 7 | 机身网格超出 MuJoCo 面数上限 | `MjModel.from_xml_path` 报 `stl_decoder: number of faces should be between 1 and 200000` | 顶点聚类抽稀 |
| 8 | 导出器丢弃 324 个未分配零件 | `export_report.json` 的 `warnings` 字段 | 已记录；这些零件不隶属于任何 link，对运动学无影响 |
| 9 | MuJoCo 加载 URDF 后 `base_link` 质量丢失 | `MjModel` 中第一个 body 是 `RL_hip` 而非 `base_link`，整机只剩 3.666 kg | MuJoCo 会把无关节根节点的子节点提升到 world。与参考包做法一致：URDF 保持 `base_link` 为根（`a2.urdf` 亦然），**自由关节写在 MJCF 中**（`a2.xml` 亦然）。实际训练加载 MJCF，质量完整 9.2106 kg |
| 10 | MJCF 在 MuJoCo 中重名 geom 报错 | —— | 所有碰撞 geom 显式命名，可视 geom 不加名 |

---

## 8.5 物理稳定性检查（§4.1.5 要求）

`python scripts/check_model_physics.py`（当前 **RESULT: PASS**）在 CPU MuJoCo 上
实际推进物理，使用与训练一致的设置（逐关节 `armature=0.03`、effort 限幅
120/180 N·m、`FULL_COLLISION` 的 condim/friction/solimp），以关节空间 PD
（Kp=120, Kd=6）维持默认站姿 4 秒：

| 检查 | 结果 |
|---|---|
| 出生时机器人对地面无穿透 | ✅ 0 处 |
| 仿真过程无 NaN / 非有限状态 | ✅ |
| 全程最大穿透 | ✅ −1.47 mm（阈值 −5 mm） |
| 机身高度是否稳定 | ✅ 均值 0.3212 m，标准差 **0.00 mm** |
| 后 2 秒是否缓慢坍塌/漂移 | ✅ dz = +0.0000 m |
| 接地的是否为四个足端球 | ✅ 恰好 4 个（`FL/FR/RL/RR_foot_collision`） |
| 是否存在机身/髋/膝/小腿触地 | ✅ 无 |
| 机身是否保持直立 | ✅ 偏离竖直 **0.37°** |
| 关节 PD 静差 | 0.107 rad（FR_thigh，A2 参考增益下约 9 kg 机体的重力下垂，非模型缺陷） |

结论：默认姿态下**无穿模、无持续抖动、无数值爆炸、无初始坍塌**。

---

## 9. 验证结果

`python scripts/verify_a2_urdf.py`（当前 **RESULT: PASS**）：

- **A. 接口一致性**：12 个转动关节的声明顺序、axis 字面量、lower/upper/effort/velocity
  逐项与 INTERFACE §2 表一致；20 个 link 名称齐全。
- **B. 加载**：URDF 与 MJCF 均可被 `mujoco.MjModel.from_xml_path` 编译；
  所有 body 质量为正、惯量有限且正定。
- **B3. MJCF**：21 个 body、13 个关节、总质量 **9.2106 kg**；
  含 `floating_base_joint`；关节顺序与 policy 顺序一致；17 个 `*_collision`
  碰撞体；`imu` 与 `RL/FL/RR/FR` 五个 site 齐备（训练配置依赖这些名字）。
- **C. 默认站姿**：四足足端高度差 **0.000 m**；各足足端位于其大腿关节正下方
  （dx = dy = 0.0 mm）；左右对称。
- **D. 运动方向**：对每个关节施加 +0.05 rad，比较足端位移方向与参考 A2 的
  同关节位移方向，余弦相似度 thigh = **1.0000**、hip ≈ 0.980、calf ≈ 0.973
  （均为正，符号约定一致；残余差异来自本模型与 A2 不同的连杆几何）。

`python scripts/check_model_physics.py`：见第 8.5 节（**PASS**）。

附加的 import 期断言：`quadruped_3320_constants.assert_policy_order()` 检查
MJCF 关节顺序，任何漂移都会立即报错。

---

## 10. 复现步骤

```bash
cd ~/RC/mjlab/unitree_rl_mjlab
source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_rl_mjlab
export MPLCONFIGDIR=/tmp/mpl

# 1) 原始 STL 包需要放在（因体积走 LFS/外部存储，未直接入库）
#    src/assets/robots/quadruped_3320/source/meshes_stl.zip

# 2) 重建 URDF + MJCF + meshes（幂等）
python scripts/build_a2_urdf.py

# 3) 校验 A2 接口一致性
python scripts/verify_a2_urdf.py
```

---

## 11. 第三方与 AI 工具声明

- 转换/校验脚本（`scripts/build_a2_urdf.py`、`scripts/verify_a2_urdf.py`）由本人
  编写，使用 AI/Agent 辅助起草与调试；全部逻辑、数值结果与取舍理由本人已复核并
  可现场解释与修改。
- 仅使用仓库既有依赖（numpy / scipy / trimesh / mujoco），未引入新的第三方代码。
- 参考只读材料：考核方 `assessment/reference/A2_urdf/`（未修改，未用于替代本人模型）。
