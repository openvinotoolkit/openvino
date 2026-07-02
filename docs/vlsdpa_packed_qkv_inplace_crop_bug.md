# VL-SDPA Packed-QKV In-Place Crop 越界读取 Bug 分析、复现与修复

> 关联：openvino.mx PR#264（原始修复）、openvino PR#36336（本分支，`ywang2/QKV_split_reshape_matcher_fuse`）
> 复现用例：`src/plugins/intel_gpu/tests/unit/test_cases/vlsdpa_gpu_test.cpp`
> 用例名：`vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression`
> 涉及源码：`impls/cm/vl_sdpa_opt.cpp`（host）、`impls/cm/cm_sdpa_vlen.cm`（wrapper kernel）
>
> **TL;DR**：`vl_sdpa` CM kernel 在 in-place crop（packed QKV）下用了 unpacked 的
> per-token stride 且 `token_offset` 公式多乘一次 `num_q_heads`，导致 Q/K/V 越界读取。
> 修复采用**三层协同**：host 侧下发 `CMFLA_IS_QKV_FUSED` 开关 + Q/K/V 各自的 base
> offset（`lower_pad[f]*head_size`），wrapper kernel 据此切换 packed stride 并拆分
> K/V 独立偏移，内层 kernel 复用其已有的 `is_qkv_fused` 能力，**无需改动**。既保住了
> in-place crop 的零拷贝收益，又对原 contiguous 路径零回归（详见 [第 5 节](#5-修复方案enable-in-place-crop三层协同)）。

## 目录导航

- [0. 概念澄清](#0-概念澄清)
- [1. 背景](#1-背景)
- [2. 根因分析](#2-根因分析)
- [3. 复现用例讲解](#3-复现用例讲解)
- [4. 测试运行示例](#4-测试运行示例)
- [5. 修复方案](#5-修复方案enable-in-place-crop三层协同)
- [6. 方案演进：用官方 Layout API 重写（方案 B）](#6-方案演进用官方-layout-api-重写方案-b当前实现)
- [7. 背景知识：核心概念与端到端流程](#7-背景知识核心概念与端到端流程)

---

## 0. 概念澄清

在读后文之前，先统一几个术语。

### 0.1 in-place crop 优化

正常情况下每个算子输出都要分配一块自己的物理 buffer。而 `crop`（切片）本质上只是
"取输入 buffer 的一部分"，数据没有变化，所以 GPU plugin 可以让 crop **不分配新
buffer**，直接复用输入内存，只记录：

- **起始偏移**（offset / `lower_pad`）：从输入 buffer 的哪个位置开始；
- **形状 + 步长（stride / pitch）**：怎么在这块内存里"跳着读"出想要的切片。

这正是 `crop_inst->can_be_optimized() == true` 的含义——运行时不执行真正的拷贝。

### 0.2 packed buffer（打包缓冲区）

"packed" 指 **Q、K、V 三份数据被塞进同一块连续内存**。`[L, 3, H, S]`（行优先）的
布局是：

```
token 0: [ Q(H*S) | K(H*S) | V(H*S) ]   ← 每 token 占 3*H*S 个元素
token 1: [ Q(H*S) | K(H*S) | V(H*S) ]
token 2: [ Q(H*S) | K(H*S) | V(H*S) ]
...
```

即**同一个 token 的 Q/K/V 相邻存放**，然后才是下一个 token。与之相对的是 unpacked
布局：Q/K/V 各自拥有**独立的连续 buffer**，每块 buffer 内相邻 token 之间步长只有
`H*S`（一份 head 数据）。示意如下：

```
unpacked Q buffer: [ Q_token0(H*S) | Q_token1(H*S) | Q_token2(H*S) | ... ]
unpacked K buffer: [ K_token0(H*S) | K_token1(H*S) | K_token2(H*S) | ... ]
unpacked V buffer: [ V_token0(H*S) | V_token1(H*S) | V_token2(H*S) | ... ]
```

**正常 unpacked 情况下**，`vl_sdpa` CM kernel 按 `H*S` 步进是正确的——三块 buffer
各自独立，每个 token 确实只隔 `H*S` 个元素。bug 的根源正是：in-place crop 之后
K/V 实际上是 packed buffer 的别名（per-token stride = `3*H*S`），但 kernel 仍按
unpacked 假设（stride = `H*S`）步进。

### 0.3 crop 输出"别名"（alias）

"别名"= **两个不同的逻辑张量指向同一块物理内存**。in-place crop 生效后，`crop_k`
没有自己的 buffer，只是 packed buffer 上"K 那一段"的视图：

```
packed buffer physical layout (per token):
  [ lower_pad(Q) | K_data | upper_pad(V) ]
    ^               ^
    |               |-- K 的逻辑起点（lower_pad 之后）
    |-- 物理起点（packed buffer base）
```

`lower_pad[f]=H` 表示跳过 H 个 heads（= H*S 个元素）才到 K 的数据起点。

### 0.4 per-token pitch（每 token 步长）

**pitch / stride** = 从一个 token 跳到下一个 token，指针要前进多少个元素。

- **packed pitch**（真实值）：token 之间隔着完整的 `[Q|K|V]`，所以是 `3*H*S`；
- **unpacked pitch**（kernel 写死的假设值）：若 K 是独立 buffer，token 之间只隔一份
  K，所以是 `H*S`（`num_kv_heads * head_size`）。

两者之比为 3，所以从第 2 个 token 起每次读取都会落在错误地址。

---

## 1. 背景

### 1.1 两条不同的 QKV-split 融合线

本分支里有两个 TransposeFusion matcher，**匹配不同的子图，末端是不同的算子**，不要
混淆：

| Matcher | 典型模型 | 输入 rank | 改写后 Split 轴 | 子图末端 |
| --- | --- | --- | --- | --- |
| `QKVSplitReshapeMatcher` | VideoChat-Flash ViT | 5D `[?,?,3,H,S]` | `axis=2` | RMSNorm / 普通 **SDPA** |
| `TransposeSplitMatcher` | Qwen-VL Vision Merger | 4D `[-1,3,H,S]` | `axis=1` | RoPE → **VL_SDPA** |

- `QKVSplitReshapeMatcher`：`Reshape([0,0,3,H,S]) → Transpose([2,0,3,1,4]) →
  Split(axis=0) → Squeeze → Transpose([0,2,1,3]) → Reshape([0,0,H*S])`，改写为
  `Split(axis=2)`。下游是 RMSNorm / 普通 SDPA。
- `TransposeSplitMatcher`：`[-1,3,H,S] → Transpose([1,0,2,3]) → Split(axis=0) →
  Reshape → RoPE → VL_SDPA`，改写为 `Split(axis=1)`。下游是 VL_SDPA。

### 1.2 本 bug 只发生在 VL_SDPA 这条线

**这个 pitch/offset bug 是 `vl_sdpa` CM kernel 独有的**，因为只有它使用了
`token_offset_q/kv` + 固定 unpacked pitch 的裸指针遍历（见
`src/plugins/intel_gpu/src/graph/impls/cm/vl_sdpa_opt.cpp`）。普通 SDPA / RMSNorm 通过
layout 的 padding/stride 正确寻址，不受影响。

因此：

- **受影响**：`TransposeSplitMatcher`（axis=1）→ VL_SDPA。
- **不受影响**：`QKVSplitReshapeMatcher`（axis=2）→ RMSNorm / 普通 SDPA。

`TransposeSplitMatcher` 改写后的子图（4D `[L, 3, H, S]`）中：

- `L`：token 数（序列长度，动态）
- `3`：QKV 三个 slot
- `H`：num_heads
- `S`：head_size

Split(axis=1) 把 packed tensor 切成三路，每路以 in-place crop 的方式**别名进原始
packed buffer**，再分别喂给 `vl_sdpa`（CM kernel）。

---

## 2. 根因分析

### 2.1 padding 的含义

**在这里，"padding" 不是深度学习里给 feature map 补零的那种 padding**，而是
OpenVINO GPU plugin 内部用来描述**"in-place crop 在物理内存中的位置"**的元数据。

**先建立一个具体的心智模型（H=2, S=72）：**

原始 packed buffer 的物理布局，每行代表一个 token：

```
物理地址→  0        144      288      432
           |--------|--------|--------|
token 0:   [ Q(H*S) | K(H*S) | V(H*S) ]    ← 共 3*H*S = 432 个 float16
token 1:   [ Q(H*S) | K(H*S) | V(H*S) ]
token 2:   [ Q(H*S) | K(H*S) | V(H*S) ]
```

in-place crop 提取 K 的那一段时，不做拷贝，只记录三件事：
1. **数据起点**：从 packed buffer 的第 `H*S = 144` 个元素处开始（K 的头部）
2. **数据长度**：每 token 有 `H*S = 144` 个逻辑元素（K 的内容）
3. **物理步长**：相邻 token 之间，物理地址前进 `3*H*S = 432`（不是 144，因为中间还夹着 Q 和 V）

这三件事在 OpenVINO 的 `data_padding` 结构里用 `lower_pad` / `upper_pad` 来表达。

#### `lower_pad[f]`、`upper_pad[f]` 是什么？

`data_padding` 把物理 buffer 分成三段（每个 token 内部）：

```
物理排列（单个 token）：
  [ lower_pad 区 | 逻辑数据区 | upper_pad 区 ]
     被遮挡的头部    真正有效的内容    被遮挡的尾部
```

- `lower_pad[f]`：在 **feature（第二维）** 轴方向，逻辑数据前面遮挡了多少个「feature 单元」
- `upper_pad[f]`：逻辑数据后面遮挡了多少个「feature 单元」
- 下标 `[f]` 对应 BFYX 格式的 F 轴（即 `data_padding._lower_size[1]`，因为 BFYX 里维度顺序是 B=0, F=1, Y=2, X=3）

**「feature 单元」在 3D [L, H, S] 格式里是什么？**

3D 张量 `[L, H, S]`（token 数 × head 数 × head_size）映射到 BFYX 时：
- B = 1（batch 维度，虚拟的）
- **F = H（head 数，这就是 feature 轴）**
- Y = L（token 数）
- X = S（head_size）

所以 `lower_pad[f] = H` 表示：**物理上每 token 的 K 数据前面被 H 个 head 的数据遮住**——正是 Q 那段（占 `H * S` 个元素）。

#### `phys_per_token_stride`（物理步长）怎么算？

物理步长 = 每 token 的物理占用 = (lower_pad + 逻辑 head 数 + upper_pad) × S

$$	ext{phys\_per\_token\_stride} = (	ext{lower\_pad}[f] + H + 	ext{upper\_pad}[f]) 	imes S$$

对 K 的 crop（K 在中间，Q=lower, V=upper）：

$$= (H + H + H) 	imes S = 3 \cdot H \cdot S \checkmark$$

#### 从 4D 到 3D：padding 为什么变了？

`TransposeSplitMatcher` 先产生一个 4D crop，然后做 reshape（squeeze 掉一个维度），
`lower_pad[f]` 和 `upper_pad[f]` 的数值都随之改变单位，但二者描述的**物理遮挡字节数不变**。

K 在 packed `[L, 3, H, S]` 里排在中间（Q 在前、V 在后），所以 lower 和 upper 各遮一段：

| 阶段 | layout | F 轴单位 | `lower_pad[f]` | `upper_pad[f]` | 物理遮挡（lower） | 物理遮挡（upper） |
|------|--------|----------|----------------|----------------|-------------------|-------------------|
| 4D crop | `[L, 1, H, S]` | 1 QKV slot | 1（遮 Q） | 1（遮 V） | 1 × H×S = H×S | 1 × H×S = H×S |
| 3D reshape | `[L, H, S]` | 1 head | H（遮 Q） | H（遮 V） | H × 1×S = H×S ✓ | H × 1×S = H×S ✓ |

**两者都从 1 变成了 H**，物理字节数完全不变——只是度量刻度由"slot"换成了"head"。

用代码对照表示：

```
4D: lower_pad[f]=1, upper_pad[f]=1
    → lower 遮挡 = 1*(H*S) = H*S   (Q 的那一 slot)
    → upper 遮挡 = 1*(H*S) = H*S   (V 的那一 slot)

3D: lower_pad[f]=H, upper_pad[f]=H
    → lower 遮挡 = H*(1*S) = H*S   ✓ 相同
    → upper 遮挡 = H*(1*S) = H*S   ✓ 相同
```

这也是 `phys_per_token_stride` 公式能用 3D 数值直接算出正确结果的原因：

```
phys_per_token_stride = (lower_pad[f] + 逻辑 head 数 + upper_pad[f]) × S
                      = (H            + H             + H            ) × S
                      = 3 × H × S   ← 与 packed buffer 完整一个 token 的长度一致 ✓
```
### 2.2 vl_sdpa_opt.cpp 的公式缺陷

#### `token_offset_kv` 是什么？

`token_offset_kv` 是 host 端（`vl_sdpa_opt.cpp`）计算、以标量形式传给 CM kernel
的一个**元素级别的偏移量**。它的语义是：

> 在 packed buffer 里，K 数据的逻辑起点距 **buffer 物理首地址**偏移了多少个 `float16` 元素？

以 H=2, S=72 为例，K 排在 Q 之后，所以正确值应等于"Q 占的元素数"：

```
正确 token_offset_kv = lower_pad[f] × head_size = H × S = 2 × 72 = 144
```

kernel 拿到这个偏移后，把它加到 K 的 SVM 指针上，就直接指向 K 数据的起点，
不需要知道 packed buffer 里 Q 和 V 的存在。

#### 缺陷：公式多乘了一次 H

`vl_sdpa_opt.cpp` 实际的计算公式：

```cpp
const auto& kv_pad = params.input_layouts[1].data_padding;  // 只读 K 的 padding
int32_t token_offset_kv = static_cast<int32_t>(kv_pad._lower_size[1]) *
                          static_cast<int32_t>(num_q_heads) *        // ← 多乘了一次 H
                          static_cast<int32_t>(head_size);
```

`lower_pad[f]` 在 3D 格式下已经是以"head"为单位的数值（= H），再乘 `num_q_heads`
就又乘了一次 H，结果变成 H²×S：

| | 计算值 | 正确值 |
|---|---|---|
| `token_offset_kv` | `H × H × S = H²S = 288` | `H × S = 144` |
| 含义 | K 指针偏移了 **2 倍** Q 的大小 | 恰好跳过 Q 到 K 头部 |

公式**多乘了一次 `num_q_heads`**，导致 K 的指针基址向后多偏了 `H×S=144` 个元素——
直接跳入了 K 数据区的中间甚至越过 K 进入 V 的区域。

### 2.3 相同偏移套用到 V

#### `kv_offset` 是什么？

`kv_offset` 是 CM kernel 内部（`cm_sdpa_vlen.cm`）计算的**当前工作项（work-item）
的读取起点**——即"我这个线程负责的 head `hkv`、从 token `kv_start` 开始，应该从
K/V buffer 的第几个元素读起"：

```c
uint kv_offset = (kv_start * num_kv_heads + hkv) * head_size + token_offset_kv;
//               |-------- token 内 head 的线性索引 ---------|   ↑
//                                                              每 token 起点的 base 偏移
```

对 unpacked（独立）K buffer，这个公式完全正确：
- `kv_start * num_kv_heads * head_size`：跳过前 `kv_start` 个 token
- `hkv * head_size`：在当前 token 内跳到第 `hkv` 个 head
- `token_offset_kv`：in-place crop 的 base 偏移（unpacked 时为 0）

#### 缺陷：K 和 V 共用同一个 `kv_offset`

CM kernel 对 K 和 V **共用同一个** `kv_offset`：

```c
// cm_sdpa_vlen.cm
uint kv_offset = (kv_start*num_kv_heads + hkv)*head_size + token_offset_kv;
// ...
key   + kv_offset   // K 用 token_offset_kv ← 已经算错（= H²S，应为 H*S）
value + kv_offset   // V 也用 token_offset_kv ← 更错（V 的正确 base 偏移是 2*H*S）
```

packed buffer 里各切片的正确 base 偏移：

```
Q base = 0           （Q 排第一，无需跳过任何东西）
K base = 1 × H×S    （跳过 Q）
V base = 2 × H×S    （跳过 Q 和 K）
```

但 kernel 对 K 和 V 均使用 `token_offset_kv`（且该值已被错误地算成 H²S），
V 本应有的独立 base 偏移 `2×H×S` 完全丢失。

### 2.4 per-token pitch 错误

CM kernel 每步 `kv_start` 前进 `num_kv_heads * head_size = H*S` 个元素，而 K/V 的
物理 per-token 步长是 `3*H*S`（packed buffer）。从 token 1 开始每个 token 的读取
都落在错误地址：

```
packed buffer 物理布局（per token，stride = 3*H*S）:
  token 0: [ pad(H*S) | K_data(H*S) | pad(H*S) ]
  token 1: [ pad(H*S) | K_data(H*S) | pad(H*S) ]
  token 2: [ pad(H*S) | K_data(H*S) | pad(H*S) ]

kernel 按 H*S 步进（应按 3*H*S），从逻辑起点算：
  token 0: 偏移 token_offset_kv = H²S (应 = H*S)  → 读到 token 1 的 lower_pad ✗
  token 1: 偏移 H²S + H*S              → 读到 token 1 的 K_data ✓ (偶然正确)
  token 2: 偏移 H²S + 2*H*S            → 读到 token 1 的 upper_pad ✗
  ...
```

综合两个错误：大多数 token 的 K 被读错，V 全部被读错，attention 输出与正确值
**全面偏离**。

---

## 3. 复现用例讲解

用例位于 `vlsdpa_gpu_test.cpp`，命名为 `packed_qkv_inplace_crop_pitch_regression`，
测试参数：H=2, S=72（与 Omni/Qwen-VL 真实 head_size 一致），L=16。

### 3.1 设备门控

```cpp
if (!engine.get_device_info().supports_immad ||
    !cldnn::check_cm_jit_support(engine, check_cfg))
    GTEST_SKIP() << "vl_sdpa CM kernel not available on this device";
```

`vl_sdpa` CM kernel 仅在支持 XMX 的 GPU + CM JIT 编译环境下可用，否则跳过。

### 3.2 测试数据

使用**显式非零数据**而非随机数，保证 attention 输出可预测、失配清晰可见：

```cpp
// q[l,h,s] = 0.5 (全部相同)
// k[l,h,s] = (l+1)*0.05 + s*0.001  (按 token 递增)
// v[l,h,s] = (l+1)*0.1              (按 token 递增)
```

### 3.3 参考拓扑（Reference，正确结果）

```
q_mem [L,H,S] ─┐
k_mem [L,H,S] ─┤→ scaled_dot_product_attention → permute → reorder → ref_out
v_mem [L,H,S] ─┘
```

三路 contiguous buffer，走普通 SDPA（非 CM 路径），作为数值"黄金参考"。

### 3.4 触发拓扑（Packed，贴合真实模型）

真实 `TransposeSplitMatcher` 产出的是**同一块 packed `[L,3,H,S]` buffer 的三个
in-place crop**，因此 Q/K/V **全部**保持 packed 的 per-token stride `3*H*S`，各自携带
编码 slice 序号的 feature padding。测试为三路分别构造这样的 padded 视图：

```
每个输入 physical per-token stride = 3*H*S：
  q_in: lower_pad[f]=0,   upper_pad[f]=2H   → 数据落在 physical[l*3HS + 0    + h*S + s]
  k_in: lower_pad[f]=H,   upper_pad[f]=H    → 数据落在 physical[l*3HS + H*S  + h*S + s]
  v_in: lower_pad[f]=2H,  upper_pad[f]=0    → 数据落在 physical[l*3HS + 2H*S + h*S + s]
```

数据通过 `mem_lock<write>` 直接写入物理 buffer 的 logical 区（pad 区保持 0），再喂给
`vl_sdpa`：

```
q_in [L,H,S] lower_pad=0  ─┐
k_in [L,H,S] lower_pad=H  ─┤→ vl_sdpa → reorder → packed_out
v_in [L,H,S] lower_pad=2H ─┘
cu_seqlens
```

### 3.5 前置 Sanity Check

```cpp
ASSERT_EQ(k_pad._lower_size[1], lp_k);  // K lower_pad[f] == H
ASSERT_EQ(v_pad._lower_size[1], lp_v);  // V lower_pad[f] == 2H
ASSERT_EQ(q_pad._upper_size[1], up_q);  // Q upper_pad[f] == 2H
```

确认三路 padding 都没有被 runtime 的 reorder 剥掉，否则 `CMFLA_IS_QKV_FUSED=0`、offset
全为 0，测试将退化为普通 contiguous、失去意义。

### 3.6 逐元素对比

```cpp
for (size_t i = 0; i < ref_ptr.size(); i++) {
    if (std::abs(float(ref_ptr[i]) - float(pkd_ptr[i])) > 0.05f)
        ++mismatches;
}
EXPECT_EQ(mismatches, 0) << "vl_sdpa produced " << mismatches << " wrong elements ...";
```

修复前该测试在未修复分支上大面积失配（FAIL）；修复后 `packed_out` 与 contiguous 参考
完全一致（PASS）。

---

## 4. 测试运行示例

### 4.1 未修复分支（预期 FAIL）

在未处理 packed / in-place crop 的分支上，per-token stride 仍按 `H*S`、且
`token_offset` 多乘一次 H，Q/K/V 全部读错地址，输出与参考大面积失配：

```
[ RUN      ] vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression
src/plugins/intel_gpu/tests/unit/test_cases/vlsdpa_gpu_test.cpp:NNN: Failure
Expected equality of these values:
  mismatches
    Which is: 2304
  0
vl_sdpa produced 2304 wrong elements (out of 2304) for packed QKV (in-place crop) ...
[  FAILED  ] vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression
 1 FAILED TEST
```

**所有 2304 个输出元素（= L×H×S = 16×2×72）均超出 0.05 阈值。**

### 4.2 修复后（本方案，实测 PASS）

应用本方案（`CMFLA_IS_QKV_FUSED` + 三个 base offset）后实测输出：

```
$ bin/intel64/Release/ov_gpu_unit_tests \
    --gtest_filter="vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression"

[==========] Running 1 test from 1 test suite.
[ RUN      ] vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression
[       OK ] vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression (275 ms)
[----------] 1 test from vlsdpa_gpu_test (275 ms total)
[  PASSED  ] 1 test.
```

同时完整 `*vlsdpa*` 套件（含 contiguous 非 packed 冒烟用例与
`prepare_buffer_fusing.in_place_crop_split_axis1_three_crops_vlsdpa_consumer`）全部通过：

```
[  PASSED  ] 18 tests.
```

说明修复对原有 contiguous 路径无回归（`is_qkv_fused=0` 时行为不变）。

### 4.3 设备不支持（SKIP）

在无 XMX 的 GPU 或不支持 CM JIT 的环境：

```
[ RUN      ] vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression
[  SKIPPED ] vlsdpa_gpu_test.packed_qkv_inplace_crop_pitch_regression
             vl_sdpa CM kernel not available on this device
```

---

## 5. 修复方案（enable in-place crop，三层协同）

关键发现：内层 `sdpa_kernel_lsc / _prefetch / sdpa_kernel`（`cm_sdpa_common.hpp`）**早已
具备** `is_qkv_fused` 模板开关——置 1 时 `q_pitch / kv_pitch` 自动变成 packed 的
`(num_heads + 2*num_kv_heads)*head_size`（即 `3*H*S`）。此前的 bug 只是外层
`cm_sdpa_vlen.cm` 从未启用它，退化成 unpacked stride，再用错误的 `token_offset` 硬凑。

因此方案是三层协同，**不修改内层 kernel**：

**① host `vl_sdpa_opt.cpp`**

```cpp
// 检测任一输入 feature 轴带 padding => packed（in-place crop 生效）
const bool is_qkv_fused =
    q_pad._lower_size[1] || k_pad._lower_size[1] || v_pad._lower_size[1] ||
    q_pad._upper_size[1] || k_pad._upper_size[1] || v_pad._upper_size[1];
jit.add(make_jit_constant("CMFLA_IS_QKV_FUSED", is_qkv_fused ? 1 : 0));

// base offset = lower_pad[f] * head_size（heads→elements，不再乘 num_q_heads）
// 且 Q/K/V 各算各的，V 不再复用 K 的偏移
int32_t token_offset_q = q_pad._lower_size[1] * head_size;  // Q slice → 0
int32_t token_offset_k = k_pad._lower_size[1] * head_size;  // K slice → H*S
int32_t token_offset_v = v_pad._lower_size[1] * head_size;  // V slice → 2H*S
// scalars: {need_wg_mapping, token_offset_q, token_offset_k, token_offset_v}
```

**② wrapper `cm_sdpa_vlen.cm`**

```c
// packed 时 per-token stride 切到 3*H*S；输出 buffer 永远非 packed
constexpr int packed_token_stride = (num_heads + num_kv_heads*2) * head_size;
constexpr int q_token_stride  = CMFLA_IS_QKV_FUSED ? packed_token_stride : num_heads    * head_size;
constexpr int kv_token_stride = CMFLA_IS_QKV_FUSED ? packed_token_stride : num_kv_heads * head_size;

uint q_in_offset = q_start * q_token_stride  + h   * head_size + token_offset_q;
uint k_offset    = kv_start * kv_token_stride + hkv * head_size + token_offset_k;  // K/V
uint v_offset    = kv_start * kv_token_stride + hkv * head_size + token_offset_v;  // 各自独立
uint o_offset    = (q_start * num_heads + h) * head_size;                          // 输出非 packed

sdpa_kernel_lsc<false, num_heads, num_kv_heads, head_size, CMFLA_IS_QKV_FUSED>(
    ..., query + q_in_offset, key + k_offset, value + v_offset, output + o_offset);
```

三处调用点（`_prefetch` / `_lsc` / 非 LSC `sdpa_kernel`）统一传入 `CMFLA_IS_QKV_FUSED`，
并把 K/V 从共用 `kv_offset` 拆成独立的 `k_offset` / `v_offset`。

**③ 内层 `cm_sdpa_common.hpp`：无需改动**（`is_qkv_fused` 已就绪）。

### 修复前后对照（H=2, S=72）

| 项 | 未修复 | 本方案 |
|---|---|---|
| per-token stride | `H*S=144`（unpacked，错） | `3*H*S=432`（packed，对）|
| `token_offset_k` | `H*num_q_heads*S=288`（错） | `H*S=144`（对）|
| `token_offset_v` | 复用 K 的偏移（错） | `2H*S=288`（独立，对）|
| output offset | 与 Q 共用（packed 时错） | 独立 `q_start*H*S`（对）|

### 为何能同时 enable in-place crop 且不回归

- **in-place crop 保留**：packed padded 布局被 `vl_sdpa` 正确寻址，无需插入 reorder 拷贝，
  保住零拷贝优化收益。
- **无回归**：无 padding 时 `is_qkv_fused=0`、三个 `token_offset` 均为 0，stride 退回
  `H*S`，与旧行为逐位一致（18/18 用例通过验证）。

---

## 6. 方案演进：用官方 Layout API 重写（方案 B，当前实现）

第 5 节的初版修复（记为**方案 A**）功能正确，但直接读取了 `data_padding` 的裸成员
`_lower_size[1]` / `_upper_size[1]`，并硬编码了 feature 轴下标 `[1]` 与 `* head_size`
的换算。方案 B 在**不改变对外行为**的前提下，把这两处替换为 OpenVINO 官方 layout
API，使其**与张量 rank 无关**、更贴近框架语义。

### 6.1 方案 A 的两个隐患

1. **硬编码 feature 轴下标 `[1]`**：`_lower_size[1]` 假设 feature 轴恒在 BFYX 的第 1
   位。一旦上游 reshape 产出的不是 3D `[L,H,S]`（例如保持 4D `[B,L,H,S]` 或其它
   rank），下标 `[1]` 指向的就不再是 head 轴，检测与偏移都会算错。
2. **硬编码 `* head_size` 换算**：`token_offset = _lower_size[1] * head_size` 把
   "head 单位 → element 单位" 的换算写死，同样内嵌了"feature 轴之后只有一个
   head_size 维度"的 rank 假设。

### 6.2 方案 B 的两处改写

**① `is_qkv_fused` 检测 → `padding::operator bool()`**

`vl_sdpa_opt.cpp` `get_jit_constants`：

```cpp
// 用 layout 的 padding 谓词替代硬编码 feature 轴索引，
// 无论 3D(HLS) 还是 4D(BHLS) 都正确。
const bool is_qkv_fused =
    static_cast<bool>(params.input_layouts[0].data_padding) ||
    static_cast<bool>(params.input_layouts[1].data_padding) ||
    static_cast<bool>(params.input_layouts[2].data_padding);
```

`padding::operator bool()`（`layout.hpp`）在任一 `lower/upper` 维度非零时返回
`true`，是"是否存在 padding"的官方语义谓词，不依赖具体轴下标。

**② `token_offset_q/k/v` → `layout::get_linear_offset()`**

`vl_sdpa_opt.cpp` `get_dispatch_data_func`：

```cpp
// get_linear_offset() 直接由 padded dims/strides 推导元素级偏移，
// 对 3D(HLS) 与 4D(BHLS) 均正确，无需硬编码 feature 轴或乘 head_size。
const int32_t token_offset_q = static_cast<int32_t>(params.input_layouts[0].get_linear_offset());
const int32_t token_offset_k = static_cast<int32_t>(params.input_layouts[1].get_linear_offset());
const int32_t token_offset_v = static_cast<int32_t>(params.input_layouts[2].get_linear_offset());
```

`get_linear_offset()` 的语义是"逻辑数据起点相对物理 buffer 首地址的元素偏移"，
其内部实现为 $\sum_{axis} \text{lower\_pad}[axis] \times \text{stride}[axis]$，
即"逐轴 lower_pad 乘该轴 stride 之和"。这正是方案 A 想手算的量，但由框架按实际
layout 计算，rank 无关、无需人工换算。

### 6.3 等价性：get_linear_offset() 恰好等于方案 A 的手算值

为什么用 stride 求和能得到和 `_lower_size[1] * head_size` 一样的结果？以 K 的 crop
（3D `[L,H,S]`，H=2, S=72）为例：

- 只有 feature（head）轴有非零 lower_pad：`lower_pad[head] = H = 2`
- head 轴的 stride = `head_size = S = 72`（head 轴之后只剩 head_size 维）
- 其余轴 lower_pad 为 0，不贡献

$$\text{get\_linear\_offset} = \text{lower\_pad}[\text{head}] \times \text{stride}[\text{head}] = H \times S = 2 \times 72 = 144$$

与方案 A 的 `_lower_size[1] * head_size = 144` 完全一致。区别在于：方案 A 把这条乘法
写死为"feature 下标 × head_size"，方案 B 让框架对**所有轴**求和——当 rank 改变、
或有多个轴带 padding 时，方案 B 仍然正确，而方案 A 会漏项或错位。

### 6.4 实测验证

改写后临时加入一行 side-by-side 对比日志（合入前已移除），实测输出：

```
[vl_sdpa dispatch] head_size=72  token_offset_q(linear)=0 (old=0)  \
    token_offset_k(linear)=144 (old=144)  token_offset_v(linear)=288 (old=288)  need_wg_mapping=0
```

`linear`（方案 B）与 `old`（方案 A 手算）**逐项一致**：

| slice | get_linear_offset() | 方案 A 手算 | 期望 |
|-------|---------------------|-------------|------|
| Q | 0 | 0 | 0 |
| K | 144 | 144 | `H*S = 144` |
| V | 288 | 288 | `2H*S = 288` |

完整 `*vlsdpa*` 套件 **19/19 全部通过**（含 contiguous 冒烟用例与 packed 回归用例），
对原有路径零回归。

### 6.5 小结

| 维度 | 方案 A（第 5 节） | 方案 B（当前实现） |
|------|-------------------|--------------------|
| `is_qkv_fused` | 读 `_lower_size[1]`/`_upper_size[1]` 裸成员 | `static_cast<bool>(data_padding)` 官方谓词 |
| `token_offset` | `_lower_size[1] * head_size` 手算 | `get_linear_offset()` 框架推导 |
| 硬编码 feature 下标 `[1]` | 有 | 无 |
| 硬编码 `* head_size` 换算 | 有 | 无 |
| rank 依赖 | 依赖 3D `[L,H,S]` | 3D/4D 均正确 |
| 数值结果 | 正确 | 与方案 A 逐位一致 |

方案 B 是方案 A 的"以官方 API 表达同一意图"的等价重写，功能与数值完全一致，但消除了
rank 假设与手工换算，更健壮、更贴近框架语义，为后续 wrapper/内层 kernel 用
`get_pitches()` 进一步替换 `CMFLA_IS_QKV_FUSED` 留出了空间（属更大范围改动，暂缓）。

### 6.6 3D/4D 鲁棒性分析

一个自然的疑问：新方案用 `get_linear_offset()`，在输入是 3D 还是 4D 时会不会算错？
结论是**不会，且 4D 恰恰是方案 A 会错、方案 B 能对的场景**。

**① `get_linear_offset()` 与 rank 无关。** 其实现（`layout.cpp`）遍历
`format.dimension()` 个轴，按 padded pitch 求和：

$$\text{offset} = \sum_{axis} \text{lower\_pad}[axis] \times \text{padded\_pitch}[axis]$$

因为乘的是**含 pad 的 pitch**，无论 QKV-slot 维是"独立的 4D 轴"还是"reshape 后折进
feature 轴"，算出的物理元素偏移都一致。以 K（H=2, S=72）为例：

| 输入形态 | layout | K 的 lower_pad | 该轴 padded pitch | offset |
|----------|--------|----------------|-------------------|--------|
| 4D crop | `[L, 3, H, S]` | slot 轴 = 1 | H×S = 144 | 1×144 = **144** ✓ |
| 3D reshape | `[L, 3H, S]` | feature 轴 = H=2 | S = 72 | 2×72 = **144** ✓ |

**② 方案 A 在 4D 下会算错。** `_lower_size[1] * head_size` 把"feature 轴 × head_size"
写死：3D 下 `H × S = 144` 碰巧对；4D 下 `_lower_size[1]` 是 slot 单位（=1），`1 × 72
= 72 ≠ 144`，K/V 指针错位。方案 B 消除了这个 rank 假设。

**③ 其它 shape 派生量已与 offset 解耦。** `head_size = key_shape[size-1]`、
`num_q_heads = query_shape[size-3]`（`vl_sdpa_opt.cpp`）用的是负向索引，依赖 transpose
后规范化成 `[...,H,L,S]`，对 3D `[H,L,S]` 与 4D `[B,H,L,S]` 一致成立；且方案 B 的
`token_offset` 已不再乘 `head_size`，与这些索引完全解耦。

**④ 成立前提（非 bug）。** `get_linear_offset()` 会把**所有轴**的 lower_pad 求和进
offset，而 kernel 里 token(L) 维靠 `kv_start * kv_token_stride` 单独推进。当前 in-place
crop 场景 padding 只落在 QKV-slot 轴、token 轴无 lower_pad，故不会重复计数。

**⑤ 实际契约说明。** `vl_sdpa_inst::calc_output_layouts` 仅 `forward_input0_shape`，
vl_sdpa 的输入契约是 **3D `[L,H,S]`**（transpose order 为 3 元素 `{1,0,2}`）。真实图中
`TransposeSplitMatcher` 产出的 4D packed crop `[L,3,H,S]` 会先 **reshape 成 3D** 再经
RoPE 喂给 vl_sdpa——**vl_sdpa 不会直接接收 4D**。因此上表的 4D 行是 API 层面的
latent-safety 属性（`get_linear_offset()` 天然覆盖），而非 vl_sdpa 的活跃执行路径。

**小结**：方案 B 对 3D/4D 两种 QKV-slot 布局都算出正确物理偏移，不引入 rank 相关 bug；
4D 场景反而暴露了方案 A 的脆弱性。由于 vl_sdpa 实际只接收 3D 输入，现有 3D 回归用例
已完整覆盖生产路径（详见 [第 3 节](#3-复现用例讲解)）。

---

## 7. 背景知识：核心概念与端到端流程

本节面向初次接触 OpenVINO GPU plugin 的读者，解释理解本 bug 所需的六个核心概念，
并用一个 Qwen-VL 模型实例串联成完整的端到端流程。

### 7.1 六个核心概念

#### Node（IR 节点 / 算子）

`ov::Node` 是 OpenVINO IR 里的**计算单元**，描述"做什么"，与硬件完全无关。

```
ov::Node
├── 类型（算子种类）：ScaledDotProductAttention, Split, Reshape, VLSDPA ...
├── 输入端口 (Input)：连接到其他 Node 的输出
├── 输出端口 (Output)：连接到下游 Node 的输入
└── 属性（算子参数）：is_causal, head_size, transpose_order ...
```

Node 之间通过端口连接形成 **有向无环图（DAG）**，即 `ov::Model`。
Node 只管"语义正确"——形状推导（`validate_and_infer_types`）在这里做，不管怎么在
GPU 上跑。例如 VLSDPA 的形状推断里就有：

```cpp
NODE_VALIDATION_CHECK(this,
    shape_q_rank.is_static() && shape_q_rank.get_length() == 3,
    "Query input rank length must be 3.");  // Q 必须是 3D，不满足直接报错
```

#### Pass 序列（图变换流水线）

**Pass** = 一次对整个 DAG 的**模式匹配 + 改写**。GPU plugin 加载模型时会顺序执行
数十个 pass：

```
Pass 的本质：识别子图 → 替换为等价的更优子图

旧子图                                新子图
─────────────────────────             ─────────────────────────
Transpose                             Split(axis=1)
    ↓                    Pass →       ↙       ↓       ↘
Split(axis=0)                       crop_q  crop_k  crop_v
    ↓                                (in-place, 携带 padding 编码偏移)
Reshape × 3
        TransposeSplitMatcher
```

Pass 之间有**顺序依赖**：`SDPAToVLSDPA` 必须在 `TransposeFusion` 之前跑，因为后者
要操作 VLSDPA 节点。所有 pass 跑完后，IR 已是 GPU 友好的形态，多余的 Transpose、
Reshape 已被折叠或消除。

#### cldnn Primitive（GPU 内部算子描述符）

每个 IR Node 被 GPU plugin 翻译成一个 **`cldnn::primitive`**——纯数据结构，只记录
"这个算子的参数"：

```cpp
// ops/vl_sdpa.cpp 里的翻译逻辑
cldnn::vl_sdpa(
    node->get_friendly_name(),          // 算子 ID
    {q_info, k_info, v_info, cu_info},  // 输入列表
    order_q, order_k, order_v, order_out  // transpose orders（已折入）
);
```

多个 primitive 组成 `cldnn::program`（GPU 执行计划图）。与 `ov::Node` 的区别：

| | `ov::Node` | `cldnn::primitive` |
|--|-----------|-------------------|
| 所在层 | OpenVINO IR（硬件无关）| GPU plugin 内部 |
| 作用 | 描述语义 | 描述 GPU 执行参数 |
| 形状信息 | `PartialShape`（含动态维）| `layout`（含 format / padding）|

#### primitive_inst（运行时实例）

`primitive_inst` = primitive 在**运行时的实体**，拥有三样东西：

```
primitive_inst
├── _impl          ← 选定的执行实现（CM? OCL? oneDNN?）
├── _outputs[]     ← GPU buffer（usm_device 内存指针）
└── _impl_params   ← 当前的 input/output layouts（含 padding —— bug 的关键所在）
```

每次推理前若 shape 发生变化，`update_impl()` 会重新选择最优实现并重新计算
dispatch 参数。

#### JIT（Just-In-Time 编译）

GPU kernel 源码里有很多**编译期常量**（如 `HEAD_SIZE`、`IS_QKV_FUSED`）。若全部做成
运行期参数，GPU 编译器无法展开循环和向量化，性能大幅下降。JIT 的做法：

```
第一次推理时：
  1. host 端计算本次推理的具体值：head_size=72, num_heads=2, is_qkv_fused=1
  2. 将这些值注入 kernel 源码作为 #define：
       #define CMFLA_HEAD_SIZE    72
       #define CMFLA_NUM_HEADS    2
       #define CMFLA_IS_QKV_FUSED 1
  3. 调用 IGC（Intel GPU Compiler）实时编译 .cm 源码 → GPU 二进制
  4. 缓存该二进制（同 shape 下直接复用）

后续推理（shape 相同）：→ 直接取缓存，跳过编译
```

`get_jit_constants()` 就是计算"要 `#define` 哪些值"的函数；`get_dispatch_data_func()`
计算每次推理动态变化的运行期标量（`token_offset_q/k/v` 等）。

#### CM kernel（C-for-Metal GPU 程序）

CM 是 Intel GPU 的**低级 C++ 方言**，比 OpenCL 更贴近硬件：

| | OpenCL kernel | CM kernel |
|--|--------------|-----------|
| 操作粒度 | 标量 / SIMD 向量 | 直接控制 GRF 寄存器（32-float 向量） |
| 内存访问 | 全局内存读写 | LSC load/store（可控 cache 策略）|
| 矩阵加速 | 无专用指令 | XMX 指令（Xe 架构硬件矩阵乘）|

`cm_sdpa_vlen.cm` 用 CM 实现 Flash Attention，直接调用 XMX 指令做 Q×K^T
矩阵乘，性能远高于等价 OpenCL 实现。

**host 与 device 的分工**：

```
host（vl_sdpa_opt.cpp，C++）              device（cm_sdpa_vlen.cm，CM，在 GPU 上并行）
────────────────────────────              ───────────────────────────────────────────
get_jit_constants()                       #define CMFLA_HEAD_SIZE 72
  → 计算编译期常量                         #define CMFLA_IS_QKV_FUSED 1
                                           template<int heads, int S, bool fused>
get_dispatch_data_func()                  void sdpa_kernel_lsc(...) {
  → 计算 token_offset_q/k/v（我们改的地方）    用 XMX 做 Q×K^T，softmax，×V
  → 设置 wgs.global 工作组维度                  写输出 buffer
                                          }
```

---

### 7.2 端到端流程：Qwen-VL 模型从 XML 到 CM kernel

#### 起点：model.xml 子图

```xml
<layer type="Parameter" name="packed_qkv"/>   <!-- 输入：[-1, 3, H, S] -->
<layer type="Transpose" .../>
<layer type="Split" axis="0" num_splits="3"/>
<layer type="Reshape" .../> ×3               <!-- → Q / K / V，各 [-1, H, S] -->
<layer type="ScaledDotProductAttention" .../>
<layer type="Parameter" name="attention_mask"/>
```

#### 阶段 0：解析 → ov::Model

```
Parameter(packed_qkv)[-1,3,H,S]
        ↓
    Transpose
        ↓
    Split(axis=0)
   ↙    ↓    ↘
Reshape Reshape Reshape        ← Q / K / V，各自独立形状
   ↘    ↓    ↙
ScaledDotProductAttention
        ↑
Parameter(attention_mask)[float]
```

全是 `ov::Node`，无任何 GPU 信息。

#### 阶段 1：Pass `SDPAToVLSDPA`

检测到 `model_type_hint == "QWenVL"` 且 attention_mask 的 consumer 全是 SDPA：

```
Before                               After
──────────────────────               ──────────────────────────────
ScaledDotProductAttention            op::internal::VLSDPA
  ├─ q                                 ├─ q
  ├─ k                                 ├─ k
  ├─ v                                 ├─ v
  └─ attention_mask (float)            └─ cu_seqlens (int32)  ← 语义替换
                                       (order_q/k/v/out 暂为 {})
```

#### 阶段 2：Pass `TransposeVLSDPAMatcher`

把 VLSDPA 前后的独立 Transpose 节点**折进** VLSDPA 的 order 属性，消掉无用节点：

```
Before                                After
─────────────────────────             ─────────────────────
Transpose([1,0,2]) ×3 → Q/K/V        Q / K / V  （直连）
        ↓                    →        VLSDPA
      VLSDPA                            order_q/k/v/out = {1,0,2}
        ↓
Transpose([1,0,2]) → output
```

#### 阶段 3：Pass `TransposeSplitMatcher`

识别 `Transpose + Split(axis=0)` 模式，改写为 `Split(axis=1)` + in-place crop：

```
Before                                   After
────────────────────────────             ──────────────────────────────────────
Param(packed_qkv)[-1,3,H,S]             Param(packed_qkv)[-1,3*H,S]
        ↓                                         ↓
    Transpose                             Split(axis=1, 3 段)
    Split(axis=0)                        ↙           ↓           ↘
   ↙    ↓    ↘                        crop_q      crop_k      crop_v
 R    R    R  (Reshape)            [L,H,S]    [L,H,S]    [L,H,S]
                                  pad=(0,2H) pad=(H,H)  pad=(2H,0)
                                       ↘           ↓           ↙
                                               VLSDPA
```

**关键**：crop_q/k/v 不拥有自己的 buffer，只记录 padding 元数据（`lower_pad[f]`
编码 slice 偏移）。这是 in-place crop，也是本 bug 的核心场景。

#### 阶段 4：ov::Model → cldnn::program

GPU plugin 遍历变换后的 ov::Model，逐节点翻译为 `cldnn::primitive`：

```cpp
// ops/vl_sdpa.cpp
auto prim = cldnn::vl_sdpa(
    "vlsdpa_0",
    {q_input, k_input, v_input, cu_input},
    {1,0,2}, {1,0,2}, {1,0,2}, {1,0,2}   // 从 VLSDPA node 属性复制
);
```

`cldnn::program` 此时知道每个节点的 **layout**（format + shape + data_padding）：
- crop_k 的 layout：`{[-1,H,S], bfyx, padding(lower=[0,H,0,0], upper=[0,H,0,0])}`
  ← 这个 padding 就是 token_offset 的来源。

#### 阶段 5：实现选择 `update_impl()`

```
Registry<vl_sdpa>::get_implementations()
  → [ VLSDPAOptImplementationManager (CM only) ]

is_supported(impl_params):
  ✓ engine.supports_immad (有 XMX 硬件)
  ✓ check_cm_jit_support  (有 CM JIT 环境)
  → 通过 → 调用 create() → primitive_inst._impl = VLSDPAOptImpl
```

若任一条件不满足，测试里的 `GTEST_SKIP()` 就在此处触发。

#### 阶段 6：JIT 编译 `get_jit_constants()`

```cpp
// 读 layout padding，判断是否 packed 路径（方案 B 的写法）
const bool is_qkv_fused =
    static_cast<bool>(params.input_layouts[0].data_padding) ||  // Q → true
    static_cast<bool>(params.input_layouts[1].data_padding) ||  // K → true
    static_cast<bool>(params.input_layouts[2].data_padding);    // V → true

jit.add({
    make_jit_constant("CMFLA_NUM_HEADS",    2),
    make_jit_constant("CMFLA_HEAD_SIZE",    72),
    make_jit_constant("CMFLA_IS_QKV_FUSED", 1),   // ← packed 路径开启
    make_jit_constant("CMFLA_SCALE_FACTOR", 0.118f),
});
// IGC 编译 cm_sdpa_vlen.cm，生成 GPU 二进制，缓存
```

#### 阶段 7：每次推理 `get_dispatch_data_func()`

```cpp
// 方案 B：从 layout padded pitch 推导物理偏移，rank 无关
const int32_t token_offset_q = params.input_layouts[0].get_linear_offset(); // 0
const int32_t token_offset_k = params.input_layouts[1].get_linear_offset(); // H*S = 144
const int32_t token_offset_v = params.input_layouts[2].get_linear_offset(); // 2H*S = 288

wgs.global = {num_q_heads, wg_count * wg_size, 1};  // GPU 工作组维度
scalars    = {0, 0, 144, 288};  // need_wg_mapping, offset_q, offset_k, offset_v
```

#### 阶段 8：CM kernel 在 GPU 上并行执行

```c
// cm_sdpa_vlen.cm（每个 GPU 工作组，线程 h=0 处理 head 0）
// 编译期已知（bake 进二进制）：HEAD_SIZE=72, IS_QKV_FUSED=1
// 运行期收到标量：token_offset_k=144, token_offset_v=288

constexpr int kv_stride =
    CMFLA_IS_QKV_FUSED ? (num_heads + num_kv_heads*2) * head_size  // 3*2*72=432
                       :  num_kv_heads * head_size;                 // unpacked: 144

uint k_offset = kv_start * kv_stride + hkv * head_size + token_offset_k;
//            = 0        * 432        + 0   * 72        + 144           = 144 ✓

// XMX 硬件矩阵乘：Q×K^T → softmax → ×V
sdpa_kernel_lsc<false, 2, 2, 72, /*is_qkv_fused=*/1>(
    query + token_offset_q,   // physical[0]   Q 起点
    key   + token_offset_k,   // physical[144] K 起点（packed buffer 中）
    value + token_offset_v,   // physical[288] V 起点（packed buffer 中）
    output + o_offset);
// 输出写回 primitive_inst._outputs[0] → 交给下一个算子
```

#### 整体链路一览

```
model.xml
  │ Frontend 解析
  ▼
ov::Model（SDPA + Transpose + Split + Reshape + attention_mask）
  │ SDPAToVLSDPA pass
  ▼
ov::Model（VLSDPA + cu_seqlens，order={}）
  │ TransposeVLSDPAMatcher pass
  ▼
ov::Model（VLSDPA，Transpose 折入 order={1,0,2}）
  │ TransposeSplitMatcher pass
  ▼
ov::Model（Split(axis=1) + in-place crop，Q/K/V 带 feature padding）
  │ GPU plugin 建图（ops/vl_sdpa.cpp）
  ▼
cldnn::program（primitive 图，每节点携带 layout + data_padding 元数据）
  │ update_impl() → Registry<vl_sdpa> → CM 唯一实现，检查 XMX+JIT
  ▼
primitive_inst._impl = VLSDPAOptImpl
  │ get_jit_constants()：is_qkv_fused=1, head_size=72 → IGC JIT 编译 .cm
  ▼
GPU 二进制（CMFLA_IS_QKV_FUSED=1 等常量已 bake 进去）
  │ get_dispatch_data_func()：token_offset_q/k/v 由 get_linear_offset() 推导
  ▼
dispatch → cm_sdpa_vlen.cm（GPU 并行）
  → kv_stride=432（packed）、k_base=144、v_base=288，正确读 packed buffer
  ▼
输出 tensor → 下一个算子
```

---

> **为什么 bug 只影响 vl_sdpa 而非其他算子？**
> 普通 SDPA、RMSNorm 等通过 OpenCL kernel 执行，框架根据 layout 的 padding/pitch
> 元数据**自动换算**访问地址，host 端不需要手算偏移。CM kernel 收到的是**裸 SVM
> 指针 + host 手算的 token_offset 标量**，框架无法兜底——host 算错了，kernel 就读
> 到错误地址，框架无感知。这正是 `token_offset_kv = lower_pad[f] * num_q_heads *
> head_size`（多乘一次 H）能悄无声息地造成大面积数值错误的根本原因。
