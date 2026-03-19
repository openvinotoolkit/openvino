# OpenVINO GPU Plugin 自定义算子开发指南（以 BevPoolV2 为例）

> 目标：把本次会话里的实战经验沉淀成一份可复用的开发/排错手册。  
> 场景：你要在 OpenVINO 的 Intel GPU 插件里新增一个自定义算子（这里用 `BevPoolV2` 作为示例）。

---

## 1. 先理解：要改哪些模块？

开发一个 GPU 自定义算子，通常要覆盖 6 个层面：

1. **OpenVINO Core 算子定义层**
   - 算子类、属性、shape/type infer、evaluate（可选但建议有）。
   - 典型位置：`src/core/include/openvino/op/<op>.hpp`、`src/core/src/op/<op>.cpp`。

2. **GPU Primitive 抽象层（插件内部图表示）**
   - 在 GPU 插件内部定义 primitive 结构，承载属性与输入输出信息。
   - 典型位置：`src/plugins/intel_gpu/include/intel_gpu/primitives/<op>.hpp`。

3. **Kernel Selector + JIT 层**
   - 定义参数结构、可选 kernel、JIT 常量、dispatch 逻辑。
   - 典型位置：
     - `src/plugins/intel_gpu/src/kernel_selector/kernels/<op>/...`
     - `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/<op>_ref.cl`

4. **GPU Graph/Impl 映射层**
   - 把 `ov::op::vXX::<Op>` 映射成 GPU primitive，并绑定 OCL implementation。
   - 典型位置：
     - `src/plugins/intel_gpu/src/plugin/ops/<op>.cpp`
     - `src/plugins/intel_gpu/src/graph/impls/ocl/<op>.cpp`

5. **工厂注册层（非常关键）**
   - 若没注册，运行时会找不到实现。
   - 典型位置：`src/plugins/intel_gpu/include/intel_gpu/plugin/primitives_list.hpp`

6. **功能测试/子图测试层**
   - 单算子测试 + compare/input map 接入。
   - 典型位置：
     - `src/plugins/intel_gpu/tests/functional/.../single_layer_tests/<op>.cpp`
     - `src/tests/functional/base_func_tests/src/base/utils/generate_inputs.cpp`
     - `src/tests/functional/base_func_tests/src/base/utils/compare_results.cpp`

---

## 2. BevPoolV2 实战里踩到的核心问题（可直接避坑）

### 问题 A：primitive 没注册，插件运行不到实现
- **现象**：图编译或运行阶段找不到对应实现。
- **处理**：在 `primitives_list.hpp` 加上：
  - `REGISTER_FACTORY(v15, BevPoolV2)`

### 问题 B：OpenCL kernel 编译失败（`itv/idx/dw` 未声明）
- **现象**：`clBuildProgram` 失败，日志提示标识符未定义。
- **根因**：`.cl` 内部按 `INPUTS_COUNT` 宏做条件编译，但 JIT 没注入这个常量。
- **处理**：在 kernel base 里补上 JIT 常量：
  - `MakeJitConstant("INPUTS_COUNT", params.inputs.size())`

### 问题 C：测试框架报 `inputMap` / `compare_map` 找不到 BevPoolV2
- **现象**：
  - `Couln't find Operation in inputMap: BevPoolV2`
  - `ASSERT_NE(it, compare_map.end())` 失败
- **根因**：测试工具按 opset/private table 自动注册，私有算子表缺少 `BevPoolV2` 条目时会漏注册。
- **处理方向**：
  1. 在 `ov_ops/opset_private_tbl.hpp` 引入并注册 `BevPoolV2`。
  2. 若仍有覆盖不到的路径，可在 `generate_inputs.cpp` / `compare_results.cpp` 增加显式映射（保险做法）。
  3. 对框架层可加 fallback（避免直接 hard fail），但**最终建议仍补齐正式映射**。

### 问题 D：OpenCL 报错信息不够直观
- **处理**：在 OCL builder 抛异常时带上 build log，便于快速定位 kernel 宏/JIT 问题。

---

## 3. 推荐开发步骤（按顺序做，成功率最高）

## Step 0：准备与基线
1. 切到仓库：
   ```bash
   cd /home/lijie/intel/intel_gpu/openvino
   ```
2. 确认你能完整构建（先不改代码）。

## Step 1：完成 Core Op 定义
1. 新增/确认 `openvino/op/bevpool_v2.hpp` 与对应实现文件。
2. 完成：属性访问、`validate_and_infer_types()`、`clone_with_new_inputs()`。
3. （建议）实现 `evaluate()`，便于 reference 路径与 CPU 侧校验。

## Step 2：完成 GPU primitive 与 plugin op 映射
1. 定义 GPU primitive 数据结构（承载算子属性）。
2. 在 plugin ops translator 中把 `ov::op::v15::BevPoolV2` 转成该 primitive。
3. 在 graph impl ocl 中绑定 kernel selector 实现。

## Step 3：完成 kernel selector 与 OpenCL kernel
1. 添加 `<op>_kernel_base.*` + selector 实现。
2. 在 `.cl` 实现计算逻辑，确保宏条件与 JIT 常量一致。
3. 重点检查 JIT 常量是否齐全（比如 `INPUTS_COUNT` 这类控制分支编译的宏）。

## Step 4：做工厂注册
1. 在 `primitives_list.hpp` 中注册 factory：
   - `REGISTER_FACTORY(v15, BevPoolV2)`
2. 否则运行时可能创建 primitive 失败。

## Step 5：接测试框架映射
1. 在 private ops table 注册 BevPoolV2：
   - include `openvino/op/bevpool_v2.hpp`
   - `_OPENVINO_OP_REG(BevPoolV2, ov::op::v15)`
2. 必要时在：
   - `generate_inputs.cpp` 增加输入生成映射
   - `compare_results.cpp` 增加 compare 映射

## Step 6：增强可观测性（推荐）
1. OCL 构建失败异常中追加 build log。
2. 这样定位 `.cl` 语法和 JIT 宏问题会快很多。

---

## 4. Clean rebuild + 复测命令（可直接复制）

> 在 `openvino` 根目录执行。

```bash
cd /home/lijie/intel/intel_gpu/openvino
rm -rf build bin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build -j"$(nproc)"
./bin/intel64/Release/ov_gpu_func_tests --gtest_filter='*BevPoolV2*' --gtest_color=yes
```

若测试目标名不确定（某些配置差异）：

```bash
cmake --build build --target help | grep -i gpu_func
find . -type f -name ov_gpu_func_tests
./bin/intel64/Release/ov_gpu_func_tests --gtest_list_tests | grep -i BevPoolV2
```

---

## 5. 排错顺序建议（非常实用）

当 BevPoolV2 测试不通过时，按这个顺序查：

1. **先看是否是 kernel 编译失败**
   - 关键词：`clBuildProgram failed`、OpenCL build log。
   - 优先检查 JIT 常量和 `.cl` 宏。

2. **再看是否是注册缺失**
   - 关键词：`find Operation in inputMap`、`compare_map.end()`。
   - 优先检查 `opset_private_tbl.hpp` 与 map 构建逻辑。

3. **最后看数值误差/精度阈值**
   - 若 map 已命中且能走到 compare，再调阈值或核对 reference。

---

## 6. 建议的提交拆分（便于 review）

建议按功能拆成 4~6 个 commit：

1. Core Op + 属性/推导
2. GPU primitive + plugin translator + graph impl
3. Kernel selector + `.cl` + JIT constants
4. Factory 注册
5. 测试接入（private table / inputMap / compareMap）
6. 调试增强（OCL build log）

这样 reviewer 能快速定位每一层改动，也更容易回滚单点问题。

---

## 7. 本次 BevPoolV2 经验结论

- GPU 自定义算子开发不是只写 kernel；**注册链路和测试链路同样关键**。  
- `INPUTS_COUNT` 这类 JIT 宏缺失，会直接导致 `.cl` 编译报看似“变量未定义”的二级错误。  
- `inputMap/compare_map` 报错通常不是算子数学逻辑错误，而是**测试框架注册缺口**。  
- 先让错误信息“可见”（带 build log），再逐层排查，效率最高。

---

## 8. 最小代码骨架模板（可直接复制改名）

> 说明：下面是“能跑通开发链路”的最小骨架，不含真实算法细节。  
> 把 `MyOp` / `my_op` / `v15` 按你的算子实际名称和版本替换。

### 8.1 Core Op 头文件

文件：`src/core/include/openvino/op/my_op.hpp`

```cpp
#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {

class OPENVINO_API MyOp : public Op {
public:
   OPENVINO_OP("MyOp", "opset15");
   MyOp() = default;
   MyOp(const OutputVector& inputs, int64_t some_attr);

   bool visit_attributes(AttributeVisitor& visitor) override;
   void validate_and_infer_types() override;
   std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

   bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
   bool has_evaluate() const override;

private:
   int64_t m_some_attr = 0;
};

}  // namespace v15
}  // namespace op
}  // namespace ov
```

### 8.2 Core Op 实现

文件：`src/core/src/op/my_op.cpp`

```cpp
#include "openvino/op/my_op.hpp"

namespace ov {
namespace op {
namespace v15 {

MyOp::MyOp(const OutputVector& inputs, int64_t some_attr) : Op(inputs), m_some_attr(some_attr) {
   constructor_validate_and_infer_types();
}

bool MyOp::visit_attributes(AttributeVisitor& visitor) {
   visitor.on_attribute("some_attr", m_some_attr);
   return true;
}

void MyOp::validate_and_infer_types() {
   OPENVINO_ASSERT(get_input_size() >= 1, "MyOp requires at least 1 input");
   set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> MyOp::clone_with_new_inputs(const OutputVector& new_args) const {
   check_new_args_count(this, new_args);
   return std::make_shared<MyOp>(new_args, m_some_attr);
}

bool MyOp::has_evaluate() const {
   return false;
}

bool MyOp::evaluate(TensorVector&, const TensorVector&) const {
   return false;
}

}  // namespace v15
}  // namespace op
}  // namespace ov
```

### 8.3 GPU primitive 定义

文件：`src/plugins/intel_gpu/include/intel_gpu/primitives/my_op.hpp`

```cpp
#pragma once

#include "intel_gpu/primitives/primitive.hpp"

namespace cldnn {

struct my_op : primitive_base<my_op> {
   CLDNN_DECLARE_PRIMITIVE(my_op)

   my_op(const primitive_id& id,
        const std::vector<input_info>& inputs,
        int64_t some_attr)
      : primitive_base(id, inputs), some_attr(some_attr) {}

   int64_t some_attr;
};

}  // namespace cldnn
```

### 8.4 Plugin translator（OV Op -> GPU primitive）

文件：`src/plugins/intel_gpu/src/plugin/ops/my_op.cpp`

```cpp
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/my_op.hpp"
#include "openvino/op/my_op.hpp"

namespace ov {
namespace intel_gpu {

static void CreateMyOpOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::MyOp>& op) {
   validate_inputs_count(op, {1});
   auto inputs = p.GetInputInfo(op);
   auto prim = cldnn::my_op(layer_type_name_ID(op), inputs, /*some_attr*/ 0);
   p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v15, MyOp);

}  // namespace intel_gpu
}  // namespace ov
```

### 8.5 Kernel selector（JIT 常量最小骨架）

文件：`src/plugins/intel_gpu/src/kernel_selector/kernels/my_op/my_op_kernel_base.cpp`

```cpp
#include "my_op_kernel_base.h"

namespace kernel_selector {

JitConstants MyOpKernelBase::GetJitConstants(const my_op_params& params) const {
   JitConstants jit = MakeBaseParamsJitConstants(params);
   jit.AddConstant(MakeJitConstant("INPUTS_COUNT", params.inputs.size()));
   jit.AddConstant(MakeJitConstant("SOME_ATTR", params.some_attr));
   return jit;
}

KernelsData MyOpKernelBase::GetCommonKernelsData(const Params& params) const {
   KernelsData kd;
   // 填充 dispatch / kernelName / jit / entry point
   return kd;
}

}  // namespace kernel_selector
```

### 8.6 OpenCL kernel 最小骨架

文件：`src/plugins/intel_gpu/src/kernel_selector/cl_kernels/my_op_ref.cl`

```c
KERNEL(my_op_ref)(OPTIONAL_SHAPE_INFO_ARG
              __global const float* input0,
              __global float* output) {
   const uint gid = (uint)get_global_id(0);
#if INPUTS_COUNT > 1
   // 多输入路径占位
#endif
   output[gid] = input0[gid];
}
```

### 8.7 OCL 实现注册（graph/impl）

文件：`src/plugins/intel_gpu/src/graph/impls/ocl/my_op.cpp`

```cpp
#include "intel_gpu/graph/impls/impl_map.hpp"
#include "intel_gpu/primitives/my_op.hpp"

namespace cldnn {
namespace ocl {

struct my_op_impl : typed_primitive_impl_ocl<my_op> {
   using parent = typed_primitive_impl_ocl<my_op>;
   using parent::parent;
   static primitive_impl* create(const my_op_node& arg, const kernel_impl_params& impl_param) {
      return new my_op_impl(arg, impl_param);
   }
};

namespace detail {
attach_my_op_impl::attach_my_op_impl() {
   implementation_map<my_op>::add(
      impl_types::ocl,
      my_op_impl::create);
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
```

### 8.8 工厂注册

文件：`src/plugins/intel_gpu/include/intel_gpu/plugin/primitives_list.hpp`

```cpp
REGISTER_FACTORY(v15, MyOp)
```

### 8.9 测试映射接入（避免 inputMap/compare_map 报错）

文件：`src/common/transformations/include/ov_ops/opset_private_tbl.hpp`

```cpp
#include "openvino/op/my_op.hpp"
_OPENVINO_OP_REG(MyOp, ov::op::v15)
```

必要时显式补充：

- `src/tests/functional/base_func_tests/src/base/utils/generate_inputs.cpp`
  ```cpp
  {ov::op::v15::MyOp::get_type_info_static(), generateInput<ov::op::v15::MyOp>},
  ```

- `src/tests/functional/base_func_tests/src/base/utils/compare_results.cpp`
  ```cpp
  {ov::op::v15::MyOp::get_type_info_static(), compareResults<ov::op::v15::MyOp>},
  ```

### 8.10 冒烟测试命令模板

```bash
cd /home/lijie/intel/intel_gpu/openvino
rm -rf build bin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build -j"$(nproc)"
./bin/intel64/Release/ov_gpu_func_tests --gtest_filter='*MyOp*' --gtest_color=yes
```

### 8.11 新算子开发 Checklist（可直接打勾）

- [ ] Core Op：属性/shape infer/clone 完成  
- [ ] Plugin translator：OV Op -> primitive 已打通  
- [ ] Kernel selector：JIT 常量齐全（尤其条件编译宏）  
- [ ] OCL kernel：最小路径可编译运行  
- [ ] Factory 注册已加  
- [ ] private ops table 已注册  
- [ ] inputMap/compare_map 不报缺失  
- [ ] `*MyOp*` 功能测试可执行

---

## 9. 本次会话总结（BevPoolV2 真实落地路径）

这一节给出“从报错到可用”的真实推进顺序，后续遇到类似问题可以按同样路径排查。

### 9.1 已完成的关键工作

1. **GPU plugin 注册链打通**
   - 增加 `REGISTER_FACTORY(v15, BevPoolV2)`，避免 primitive 找不到实现。

2. **Kernel 编译阻塞修复**
   - 在 kernel selector JIT 常量里补齐 `INPUTS_COUNT`。
   - 解决 `.cl` 编译阶段变量未声明等连锁错误。

3. **OCL 错误可观测性增强**
   - 在 OCL build 异常路径补充 build log，定位问题速度明显提升。

4. **ONNX 合同统一（4 输入）**
   - 统一输入语义为：`feat`, `depth`, `indices`, `intervals`。
   - 统一 layout：`feat=NHWC`，`depth=NCHW`。

5. **参考真值路径建立（SYCL）**
   - 在 deploy 侧新增 `test_bev_pool_ref_gen`，可生成：
     - `camera_features.bin`
     - `camera_depth_weights.bin`
     - `indices.bin`
     - `intervals.bin`
     - `bev_ref_output.bin`
     - `meta.txt`

6. **OpenVINO 对比脚本建立**
   - 新增 `compare_bevpool_ref.py`，支持 CPU/GPU 与参考输出逐元素误差统计。

### 9.2 关键经验

- GPU 自定义算子开发，**工程链路问题**（注册/JIT/测试映射）通常比数学逻辑问题更早出现。  
- ONNX 侧要保证输入不被裁剪，导出脚本必须确保 graph 对每个输入存在显式依赖。  
- 先建立可重复的 reference（SYCL/CPU evaluate），再谈性能优化，能避免“快但不准”。

---

## 10. 新开发一个 OpenVINO GPU plugin 算子的完整工作分解（以 BevPoolV2 为例）

建议按“里程碑”推进，每个里程碑都要有可验证产物。

### 里程碑 M1：Core Op 完整定义

**目标**：模型前端能识别，shape/type infer 正常。

必做项：
- Op 类定义（属性、构造、访问器）。
- `validate_and_infer_types()`。
- `clone_with_new_inputs()`。
- 推荐实现 `evaluate()`（用于 reference 对齐）。

验收标准：
- Core 单元和基础推导路径无报错。
- 异常输入能报清晰错误。

### 里程碑 M2：GPU primitive + translator + impl

**目标**：`ov::op::v15::BevPoolV2` 能在 GPU 插件内完成映射。

必做项：
- primitive 结构体定义属性。
- plugin translator 正确读取 Op 属性并构造 primitive。
- graph impl ocl 绑定到 kernel selector。

验收标准：
- 编译阶段无 symbol/注册缺失。
- 图编译路径可走到 kernel selector。

### 里程碑 M3：Kernel selector + OpenCL kernel

**目标**：OCL kernel 可编译可执行。

必做项：
- selector 参数、dispatch、优先级逻辑。
- `.cl` kernel 完成输入/输出寻址与计算。
- JIT 常量完整（尤其 `INPUTS_COUNT`、shape 相关宏）。

验收标准：
- `clBuildProgram` 通过。
- kernel 能产生非零、合理分布输出。

### 里程碑 M4：工厂/私有表/测试映射全接入

**目标**：测试框架与运行时都能稳定识别 BevPoolV2。

必做项：
- `primitives_list.hpp` 注册。
- `opset_private_tbl.hpp` 注册。
- 必要时补 `generate_inputs.cpp` / `compare_results.cpp` 映射。

验收标准：
- 无 `inputMap` / `compare_map` 缺失错误。

### 里程碑 M5：端到端验证（功能 + ONNX + benchmark）

**目标**：同一算子在 Core/GPU/ONNX 路径结果一致。

必做项：
- `ov_gpu_func_tests` 通过。
- ONNX 四输入导出可用。
- `benchmark_app` 在 CPU/GPU 可运行。
- 与 SYCL reference 对比误差在阈值内。

---

## 11. 测试体系（重点）

这一节是“怎么测才算完整”的核心，分三层：kernel 级、插件功能级、ONNX 端到端级。

### 11.1 Kernel function 级检测（最小闭环）

目标：先证明 kernel 编译与基本执行正确，再进入更复杂测试。

建议顺序：

1. **编译可用性检测**
   - 观察是否存在 `clBuildProgram failed`。
   - 必须打开/保留 OCL build log（已在会话中增强过）。
   - 重点检查 `.cl` 中依赖宏：`INPUTS_COUNT`、shape、layout 相关宏。

2. **JIT 常量完整性检测**
   - 在 kernel selector 中逐项核对 `MakeJitConstant(...)`。
   - 常见误区：`params.inputs.size()` 与 `.cl` 条件宏不一致。

3. **最小输入执行检测**
   - 用极小 shape（例如 `N=1,D=2,H=3,W=5,C=4`）先跑通。
   - 检查输出是否 NaN/Inf、是否全零（若预期应有非零）。

4. **索引/区间边界检测**
   - 覆盖空区间、单元素区间、跨块区间、最后一段区间。
   - 覆盖 `indices` 越界保护路径（应 fail fast）。

5. **layout 对齐检测**
   - 固定使用：`feat=NHWC`，`depth=NCHW`。
   - 如插件内部需要变换，必须显式验证变换前后一致性。

### 11.2 `ov_gpu_func_tests` 功能测试（OpenVINO 插件侧）

#### 11.2.1 编译与执行

> 推荐在本机稳定参数下用 `-j4`。

```bash
cd /home/lijie/intel/intel_gpu/openvino
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build -j4
./bin/intel64/Release/ov_gpu_func_tests --gtest_filter='*BevPoolV2*' --gtest_color=yes
```

#### 11.2.2 常见失败与对应处理

1. **`inputMap` 缺失**
   - 现象：`Couldn't find Operation in inputMap: BevPoolV2`
   - 处理：补 private table 与 input map 注册。

2. **`compare_map` 缺失**
   - 现象：`ASSERT_NE(it, compare_map.end())`。
   - 处理：补 compare map 映射。

3. **kernel 编译失败**
   - 现象：OpenCL build error。
   - 处理：回到 11.1，优先查 JIT 常量和宏条件。

4. **数值误差偏大**
   - 处理：
     - 先与 reference 对比确定是实现误差还是输入布局错误；
     - 再评估容差设置（不要先放宽阈值掩盖逻辑问题）。

#### 11.2.3 推荐测试矩阵

- shape 维度：小/中/目标业务尺寸。
- 数据分布：随机、全零、稀疏深度、极值。
- interval 分布：均匀、长尾、空洞。
- 设备：CPU reference、GPU plugin。

### 11.3 `benchmark_app` + ONNX 端到端测试（前后端契约）

#### 11.3.1 生成 ONNX（4 输入）

当前导出脚本：
- `feat`（NHWC）
- `depth`（NCHW）
- `indices`（K）
- `intervals`（M,3）

导出示例（使用指定虚拟环境）：

```bash
cd /home/lijie/intel/intel_gpu/openvino
/home/lijie/envs/ovenv/bin/python export_bevpool_v2_custom_op.py \
  --out bevpool_v2_custom_ref4.onnx \
  --N 1 --D 90 --H 54 --W 96 --C 80 \
   --K 466560 --M 7313 \
  --out-height 128 --out-width 128
```

检查输入确实是 4 个：

```bash
cd /home/lijie/intel/intel_gpu/openvino
/home/lijie/envs/ovenv/bin/python - <<'PY'
import onnx
m = onnx.load('bevpool_v2_custom_ref4.onnx')
print('num_inputs =', len(m.graph.input))
for i, inp in enumerate(m.graph.input):
    print(i, inp.name)
PY
```

#### 11.3.2 `benchmark_app` 冒烟（CPU/GPU）

> 目标是验证前后端连通和输入契约，不是做性能结论。

```bash
cd /home/lijie/intel/intel_gpu/openvino
./bin/intel64/Release/benchmark_app \
  -m ./bevpool_v2_custom_ref4.onnx \
  -d CPU \
   -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7313,3]" \
  -niter 5

./bin/intel64/Release/benchmark_app \
  -m ./bevpool_v2_custom_ref4.onnx \
  -d GPU \
   -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7313,3]" \
  -niter 5
```

### 11.4 与 SYCL reference 的逐元素误差比对

#### 11.4.1 生成参考数据

在 deploy 工程：

```bash
cd /mnt/project/garnet_park/bev/bev_latest/deploy
./build/test_bev_pool_ref_gen ./build/bevpool_ref_case1 2026
```

#### 11.4.2 在 OpenVINO 侧做误差对比

```bash
cd /home/lijie/intel/intel_gpu/openvino
/home/lijie/envs/ovenv/bin/python compare_bevpool_ref.py \
  --model ./bevpool_v2_custom_ref4.onnx \
  --ref-dir /mnt/project/garnet_park/bev/bev_latest/deploy/build/bevpool_ref_case1 \
  --topk 10
```

输出重点关注：
- `CPU vs REF`: `max_abs`, `mean_abs`, `max_rel`, `mean_rel`
- `GPU vs REF`: 同上
- `CPU vs GPU`: 插件一致性
- `Top-k mismatch`: 最大误差位置定位

---

## 12. 一份可执行的“新算子开发交付清单”

功能实现：
- [ ] Core Op（属性、infer、clone、evaluate）
- [ ] GPU primitive + translator + impl
- [ ] kernel selector + OCL kernel
- [ ] JIT 常量全覆盖（含条件编译宏）
- [ ] factory 注册

测试接入：
- [ ] private ops table 注册
- [ ] input/compare map 注册
- [ ] `ov_gpu_func_tests` 指定用例可通过

前后端验证：
- [ ] ONNX 导出为 4 输入（feat/depth/indices/intervals）
- [ ] `benchmark_app` CPU/GPU 均可运行
- [ ] 与 SYCL reference 完成逐元素误差对比

排错能力：
- [ ] OCL build log 可见
- [ ] 失败路径有明确日志（输入名、shape、layout、attr）

文档与复现：
- [ ] 命令可一键复现
- [ ] 环境（Python/依赖/路径）写清楚
- [ ] 数据格式（bin header、layout）写清楚

---

## 13. 结语：推荐工程策略

1. **先正确，后性能**：先让 reference 对齐，再做 kernel 优化。  
2. **先小 shape 打通，再上业务 shape**：减少排错复杂度。  
3. **每层都可观测**：Core、plugin、kernel、frontend、benchmark 都要能定位。  
4. **提交按层拆分**：让回归与 code review 更高效。  
5. **固定契约**：输入名、layout、dtype、bin 格式一旦确定，脚本和测试统一维护。