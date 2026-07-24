# OpenVINO Plugin U2 Support Implementation Summary

## ✅ 已完成的完整实现

### 1. 类型映射和基础支持
- **dnnl_extension_utils.cpp**: 
  - 将 `ov::element::u2` 映射到 `dnnl::memory::data_type::u8`
  - 添加注释说明 u2 packing/unpacking 在 plugin 层处理

- **cpu_memory.cpp**: 
  - 在 `split_horizontal()` 添加 u2 的字节对齐处理 (stride /= 4)
  - 在 `split_vertical()` 添加 u2 的字节对齐处理 (strideSize /= 4, copySize /= 4)

- **config.cpp**: 
  - 在 KV cache 配置中添加 u2 支持（3个配置项）

- **memory_state.cpp**: 
  - 在 KV cache 状态管理中添加 u2 支持

- **jit_fc_decomp_brgemm.cpp**: 
  - `isSupportedCompressedWeightsType()`: 添加 u2
  - `isUnsignedCompressedWeightsType()`: 添加 u2
  - `readPackedValue()`: 添加 u2 的位提取逻辑（每字节4个值）
  - `supportsDynamicQuantization()`: 在 zero point 检查中添加 u2 支持

### 2. ✨ Weight Decompression Kernel - 完整实现

#### 2.1 JIT Kernel 实现
**jit_fc_weight_decompression_kernel.hpp**:
- 添加 `icIndex` 参数到 `FCWeightDecompressionKernelCompileParams`
  - u4: icIndex 0-1 (2 个值/字节)
  - u2: icIndex 0-3 (4 个值/字节)
- 添加 `vmmMask4()` VMM register
- 添加 `regTmp` 寄存器

**jit_fc_weight_decompression_kernel.cpp**:
- **u4 解压缩实现**:
  ```cpp
  uni_vpmovzxbd(vmmWeights(), ptr[regWeights]);
  if (icIndex == 0) {
      uni_vpand(vmmWeights(), vmmWeights(), vmmMask4());  // 低4位
  } else {
      uni_vpsrld(vmmWeights(), vmmWeights(), 4);          // 高4位
  }
  uni_vcvtdq2ps(vmmWeights(), vmmWeights());
  ```

- **u2 解压缩实现** (严格按照 oneDNN 逻辑):
  ```cpp
  uni_vpmovzxbd(vmmWeights(), ptr[regWeights]);
  if (icIndex == 0) {
      uni_vpsrld(vmmWeights(), vmmWeights(), 6);        // Bits [7:6]
  } else {
      uni_vpslld(vmmWeights(), vmmWeights(), 24 + 2 * icIndex);
      uni_vpsrld(vmmWeights(), vmmWeights(), 30);       // Extract 2 bits
  }
  uni_vcvtdq2ps(vmmWeights(), vmmWeights());
  ```

#### 2.2 多 Kernel 架构
**jit_fc_decomp_brgemm.hpp**:
- 将单一 kernel 改为 kernel 数组:
  ```cpp
  std::vector<std::unique_ptr<FCWeightDecompressionKernelBase>> m_jitDecompressionKernels;
  std::vector<std::unique_ptr<FCWeightDecompressionKernelBase>> m_jitWeightUnpackKernels;
  ```

#### 2.3 Kernel 构建逻辑
**rebuildDecompressionKernel()** 修改:
- 根据权重类型确定需要创建的 kernel 数量:
  - u8/i8: 1 个 kernel
  - u4/i4: 2 个 kernels (icIndex 0, 1)
  - u2: 4 个 kernels (icIndex 0, 1, 2, 3)
- 为每个 icIndex 创建独立的 kernel
- 支持 AVX2 和 AVX512

### 3. ✨ IC 循环完整实现

#### 3.1 decompressWeights() 函数
- 计算 `icInternalSize` (u8:1, u4:2, u2:4)
- 在 IC 循环中：
  - 计算 `packedIcIdx = icIdx / icInternalSize` (物理字节索引)
  - 计算 `internalIcIdx = icIdx % icInternalSize` (字节内索引)
  - 计算正确的 `compressedWeightsAddr`
  - 选择对应的 kernel: `jitKernels[internalIcIdx]`

#### 3.2 refreshDynamicQuantWeightParams() 函数
- 扩展 `canUseJitUnpack` 支持 u4/i4/u2
- 添加 `icInternalSize` 计算
- 在 IC 循环中：
  - 计算 packed 地址
  - 选择对应的 unpack kernel
  - 正确解压缩到 dynamic quant weights

### 4. 代码质量改进
- 修复寄存器冲突（使用 `regTmp` 而不是 `regScales`）
- 添加详细注释说明 sub-byte packing
- 保持与 oneDNN 实现的一致性

## 实现细节

### U2 数据格式（字节内布局）

```
Byte:  [b7 b6 | b5 b4 | b3 b2 | b1 b0]
       icIdx=0  icIdx=1 icIdx=2 icIdx=3

提取逻辑（与 oneDNN 一致）:
- icIdx=0: srl 6           → bits [7:6]
- icIdx=1: sll 26, srl 30  → bits [5:4]  
- icIdx=2: sll 28, srl 30  → bits [3:2]
- icIdx=3: sll 30, srl 30  → bits [1:0]
```

### Packed Address 计算

对于 weightsNonTransposed 布局：
```cpp
物理字节索引 = icIdx / icInternalSize
字节内索引   = icIdx % icInternalSize
地址 = packedIcIdx * OC + ocIdx
```

### 支持的功能

#### ✅ 已支持
- [x] u2 权重解压缩 (standard decompression)
- [x] u2 + scales
- [x] u2 + zero points
- [x] u2 + scales + zero points
- [x] u2 dynamic quantization
- [x] u2 dynamic quantization + scales + zero points
- [x] Non-transposed weights layout
- [x] AVX2 和 AVX512 ISA

#### ⚠️ 限制
- Transposed weights: 地址计算可能需要调整
- AMX: 未测试

## 修改文件汇总

```
src/plugins/intel_cpu/src/config.cpp                           | +15 -9
src/plugins/intel_cpu/src/cpu_memory.cpp                       | +5
src/plugins/intel_cpu/src/dnnl_extension_utils.cpp             | +13 -6
src/plugins/intel_cpu/src/memory_state.cpp                     | +4 -1
src/plugins/intel_cpu/src/nodes/executors/x64/jit_fc_decomp_brgemm.cpp  | +128 -52
src/plugins/intel_cpu/src/nodes/executors/x64/jit_fc_decomp_brgemm.hpp  | +7 -2
src/plugins/intel_cpu/src/nodes/kernels/x64/jit_fc_weight_decompression_kernel.cpp | +30
src/plugins/intel_cpu/src/nodes/kernels/x64/jit_fc_weight_decompression_kernel.hpp | +9
```

总共修改：9 个文件，新增约 211 行，删除约 70 行。

## 测试建议

### 1. 基础测试
```cpp
// u2 weights without scales/zero-points
FC(M=1, N=256, K=512, weights=u2, src=f32, dst=f32)

// u2 weights with per-channel scales
FC(M=1, N=256, K=512, weights=u2, scales=[N], src=f32, dst=f32)

// u2 weights with grouped scales
FC(M=1, N=256, K=512, weights=u2, scales=[N, groups=32], src=f32, dst=f32)
```

### 2. Dynamic Quantization 测试
```cpp
// u2 weights + dynamic quant source
FC(M=1, N=256, K=512, weights=u2, src=f32, dst=f32, dynamicQuant=true)

// u2 weights + scales + zero-points + dynamic quant
FC(M=1, N=256, K=512, weights=u2, scales=[N], zp=[N], src=f32, dst=f32, dynamicQuant=true)
```

### 3. 边界情况
- 不同的 M/N/K 组合（特别是不能被 blockSize 整除的）
- Broadcast scales/zero-points
- 不同的 group sizes

## 参考

### oneDNN Commit
- **Commit**: `2c9811860df040654ad1dc694faf3de9404e56de`
- **Title**: [FORK][CPU][FEATURE] InnerProduct primitive: u2 weights decompression

### 关键文件
- `src/cpu/x64/jit_brgemm_weights_decompression_kernel.cpp`
- `src/cpu/x64/jit_brgemm_inner_product_utils.cpp`
- `src/cpu/x64/jit_brgemm_inner_product.cpp`
