// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

// Xbyak AArch64 JIT
#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu {

struct jit_conv3d_call_args {
    const uint16_t* src;   // f16 base ptr
    const uint16_t* wei;   // f16 base ptr
    const uint16_t* wei2;  // optional second oc f16 base ptr (can be null)
    const uint16_t* wei3;  // optional third oc f16 base ptr (can be null)
    const uint16_t* wei4;  // optional fourth oc f16 base ptr (can be null)
    size_t repeats;        // number of full 8-channel blocks
    size_t tail;           // remaining channels (< 8)
    size_t src_stride;     // stride between channels in bytes
    size_t wei_stride;     // stride between channels in bytes
    size_t src_blk_stride; // stride between successive 8-channel blocks in bytes
    size_t wei_blk_stride; // stride between successive 8-channel blocks in bytes
    float* acc;            // f32 accumulator
    float* acc2;           // optional second f32 accumulator (can be null)
    size_t kw_cnt;         // number of taps along W to iterate (stride=1 fast path); 0 or 1 -> single
    size_t src_dx;         // bytes to advance src base between successive kx taps
    size_t wei_dx;         // bytes to advance weights base between successive kx taps
    float* acc3;           // optional third f32 accumulator (can be null)
    float* acc4;           // optional fourth f32 accumulator (can be null)
    size_t kh_cnt;         // number of taps along H to iterate (stride=1 fast path); 0 or 1 -> single
    size_t src_dy;         // bytes to advance src base between successive ky taps
    size_t wei_dy;         // bytes to advance weights base between successive ky taps
};

class JitConv3DKernelF16 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitConv3DKernelF16)
    using jit_fn = void (*)(const jit_conv3d_call_args*);

    JitConv3DKernelF16();

    void create_ker();
    inline void operator()(const jit_conv3d_call_args* p) const {
        ker_(p);
    }

private:
    void generate() override;

    jit_fn ker_{nullptr};
};

// AArch64 JIT Convolution (FP16) executor for 3D conv (NCDHW)
class JitConv3DExecutor : public Executor {
public:
    JitConv3DExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override {
        m_memory = memory;
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    void execute() override { execute(m_memory); }
    void exec([[maybe_unused]] const std::vector<MemoryCPtr>& src,
              [[maybe_unused]] const std::vector<MemoryPtr>& dst) override {}

    [[nodiscard]] impl_desc_type implType() const override { return impl_desc_type::jit_asimd; }

    static bool supports(const ConvConfig& cfg);

private:
    // Simple reference fallback (parallelized) using FP16 data; correctness-first
    void run_naive_fp16(const MemoryArgs& memory);
    void ensure_weights_packed(const MemoryArgs& memory);

    // Minimal inner-product kernel (fp16 x fp16 -> f32 accumulation)
    std::unique_ptr<JitConv3DKernelF16> m_ip_kernel;

    ConvAttrs m_attrs;
    MemoryArgs m_memory;
    size_t m_threadsNum{0};

    // Packed weights: layout [OC, KD, KH, KW, Ct] where Ct is 8-lane channel tiles
    std::vector<uint16_t> m_wei_packed;
    bool m_wei_packed_ready{false};
    size_t m_padded_C{0};

    // Optional fused PReLU (per-tensor or per-channel). Extracted from attrs.postOps.
    bool m_has_prelu{false};
    std::vector<float> m_prelu_slopes;  // size 1 (per-tensor) or OC (per-channel)

    // Gate executor-side post-ops (bias, PReLU). Disabled per user request for measurements.
    bool m_apply_post_ops{false};
};

using JitConv3DExecutorPtr = std::shared_ptr<JitConv3DExecutor>;

}  // namespace ov::intel_cpu
