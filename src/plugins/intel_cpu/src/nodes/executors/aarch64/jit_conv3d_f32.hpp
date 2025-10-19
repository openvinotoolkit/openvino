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

// Xbyak AArch64 JIT
#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu {

struct jit_conv3d_f32_call_args {
    const float* src;       // f32 base ptr
    const float* wei;       // f32 base ptr (oc0)
    const float* wei2;      // optional second oc f32 base ptr (can be null)
    size_t repeats;         // number of full 4-channel blocks
    size_t tail;            // remaining channels (< 4)
    size_t src_stride;      // stride between channels in bytes
    size_t wei_stride;      // stride between channels in bytes
    size_t src_blk_stride;  // stride between successive 4-channel blocks in bytes
    size_t wei_blk_stride;  // stride between successive 4-channel blocks in bytes
    float* acc;             // f32 accumulator
    float* acc2;            // optional second f32 accumulator (can be null)
    size_t kw_cnt;          // number of taps along W to iterate (stride=1 fast path); 0 or 1 -> single
    size_t src_dx;          // bytes to advance src base between successive kx taps
    size_t wei_dx;          // bytes to advance weights base between successive kx taps
};

class JitConv3DKernelF32 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitConv3DKernelF32)
    using jit_fn = void (*)(const jit_conv3d_f32_call_args*);

    JitConv3DKernelF32() = default;

    void create_ker();
    inline void operator()(const jit_conv3d_f32_call_args* p) const {
        ker_(p);
    }

private:
    void generate() override;

    jit_fn ker_{nullptr};
};

// AArch64 JIT Convolution (FP32) executor for 3D conv (NCDHW)
class JitConv3DExecutorF32 : public Executor {
public:
    JitConv3DExecutorF32(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override {
        m_memory = memory;
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    void execute() override {
        execute(m_memory);
    }
    void exec([[maybe_unused]] const std::vector<MemoryCPtr>& src,
              [[maybe_unused]] const std::vector<MemoryPtr>& dst) override {}

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::jit_asimd;
    }

    static bool supports(const ConvConfig& cfg);

private:
    void run_naive_fp32(const MemoryArgs& memory);
    void ensure_weights_packed(const MemoryArgs& memory);

    std::unique_ptr<JitConv3DKernelF32> m_ip_kernel;

    ConvAttrs m_attrs;
    MemoryArgs m_memory;

    std::vector<float> m_wei_packed;  // [OC, KD, KH, KW, Ct=4]
    bool m_wei_packed_ready{false};
    size_t m_padded_C{0};
};

using JitConv3DExecutorF32Ptr = std::shared_ptr<JitConv3DExecutorF32>;

}  // namespace ov::intel_cpu
