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
#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu {

struct jit_conv3d_call_args {
    const uint16_t* src;    // f16 base ptr
    const uint16_t* wei;    // f16 base ptr
    const uint16_t* wei2;   // optional second oc f16 base ptr (can be null)
    const uint16_t* wei3;   // optional third oc f16 base ptr (can be null)
    const uint16_t* wei4;   // optional fourth oc f16 base ptr (can be null)
    size_t repeats;         // number of full 8-channel blocks
    size_t tail;            // remaining channels (< 8)
    size_t src_stride;      // stride between channels in bytes
    size_t wei_stride;      // stride between channels in bytes
    size_t src_blk_stride;  // stride between successive 8-channel blocks in bytes
    size_t wei_blk_stride;  // stride between successive 8-channel blocks in bytes
    float* acc;             // f32 accumulator
    float* acc2;            // optional second f32 accumulator (can be null)
    size_t kw_cnt;          // number of taps along W to iterate (stride=1 fast path); 0 or 1 -> single
    size_t src_dx;          // bytes to advance src base between successive kx taps
    size_t wei_dx;          // bytes to advance weights base between successive kx taps
    float* acc3;            // optional third f32 accumulator (can be null)
    float* acc4;            // optional fourth f32 accumulator (can be null)
    size_t kh_cnt;          // number of taps along H to iterate (stride=1 fast path); 0 or 1 -> single
    size_t src_dy;          // bytes to advance src base between successive ky taps
    size_t wei_dy;          // bytes to advance weights base between successive ky taps
};

struct jit_conv3d_f32_call_args {
    const float* src;
    const float* wei;
    const float* wei2;
    size_t repeats;
    size_t tail;
    size_t src_stride;
    size_t wei_stride;
    size_t src_blk_stride;
    size_t wei_blk_stride;
    float* acc;
    float* acc2;
    size_t kw_cnt;
    size_t src_dx;
    size_t wei_dx;
    // Optional extra outputs for quad-OC scheduling. When acc3 is non-null,
    // the kernel can update up to 4 output channels in a single pass.
    const float* wei3{nullptr};
    const float* wei4{nullptr};
    float* acc3{nullptr};
    float* acc4{nullptr};
};

class JitConv3DKernelF16 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitConv3DKernelF16)
    using jit_fn = void (*)(const jit_conv3d_call_args*);

    JitConv3DKernelF16();

    void create_ker();
    void set_use_fhm(bool v) { m_use_fhm_ = v; }
    inline void operator()(const jit_conv3d_call_args* p) const {
        ker_(p);
    }

private:
    void generate() override;

    void gen_minimal_kernel();
    void gen_optimized_kernel();

    jit_fn ker_{nullptr};
    bool m_force_single_kh_{true};
    bool m_use_fhm_{true};
public:
    void set_force_single_kh(bool v) { m_force_single_kh_ = v; }
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

class JitConv3DExecutor : public Executor {
public:
    JitConv3DExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

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

    void prepare_weights_early(const MemoryArgs& memory);

private:
    void run_naive_fp16(const MemoryArgs& memory);
    void run_naive_fp16_fallback(const MemoryArgs& memory);
    void ensure_weights_packed(const MemoryArgs& memory);
    void run_naive_fp32(const MemoryArgs& memory);
    void ensure_weights_packed_f32(const MemoryArgs& memory);

    std::unique_ptr<JitConv3DKernelF16> m_ip_kernel;
    std::unique_ptr<JitConv3DKernelF32> m_ip_kernel_f32;

    ConvAttrs m_attrs;
    MemoryArgs m_memory;
    size_t m_threadsNum{0};
    bool m_is_fp32{false};

    std::vector<uint16_t> m_wei_packed;
    bool m_wei_packed_ready{false};
    size_t m_padded_C{0};
    std::vector<float> m_wei_packed_f32;
    bool m_wei_packed_ready_f32{false};
    size_t m_padded_C_f32{0};

    bool m_has_prelu{false};
    std::vector<float> m_prelu_slopes;
};

using JitConv3DExecutorPtr = std::shared_ptr<JitConv3DExecutor>;

}  // namespace ov::intel_cpu
