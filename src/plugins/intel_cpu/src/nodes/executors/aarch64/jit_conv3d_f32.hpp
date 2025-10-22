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

#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu {

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
