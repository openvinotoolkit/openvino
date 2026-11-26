// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/kernels/aarch64/jit_sdot_convolution.hpp"

namespace ov::intel_cpu::aarch64 {

class Oc1DotConvolution : public Executor {
public:
    Oc1DotConvolution(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config);
    static bool supportsGeometry(const VectorDims& src,
                                 const VectorDims& weights,
                                 const VectorDims& dst,
                                 const std::vector<size_t>& strides,
                                 const std::vector<size_t>& dilations,
                                 const std::vector<ptrdiff_t>& padsBegin,
                                 const std::vector<ptrdiff_t>& padsEnd);
    bool update(const MemoryArgs&) override {
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::jit_asimd;
    }

private:
    std::shared_ptr<CpuParallel> m_parallel;
    size_t m_channels;
    bool m_unsigned_input;
    int32_t m_zero_point;
    size_t m_pad_top;
    size_t m_pad_left;
    float m_scale;
    bool m_with_bias;
    Oc1DotKernel m_kernel;
};

}  // namespace ov::intel_cpu::aarch64
