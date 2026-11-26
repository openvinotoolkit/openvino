// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/kernels/aarch64/jit_sdot_convolution.hpp"

namespace ov::intel_cpu::aarch64 {

class U8Oc16DotConvolution : public Executor {
public:
    U8Oc16DotConvolution(const ConvAttrs&, const MemoryArgs&, const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config);
    bool update(const MemoryArgs&) override {
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::jit_asimd;
    }

private:
    std::shared_ptr<CpuParallel> m_parallel;
    U8Oc16DotKernel m_kernel;
};

}  // namespace ov::intel_cpu::aarch64
