// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_types.h"
#include "node.h"
#include "nodes/executors/eltwise.hpp"

namespace ov::intel_cpu::executors::aarch64 {

class JitEltwiseExecutor : public EltwiseExecutor {
public:
    explicit JitEltwiseExecutor(const ExecutorContext::CPtr context);

    static bool isSupported(const Algorithm& algorithm,
                            const std::vector<ov::element::Type>& input_precisions,
                            const std::vector<ov::element::Type>& output_precisions,
                            const float alpha,
                            const float beta,
                            const float gamma);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return impl_desc_type::asimd;
    }

private:
    std::function<void()> exec_func;
};

}  // namespace ov::intel_cpu::executors::aarch64
