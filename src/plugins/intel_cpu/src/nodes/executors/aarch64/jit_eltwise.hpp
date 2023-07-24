// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_types.h"
#include "../executor.hpp"
#include "../eltwise.hpp"

namespace ov {
namespace intel_cpu {
namespace executors {
namespace aarch64 {

using namespace InferenceEngine;

class JitEltwiseExecutor : public EltwiseExecutor {
public:
    explicit JitEltwiseExecutor(const ExecutorContext::CPtr context);
    static bool isSupported(const Algorithm& algorithm);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return impl_desc_type::asimd;
    }
private:
    std::function<void()> exec_func;
};

}   // namespace aarch64
}   // namespace executors
}   // namespace intel_cpu
}   // namespace ov
