// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise.hpp"
#include <vector>

namespace ov {
namespace intel_cpu {
namespace executors {
namespace aarch64 {

bool JitEltwiseExecutor::isSupported(
    const Algorithm& algorithm,
    const float alpha,
    const float beta,
    const float gamma) {
    const auto is_supported = one_of(algorithm,
                                    Algorithm::EltwiseAdd,
                                    Algorithm::EltwiseDivide,
                                    Algorithm::EltwiseMultiply,
                                    Algorithm::EltwiseMulAdd,
                                    Algorithm::EltwisePowerStatic,
                                    Algorithm::EltwiseRelu,
                                    Algorithm::EltwiseSigmoid,
                                    Algorithm::EltwiseSubtract);
    if (!is_supported) {
        return false;
    }

    if ((algorithm == Algorithm::EltwiseRelu) && ((alpha != 0.f) || (beta != 0.f) || (gamma != 0.f))) {
        return false;
    }

    return true;
}

JitEltwiseExecutor::JitEltwiseExecutor(const ExecutorContext::CPtr context) : EltwiseExecutor(context) {}

bool JitEltwiseExecutor::init(const EltwiseAttrs &eltwiseAttrs,
                              const std::vector<MemoryDescPtr> &srcDescs,
                              const std::vector<MemoryDescPtr> &dstDescs,
                              const std::vector<EltwisePostOp> &postOps) {
    return true;
}

void JitEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src,
                              const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    exec_func();
}

}   // namespace aarch64
}   // namespace executors
}   // namespace intel_cpu
}   // namespace ov
