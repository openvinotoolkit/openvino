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
    const std::vector<ov::element::Type>& input_precisions,
    const std::vector<ov::element::Type>& output_precisions,
    const float alpha,
    const float beta,
    const float gamma) {
    const auto is_supported = one_of(algorithm,
                                     Algorithm::EltwiseAbs,
                                     Algorithm::EltwiseAdd,
                                     Algorithm::EltwiseClamp,
                                     Algorithm::EltwiseDivide,
                                     Algorithm::EltwiseEqual,
                                     Algorithm::EltwiseExp,
                                     Algorithm::EltwiseMultiply,
                                     Algorithm::EltwiseMulAdd,
                                     Algorithm::EltwisePowerStatic,
                                     Algorithm::EltwisePrelu,
                                     Algorithm::EltwiseRelu,
                                     Algorithm::EltwiseSelect,
                                     Algorithm::EltwiseSigmoid,
                                     Algorithm::EltwiseSubtract,
                                     Algorithm::EltwiseSwish,
                                     Algorithm::EltwiseTanh);
    if (!is_supported) {
        return false;
    }

    if ((algorithm == Algorithm::EltwiseRelu) && ((alpha != 0.f) || (beta != 0.f) || (gamma != 0.f))) {
        return false;
    }

    const auto check_precisions = [](
            const std::vector<ov::element::Type>& input_precisions,
            const std::vector<ov::element::Type>& output_precisions,
            const std::set<ov::element::Type>& supported_precisions) {
        if (std::any_of(input_precisions.begin(),
                        input_precisions.end(),
                        [&supported_precisions](const ov::element::Type& precision) {
                            return supported_precisions.find(precision) == supported_precisions.end();
                        })) {
            return false;
        }

        if (std::any_of(output_precisions.begin(),
                        output_precisions.end(),
                        [&supported_precisions](const ov::element::Type& precision) {
                            return supported_precisions.find(precision) == supported_precisions.end();
                        })) {
            return false;
        }

        return true;
    };

    const std::set<ov::element::Type> supported_precisions =
        (algorithm == Algorithm::EltwiseDivide) ?
            // Divide operation doesn't support int32 tensor inference in fp32 precision.
            // As result Divide operation supports fp16 and fp32 only.
            std::set<ov::element::Type> { ov::element::f16, ov::element::f32 } :
            std::set<ov::element::Type> {
                ov::element::f16,
                ov::element::f32,
                ov::element::i32,
                ov::element::i8,
                ov::element::u8
            };

    if (!check_precisions(input_precisions, output_precisions, supported_precisions)) {
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
