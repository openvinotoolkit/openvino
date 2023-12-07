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
    const Node* node,
    const float alpha,
    const float beta,
    const float gamma) {
    const Algorithm& algorithm = node->getAlgorithm();
    const auto is_supported = one_of(algorithm,
                                    Algorithm::EltwiseAdd,
                                    Algorithm::EltwiseMultiply,
                                    Algorithm::EltwiseMulAdd,
                                    Algorithm::EltwisePowerStatic,
                                    Algorithm::EltwiseRelu);
    if (!is_supported) {
        return false;
    }

    const auto check_precisions = [&node](const std::set<ov::element::Type>& precisions) {
        const auto& input_precisions = node->getOriginalInputPrecisions();
        if (std::any_of(input_precisions.begin(),
                        input_precisions.end(),
                        [&precisions](const ov::element::Type& precision) {
                            return precisions.find(precision) == precisions.end();
                        })) {
            return false;
        }

        const auto& output_precisions = node->getOriginalOutputPrecisions();
        if (std::any_of(output_precisions.begin(),
                        output_precisions.end(),
                        [&precisions](const ov::element::Type& precision) {
                            return precisions.find(precision) == precisions.end();
                        })) {
            return false;
        }

        return true;
    };

    const std::set<ov::element::Type> supported_precisions = {
        ov::element::f16,
        ov::element::f32
    };

    const auto parent = node->getParentEdgeAt(0)->getParent();
    if (parent->getType() == ov::intel_cpu::Type::Convert) {
        const auto& input_precisions = parent->getOriginalInputPrecisions();
        if (input_precisions.size() != 1ull) {
            return false;
        }
        // input precision will be changed after fuse
        if (supported_precisions.find(input_precisions[0]) == supported_precisions.end()) {
            return false;
        }
    }

    if (!check_precisions(supported_precisions)) {
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
