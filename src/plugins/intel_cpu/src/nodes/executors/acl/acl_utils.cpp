// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"

#include "support/Mutex.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

void configureThreadSafe(const std::function<void(void)>& config) {
    // Issue: CVS-123514
    static arm_compute::Mutex mtx_config;
    arm_compute::lock_guard<arm_compute::Mutex> _lock{mtx_config};
    config();
}

arm_compute::ActivationLayerInfo getActivationLayerInfo(Algorithm algorithm,
                                                        float alpha = 0.0,
                                                        float beta = 0.0,
                                                        float gamma = 0.0) {
    switch (algorithm) {
    case Algorithm::EltwiseRelu:
        if (alpha == 0) {
            return arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
        } else {
            return {arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, alpha};
        }
    case Algorithm::EltwiseGeluErf:
        return arm_compute::ActivationLayerInfo::ActivationFunction::GELU;
    case Algorithm::EltwiseElu:
        return {arm_compute::ActivationLayerInfo::ActivationFunction::ELU, alpha};
    case Algorithm::EltwiseTanh:
        return {arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f};
    case Algorithm::EltwiseSigmoid:
        return arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
    case Algorithm::EltwiseSqrt:
        return arm_compute::ActivationLayerInfo::ActivationFunction::SQRT;
    case Algorithm::EltwiseSoftRelu:
        return arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU;
    case Algorithm::EltwiseClamp:
        return {arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, beta, alpha};
    case Algorithm::EltwiseSwish:
        return {arm_compute::ActivationLayerInfo::ActivationFunction::SWISH, alpha};
    case Algorithm::EltwiseHswish:
        return arm_compute::ActivationLayerInfo::ActivationFunction::HARD_SWISH;
    default:
        OPENVINO_THROW("Unsupported operation type for ACL Eltwise executor: ", static_cast<int>(algorithm));
    }
}

bool checkActivationLayerInfo(Algorithm algorithm) {
    switch (algorithm) {
    case Algorithm::EltwiseRelu:
    case Algorithm::EltwiseGeluErf:
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwiseHswish:
        return true;
    default:
        return false;
    }
}

}  // namespace ov::intel_cpu
