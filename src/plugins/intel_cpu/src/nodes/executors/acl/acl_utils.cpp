// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "support/Mutex.h"

namespace ov {
namespace intel_cpu {

void configureThreadSafe(const std::function<void(void)>& config) {
    // Issue: CVS-123514
    static arm_compute::Mutex mtx_config;
    arm_compute::lock_guard<arm_compute::Mutex> _lock{mtx_config};
    config();
}

bool getActivationLayerInfo(Algorithm algorithm,
                            arm_compute::ActivationLayerInfo &activationLayerInfo,
                            float alpha = 0.0,
                            float beta  = 0.0,
                            float gamma = 0.0) {
    switch (algorithm) {
        case Algorithm::EltwiseRelu:
            if (alpha == 0) {
                activationLayerInfo = arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
            } else {
                activationLayerInfo = {arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, alpha};
            }
            return true;
        case Algorithm::EltwiseGeluErf:
            activationLayerInfo = arm_compute::ActivationLayerInfo::ActivationFunction::GELU;
            return true;
        case Algorithm::EltwiseElu:
            activationLayerInfo = {arm_compute::ActivationLayerInfo::ActivationFunction::ELU, alpha};
            return true;
        case Algorithm::EltwiseTanh:
            activationLayerInfo = {arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f};
            return true;
        case Algorithm::EltwiseSigmoid:
            activationLayerInfo = arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC;
            return true;
        case Algorithm::EltwiseSqrt:
            activationLayerInfo = arm_compute::ActivationLayerInfo::ActivationFunction::SQRT;
            return true;
        case Algorithm::EltwiseSoftRelu:
            activationLayerInfo = arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU;
            return true;
        case Algorithm::EltwiseClamp:
            activationLayerInfo = {arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, beta, alpha};
            return true;
        case Algorithm::EltwiseSwish:
            activationLayerInfo = {arm_compute::ActivationLayerInfo::ActivationFunction::SWISH, alpha};
            return true;
        case Algorithm::EltwiseHswish:
            activationLayerInfo = arm_compute::ActivationLayerInfo::ActivationFunction::HARD_SWISH;
            return true;
        default:
            return false;
    }
}

}   // namespace intel_cpu
}   // namespace ov
