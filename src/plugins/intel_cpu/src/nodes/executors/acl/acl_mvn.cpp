// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"
#include "acl_utils.hpp"

namespace ov::intel_cpu {

bool ACLMVNExecutor::supports(const MVNConfig &config) {
    if (config.attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT) {
        DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support OUTSIDE_SQRT mode");
        return false;
    }
    if (!config.attrs.normalizeVariance_) {
        DEBUG_LOG("NEMeanStdDevNormalizationLayer supports normalize_variance=true only");
        return false;
    }
    return true;
}

void ACLMVNExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    const auto srcDims = aclMemoryShapes[ACLArgs::ACL_SRC_0];
    const auto srcNumDim = aclMemoryShapes[ACLArgs::ACL_SRC_0].num_dimensions();

    size_t X, Y;
    if (aclMVNAtrrs.initAcrossChannels_) {
        if (srcDims.num_dimensions() >= 2u) {
            Y = srcDims[srcNumDim - 1];
            X = srcDims[srcNumDim - 2];
            for (size_t i = 2; i < srcDims.num_dimensions(); i++) {
                X *= srcDims[srcNumDim - i - 1];
            }
        } else {
            Y = 1;
            X = srcDims[srcNumDim - 1];
        }
    } else {
        if (srcDims.num_dimensions() > 2u) {
            Y = srcDims[srcNumDim - 1] * srcDims[srcNumDim - 2];
            X = srcDims[srcNumDim - 3];
            for (size_t i = 3; i < srcDims.num_dimensions(); i++) {
                X *= srcDims[srcNumDim - i - 1];
            }
        } else if (srcDims.num_dimensions() == 2u) {
            Y = srcDims[srcNumDim - 1] * srcDims[srcNumDim - 2];
            X = 1;
        } else {
            Y = srcDims[srcNumDim - 1];
            X = 1;
        }
    }
    aclMemoryShapes[ACLArgs::ACL_SRC_0] = aclMemoryShapes[ACLArgs::ACL_DST] = arm_compute::TensorShape(X, Y);
}

arm_compute::Status ACLMVNExecutor::validateTensorsInfo(const ACLInfos &aclMemoryInfos) {
    if (!aclMVNAtrrs.initAcrossChannels_ &&
        aclMemoryInfos[ACLArgs::ACL_SRC_0]->data_layout() == arm_compute::DataLayout::NHWC) {
        std::string error_description = "initAcrossChannels = false is not supported by ACL for NHWC layout";
        DEBUG_LOG(error_description);
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR, error_description);
    }
    return arm_compute::NEMeanStdDevNormalizationLayer::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            aclMVNAtrrs.epsValue_);
}

ACLFunction ACLMVNExecutor::configureFunction(const ACLTensors & aclMemoryTensors) {
    auto neMVN = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    neMVN->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
            aclMemoryTensors[ACLArgs::ACL_DST].get(),
            aclMVNAtrrs.epsValue_);
    return neMVN;
}

} // namespace ov::intel_cpu

