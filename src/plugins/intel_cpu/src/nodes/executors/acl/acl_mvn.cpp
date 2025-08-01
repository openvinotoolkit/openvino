// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "acl_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

bool ACLMVNExecutor::supports(const MVNConfig& config) {
    DEBUG_LOG("ACL MVN supports() called");

    // Check precision - ACL MVN supports f16 and f32
    const auto& srcDesc = config.descs.at(ARG_SRC_0);
    const auto& dstDesc = config.descs.at(ARG_DST);

    // Check supported precisions
    auto srcPrecision = srcDesc->getPrecision();
    auto dstPrecision = dstDesc->getPrecision();

    DEBUG_LOG("ACL MVN: srcPrecision=", srcPrecision.get_type_name(), " dstPrecision=", dstPrecision.get_type_name());
    DEBUG_LOG("ACL MVN: normalizeVariance=",
              config.attrs.normalizeVariance_,
              " initAcrossChannels=",
              config.attrs.initAcrossChannels_,
              " execAcrossChannels=",
              config.attrs.execAcrossChannels_,
              " epsValue=",
              config.attrs.epsValue_,
              " epsMode=",
              static_cast<int>(config.attrs.epsMode_));

    const bool unsupported_src_precision = srcPrecision != ov::element::f32 && srcPrecision != ov::element::f16;
    const bool unsupported_dst_precision = dstPrecision != ov::element::f32 && dstPrecision != ov::element::f16;

    if (unsupported_src_precision || unsupported_dst_precision) {
        DEBUG_LOG("ACL MVN: Unsupported precision");
        return false;
    }

    // Input and output precisions must match
    if (srcPrecision != dstPrecision) {
        DEBUG_LOG("ACL MVN: Precision mismatch");
        return false;
    }

    if (config.attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT) {
        DEBUG_LOG("ACL MVN: OUTSIDE_SQRT not supported");
        return false;
    }
    if (!config.attrs.normalizeVariance_) {
        DEBUG_LOG("ACL MVN: normalize_variance=false not supported");
        return false;
    }

    // Check layout compatibility
    const bool ncsp_mismatch = srcDesc->hasLayoutType(LayoutType::ncsp) && !dstDesc->hasLayoutType(LayoutType::ncsp);
    const bool nspc_mismatch = srcDesc->hasLayoutType(LayoutType::nspc) && !dstDesc->hasLayoutType(LayoutType::nspc);

    if (ncsp_mismatch || nspc_mismatch) {
        DEBUG_LOG("ACL MVN: Layout mismatch");
        return false;
    }

    // Original conditions from master: NHWC with initAcrossChannels=false is not supported
    if (!config.attrs.initAcrossChannels_ && srcDesc->hasLayoutType(LayoutType::nspc)) {
        DEBUG_LOG("ACL MVN: NHWC with initAcrossChannels=false not supported");
        return false;
    }

    DEBUG_LOG("ACL MVN: supports() returning true");
    return true;
}

void ACLMVNExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    DEBUG_LOG("ACL MVN updateTensorsShapes called");

    // Get original shape from ACL tensor
    const auto& srcShape = aclMemoryShapes[ACLArgs::ACL_SRC_0];

    // Convert ACL shape to VectorDims for easier manipulation
    VectorDims srcDims;
    for (size_t i = 0; i < srcShape.num_dimensions(); i++) {
        srcDims.push_back(srcShape[srcShape.num_dimensions() - 1 - i]);
    }

    // Original logic from master branch
    size_t X = 0, Y = 0;
    if (aclMVNAtrrs.initAcrossChannels_) {
        if (srcDims.size() >= 2U) {
            Y = srcDims[0];
            X = srcDims[1];
            for (size_t i = 2; i < srcDims.size(); i++) {
                X *= srcDims[i];
            }
        } else {
            Y = 1;
            X = srcDims[0];
        }
    } else {
        if (srcDims.size() > 2U) {
            Y = srcDims[0] * srcDims[1];
            X = srcDims[2];
            for (size_t i = 3; i < srcDims.size(); i++) {
                X *= srcDims[i];
            }
        } else if (srcDims.size() == 2U) {
            Y = srcDims[0] * srcDims[1];
            X = 1;
        } else {
            Y = srcDims[0];
            X = 1;
        }
    }

    // ACL expects shape in (width, height) format
    arm_compute::TensorShape newShape(X, Y);

    aclMemoryShapes[ACLArgs::ACL_SRC_0] = newShape;
    aclMemoryShapes[ACLArgs::ACL_DST] = newShape;
}

arm_compute::Status ACLMVNExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    DEBUG_LOG("ACL MVN validateTensorsInfo called");

    // We handle NHWC with initAcrossChannels=false by shape transformation in updateTensorsShapes
    // So we don't need to reject it here

    auto status = arm_compute::NEMeanStdDevNormalizationLayer::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                                                        aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                                                        aclMVNAtrrs.epsValue_);

    if (!status) {
        DEBUG_LOG("ACL MVN validation failed: ", status.error_description());
    } else {
        DEBUG_LOG("ACL MVN validation succeeded");
    }

    return status;
}

ACLFunction ACLMVNExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    // ACL may have issues with very small epsilon values
    // Use a minimum epsilon value that works well with ACL
    float aclEpsilon = std::max(aclMVNAtrrs.epsValue_, 1e-6F);

    DEBUG_LOG("ACL MVN configureFunction called, original epsilon=",
              aclMVNAtrrs.epsValue_,
              ", ACL epsilon=",
              aclEpsilon);

    auto neMVN = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    neMVN->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(), aclMemoryTensors[ACLArgs::ACL_DST].get(), aclEpsilon);
    DEBUG_LOG("ACL MVN configureFunction completed");
    return neMVN;
}

}  // namespace ov::intel_cpu
