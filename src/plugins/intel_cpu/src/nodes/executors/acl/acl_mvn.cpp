// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h>

#include <algorithm>
#include <cstddef>
#include <memory>

#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/debug_messages.hpp"
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

    const auto& srcPrecision = srcDesc->getPrecision();
    const auto& dstPrecision = dstDesc->getPrecision();

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

    VERIFY((srcPrecision == ov::element::f32 || srcPrecision == ov::element::f16) &&
               (dstPrecision == ov::element::f32 || dstPrecision == ov::element::f16),
           UNSUPPORTED_SRC_PRECISIONS);

    // Input and output precisions must match
    VERIFY(srcPrecision == dstPrecision, UNSUPPORTED_DST_PRECISIONS);

    // ACL supports only INSIDE_SQRT with normalizeVariance=true
    VERIFY(config.attrs.epsMode_ != MVNEpsMode::OUTSIDE_SQRT, UNSUPPORTED_ATTRIBUTE);
    VERIFY(config.attrs.normalizeVariance_, UNSUPPORTED_ATTRIBUTE);

    // Check layout compatibility
    // Require src and dst layouts to match (either ncsp or nspc)
    const bool both_ncsp = srcDesc->hasLayoutType(LayoutType::ncsp) && dstDesc->hasLayoutType(LayoutType::ncsp);
    const bool both_nspc = srcDesc->hasLayoutType(LayoutType::nspc) && dstDesc->hasLayoutType(LayoutType::nspc);
    VERIFY(both_ncsp || both_nspc, MEMORY_FORMAT_MISMATCH);

    // Original conditions from master: NHWC with initAcrossChannels=false is not supported
    VERIFY(config.attrs.initAcrossChannels_ || !srcDesc->hasLayoutType(LayoutType::nspc), UNSUPPORTED_ATTRIBUTE);

    return true;
}

void ACLMVNExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    DEBUG_LOG("ACL MVN updateTensorsShapes called");

    // Get the original shape from ACL tensor
    const auto& srcShape = aclMemoryShapes[ACLArgs::ACL_SRC_0];

    // Convert ACL shape to VectorDims for easier manipulation
    VectorDims srcDims;
    for (size_t i = 0; i < srcShape.num_dimensions(); i++) {
        srcDims.push_back(srcShape[srcShape.num_dimensions() - 1 - i]);
    }

    // Original logic from the master branch
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
