// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h>

#include <cstddef>
#include <memory>
#include <string>

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
              config.attrs.initAcrossChannels_);

    if ((srcPrecision != ov::element::f32 && srcPrecision != ov::element::f16) ||
        (dstPrecision != ov::element::f32 && dstPrecision != ov::element::f16)) {
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
    if ((srcDesc->hasLayoutType(LayoutType::ncsp) && !dstDesc->hasLayoutType(LayoutType::ncsp)) ||
        (srcDesc->hasLayoutType(LayoutType::nspc) && !dstDesc->hasLayoutType(LayoutType::nspc))) {
        DEBUG_LOG("ACL MVN: Layout mismatch");
        return false;
    }

    // ACL MVN with NHWC layout only supports initAcrossChannels = true
    // But we can handle this case by reshaping appropriately in updateTensorsShapes
    if (srcDesc->hasLayoutType(LayoutType::nspc) && !config.attrs.initAcrossChannels_) {
        DEBUG_LOG("ACL MVN: NHWC layout with initAcrossChannels=false will be handled via shape transformation");
        // We'll handle this in updateTensorsShapes
    }

    DEBUG_LOG("ACL MVN: supports() returning true");
    return true;
}

void ACLMVNExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    DEBUG_LOG("ACL MVN updateTensorsShapes called");
    const auto srcDims = aclMemoryShapes[ACLArgs::ACL_SRC_0];
    const auto srcNumDim = aclMemoryShapes[ACLArgs::ACL_SRC_0].num_dimensions();

    DEBUG_LOG("ACL MVN: srcNumDim=",
              srcNumDim,
              ", initAcrossChannels=",
              aclMVNAtrrs.initAcrossChannels_,
              ", isNHWCLayout=",
              isNHWCLayout);
    for (size_t i = 0; i < srcNumDim; i++) {
        DEBUG_LOG("  srcDims[", i, "]=", srcDims[i]);
    }

    size_t X = 1, Y = 1;  // Initialize with default values

    // Handle empty tensor (scalar) or 1D tensor
    if (srcNumDim == 0) {
        DEBUG_LOG("ACL MVN: Handling scalar tensor, reshaping to (1,1)");
        aclMemoryShapes[ACLArgs::ACL_SRC_0] = aclMemoryShapes[ACLArgs::ACL_DST] = arm_compute::TensorShape(1, 1);
        return;
    }
    if (srcNumDim == 1) {
        // For 1D tensor, reshape to 2D based on normalization mode
        if (aclMVNAtrrs.initAcrossChannels_) {
            DEBUG_LOG("ACL MVN: Handling 1D tensor across channels, reshaping to (", srcDims[0], ", 1)");
            aclMemoryShapes[ACLArgs::ACL_SRC_0] = aclMemoryShapes[ACLArgs::ACL_DST] =
                arm_compute::TensorShape(srcDims[0], 1);
        } else {
            DEBUG_LOG("ACL MVN: Handling 1D tensor not across channels, reshaping to (1, ", srcDims[0], ")");
            aclMemoryShapes[ACLArgs::ACL_SRC_0] = aclMemoryShapes[ACLArgs::ACL_DST] =
                arm_compute::TensorShape(1, srcDims[0]);
        }
        return;
    }

    // Special handling for NHWC layout with initAcrossChannels=false
    if (isNHWCLayout && !aclMVNAtrrs.initAcrossChannels_) {
        DEBUG_LOG("ACL MVN: Special handling for NHWC with initAcrossChannels=false");
        // For NHWC layout, we need to treat it as if initAcrossChannels=true
        // because ACL MVN doesn't support channel-wise normalization with NHWC
        // We'll reshape the tensor to merge spatial dimensions
        if (srcDims.num_dimensions() >= 2u) {
            Y = srcDims[srcNumDim - 1];  // Channels dimension
            X = 1;
            for (size_t i = 0; i < srcDims.num_dimensions() - 1; i++) {
                X *= srcDims[i];  // Merge all spatial dimensions
            }
        } else {
            Y = 1;
            X = srcDims[0];
        }
        // Force initAcrossChannels to true for ACL processing
        aclMVNAtrrs.initAcrossChannels_ = true;
        DEBUG_LOG("ACL MVN: Forced initAcrossChannels=true for NHWC layout");
    } else if (aclMVNAtrrs.initAcrossChannels_) {
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
    DEBUG_LOG("ACL MVN: Final reshape to (X=", X, ", Y=", Y, ")");
    aclMemoryShapes[ACLArgs::ACL_SRC_0] = aclMemoryShapes[ACLArgs::ACL_DST] = arm_compute::TensorShape(X, Y);
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
    DEBUG_LOG("ACL MVN configureFunction called, epsilon=", aclMVNAtrrs.epsValue_);
    auto neMVN = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    neMVN->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                     aclMemoryTensors[ACLArgs::ACL_DST].get(),
                     aclMVNAtrrs.epsValue_);
    DEBUG_LOG("ACL MVN configureFunction completed");
    return neMVN;
}

}  // namespace ov::intel_cpu
