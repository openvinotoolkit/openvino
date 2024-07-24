// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_gemm.hpp"

#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"

namespace ov {
namespace intel_cpu {

ACLGEMMExecutor::ACLGEMMExecutor(const GEMMAttrs &attrs,
                                 const PostOps &postOps,
                                 const MemoryArgs &memory,
                                 const ExecutorContext::CPtr context) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    OPENVINO_ASSERT(!attrs.transpose_a && !attrs.transpose_b, "not supported");
}

bool ACLGEMMExecutor::supports(const GEMMConfig &config) {
    // TODO: check weights layout
    const auto attrs = static_cast<GEMMAttrs>(config.attrs);
    if (attrs.transpose_a || attrs.transpose_b) {
        return false;
    }
    if (std::any_of(
            attrs.dequantizationScales.begin(),
            attrs.dequantizationScales.end(),
            [](float value) { return value != 1.f;})) {
        return false;
    }

    const auto src1_dims = std::dynamic_pointer_cast<BlockedMemoryDesc>(config.descs.at(ARG_SRC))->getBlockDims();
    const auto src2_dims = std::dynamic_pointer_cast<BlockedMemoryDesc>(config.descs.at(ARG_WEI))->getBlockDims();

    VERIFY(one_of(srcType(config), ov::element::i8, ov::element::u8), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(postOpsNumbers(config) == 0, UNSUPPORTED_NUMBER_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U, 4U), UNSUPPORTED_WEI_RANK);
    VERIFY(static_cast<GEMMAttrs>(config.attrs).dequantizationScales.size() <= 1, UNSUPPORTED_PER_CHANNEL_QUANTIZATION);
    return true;
}

void ACLGEMMExecutor::updateTensorsShapes(ACLMemoryShapes& aclMemoryShapes) {
    if (gemmInfo.pretranspose_B()) {
        std::swap(aclMemoryShapes[ACLArgs::ACL_WEI][0], aclMemoryShapes[ACLArgs::ACL_WEI][1]);
    }
}

arm_compute::Status ACLGEMMExecutor::validateTensorsInfo(const ACLMemoryInfo & aclMemoryInfos) {
    const auto matMulValid = arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            gemmInfo);
    return matMulValid;
}

ACLFunction ACLGEMMExecutor::configureFunction(const ACLMemoryTensors & aclMemoryTensors) {
    auto gemm = std::make_unique<arm_compute::NEGEMMLowpMatrixMultiplyCore>();
    gemm->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
            aclMemoryTensors[ACLArgs::ACL_WEI].get(),
            aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
            aclMemoryTensors.at(ACLArgs::ACL_DST).get(),
            gemmInfo);
    return gemm;
}

ACLInfo ACLGEMMExecutor::initTensorInfo(const arm_compute::TensorShape& tensorShape,
                                        const arm_compute::DataType& dataType,
                                        const arm_compute::DataLayout& dataLayout) {
    arm_compute::DataType result;
    switch (dataType) {
        case arm_compute::DataType::S8: {
            result = arm_compute::DataType::QASYMM8_SIGNED;
            break;
        }
        case arm_compute::DataType::U8: {
            result = arm_compute::DataType::QASYMM8;
            break;
        }
        default: {
            result = dataType;
            break;
        }
    }

    return ACLCommonExecutor::initTensorInfo(tensorShape, result, dataLayout);
}

}   // namespace intel_cpu
}   // namespace ov
