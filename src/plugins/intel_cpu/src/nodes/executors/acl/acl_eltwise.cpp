// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_eltwise.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Dimensions.h>
#include <arm_compute/core/Rounding.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPReluLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>
#include <arm_compute/runtime/Tensor.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

using namespace arm_compute;

inline VectorDims reshape_sizes(VectorDims dims) {
    static constexpr size_t MAX_NUM_SHAPE = arm_compute::MAX_DIMS;

    if (dims.size() < MAX_NUM_SHAPE) {
        return dims;
    }

    VectorDims result_dims(MAX_NUM_SHAPE - 1);

    for (size_t i = 0; i < MAX_NUM_SHAPE - 1; i++) {
        result_dims[i] = dims[i];
    }
    for (size_t i = MAX_NUM_SHAPE - 1; i < dims.size(); i++) {
        result_dims[MAX_NUM_SHAPE - 2] *= dims[i];
    }

    return result_dims;
}

inline void log_unsupported_prec(const std::vector<MemoryDescPtr>& srcDescs,
                                 const std::vector<MemoryDescPtr>& dstDescs,
                                 const Algorithm eltwiseAlgorithm) {
    std::string srcPrec;
    for (const auto& srcDesc : srcDescs) {
        srcPrec += srcDesc->getPrecision().to_string() + " ";
    }
    DEBUG_LOG(algToString(eltwiseAlgorithm),
              ": provided combination of src precisions: [",
              srcPrec,
              "] and dst precision: ",
              dstDescs[0]->getPrecision().to_string(),
              " is not supported");
}

bool ACLEltwiseExecutor::supports(const EltwiseConfig& config) {
    // Check for post-ops support
    if (!config.attrs.postOps.empty()) {
        DEBUG_LOG("Eltwise ACL executor does not support post-ops");
        return false;
    }

    std::vector<MemoryDescPtr> srcDescs(config.descs.size() - 1);
    std::vector<MemoryDescPtr> dstDescs{config.descs.at(ARG_DST)};

    for (const auto& [argId, desc] : config.descs) {
        if (argId == ARG_DST) {
            continue;
        }
        srcDescs[argId - ARG_SRC] = desc;
    }

    auto checkPrecision = [&srcDescs, &dstDescs](std::vector<ov::element::Type> srcVecPrc,
                                                 ov::element::Type dstPrc) -> bool {
        for (size_t i = 0; i < srcDescs.size(); i++) {
            if (srcDescs[i]->getPrecision() != srcVecPrc[i]) {
                return false;
            }
        }
        return dstDescs[0]->getPrecision() == dstPrc;
    };

    const auto& eltwiseAttrs = config.attrs;

    switch (eltwiseAttrs.data.algo) {
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseRelu:
#ifdef OPENVINO_ARCH_ARM64
    case Algorithm::EltwiseGeluErf:
#endif
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwisePrelu:
    case Algorithm::EltwiseHswish:
        if (!checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) &&
            !checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32)) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseAbs:
    case Algorithm::EltwiseExp:
    case Algorithm::EltwiseLog:
        if (!checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) &&
            !checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) &&
            !checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32)) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseMaximum:
    case Algorithm::EltwiseMinimum:
    case Algorithm::EltwiseSquaredDifference:
        if (!checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) &&
            !checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) &&
            !checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) &&
            !checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32)) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
        if (!checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) &&
            !checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) &&
            !checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) &&
            !checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) &&
            !checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32)) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    case Algorithm::EltwiseMultiply:
        if (!checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) &&
            !checkPrecision({ov::element::u8, ov::element::u8}, ov::element::i16) &&
            !checkPrecision({ov::element::u8, ov::element::i16}, ov::element::i16) &&
            !checkPrecision({ov::element::i16, ov::element::u8}, ov::element::i16) &&
            !checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) &&
            !checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) &&
            !checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32)) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    // ACL supports only U8 precision on output for comparison operations
    case Algorithm::EltwiseEqual:
    case Algorithm::EltwiseNotEqual:
    case Algorithm::EltwiseGreater:
    case Algorithm::EltwiseGreaterEqual:
    case Algorithm::EltwiseLess:
    case Algorithm::EltwiseLessEqual:
        if (!checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) &&
            !checkPrecision({ov::element::i16, ov::element::i16}, ov::element::u8) &&
            !checkPrecision({ov::element::i32, ov::element::i32}, ov::element::u8) &&
            !checkPrecision({ov::element::f16, ov::element::f16}, ov::element::u8) &&
            !checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32)) {
            log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.data.algo);
            return false;
        }
        break;
    default:
        DEBUG_LOG("Eltwise algorithm ", algToString(eltwiseAttrs.data.algo), " is not supported");
        return false;
    }

    for (const auto& srcDesc : srcDescs) {
        if (getAclDataLayoutByMemoryDesc(srcDesc) == arm_compute::DataLayout::UNKNOWN) {
            DEBUG_LOG("src descriptor layout is unsupported by ACL: ", srcDesc->serializeFormat());
            return false;
        }
    }
    for (const auto& dstDesc : dstDescs) {
        if (getAclDataLayoutByMemoryDesc(dstDesc) == arm_compute::DataLayout::UNKNOWN) {
            DEBUG_LOG("dst descriptor layout is unsupported by ACL: ", dstDesc->serializeFormat());
            return false;
        }
    }

    return true;
}

ACLEltwiseExecutor::ACLEltwiseExecutor(const EltwiseAttrs& attrs,
                                       [[maybe_unused]] const MemoryArgs& memory,
                                       [[maybe_unused]] const ExecutorContext::CPtr& context)
    : aclEltwiseAttrs(attrs) {
    // Set max dimensions for tensor shape handling
    aclTensorAttrs.maxDimsShape = arm_compute::MAX_DIMS;
    
    // Check if any source has NHWC layout
    for (const auto& [argId, mem] : memory) {
        if (argId != ARG_DST && mem->getDescPtr()->hasLayoutType(LayoutType::nspc)) {
            aclTensorAttrs.hasLayoutTypeNHWC = true;
            break;
        }
    }
}

void ACLEltwiseExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    // Reshape dimensions if needed
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        if (aclMemoryShapes[i].num_dimensions() > 0) {
            VectorDims dims;
            for (unsigned int d = 0; d < aclMemoryShapes[i].num_dimensions(); d++) {
                dims.push_back(aclMemoryShapes[i][d]);
            }
            auto reshaped_dims = reshape_sizes(dims);
            aclMemoryShapes[i] = shapeCast(reshaped_dims);
        }
    }

    // Handle special case for binary operations with NHWC layout
    if (aclTensorAttrs.hasLayoutTypeNHWC && 
        aclMemoryShapes[ACL_SRC_0].num_dimensions() > 0 && 
        aclMemoryShapes[ACL_SRC_1].num_dimensions() > 0 &&
        aclMemoryShapes[ACL_SRC_0] != aclMemoryShapes[ACL_SRC_1]) {
        changeLayoutToNH_C({&aclMemoryShapes[ACL_SRC_0], &aclMemoryShapes[ACL_SRC_1], &aclMemoryShapes[ACL_DST]});
    }
}

arm_compute::Status ACLEltwiseExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    // Validate based on the algorithm type
    switch (aclEltwiseAttrs.data.algo) {
    case Algorithm::EltwiseAdd:
        return NEArithmeticAddition::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                             aclMemoryInfos[ACL_SRC_1].get(),
                                             aclMemoryInfos[ACL_DST].get(),
                                             ConvertPolicy::SATURATE);
    case Algorithm::EltwiseMultiply:
        return NEPixelWiseMultiplication::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                  aclMemoryInfos[ACL_SRC_1].get(),
                                                  aclMemoryInfos[ACL_DST].get(),
                                                  1.0F,
                                                  ConvertPolicy::SATURATE,
                                                  RoundingPolicy::TO_ZERO);
    case Algorithm::EltwiseSubtract:
        return NEArithmeticSubtraction::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ConvertPolicy::SATURATE);
    case Algorithm::EltwiseDivide:
        return NEElementwiseDivision::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                              aclMemoryInfos[ACL_SRC_1].get(),
                                              aclMemoryInfos[ACL_DST].get());
    case Algorithm::EltwiseMaximum:
        return NEElementwiseMax::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                         aclMemoryInfos[ACL_SRC_1].get(),
                                         aclMemoryInfos[ACL_DST].get());
    case Algorithm::EltwiseMinimum:
        return NEElementwiseMin::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                         aclMemoryInfos[ACL_SRC_1].get(),
                                         aclMemoryInfos[ACL_DST].get());
    case Algorithm::EltwiseSquaredDifference:
        return NEElementwiseSquaredDiff::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                 aclMemoryInfos[ACL_SRC_1].get(),
                                                 aclMemoryInfos[ACL_DST].get());
    case Algorithm::EltwiseEqual:
        return NEElementwiseComparison::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ComparisonOperation::Equal);
    case Algorithm::EltwiseNotEqual:
        return NEElementwiseComparison::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ComparisonOperation::NotEqual);
    case Algorithm::EltwiseGreater:
        return NEElementwiseComparison::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ComparisonOperation::Greater);
    case Algorithm::EltwiseGreaterEqual:
        return NEElementwiseComparison::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ComparisonOperation::GreaterEqual);
    case Algorithm::EltwiseLess:
        return NEElementwiseComparison::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ComparisonOperation::Less);
    case Algorithm::EltwiseLessEqual:
        return NEElementwiseComparison::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                                aclMemoryInfos[ACL_SRC_1].get(),
                                                aclMemoryInfos[ACL_DST].get(),
                                                ComparisonOperation::LessEqual);
    case Algorithm::EltwiseAbs:
        return NEAbsLayer::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                   aclMemoryInfos[ACL_DST].get());
    case Algorithm::EltwiseExp:
        return NEExpLayer::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                   aclMemoryInfos[ACL_DST].get());
    case Algorithm::EltwisePrelu:
        return NEPReluLayer::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                    aclMemoryInfos[ACL_SRC_1].get(),
                                    aclMemoryInfos[ACL_DST].get());
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
        return NEActivationLayer::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                          aclMemoryInfos[ACL_DST].get(),
                                          getActivationLayerInfo(aclEltwiseAttrs.data.algo,
                                                                aclEltwiseAttrs.data.alpha,
                                                                aclEltwiseAttrs.data.beta,
                                                                aclEltwiseAttrs.data.gamma));
    case Algorithm::EltwiseLog:
        return NELogLayer::validate(aclMemoryInfos[ACL_SRC_0].get(),
                                   aclMemoryInfos[ACL_DST].get());
    default:
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                  "Unsupported Eltwise algorithm");
    }
}

ACLFunction ACLEltwiseExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    switch (aclEltwiseAttrs.data.algo) {
    case Algorithm::EltwiseAdd: {
        auto acl_op = std::make_unique<NEArithmeticAddition>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ConvertPolicy::SATURATE);
        return acl_op;
    }
    case Algorithm::EltwiseMultiply: {
        auto acl_op = std::make_unique<NEPixelWiseMultiplication>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         1.0F,
                         ConvertPolicy::SATURATE,
                         RoundingPolicy::TO_ZERO);
        return acl_op;
    }
    case Algorithm::EltwiseSubtract: {
        auto acl_op = std::make_unique<NEArithmeticSubtraction>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ConvertPolicy::SATURATE);
        return acl_op;
    }
    case Algorithm::EltwiseDivide: {
        auto acl_op = std::make_unique<NEElementwiseDivision>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwiseMaximum: {
        auto acl_op = std::make_unique<NEElementwiseMax>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwiseMinimum: {
        auto acl_op = std::make_unique<NEElementwiseMin>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwiseSquaredDifference: {
        auto acl_op = std::make_unique<NEElementwiseSquaredDiff>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwiseEqual: {
        auto acl_op = std::make_unique<NEElementwiseComparison>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ComparisonOperation::Equal);
        return acl_op;
    }
    case Algorithm::EltwiseNotEqual: {
        auto acl_op = std::make_unique<NEElementwiseComparison>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ComparisonOperation::NotEqual);
        return acl_op;
    }
    case Algorithm::EltwiseGreater: {
        auto acl_op = std::make_unique<NEElementwiseComparison>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ComparisonOperation::Greater);
        return acl_op;
    }
    case Algorithm::EltwiseGreaterEqual: {
        auto acl_op = std::make_unique<NEElementwiseComparison>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ComparisonOperation::GreaterEqual);
        return acl_op;
    }
    case Algorithm::EltwiseLess: {
        auto acl_op = std::make_unique<NEElementwiseComparison>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ComparisonOperation::Less);
        return acl_op;
    }
    case Algorithm::EltwiseLessEqual: {
        auto acl_op = std::make_unique<NEElementwiseComparison>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         ComparisonOperation::LessEqual);
        return acl_op;
    }
    case Algorithm::EltwiseAbs: {
        auto acl_op = std::make_unique<NEAbsLayer>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwiseExp: {
        auto acl_op = std::make_unique<NEExpLayer>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwisePrelu: {
        auto acl_op = std::make_unique<NEPReluLayer>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_SRC_1].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    case Algorithm::EltwiseRelu:
    case Algorithm::EltwiseGeluErf:
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwiseHswish: {
        auto acl_op = std::make_unique<NEActivationLayer>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_DST].get(),
                         getActivationLayerInfo(aclEltwiseAttrs.data.algo,
                                              aclEltwiseAttrs.data.alpha,
                                              aclEltwiseAttrs.data.beta,
                                              aclEltwiseAttrs.data.gamma));
        return acl_op;
    }
    case Algorithm::EltwiseLog: {
        auto acl_op = std::make_unique<NELogLayer>();
        acl_op->configure(aclMemoryTensors[ACL_SRC_0].get(),
                         aclMemoryTensors[ACL_DST].get());
        return acl_op;
    }
    default:
        OPENVINO_THROW("Unsupported operation type for ACL Eltwise executor: ",
                       static_cast<int>(aclEltwiseAttrs.data.algo));
    }
}

}  // namespace ov::intel_cpu