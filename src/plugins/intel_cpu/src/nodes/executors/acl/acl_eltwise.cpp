// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_eltwise.hpp"
#include "acl_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

inline VectorDims reshape_sizes(VectorDims dims) {
    const size_t MAX_NUM_SHAPE = arm_compute::MAX_DIMS;
    VectorDims result_dims(MAX_NUM_SHAPE - 1);
    if (dims.size() >= MAX_NUM_SHAPE) {
        for (size_t i = 0; i < MAX_NUM_SHAPE - 1; i++) {
            result_dims[i] = dims[i];
        }
        for (size_t i = MAX_NUM_SHAPE - 1; i < dims.size(); i++) {
            result_dims[MAX_NUM_SHAPE - 2] *= dims[i];
        }
    } else {
        result_dims = dims;
    }
    return result_dims;
}

inline void log_unsupported_prec(const std::vector<MemoryDescPtr>& srcDescs,
                                 const std::vector<MemoryDescPtr>& dstDescs,
                                 const Algorithm eltwiseAlgorithm) {
    std::string srcPrec;
    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcPrec += srcDescs[i]->getPrecision().to_string() + " ";
    }
    DEBUG_LOG(algToString(eltwiseAlgorithm), ": provided combination of src precisions: [", srcPrec,
                          "] and dst precision: ", dstDescs[0]->getPrecision().to_string(), " is not supported");
}

bool AclEltwiseExecutor::isEltwiseAlgorithmSupported(Algorithm algorithm) {
    if (one_of(algorithm, Algorithm::EltwiseSqrt,
                          Algorithm::EltwiseDivide,
                          Algorithm::EltwiseRelu,
#ifdef OPENVINO_ARCH_ARM64
                          Algorithm::EltwiseGeluErf,
#endif
                          Algorithm::EltwiseElu,
                          Algorithm::EltwiseTanh,
                          Algorithm::EltwiseSigmoid,
                          Algorithm::EltwiseSoftRelu,
                          Algorithm::EltwiseClamp,
                          Algorithm::EltwiseSwish,
                          Algorithm::EltwisePrelu,
                          Algorithm::EltwiseHswish,
                          Algorithm::EltwiseAbs,
                          Algorithm::EltwiseExp,
                          Algorithm::EltwiseLog,
                          Algorithm::EltwiseMaximum,
                          Algorithm::EltwiseMinimum,
                          Algorithm::EltwiseSquaredDifference,
                          Algorithm::EltwiseAdd,
                          Algorithm::EltwiseSubtract,
                          Algorithm::EltwiseMultiply,
                          Algorithm::EltwiseEqual,
                          Algorithm::EltwiseNotEqual,
                          Algorithm::EltwiseGreater,
                          Algorithm::EltwiseGreaterEqual,
                          Algorithm::EltwiseLess,
                          Algorithm::EltwiseLessEqual)) {
        return true;
    }
    return false;
}

bool AclEltwiseExecutorBuilder::isSupported(const EltwiseAttrs& eltwiseAttrs,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs) const {
    auto checkPrecision = [&srcDescs, &dstDescs](std::vector<ov::element::Type> srcVecPrc, ov::element::Type dstPrc) -> bool {
        for (size_t i = 0; i < srcDescs.size(); i++) {
            if (srcDescs[i]->getPrecision() != srcVecPrc[i]) return false;
        }
        if (dstDescs[0]->getPrecision() != dstPrc) { return false; }
        return true;
    };

    switch (eltwiseAttrs.algorithm) {
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
            if (!(checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
                  checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
                return false;
            }
            break;
        case Algorithm::EltwiseAbs:
        case Algorithm::EltwiseExp:
        case Algorithm::EltwiseLog:
            if (!(checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) ||
                  checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
                  checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
                return false;
            }
            break;
        case Algorithm::EltwiseMaximum:
        case Algorithm::EltwiseMinimum:
        case Algorithm::EltwiseSquaredDifference:
            if (!(checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) ||
                  checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) ||
                  checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
                  checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
                return false;
            }
            break;
        case Algorithm::EltwiseAdd:
        case Algorithm::EltwiseSubtract:
            if (!(checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) ||
                  checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) ||
                  checkPrecision({ov::element::i32, ov::element::i32}, ov::element::i32) ||
                  checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
                  checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
                return false;
            }
            break;
        case Algorithm::EltwiseMultiply:
            if (!(checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) ||
                  checkPrecision({ov::element::u8, ov::element::u8}, ov::element::i16) ||
                  checkPrecision({ov::element::u8, ov::element::i16}, ov::element::i16) ||
                  checkPrecision({ov::element::i16, ov::element::u8}, ov::element::i16) ||
                  checkPrecision({ov::element::i16, ov::element::i16}, ov::element::i16) ||
                  checkPrecision({ov::element::f16, ov::element::f16}, ov::element::f16) ||
                  checkPrecision({ov::element::f32, ov::element::f32}, ov::element::f32))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
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
            if (!(checkPrecision({ov::element::u8, ov::element::u8}, ov::element::u8) ||
                  checkPrecision({ov::element::i16, ov::element::i16}, ov::element::u8) ||
                  checkPrecision({ov::element::i32, ov::element::i32}, ov::element::u8) ||
                  checkPrecision({ov::element::f16, ov::element::f16}, ov::element::u8) ||
                  checkPrecision({ov::element::f32, ov::element::f32}, ov::element::u8))) {
                log_unsupported_prec(srcDescs, dstDescs, eltwiseAttrs.algorithm);
                return false;
            }
            break;
        default:
            DEBUG_LOG("Eltwise algorithm ", algToString(eltwiseAttrs.algorithm), " is not supported");
            return false;
    }

    for (const auto & srcDesc : srcDescs) {
        if (getAclDataLayoutByMemoryDesc(srcDesc) == arm_compute::DataLayout::UNKNOWN) {
            DEBUG_LOG("src descriptor layout is unsupported by ACL: ", srcDesc->serializeFormat());
            return false;
        }
    }
    for (const auto & dstDesc : dstDescs) {
        if (getAclDataLayoutByMemoryDesc(dstDesc) == arm_compute::DataLayout::UNKNOWN) {
            DEBUG_LOG("dst descriptor layout is unsupported by ACL: ", dstDesc->serializeFormat());
            return false;
        }
    }

    return true;
}

AclEltwiseExecutor::AclEltwiseExecutor(const ExecutorContext::CPtr context) : EltwiseExecutor(context) {}

bool AclEltwiseExecutor::init(const EltwiseAttrs &eltwiseAttrs, const std::vector<MemoryDescPtr> &srcDescs,
                              const std::vector<MemoryDescPtr> &dstDescs,
                              const std::vector<EltwisePostOp> &postOps) {
    if (!postOps.empty()) { return false; }
    aclEltwiseAttrs = eltwiseAttrs;

    std::vector<arm_compute::TensorShape> srcVecDims(srcDescs.size()), dstVecDims(dstDescs.size());
    std::vector<arm_compute::DataLayout> srcDataLayout(srcDescs.size()), dstDataLayout(dstDescs.size());
    std::vector<arm_compute::TensorInfo> srcTensorsInfo(srcDescs.size()), dstTensorsInfo(dstDescs.size());
    srcTensors = std::vector<arm_compute::Tensor>(srcDescs.size());
    dstTensors = std::vector<arm_compute::Tensor>(dstDescs.size());

    for (size_t i = 0; i < srcVecDims.size(); i++) {
        srcVecDims[i] = shapeCast(reshape_sizes(srcDescs[i]->getShape().getDims()));
    }
    for (size_t i = 0; i < dstVecDims.size(); i++) {
        dstVecDims[i] = shapeCast(reshape_sizes(dstDescs[i]->getShape().getDims()));
    }

    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcDataLayout[i] = getAclDataLayoutByMemoryDesc(srcDescs[i]);
        if (srcDataLayout[i] == arm_compute::DataLayout::UNKNOWN) { return false; }
    }
    for (size_t i = 0; i < dstDescs.size(); i++) {
        dstDataLayout[i] = getAclDataLayoutByMemoryDesc(dstDescs[i]);
        if (dstDataLayout[i] == arm_compute::DataLayout::UNKNOWN) { return false; }
    }

    if (srcDescs.size() == 2 &&
        srcDescs[0]->hasLayoutType(LayoutType::nspc) && srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
        srcDescs[0]->getShape().getDims() != srcDescs[1]->getShape().getDims()) {
        if (srcVecDims[0].num_dimensions() < 5) {
            srcDataLayout[0] = srcDataLayout[1] = dstDataLayout[0] = DataLayout::NCHW;
        } else {
            srcDataLayout[0] = srcDataLayout[1] = dstDataLayout[0] = DataLayout::NCDHW;
        }
        changeLayoutToNH_C({&(srcVecDims[0]), &(srcVecDims[1]), &(dstVecDims[0])});
    }

    for (size_t i = 0; i < srcVecDims.size(); i++) {
        srcTensorsInfo[i] = TensorInfo(srcVecDims[i], 1,
                                       precisionToAclDataType(srcDescs[i]->getPrecision()),
                                       srcDataLayout[i]);
        srcTensors[i].allocator()->init(srcTensorsInfo[i]);
    }

    for (size_t i = 0; i < dstVecDims.size(); i++) {
        dstTensorsInfo[i] = TensorInfo(dstVecDims[i], 1,
                                       precisionToAclDataType(dstDescs[i]->getPrecision()),
                                       dstDataLayout[i]);
        dstTensors[i].allocator()->init(dstTensorsInfo[i]);
    }

    std::function<std::unique_ptr<IFunction>(void)> exec_func;
    switch (aclEltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd:
            if (!NEArithmeticAddition::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ConvertPolicy::SATURATE))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEArithmeticAddition>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ConvertPolicy::SATURATE);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseMultiply:
            if (!NEPixelWiseMultiplication::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0],
                                                     1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEPixelWiseMultiplication>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], 1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseSubtract:
            if (!NEArithmeticSubtraction::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ConvertPolicy::SATURATE))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEArithmeticSubtraction>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ConvertPolicy::SATURATE);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseDivide:
            if (!NEElementwiseDivision::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseDivision>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0]);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseMaximum:
            if (!NEElementwiseMax::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseMax>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0]);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseMinimum:
            if (!NEElementwiseMin::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseMin>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0]);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseSquaredDifference:
            if (!NEElementwiseSquaredDiff::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseSquaredDiff>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0]);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseEqual:
            if (!NEElementwiseComparison::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ComparisonOperation::Equal))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ComparisonOperation::Equal);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseNotEqual:
            if (!NEElementwiseComparison::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ComparisonOperation::NotEqual))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ComparisonOperation::NotEqual);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseGreater:
            if (!NEElementwiseComparison::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ComparisonOperation::Greater))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ComparisonOperation::Greater);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseGreaterEqual:
            if (!NEElementwiseComparison::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ComparisonOperation::GreaterEqual))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ComparisonOperation::GreaterEqual);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseLess:
            if (!NEElementwiseComparison::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ComparisonOperation::Less))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ComparisonOperation::Less);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseLessEqual:
            if (!NEElementwiseComparison::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0], ComparisonOperation::LessEqual))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0], ComparisonOperation::LessEqual);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseAbs:
            if (!NEAbsLayer::validate(&srcTensorsInfo[0], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEAbsLayer>();
                acl_op->configure(&srcTensors[0], &dstTensors[0]);
                return acl_op;
            };
            break;
        case Algorithm::EltwiseExp:
            if (!NEExpLayer::validate(&srcTensorsInfo[0], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEExpLayer>();
                acl_op->configure(&srcTensors[0], &dstTensors[0]);
                return acl_op;
            };
            break;
        case Algorithm::EltwisePrelu:
            if (!NEPReluLayer::validate(&srcTensorsInfo[0], &srcTensorsInfo[1], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEPReluLayer>();
                acl_op->configure(&srcTensors[0], &srcTensors[1], &dstTensors[0]);
                return acl_op;
            };
            break;
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
            if (!NEActivationLayer::validate(&srcTensorsInfo[0], &dstTensorsInfo[0], getActivationLayerInfo(aclEltwiseAttrs.algorithm,
                                                                                                            aclEltwiseAttrs.alpha,
                                                                                                            aclEltwiseAttrs.beta,
                                                                                                            aclEltwiseAttrs.gamma)))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&srcTensors[0], &dstTensors[0], getActivationLayerInfo(aclEltwiseAttrs.algorithm,
                                                                                         aclEltwiseAttrs.alpha,
                                                                                         aclEltwiseAttrs.beta,
                                                                                         aclEltwiseAttrs.gamma));
                return acl_op;
            };
            break;
        case Algorithm::EltwiseLog:
            if (!NELogLayer::validate(&srcTensorsInfo[0], &dstTensorsInfo[0]))
                return false;
            exec_func = [this]() -> std::unique_ptr<IFunction> {
                auto acl_op = std::make_unique<NELogLayer>();
                acl_op->configure(&srcTensors[0], &dstTensors[0]);
                return acl_op;
            };
            break;
        default:
            OPENVINO_THROW("Unsupported operation type for ACL Eltwise executor: ",
                           static_cast<int>(aclEltwiseAttrs.algorithm));
    }

    configureThreadSafe([&] { ifunc = exec_func(); });
    return true;
}

void AclEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    for (size_t i = 0; i < src.size(); i++) {
        srcTensors[i].allocator()->import_memory(src[i]->getData());
    }
    for (size_t i = 0; i < dst.size(); i++) {
        dstTensors[i].allocator()->import_memory(dst[i]->getData());
    }

    ifunc->run();

    for (size_t i = 0; i < src.size(); i++) {
        srcTensors[i].allocator()->free();
    }
    for (size_t i = 0; i < dst.size(); i++) {
        dstTensors[i].allocator()->free();
    }
}
}   // namespace intel_cpu
}   // namespace ov
