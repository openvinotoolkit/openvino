// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_mm.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "kleidiai_common.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/precision_support.h"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu {

using namespace executor;
using namespace ov::element;

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, static_cast<T>(1), std::multiplies<T>()),
            dims[dims.size() - 1]};
}

static bool useDynamicQuantizationImpl(const FCAttrs& attrs, const MemoryDescPtr& weightDesc) {
    if (attrs.dynamicQuantizationGroupSize != std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    if (!hasIntDotProductSupport() && !hasInt8MMSupport()) {
        return false;
    }

    return weightDesc->getPrecision() == element::i8 || weightDesc->getPrecision() == element::i4;
}

bool MatMulKleidiAIExecutor::supports(const FCConfig& config) {
    bool returnValue = config.descs.at(ARG_WEI)->getPrecision() == element::f32 ||
                       useDynamicQuantizationImpl(config.attrs, config.descs.at(ARG_WEI));
    return returnValue;
}

bool MatMulKleidiAIExecutor::isGroupQuantizationEnabled(const MemoryArgs& memory) {
    auto scales = memory.at(ARG_WEI | ARG_ATTR_SCALES)->getDesc().getShape().getStaticDims();
    return (scales[1] > 1);
}

MatMulKleidiAIExecutor::MatMulKleidiAIExecutor(const FCAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : executorContext(context) {
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto N = weiDims[0];
    auto K = weiDims[1];

    const bool hasBias = !memory.at(ARG_BIAS)->getDesc().empty();
    if (hasBias) {
        biasMem = memory.at(ARG_BIAS);
    } else {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(f32, Shape({N}));
        biasMem = std::make_shared<Memory>(context->getEngine(), biasDesc);
        biasMem->nullify();
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }

    auto originalWeightsDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& wgtDims = originalWeightsDesc->getShape().getStaticDims();
    const VectorDims wgtDims2D = reshapeDownToRank<2>(wgtDims);
    originalWeightsDesc = std::make_shared<CpuBlockedMemoryDesc>(originalWeightsDesc->getPrecision(), Shape{wgtDims2D});
    auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);

    bool isTransposed = false;
    float* rhs_scales = nullptr;

    // Whether dynamic quantization is enabled
    bool useDynamicQuant = useDynamicQuantizationImpl(attrs, originalWeightsDesc);

    if (!useDynamicQuant) {
        _kernel = std::make_shared<kai_common::uKernel<kai_common::KAIKernelTag::F32_NEON_MLA>>(N, K);

        auto dstDesc = originalWeightsDesc->cloneWithNewPrecision(memory.at(ARG_SRC)->getDescPtr()->getPrecision());
        auto dnnlDstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc);

        if (!attrs.weightsNonTransposed) {
            dnnlDstDesc = acl_fc_executor::makeTransposedWeightDescriptor(dnnlDstDesc, dnnlSrcDesc);
            aclfcAttrs.isWeightsRepacked = true;
        }
        MemoryCPtr packedWeights =
            acl_fc_executor::reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);
        const size_t rhsPackedSize = _kernel->get_rhsPackedSize();
        auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(u8, Shape({rhsPackedSize}));
        rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);

        _kernel->packData(false, packedWeights, biasMem, hasBias, nullptr, rhsPackedMem);
    } else {
        MemoryPtr weightsMemory = memory.at(ARG_WEI);
        // Check if weights are in int4 or int8
        if (weightsMemory->getDescPtr()->getPrecision() == element::i4) {
            isTransposed = attrs.weightsNonTransposed;
            if (isGroupQuantizationEnabled(memory)) {
                _kernel =
                    std::make_shared<kai_common::uKernel<kai_common::KAIKernelTag::I4_NEON_IMM_GROUP>>(N,
                                                                                                       K,
                                                                                                       lhsPackedMem,
                                                                                                       memory);
            } else {
                if (hasInt8MMSupport()) {
                    _kernel =
                        std::make_shared<kai_common::uKernel<kai_common::KAIKernelTag::I4_NEON_IMM>>(N,
                                                                                                     K,
                                                                                                     lhsPackedMem);
                } else {
                    _kernel =
                        std::make_shared<kai_common::uKernel<kai_common::KAIKernelTag::I4_NEON_DOTPROD>>(N,
                                                                                                         K,
                                                                                                         lhsPackedMem);
                }
            }
            const size_t rhsPackedSize = _kernel->get_rhsPackedSize();
            auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({rhsPackedSize}));
            rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);
        } else {
            if (hasInt8MMSupport()) {
                _kernel =
                    std::make_shared<kai_common::uKernel<kai_common::KAIKernelTag::I8_NEON_IMM>>(N, K, lhsPackedMem);
            } else {
                _kernel =
                    std::make_shared<kai_common::uKernel<kai_common::KAIKernelTag::I8_NEON_DOTPROD>>(N,
                                                                                                     K,
                                                                                                     lhsPackedMem);
            }

            if (!attrs.weightsNonTransposed) {
                auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);
                auto dnnlDstDesc = acl_fc_executor::makeTransposedWeightDescriptor(dnnlSrcDesc, dnnlSrcDesc);
                weightsMemory = acl_fc_executor::reorderData(dnnlSrcDesc, dnnlDstDesc, memory.at(ARG_WEI), context);
            }
            const size_t rhsPackedSize = _kernel->get_rhsPackedSize();
            auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({rhsPackedSize}));
            rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);
        }

        rhs_scales = static_cast<float*>(memory.at(ARG_WEI | ARG_ATTR_SCALES)->getData());
        _kernel->packData(isTransposed, weightsMemory, biasMem, hasBias, rhs_scales, rhsPackedMem);

        // Create scratchpad to initialize memory for LHS in update()
        scratchPad = context->getScratchPad();
    }
}

bool MatMulKleidiAIExecutor::update(const MemoryArgs& memory) {
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    const auto& wgtDims = weiDesc->getShape().getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    K = wgtDims[1];
    N = wgtDims[0];

    const auto& outDims = dstDesc->getShape().getStaticDims();
    if (outDims.size() > 2) {
        M = std::accumulate(outDims.begin(), outDims.end() - 1, 1, std::multiplies<>());
    } else {
        M = outDims[0];
    }
    // Assign LHS memory using newer logic
    size_t lhsPackedSize = _kernel->getLHSPackedSize(M);
    if (lhsPackedSize != 0) {
        auto lhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({lhsPackedSize}));
        lhsPackedMem = scratchPad->createScratchPadMem(lhsPackedDesc);
    }
    return true;
}

void MatMulKleidiAIExecutor::execute(const MemoryArgs& memory) {
    auto srcMem = memory.at(ARG_SRC);
    auto dstMem = memory.at(ARG_DST);
    auto srcDims = normalizeDimsTo2D(srcMem->getDesc().getShape().getDims());
    _kernel->execute(executorContext->getCpuParallel(), srcDims[0], dstMem, srcMem);
}

void MatMulKleidiAIExecutor::moveMemToNumaNode([[maybe_unused]] int numaNodeID) {
    OPENVINO_THROW_NOT_IMPLEMENTED("'moveMemToNumaNode' is not implemented by the executor");
}

}  // namespace ov::intel_cpu
