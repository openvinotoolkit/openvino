// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_stateful_executor.hpp"

#include <algorithm>
#include <any>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/blocked_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/ref/eltwise.hpp"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/debug_capabilities.h"

#ifndef OPENVINO_ARCH_ARM64
#    include "nodes/executors/implementation_utils.hpp"
#endif

namespace ov::intel_cpu {

EltwiseStatefulExecutor::EltwiseStatefulExecutor(EltwiseAttrs attrs,
                                                 const MemoryArgs& memory,
                                                 ExecutorContext::CPtr context)
    : m_attrs(std::move(attrs)),
      m_context(std::move(context)) {
    std::vector<ov::element::Type> input_precisions(memory.size() - 1);                    // -1 for output precision
    std::vector<ov::element::Type> output_precisions{memory.at(ARG_DST)->getPrecision()};  // -1 for output precision

    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;  // Skip output precision
        }

        input_precisions[argId - ARG_SRC] = mem->getPrecision();
    }
    bool canUseOptimizedImpl = EltwiseJitExecutor::supports(m_attrs,
                                                            memory.at(ARG_SRC)->getShape().getRank(),
                                                            input_precisions,
                                                            output_precisions);
    bool canUseOptimizedShapeAgnosticImpl = canUseOptimizedImpl && memory.at(ARG_DST)->getShape().isDynamic();
#ifndef OPENVINO_ARCH_ARM64
    canUseOptimizedShapeAgnosticImpl &= !hasPostOp<FakeQuantizePostOp>(m_attrs.postOps);
#endif
    if (canUseOptimizedShapeAgnosticImpl) {
        eltwiseImplType = EltwiseImplType::optimizedShapeAgnostic;
    } else if (canUseOptimizedImpl) {
        eltwiseImplType = EltwiseImplType::optimized;
    } else {
        eltwiseImplType = EltwiseImplType::reference;
    }

    size_t inputNum = memory.size() - 1;  // -1 for output

    for (size_t i = 0; i < inputNum; i++) {
        const auto desc = memory.at(i + ARG_SRC)->getDescWithType<BlockedMemoryDesc>();
        m_srcOffsets.push_back(desc->getOffsetPadding() * desc->getPrecision().size());
    }

    const auto desc = memory.at(ARG_DST)->getDescWithType<BlockedMemoryDesc>();
    m_dstOffset = desc->getOffsetPadding() * desc->getPrecision().size();

    for (size_t i = 0; i < inputNum; ++i) {
        m_inpPrc.push_back(memory.at(i + ARG_SRC)->getDesc().getPrecision());
    }
    m_outPrc = memory.at(ARG_DST)->getDesc().getPrecision();
    // insert ourselves first
    m_shapeAgnosticData.eltwise_data.push_back(m_attrs.data);
    m_shapeAgnosticData.ops_list.push_back(Type::Eltwise);

    // fill shape agnostic data from postOps (only non-eltwise operations)
    for (const auto& po : m_attrs.postOps) {
        ov::intel_cpu::Type type = po.type() == typeid(FakeQuantizePostOp) ? Type::FakeQuantize : Type::Eltwise;
        m_shapeAgnosticData.ops_list.push_back(type);

        switch (type) {
        case Type::Eltwise: {
            if (const auto* eltwise = std::any_cast<ActivationPostOp>(&po)) {
                m_shapeAgnosticData.eltwise_data.push_back({convertToEltwiseAlgorithm(eltwise->type()),
                                                            convertToDnnlAlgorithm(eltwise->type()),
                                                            eltwise->alpha(),
                                                            eltwise->beta(),
                                                            eltwise->gamma()});
            } else if (const auto* eltwise = std::any_cast<ScaleShiftPostOp>(&po)) {
                m_shapeAgnosticData.eltwise_data.push_back(
                    {convertToEltwiseAlgorithm(eltwise->type()), dnnl::algorithm::undef, 0, 0, 0});
            } else {
                OPENVINO_THROW("has unexpected fused op");
            }
            break;
        }
        case Type::FakeQuantize: {
            size_t channelAxis = 1;
            PostOps fqPostOps = {po};
            DnnlPostOpsComposer legacyPostOpsOriginalZeroPoints(
                fqPostOps,
                m_context->getEngine(),
                memory.at(ARG_DST)->getShape().getMaxDims(),
                channelAxis,
                false,
                0,
                memory,
                DnnlExtensionUtils::ElementTypeToDataType(memory.at(ARG_DST)->getPrecision()),
                {},
                PostOpsMode::ForcedLegacy,
                true,
                m_shapeAgnosticData.postOps);
            DnnlPrimitiveAttrs postOps = legacyPostOpsOriginalZeroPoints.compose();
            m_fqMemory.push_back(postOps.cpuArgs.begin()->second);
            break;
        }
        default:
            OPENVINO_THROW("has unexpected fused op");
        }
    }

    for (const auto& mem : m_fqMemory) {
        m_fqDataPtrs.push_back(mem->getData());
    }
}

std::vector<VectorDims> EltwiseStatefulExecutor::updateInputBlockDims(const MemoryArgs& memory) {
    std::vector<VectorDims> dims_in;
    auto outBlockingDesc = memory.at(ARG_DST)->getDescWithType<BlockedMemoryDesc>();
    const auto& outOrder = outBlockingDesc->getOrder();
    size_t inputNum = memory.size() - 1;  // -1 for output

    const auto& currentOutBlkDims = outBlockingDesc->getBlockDims();
    size_t input_size = std::max(static_cast<size_t>(6), currentOutBlkDims.size());
    // init dims
    dims_in.resize(inputNum);
    for (size_t i = 0; i < inputNum; i++) {
        dims_in[i].resize(input_size, 1);
    }

    if (m_currentInBlkDims.empty()) {
        m_currentInBlkDims.resize(inputNum);
    }

    size_t outRank = currentOutBlkDims.size();

    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;
        }

        const int i = argId - ARG_SRC;
        auto inBlockingDesc = memory.at(argId)->getDescWithType<BlockedMemoryDesc>();
        m_currentInBlkDims[i] = inBlockingDesc->getBlockDims();
        size_t inRank = m_currentInBlkDims[i].size();

        // WA to normalize blocked and planar layouts
        const auto& inOrder = inBlockingDesc->getOrder();
        size_t startOff = outOrder.size() != outBlockingDesc->getShape().getRank() &&
                                  outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1]
                              ? 1
                              : 0;

        // WA to handle nspc layout with 1D tensors
        if (1 == inRank) {
            if (outRank > 2 && 1 == outOrder.back()) {
                startOff = 1;
            }
        }

        for (size_t j = 0; j < inRank; j++) {
            dims_in[i][dims_in[i].size() - 1 - j - startOff] = m_currentInBlkDims[i][inRank - 1 - j];
        }
    }

    return dims_in;
}

// we can skip searching in the cache if broadcast policy for last input dims is not changed
// last input dim == 1 means broadcasted (also if output dim == 1)
// last input dim != 1 means not broadcasted
bool EltwiseStatefulExecutor::canReuseCurrentExecutor(const std::vector<VectorDims>& dims_in) {
    if (eltwiseImplType != EltwiseImplType::optimizedShapeAgnostic) {
        return false;
    }

    const size_t inputNum = dims_in.size();
    bool canSkipSearchInCache = false;
    if (m_executor) {
        canSkipSearchInCache = true;
        // check broadcast policy
        for (size_t i = 0; i < inputNum; i++) {
            if (m_broadcastPolicy[i] != (dims_in[i].back() == 1)) {
                m_broadcastPolicy[i] = (dims_in[i].back() == 1);
                canSkipSearchInCache = false;
            }
        }
    } else {
        // fill broadcast policy
        m_broadcastPolicy.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            m_broadcastPolicy[i] = (dims_in[i].back() == 1);
        }
    }

    return canSkipSearchInCache;
}

void EltwiseStatefulExecutor::updateExecutionParams(const std::vector<VectorDims>& inDims,
                                                    const VectorDims& currentOutBlkDims) {
    auto& outDims = m_execParams.outDims;
    auto& inOffsets = m_execParams.inOffsets;
    auto& outOffsets = m_execParams.outOffsets;
    const size_t outRank = currentOutBlkDims.size();
    // outDims recalculation
    outDims.resize(inDims[0].size(), 1);
    for (size_t i = 0; i < outRank; i++) {
        outDims[outDims.size() - 1 - i] = currentOutBlkDims[outRank - 1 - i];
    }
    // offsets recalculation
    auto offset_out_calc = [](VectorDims& offset, const VectorDims& dims) {
        int k = 1;
        for (int i = offset.size() - 1; i >= 0; i--) {
            offset[i] = k;
            k *= dims[i];
        }
    };

    auto offset_in_calc = [](VectorDims& offset, const VectorDims& inDims, const VectorDims& dims_out) {
        int k = 1;
        for (int i = offset.size() - 1; i >= 0; i--) {
            offset[i] = (inDims[i] == dims_out[i]) ? k : 0;
            k *= inDims[i];
        }
    };

    auto inputSize = inDims.front().size();
    outOffsets.resize(inputSize, 1);
    offset_out_calc(outOffsets, outDims);
    for (size_t j = 0; j < inputSize; j++) {
        outOffsets[j] *= m_outPrc.size();
    }

    auto inputsNumber = inDims.size();
    inOffsets.resize(inputsNumber);
    for (size_t i = 0; i < inputsNumber; i++) {
        inOffsets[i].resize(inputSize, 1);
        offset_in_calc(inOffsets[i], inDims[i], outDims);
        for (size_t j = 0; j < inputSize; j++) {
            inOffsets[i][j] *= m_inpPrc[i].size();
        }
    }
}

bool EltwiseStatefulExecutor::update(const MemoryArgs& memory) {
    const std::vector<VectorDims> inDims = updateInputBlockDims(memory);

    const auto& currentOutBlkDims = memory.at(ARG_DST)->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    if (eltwiseImplType == EltwiseImplType::optimizedShapeAgnostic) {
        updateExecutionParams(inDims, currentOutBlkDims);

        const bool reuse = canReuseCurrentExecutor(inDims);
        if (m_executor && reuse) {  // use current executor, just update execution params
            return true;
        }
    }

    m_executor = eltwiseImplType == EltwiseImplType::reference
                     ? createEltwiseRefExecutor(inDims, currentOutBlkDims, m_outPrc, m_context, m_shapeAgnosticData)
                     : EltwiseJitExecutor::create(memory,
                                                  inDims,
                                                  currentOutBlkDims,
                                                  m_outPrc,
                                                  m_context,
                                                  m_shapeAgnosticData,
                                                  eltwiseImplType);
    OPENVINO_DEBUG_ASSERT(m_executor, "Failed to create Eltwise executor");

    return true;
}

void EltwiseStatefulExecutor::execute(const MemoryArgs& memory) {
    // Convert MemoryArgs to jit_eltwise_call_args_ptrs
    jit_eltwise_call_args_ptrs args_ptrs = {};

    const VectorDims& outDims =
        eltwiseImplType == EltwiseImplType::optimizedShapeAgnostic ? m_execParams.outDims : m_executor->getOutDims();
    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            args_ptrs.dst_ptr = mem->getDataAs<uint8_t>() + m_dstOffset;
        } else {
            const int i = argId - ARG_SRC;
            args_ptrs.src_ptr[i] = mem->getDataAs<const uint8_t>() + m_srcOffsets[i];
        }
    }

    args_ptrs.post_op_data = m_fqDataPtrs.data();

    if (eltwiseImplType == EltwiseImplType::optimizedShapeAgnostic) {
        args_ptrs.work_amount = outDims.back();
        for (size_t i = 0; i < m_execParams.inOffsets.size(); i++) {
            args_ptrs.src_offsets[i] = m_execParams.inOffsets[i].data();
        }
        args_ptrs.dst_offsets = m_execParams.outOffsets.data();
    }

    m_executor->exec(args_ptrs, outDims);
}

impl_desc_type EltwiseStatefulExecutor::implType() const {
    if (eltwiseImplType == EltwiseImplType::reference) {
        return impl_desc_type::ref;
    }

    return EltwiseJitExecutor::implType();
}

}  // namespace ov::intel_cpu
