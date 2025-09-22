// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.h"

#include <algorithm>
#include <cassert>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cpu_shape.h"
#include "cpu_types.h"
#include "eltwise.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/matmul.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/matmul.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool MatMul::canBeExecutedInInt8() const {
    auto firstInputPrecision = getOriginalInputPrecisionAtPort(0);
    auto secondInputPrecision = getOriginalInputPrecisionAtPort(1);

    return any_of(firstInputPrecision, ov::element::u8, ov::element::i8) && secondInputPrecision == ov::element::i8;
}

bool MatMul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = ov::as_type_ptr<const ov::op::v0::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only v0 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage =
                    "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MatMul::MatMul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, MMShapeInferFactory(op)) {
    std::string errorMessage;

    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto matMul = ov::as_type_ptr<const ov::op::v0::MatMul>(op);

    if (!matMul) {
        OPENVINO_THROW_NOT_IMPLEMENTED("Operation with name ",
                                       op->get_friendly_name(),
                                       ":",
                                       op->get_type_name(),
                                       " is not an instance of MatMul from v0");
    }

    m_attrs.transposeA = matMul->get_transpose_a();
    m_attrs.transposeB = matMul->get_transpose_b();

    m_atoi[ARG_SRC] = 0;
    m_atoi[ARG_WEI] = 1;
}

bool MatMul::canFuse(const NodePtr& node) const {
    // WA for CVS-84056: oneDNN brgemm impl has problem with per-OC binary-postOps for MatMul with 6D inputs
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->getBroadcastingPolicy() == EltwiseBroadcastingPolicy::PerChannel) {
                auto rank = getInputShapeAtPort(0).getRank();
                if (rank > 4) {
                    DEBUG_LOG("skip fusing non-perTensor Eltwise:",
                              eltwiseNode->getName(),
                              " into 6D MatMul:",
                              getName());
                    return false;
                }
            }
        }
    }

    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32. If fusing FQ into matmul, there would be reorder
    //  inserted after matmul. In some bert model, this reorder causes great perf degradation. Todo: Remove this if
    //  onednn primitive support U8 output with floating input.
    if (node->getType() == Type::FakeQuantize &&
        any_of(node->getOriginalOutputPrecisionAtPort(0), ov::element::i8, ov::element::u8) && !canBeExecutedInInt8() &&
        getOriginalInputPrecisionAtPort(0) == ov::element::f32) {
        return false;
    }
    return canFuseSimpleOperation(node);
}

std::tuple<VecMemoryDescs, MemoryDescPtr> MatMul::initMemoryDescriptors(ov::element::Type dstType) const {
    const auto& srcTypes = getOriginalInputPrecisions();

    VecMemoryDescs srcDescs;
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        if (srcTypes[i] == ov::element::dynamic) {
            srcDescs.push_back(MemoryDescUtils::makeEmptyDesc());
            continue;
        }
        auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        srcDescs.push_back(srcDesc);
    }

    auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstType, getOutputShapeAtPort(0));

    return {srcDescs, dstDesc};
}

ExecutorFactoryPtr<MatMulAttrs> MatMul::createExecutorFactory(const MemoryDescArgs& descs, const MatMulAttrs& attrs) {
    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    return std::make_shared<ExecutorFactory<MatMulAttrs>>(attrs, executionContext, descs, memoryFormatFilter);
}

void MatMul::initSupportedPrimitiveDescriptors() {
    m_attrs.withBias = getOriginalInputsNumber() == 3;
    if (m_attrs.withBias) {
        m_atoi[ARG_BIAS] = 2;
    }

    m_attrs.dqScales = getDQScales();
    m_attrs.postOps = getPostOps(fusedWith, ov::element::dynamic);

    auto dstType = getOriginalOutputPrecisionAtPort(0);

    // make sure dst type is equal to the output type of the last fused node
    if (!fusedWith.empty()) {
        dstType = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    auto [srcDescs, dstDesc] = initMemoryDescriptors(dstType);

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[0]},
        {ARG_WEI, srcDescs[1]},
        {ARG_BIAS, m_attrs.withBias ? srcDescs[2] : MemoryDescUtils::makeEmptyDesc()},
        {ARG_DST, dstDesc},
    };

    m_factory = createExecutorFactory(descs, m_attrs);

    const std::vector<MemoryDescArgs> nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);

    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcDescs.size());

        auto getBlockedMask = [](const std::shared_ptr<MemoryDesc>& memDesc) {
            if (memDesc->getType() & MemoryDescType::Blocked) {
                return BlockedMemoryDesc::EMPTY_MASK;
            }
            return BlockedMemoryDesc::FULL_MASK;
        };

        for (const auto& desc : nodeDescriptors) {
            if (auto it = m_atoi.find(desc.first); it != m_atoi.end()) {
                const auto& inputDesc = desc.second;
                nodeConfig.inConfs[it->second] = PortConfig{inputDesc, getBlockedMask(inputDesc)};
            }
        }

        for (size_t i = 3; i < srcDescs.size(); i++) {
            nodeConfig.inConfs[i] = PortConfig{srcDescs[i]};
        }

        const auto& outputDesc = nodeDescriptors.at(ARG_DST);
        nodeConfig.outConfs.emplace_back(outputDesc, getBlockedMask(outputDesc), -1);

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

ov::element::Type MatMul::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(parentEdge->getMemory().getDesc().getPrecision());
        }
    }

    return getMaxPrecision(inputPrecisions);
}

const std::vector<impl_desc_type>& MatMul::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,       impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512, impl_desc_type::brgemm_avx2,
        impl_desc_type::gemm_acl,      impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,   impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,      impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_any,      impl_desc_type::gemm,
        impl_desc_type::jit_gemm,      impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,   impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw, impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,    impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,  impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,    impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,       impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1, impl_desc_type::jit_sse42,
        impl_desc_type::ref,
    };

    return priorities;
}

void MatMul::createPrimitive() {
    for (const auto& entry : m_atoi) {
        const auto argumentId = entry.first;
        const auto inputId = entry.second;
        m_memory[argumentId] = getSrcMemoryAtPort(inputId);
    }

    if (!m_attrs.withBias) {
        m_memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
    }

    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor = m_factory->make(m_memory, false);

    Node::createPrimitive();
}

void MatMul::prepareParams() {
    // check for a degenerate case. In this context the degenerate case is a matrix multiplication where the
    // collapsing dimension is zero, e.g., AB=C, where A has the shape [10, 0] and B has the shape [0, 20],
    // consequently C has shape [10, 20]. In this scenario C is a null matrix (a matrix filled with zeroes)
    // according to the empty sum convention.
    if (m_memory[ARG_SRC]->getDesc().getShape().hasZeroDims() &&
        m_memory[ARG_WEI]->getDesc().getShape().hasZeroDims() &&
        !m_memory[ARG_DST]->getDesc().getShape().hasZeroDims()) {
        // todo: obviously we need a special executor that would process fused ops providing a correct result
        CPU_NODE_ASSERT(m_memory[ARG_BIAS]->getDesc().empty(),
                        "Matmul doesn't support a degenerate case when bias is provided");
        CPU_NODE_ASSERT(fusedWith.empty(), "Matmul doesn't support a degenerate case when other ops are fused");
        return;
    }

    const auto& executor = m_executor;
    assert(executor);
    executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());
}

void MatMul::execute(const dnnl::stream& /*strm*/) {
    if (hasEmptyInputTensors()) {  // this is a degenerate case, fill output with zeroes
        getDstMemoryAtPort(0)->nullify();
        return;
    }

    const auto& executor = m_executor;
    assert(executor);

    executor->execute(m_memory);
}

void MatMul::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool MatMul::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroOutputDims();
}

bool MatMul::isExecutable() const {
    return !hasEmptyOutputTensors();
}

}  // namespace ov::intel_cpu::node
