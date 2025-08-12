// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv.h"

#include <oneapi/dnnl/dnnl_common_types.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "allocation_context.hpp"
#include "common/cpu_convert.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "edge.h"
#include "eltwise.h"
#include "graph.h"
#include "graph_context.h"
#include "input.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/node_config.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "post_ops.hpp"
#include "shape_inference/custom/convolution.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

class Convolution::FusedSubgraph {
public:
    FusedSubgraph(const std::vector<NodePtr>& opList, const Convolution& conv, const GraphContext::CPtr& context) {
        _graph = std::make_unique<Graph>();

        std::unordered_set<NodePtr> nodesSet;
        std::vector<EdgePtr> edges;

        auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
            auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
            Node::addEdge(edge);
            edges.push_back(edge);
            nodesSet.insert(parent);
            nodesSet.insert(child);
        };

        // Make inputs
        const auto& inpMemDesc1 = conv.getBaseMemDescAtOutputPort(0);
        auto inp0 = std::make_shared<Input>(inpMemDesc1, "inp0", "Parameter", context);
        inputs.push_back(inp0);
        const size_t sumPortNum = conv.getParentEdges().size() - 1;
        const auto& inpMemDesc2 = conv.getBaseMemDescAtInputPort(sumPortNum);
        auto inp1 = std::make_shared<Input>(inpMemDesc2, "inp1", "Parameter", context);
        inputs.push_back(inp1);

        auto itr = std::find_if(opList.begin(), opList.end(), [](const NodePtr& node) {
            if (auto eltwise = std::dynamic_pointer_cast<Eltwise>(node)) {
                return eltwise->isSpecialConvolutionAddFusing();
            }
            return false;
        });

        if (itr == opList.end()) {
            return;
        }

        auto sumNode = *itr;
        addEdge(inp0, sumNode, 0, 0);
        addEdge(inp1, sumNode, 0, 1);

        // Replicate the rest of the subgraph
        auto parentItr = itr;
        while (++itr != opList.end()) {
            auto parentNode = *parentItr;
            const auto& currentNode = *itr;
            if (Type::FakeQuantize == currentNode->getType()) {
                parentNode->addFusedNode(currentNode);
            } else {
                addEdge(parentNode, currentNode, 0, 0);
                auto constantsItr = conv.fusedConstNodes.find(currentNode);
                if (constantsItr != conv.fusedConstNodes.end()) {
                    size_t inpPort = 1LU;
                    for (const auto& item : constantsItr->second) {
                        addEdge(item, currentNode, 0, inpPort++);
                    }
                }
                parentItr = itr;
            }
        }

        // Make output
        const auto& outMemDesc = conv.getBaseMemDescAtOutputPort(0);
        auto out = std::make_shared<Input>(outMemDesc, "out", "Result", context);
        addEdge(*parentItr, out, 0, 0);
        outputs.push_back(out);

        std::vector<NodePtr> nodes(nodesSet.begin(), nodesSet.end());

        _graph->Init(nodes, edges, context, "fused_subgraph");
    }

    int RegisterToAllocationContext(int offset, AllocationContext& context) {
        return _graph->RegisterToAllocationContext(offset, context);
    }

    void Activate() const {
        _graph->Activate();
    }

    [[nodiscard]] std::shared_ptr<Input> getInput(size_t idx) const {
        OPENVINO_ASSERT(idx < inputs.size(),
                        "OutOfBounds: Unexpected input index in Convolution::fusedSubgraph::getInput idx=",
                        idx,
                        " inputs.size()=",
                        inputs.size());
        return inputs[idx];
    }

    [[nodiscard]] std::shared_ptr<Input> getOutput(size_t idx) const {
        OPENVINO_ASSERT(idx < outputs.size(),
                        "OutOfBounds: Unexpected output index in Convolution::fusedSubgraph::getInput idx=",
                        idx,
                        " outputs.size()=",
                        outputs.size());
        return outputs[idx];
    }

    void infer() {
        _graph->ResetInferCount();
        _graph->Infer();
    }

private:
    std::unique_ptr<Graph> _graph;
    std::vector<std::shared_ptr<Input>> inputs;
    std::vector<std::shared_ptr<Input>> outputs;
};

bool Convolution::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type_any_of<ov::op::v1::Convolution, ov::op::v1::GroupConvolution>(op)) {
            errorMessage = "Only opset1 Convolution and GroupConvolution operations are supported";
            return false;
        }
        size_t ndims = op->get_input_partial_shape(0).rank().get_length();
        if ((ndims < 3) || (ndims > 5)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(ndims);
            return false;
        }
        if (op->get_input_partial_shape(1).is_dynamic()) {
            errorMessage = "Doesn't support dynamic weights shape";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Convolution::Convolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, ConvolutionShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    m_atoi[ARG_SRC] = DATA;
    m_atoi[ARG_WEI] = WEIGHTS;

    auto convolutionOp = ov::as_type_ptr<ov::op::v1::Convolution>(op);
    auto groupConvolutionOp = ov::as_type_ptr<ov::op::v1::GroupConvolution>(op);

    if (convolutionOp) {
        algorithm = Algorithm::ConvolutionCommon;

        groupNum = 1;
        m_attrs.isGrouped = false;

        const auto& weightDims = convolutionOp->input_value(1).get_shape();

        IC = weightDims[1];
        groupIC = IC;
        groupOC = weightDims[0];

        for (size_t i : convolutionOp->get_strides()) {
            m_attrs.stride.push_back(i);
        }
        for (size_t i : convolutionOp->get_dilations()) {
            m_attrs.dilation.push_back(static_cast<ptrdiff_t>(i) - 1);
        }
        m_attrs.paddingL = convolutionOp->get_pads_begin();
        m_attrs.paddingR = convolutionOp->get_pads_end();
        if (convolutionOp->get_auto_pad() == ov::op::PadType::SAME_UPPER) {
            m_attrs.autoPadding = AutoPaddingType::SAME_UPPER;
        } else if (convolutionOp->get_auto_pad() == ov::op::PadType::SAME_LOWER) {
            m_attrs.autoPadding = AutoPaddingType::SAME_LOWER;
        } else {
            m_attrs.autoPadding = AutoPaddingType::None;
        }
    } else if (groupConvolutionOp) {
        algorithm = Algorithm::ConvolutionGrouped;
        m_attrs.isGrouped = true;

        groupNum = groupConvolutionOp->input_value(1).get_shape()[0];

        const auto& weightDims = groupConvolutionOp->input_value(1).get_shape();

        groupIC = weightDims[2];
        IC = groupIC * groupNum;
        groupOC = weightDims[1];

        for (size_t i : groupConvolutionOp->get_strides()) {
            m_attrs.stride.push_back(i);
        }
        for (size_t i : groupConvolutionOp->get_dilations()) {
            m_attrs.dilation.push_back(i - 1);
        }
        m_attrs.paddingL = groupConvolutionOp->get_pads_begin();
        m_attrs.paddingR = groupConvolutionOp->get_pads_end();
        if (groupConvolutionOp->get_auto_pad() == ov::op::PadType::SAME_UPPER) {
            m_attrs.autoPadding = AutoPaddingType::SAME_UPPER;
        } else if (groupConvolutionOp->get_auto_pad() == ov::op::PadType::SAME_LOWER) {
            m_attrs.autoPadding = AutoPaddingType::SAME_LOWER;
        } else {
            m_attrs.autoPadding = AutoPaddingType::None;
        }
    }
    // Only apply this heuristic logic on FP32 IR. IC=1 ,OC=1 would disable brgconv on avx2.
    const bool isAvx2FP32 = !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
                            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !context->isGraphQuantized();
    useJitPlanar = ((all_of(1U, IC, groupOC * groupNum)) && isAvx2FP32);
}

bool Convolution::canBeExecutedInInt8() const {
    auto inputDataType = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(0));
    auto weightsDataType = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(1));

    if (!legacyInputZeroPoints.empty()) {
        inputDataType = memory::data_type::u8;
    }

    if (!legacyWeightsZeroPoints.empty()) {
        weightsDataType = memory::data_type::s8;
    }

    return any_of(inputDataType, memory::data_type::u8, memory::data_type::s8) &&
           weightsDataType == memory::data_type::s8;
}

const std::vector<impl_desc_type>& Convolution::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::dw_acl,
        impl_desc_type::winograd_acl,
        impl_desc_type::gemm_acl,
        impl_desc_type::acl,
        impl_desc_type::brgconv_avx512_dw,
        impl_desc_type::brgconv_avx512_amx_1x1,
        impl_desc_type::brgconv_avx512_amx,
        impl_desc_type::jit_avx512_amx_dw,
        impl_desc_type::jit_avx512_amx_1x1,
        impl_desc_type::jit_avx512_amx,
        impl_desc_type::brgconv_avx512_1x1,
        impl_desc_type::brgconv_avx512,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::brgconv_avx2_dw,
        impl_desc_type::brgconv_avx2_1x1,
        impl_desc_type::brgconv_avx2,
        impl_desc_type::jit_avx2_1x1_dw,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,
        impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::jit_gemm,
        impl_desc_type::ref_any,
        impl_desc_type::ref,
    };
    // WA heuristic to avoid regressions introduced by avx2 brgconv.
    const bool isBrgConvAvailable = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !useJitPlanar;
    if (isBrgConvAvailable) {
        return priorities;
    }

    static const std::vector<impl_desc_type> priorities_wo_brgemm = [&] {
        std::vector<impl_desc_type> result;
        std::copy_if(priorities.begin(), priorities.end(), std::back_inserter(result), [](impl_desc_type type) {
            return (type & impl_desc_type::brgconv) == 0;
        });
        return result;
    }();
    return priorities_wo_brgemm;
}

void Convolution::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
    /* preemptively create a fallback subgraph to include it into global memory reuse
     * pros:
     * - less total memory usage when fallback is actually needed (by size of intermediate memory)
     * - no runtime overhead of graph creation when fallback is needed for the first time
     * cons:
     * - more total memory usage when fallback is not needed (by size of a graph data structure itself)
     */
    if (withSum && isDynamicNode()) {
        subgraph = std::make_shared<FusedSubgraph>(fusedWith, *this, context);
    }
}

static MemoryDescPtr getSumMemDesc(const MemoryDescPtr& outputDesc,
                                   const Shape& sumShape,
                                   ov::element::Type sumPrecision) {
    if (outputDesc->getShape().isStatic()) {
        return outputDesc->cloneWithNewPrecision(sumPrecision);
    }

    // When we set the input shape with ranged dimensions, the sum node's input shape may mismatch with the output
    // shape, we just change ranged min value to 1 to meet this case. For example: Output shape = {1, 160, {128, 256},
    // {128, 256}} Sum input shape = {1, 160, 1, 1} Update sum shape to {1, 160, {1, 256}, {1, 256}}
    const auto& shape = outputDesc->getShape();
    if (shape.getRank() != sumShape.getRank()) {
        return outputDesc->cloneWithNewPrecision(sumPrecision);
    }

    const auto& sumDims = sumShape.getDims();
    const auto& maxDims = shape.getMaxDims();
    auto minDims = shape.getMinDims();

    for (size_t i = 0; i < maxDims.size(); i++) {
        if ((maxDims[i] > minDims[i]) && sumDims[i] == 1) {
            minDims[i] = 1;
        }
    }

    auto* blockedOutputDesc = outputDesc->as<BlockedMemoryDesc>();

    return std::make_shared<CpuBlockedMemoryDesc>(sumPrecision,
                                                  Shape(minDims, maxDims),
                                                  blockedOutputDesc->getBlockDims(),
                                                  blockedOutputDesc->getOrder(),
                                                  blockedOutputDesc->getOffsetPadding(),
                                                  blockedOutputDesc->getOffsetPaddingToData(),
                                                  blockedOutputDesc->getStrides());
}

std::tuple<VecMemoryDescs, MemoryDescPtr> Convolution::initMemoryDescriptors(ov::element::Type dstType) const {
    const auto& srcTypes = getOriginalInputPrecisions();

    VecMemoryDescs srcDescs;
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        if (srcTypes[i] == element::dynamic) {
            srcDescs.push_back(MemoryDescUtils::makeEmptyDesc());
            continue;
        }
        auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        srcDescs.push_back(srcDesc);
    }

    auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstType, getOutputShapeAtPort(0));

    return {srcDescs, dstDesc};
}

ExecutorFactoryPtr<ConvAttrs> Convolution::createExecutorFactory(const MemoryDescArgs& descs, const ConvAttrs& attrs) {
    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    return std::make_shared<ExecutorFactory<ConvAttrs>>(attrs, executionContext, descs, memoryFormatFilter);
}

std::tuple<ov::element::Type, ov::element::Type> Convolution::getDstAndSumPrecision() {
    auto getSumDataType = [](const std::shared_ptr<node::Eltwise>& eltwise) {
        int fusingPort = eltwise->getFusingPort();
        switch (fusingPort) {
        case 0:
            return eltwise->getOriginalInputPrecisionAtPort(1);
        case 1:
            return eltwise->getOriginalInputPrecisionAtPort(0);
        default:
            OPENVINO_THROW("Unexpected sum fusing port: ", fusingPort);
        }
    };

    auto dstType = getOriginalOutputPrecisionAtPort(0);

    // make sure dst type is equal to the output type of the last fused node
    if (!fusedWith.empty()) {
        dstType = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    auto ndims = getInputShapeAtPort(0).getRank();

    // Make sure that convolution output and the Sum input have equal precision sizes
    // since they use the same physical memory. Upscale in case precisions are different
    for (auto& node : fusedWith) {
        if (node->getAlgorithm() != Algorithm::EltwiseAdd) {
            continue;
        }

        if (auto eltwiseNode = std::dynamic_pointer_cast<Eltwise>(node)) {
            if (!eltwiseNode->isSpecialConvolutionAddFusing()) {
                continue;
            }

            ov::element::Type eltwisePrecision = getSumDataType(eltwiseNode);
            if (canBeExecutedInInt8() && dstType.size() != eltwisePrecision.size()) {
                return {ov::element::f32, ov::element::f32};
            }

            if (isDepthWise() && ndims == 5) {
                return {ov::element::f32, ov::element::f32};
            }

            if (any_of(dstType, ov::element::f32, ov::element::bf16, ov::element::f16)) {
                return {dstType, dstType};
            }

            return {dstType, eltwisePrecision};
        }
    }

    return {dstType, ov::element::dynamic};
}

void Convolution::initSupportedPrimitiveDescriptors() {
    m_attrs.withBias = getOriginalInputsNumber() == 3;
    if (m_attrs.withBias) {
        m_atoi[ARG_BIAS] = BIAS;
    }

    m_attrs.isGraphQuantized = context->isGraphQuantized();
    m_attrs.fcSemantic = false;
    m_attrs.nonConstantWeights = !getParentEdgeAt(WEIGHTS)->getParent()->isConstant();
    m_attrs.weightsNonTransposed = false;
    m_attrs.dqScales = getDQScales();

    const auto [dstType, sumType] = getDstAndSumPrecision();

    m_attrs.postOps = getPostOps(fusedWith, sumType);

    auto [srcDescs, dstDesc] = initMemoryDescriptors(dstType);

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[DATA]},
        {ARG_WEI, srcDescs[WEIGHTS]},
        {ARG_BIAS, m_attrs.withBias ? srcDescs[BIAS] : MemoryDescUtils::makeEmptyDesc()},
        {ARG_DST, dstDesc},
    };

    m_factory = createExecutorFactory(descs, m_attrs);

    const std::vector<MemoryDescArgs> nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);

    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcDescs.size());

        auto getBlockedMask = [](const std::shared_ptr<MemoryDesc>& memDesc, const bool isGrouped) {
            if (memDesc->getType() & MemoryDescType::Blocked && !isGrouped) {
                return BlockedMemoryDesc::EMPTY_MASK;
            }
            return BlockedMemoryDesc::FULL_MASK;
        };

        for (const auto& desc : nodeDescriptors) {
            if (auto it = m_atoi.find(desc.first); it != m_atoi.end()) {
                const auto& inputDesc = desc.second;
                nodeConfig.inConfs[it->second] = PortConfig(inputDesc, getBlockedMask(inputDesc, m_attrs.isGrouped));
            }
        }

        for (size_t i = 3; i < srcDescs.size(); i++) {
            nodeConfig.inConfs[i] = PortConfig(srcDescs[i]);
        }

        const int inPlaceOutPort = withSum ? static_cast<int>(getParentEdges().size()) - 1 : -1;
        const auto& outputDesc = nodeDescriptors.at(ARG_DST);
        nodeConfig.outConfs.emplace_back(outputDesc, getBlockedMask(outputDesc, m_attrs.isGrouped), inPlaceOutPort);

        if (withDWConv) {
            constexpr size_t X_AXIS = 0;
            constexpr size_t Y_AXIS = 1;
            const std::vector<size_t> dwWeightsDims{dw_conv_oc, 1, 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]};
            const std::vector<size_t> dwBiasesDims{dw_conv_oc};

            const auto dwWeightsPrc = DnnlExtensionUtils::ElementTypeToDataType(
                dw_conv_in_dt == dnnl_u8 ? ov::element::i8 : ov::element::f32);
            const auto dwWeightsDesc = std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwWeightsDims),
                                                                               dwWeightsPrc,
                                                                               memory::format_tag::Goihw8g);
            nodeConfig.inConfs.emplace_back(dwWeightsDesc);

            const auto dwBiasPrc = memory::data_type::f32;
            const auto dwBiasDesc =
                std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwBiasesDims), dwBiasPrc, memory::format_tag::x);
            nodeConfig.inConfs.emplace_back(dwBiasDesc);
        }

        if (withSum) {
            auto sumDesc =
                getSumMemDesc(nodeDescriptors.at(ARG_DST), getInputShapeAtPort(getParentEdges().size() - 1), sumType);
            nodeConfig.inConfs.emplace_back(sumDesc, BlockedMemoryDesc::FULL_MASK, -1);
        }

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

bool Convolution::created() const {
    return getType() == Type::Convolution;
}

template <typename T>
static MemoryPtr memoryViewToVector(const std::vector<T>& vec, const dnnl::engine& engine) {
    const auto type = ov::element::from<T>();
    DnnlBlockedMemoryDesc memoryDesc(type, {vec.size()});
    return std::make_shared<Memory>(engine, memoryDesc, vec.data());
}

bool Convolution::canFuse(const NodePtr& node) const {
#if defined(OV_CPU_WITH_ACL)
    if (!fusedWith.empty()) {
        return false;
    }
#endif
    return canFuseSimpleOperation(node);
}

ov::element::Type Convolution::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(
                DnnlExtensionUtils::DataTypeToElementType((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

int Convolution::registerToAllocationContext(int offset, AllocationContext& context) {
    if (subgraph) {
        return subgraph->RegisterToAllocationContext(offset, context);
    }

    return Node::registerToAllocationContext(offset, context);
}

void Convolution::createPrimitive() {
    for (const auto& entry : m_atoi) {
        const auto argumentId = entry.first;
        const auto inputId = entry.second;
        m_memory[argumentId] = getSrcMemoryAtPort(inputId);
    }

    if (!m_attrs.withBias) {
        m_memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
    }

    if (withDWConv) {
        m_memory[ARG_ATTR_POST_OP_DW | ARG_WEI] = getSrcMemoryAtPort(getOriginalInputsNumber() + 0);
        m_memory[ARG_ATTR_POST_OP_DW | ARG_BIAS] = getSrcMemoryAtPort(getOriginalInputsNumber() + 1);
    }

    if (!legacyInputZeroPoints.empty()) {
        m_memory[ARG_ATTR_ZERO_POINTS | ARG_SRC] = memoryViewToVector(legacyInputZeroPoints, getEngine());
    }

    if (!legacyWeightsZeroPoints.empty()) {
        m_memory[ARG_ATTR_ZERO_POINTS | ARG_WEI] = memoryViewToVector(legacyWeightsZeroPoints, getEngine());
    }

    if (!legacyOutputCompensation.empty()) {
        m_memory[ARG_ATTR_ZERO_POINTS | ARG_DST] = memoryViewToVector(legacyOutputCompensation, getEngine());
    }

    if (!inputZeroPoints.empty()) {
        // WA Pass different representation of zero points using different identifier ARG_SRC_3
        // which is normally not used by convolution
        m_memory[ARG_ATTR_ZERO_POINTS | ARG_SRC_3] = memoryViewToVector(inputZeroPoints, getEngine());
    }

    if (withSum) {
        m_memory[ARG_SUM] = getSrcMemoryAtPort(getParentEdges().size() - 1);
    }

    m_memory[ARG_DST] = getDstMemoryAtPort(0);

    m_executor = m_factory->make(m_memory);

    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());

    if (subgraph) {
        subgraph->Activate();
    }

    Node::createPrimitive();
}

ExecutorPtr Convolution::createFallbackExecutor() {
    if (fallbackExecutor) {
        return fallbackExecutor;
    }

    ConvAttrs fallbackAttrs = m_attrs;
    PostOps& fallbackPostOps = fallbackAttrs.postOps;
    // remove sum post-op from fallback post-ops
    auto sumPostOp = std::find_if(fallbackPostOps.begin(), fallbackPostOps.end(), [](const auto& postOp) {
        return typeid(SumPostOp) == postOp.type();
    });

    fallbackPostOps.erase(sumPostOp, fallbackPostOps.end());

    CPU_NODE_ASSERT(fallbackPostOps.size() < m_attrs.postOps.size(),
                    "Unexpected post-ops size after sum post-op removal");

    auto dstType = getOriginalInputPrecisionAtPort(0);

    if (!fusedWith.empty()) {
        dstType = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    auto [srcDescs, dstDesc] = initMemoryDescriptors(dstType);

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[DATA]},
        {ARG_WEI, srcDescs[WEIGHTS]},
        {ARG_BIAS, m_attrs.withBias ? srcDescs[BIAS] : MemoryDescUtils::makeEmptyDesc()},
        {ARG_DST, dstDesc},
    };

    auto fallbackFactory = createExecutorFactory(descs, fallbackAttrs);
    fallbackExecutor = fallbackFactory->make(m_memory);
    return fallbackExecutor;
}

void Convolution::prepareParams() {
    // In case of fallback dst memory is a subgraph input
    m_memory[ARG_DST] = withSumBroadcast ? subgraph->getInput(0)->getDstMemoryAtPort(0) : getDstMemoryAtPort(0);
    const auto& executor = withSumBroadcast ? createFallbackExecutor() : m_executor;
    assert(executor);
    executor->update(m_memory);
}

void Convolution::redefineOutputMemory(const std::vector<VectorDims>& newOutputShapes) {
    if (!withSum) {  // fast path
        Node::redefineOutputMemory(newOutputShapes);
        return;
    }

    const size_t sumPortNum = getParentEdges().size() - 1;
    const auto& sumInpMem = getParentEdgeAt(sumPortNum)->getMemory();
    if (newOutputShapes.front() != sumInpMem.getStaticDims()) {
        withSumBroadcast = true;

        auto inp0 = subgraph->getInput(0);
        inp0->redefineOutputMemory(newOutputShapes);

        auto inp1 = subgraph->getInput(1);
        inp1->redefineOutputMemory({sumInpMem.getStaticDims()});
        // postpone output memory reallocation since it is the same memory with the sum
        // second input
        return;
    }

    withSumBroadcast = false;

    Node::redefineOutputMemory(newOutputShapes);
}

void Convolution::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto& executor = withSumBroadcast ? fallbackExecutor : m_executor;
    assert(executor);
    executor->execute(m_memory);
}

void Convolution::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);

    if (!withSumBroadcast) {
        return;
    }

    CPU_NODE_ASSERT(subgraph, "Fused ops subgraph has not been created");

    const size_t sumPortNum = getParentEdges().size() - 1;
    const auto& sumInpMem = getParentEdgeAt(sumPortNum)->getMemory();
    auto inp1 = subgraph->getInput(1);
    auto inp1Mem = inp1->getDstMemoryAtPort(0);
    inp1Mem->getMemoryBlock()->setExtBuff(sumInpMem.getData(), sumInpMem.getSize());

    subgraph->infer();

    auto out = subgraph->getOutput(0);
    const auto& outMem = out->getParentEdgeAt(0)->getMemory();
    auto convOutMem = getDstMemoryAtPort(0);
    Node::redefineOutputMemory({outMem.getStaticDims()});
    convOutMem->load(outMem, true, false);
}

void Convolution::addFusedNode(const NodePtr& fusingNode) {
    if (Type::Eltwise == fusingNode->getType()) {
        if (fusingNode->getAlgorithm() == Algorithm::EltwiseAdd) {
            auto eltwiseNode = std::dynamic_pointer_cast<Eltwise>(fusingNode);
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                withSum = true;
            }
        }
        if (withSum && isDynamicNode()) {
            for (size_t i = 0; i < fusingNode->getParentEdges().size(); ++i) {
                auto edge = fusingNode->getParentEdgeAt(i);
                auto parent = edge->getParent();
                if ("Constant" == parent->getTypeStr()) {
                    fusedConstNodes[fusingNode].push_back(parent);
                }
            }
        }
    }

    if (fusingNode->getType() == Type::Convolution) {
        auto convolutionNode = std::dynamic_pointer_cast<Convolution>(fusingNode);
        CPU_NODE_ASSERT(convolutionNode, "Unexpected dynamic node type");
        withDWConv = true;
        const auto& inActivationDims = convolutionNode->inputShapes[0].getStaticDims();
        dw_conv_ih = inActivationDims[convolutionNode->inputShapes[0].getRank() - 2];
        dw_conv_iw = inActivationDims[convolutionNode->inputShapes[0].getRank() - 1];

        const auto& outDims = convolutionNode->outputShapes[0].getStaticDims();
        dw_conv_oc = outDims[1];

        const auto& dwWeightsDims = convolutionNode->inputShapes[1].getStaticDims();
        dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 1]);
        dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 2]);
        dw_conv_strides = convolutionNode->getStride();

        if (canBeExecutedInInt8()) {
            if (fusedWith.empty()) {
                dw_conv_in_dt = DnnlExtensionUtils::ElementTypeToDataType(getOriginalOutputPrecisionAtPort(0));
            } else {
                dw_conv_in_dt =
                    DnnlExtensionUtils::ElementTypeToDataType(fusedWith.back()->getOriginalOutputPrecisionAtPort(0));
            }
        } else {
            dw_conv_in_dt = memory::data_type::f32;
        }

        const auto& weightDims = getInputShapeAtPort(1).getStaticDims();
        // @todo padding should be updated by the graph optimizer / transformation
        for (size_t j = 0; j < m_attrs.paddingR.size(); j++) {
            int with_group = m_attrs.isGrouped ? 1 : 0;
            int krn = weightDims[with_group + 2 + j];
            int src = getInputShapeAtPort(0).getStaticDims()[2 + j];
            int dst = fusingNode->getOutputShapeAtPort(0).getStaticDims()[2 + j];

            krn = (krn - 1) * (m_attrs.dilation[j] + 1) + 1;
            int calc_dst = (src - krn + m_attrs.paddingL[j]) / m_attrs.stride[j] + 1;
            m_attrs.paddingR[j] = (dst - calc_dst) * m_attrs.stride[j];
        }
    }

    Node::addFusedNode(fusingNode);
}

void Convolution::initializeInputZeroPoints(const uint8_t* inputZpData, const size_t inputZpSize) {
    const bool zeroPointsNotSet = inputZeroPoints.empty() && legacyInputZeroPoints.empty();
    CPU_NODE_ASSERT(zeroPointsNotSet, "input zero points are not empty");

    if (inputZpSize) {
        m_attrs.inputZeroPointsType = ZeroPointsType::PerTensor;
    }

    for (size_t j = 0; j < inputZpSize; j++) {
        legacyInputZeroPoints.push_back(inputZpData[j]);
        if (inputZpData[j] != inputZpData[0]) {
            m_attrs.inputZeroPointsType = ZeroPointsType::PerChannel;
        }
    }
    // Only enable per-tensor zero point on avx512-amx and avx512-core-vnni, avx2_vnni_2.
    // avx2_vnni is not enabled per-tensor z because of perf regression brgconv with per-tensor zpcompared with jit
    // per-channel zp If zero point is pertensor, both legacy zp and stock zp would be passed into conv node. The conv
    // node would determine how to create post-ops attribute and prioritize to choose final onednn kernel.
    if (m_attrs.inputZeroPointsType == ZeroPointsType::PerTensor &&
        (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx) ||
         impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_vnni) ||
         impl::cpu::x64::mayiuse(impl::cpu::x64::avx2_vnni_2))) {
        inputZeroPoints.push_back(static_cast<int32_t>(inputZpData[0]));
    } else {
        m_attrs.inputZeroPointsType = ZeroPointsType::PerChannel;
    }
}

}  // namespace ov::intel_cpu::node
