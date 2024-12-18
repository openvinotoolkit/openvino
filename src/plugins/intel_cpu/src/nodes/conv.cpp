// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/cpu_convert.h"
#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/primitive_hashing_utils.hpp"
#include "concat.h"
#include "cpu/cpu_primitive.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "dnnl_types.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "graph.h"
#include "input.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "onednn/dnnl.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "pooling.h"
#include "reorder.h"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov {
namespace intel_cpu {
namespace node {

bool Convolution::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v1::Convolution>(op) && !ov::is_type<ov::op::v1::GroupConvolution>(op)) {
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
    : Node(op, context, NgraphShapeInferFactory(op)),
      withBiases(false),
      withSum(false),
      withDWConv(false),
      dw_conv_oc(0),
      dw_conv_ih(0),
      dw_conv_iw(0),
      dw_conv_in_dt(memory::data_type::undef),
      groupNum(1lu),
      IC(1),
      groupIC(1),
      groupOC(1) {
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

        for (size_t i = 0; i < convolutionOp->get_strides().size(); i++) {
            m_attrs.stride.push_back(convolutionOp->get_strides()[i]);
        }
        for (size_t i = 0; i < convolutionOp->get_dilations().size(); i++) {
            m_attrs.dilation.push_back(static_cast<ptrdiff_t>(convolutionOp->get_dilations()[i]) - 1);
        }
        m_attrs.paddingL = convolutionOp->get_pads_begin();
        m_attrs.paddingR = convolutionOp->get_pads_end();
        m_attrs.autoPadding =
            convolutionOp->get_auto_pad() == ov::op::PadType::SAME_UPPER
                ? AutoPaddingType::SAME_UPPER
                : (convolutionOp->get_auto_pad() == ov::op::PadType::SAME_LOWER ? AutoPaddingType::SAME_LOWER
                                                                                : AutoPaddingType::None);
    } else if (groupConvolutionOp) {
        algorithm = Algorithm::ConvolutionGrouped;
        m_attrs.isGrouped = true;

        groupNum = groupConvolutionOp->input_value(1).get_shape()[0];

        const auto& weightDims = groupConvolutionOp->input_value(1).get_shape();

        groupIC = weightDims[2];
        IC = groupIC * groupNum;
        groupOC = weightDims[1];

        for (size_t i = 0; i < groupConvolutionOp->get_strides().size(); i++) {
            m_attrs.stride.push_back(groupConvolutionOp->get_strides()[i]);
        }
        for (size_t i = 0; i < groupConvolutionOp->get_dilations().size(); i++) {
            m_attrs.dilation.push_back(groupConvolutionOp->get_dilations()[i] - 1);
        }
        m_attrs.paddingL = groupConvolutionOp->get_pads_begin();
        m_attrs.paddingR = groupConvolutionOp->get_pads_end();
        m_attrs.autoPadding =
            groupConvolutionOp->get_auto_pad() == ov::op::PadType::SAME_UPPER
                ? AutoPaddingType::SAME_UPPER
                : (groupConvolutionOp->get_auto_pad() == ov::op::PadType::SAME_LOWER ? AutoPaddingType::SAME_LOWER
                                                                                     : AutoPaddingType::None);
    }
    // Only apply this heuristic logic on FP32 IR. IC=1 ,OC=1 would disable brgconv on avx2.
    const bool isAvx2FP32 = !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
                            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !context->isGraphQuantized();
    useJitPlanar = ((IC == 1 && groupOC * groupNum == 1) && isAvx2FP32);
}

bool Convolution::canBeExecutedInInt8() const {
    auto inputDataType = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(0));
    auto weightsDataType = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(1));

    if (!legacyInputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    if (!legacyWeightsZeroPoints.empty())
        weightsDataType = memory::data_type::s8;

    return one_of(inputDataType, memory::data_type::u8, memory::data_type::s8) &&
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

    const bool isBrgConvAvailable = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !useJitPlanar;
    if (isBrgConvAvailable)
        return priorities;

    static const std::vector<impl_desc_type> priorities_wo_brgemm = [&] {
        std::vector<impl_desc_type> result;
        std::copy_if(priorities.begin(), priorities.end(), std::back_inserter(result), [](impl_desc_type type) {
            return !(type & impl_desc_type::brgconv);
        });
        return result;
    }();
    return priorities_wo_brgemm;
}

void Convolution::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
}

static MemoryDescPtr getSumMemDesc(const MemoryDescPtr& outputDesc, const Shape& sumShape) {
    if (outputDesc->getShape().isStatic()) {
        return outputDesc;
    }

    // When we set input shape with ranged dims, sum node input shape maybe mismatch with output shape, we just
    // change ranged min value to 1 to meet this case. For example: Output shape = {1, 160, {128, 256}, {128, 256}}
    // Sum input shape = {1, 160, 1, 1}
    // Update sum shape to {1, 160, {1, 256}, {1, 256}}
    auto shape = outputDesc->getShape();
    auto blockedOutputDesc = outputDesc->as<BlockedMemoryDesc>();
    if (shape.getRank() != sumShape.getRank()) {
        return std::make_shared<CpuBlockedMemoryDesc>(outputDesc->getPrecision(),
                                                      shape,
                                                      blockedOutputDesc->getBlockDims(),
                                                      blockedOutputDesc->getOrder(),
                                                      blockedOutputDesc->getOffsetPadding(),
                                                      blockedOutputDesc->getOffsetPaddingToData(),
                                                      blockedOutputDesc->getStrides());
    }

    const auto& sumDims = sumShape.getDims();
    const auto& maxDims = shape.getMaxDims();
    auto minDims = shape.getMinDims();

    for (size_t i = 0; i < maxDims.size(); i++) {
        if ((maxDims[i] > minDims[i]) && sumDims[i] == 1) {
            minDims[i] = 1;
        }
    }

    return std::make_shared<CpuBlockedMemoryDesc>(outputDesc->getPrecision(),
                                                  Shape(minDims, maxDims),
                                                  blockedOutputDesc->getBlockDims(),
                                                  blockedOutputDesc->getOrder(),
                                                  blockedOutputDesc->getOffsetPadding(),
                                                  blockedOutputDesc->getOffsetPaddingToData(),
                                                  blockedOutputDesc->getStrides());
}

void Convolution::initSupportedPrimitiveDescriptors() {
    m_attrs.withBias = getOriginalInputsNumber() == 3;
    if (m_attrs.withBias)
        m_atoi[ARG_BIAS] = BIAS;

    m_attrs.isGraphQuantized = context->isGraphQuantized();
    m_attrs.fcSemantic = false;
    m_attrs.nonConstantWeights = !getParentEdgeAt(WEIGHTS)->getParent()->isConstant();
    m_attrs.weightsNonTransposed = false;
    m_attrs.inputZeroPointsType = inputZeroPointType;
    m_attrs.dqScales = getDQScales();

    postOps = getPostOps(fusedWith);

    const auto& srcTypes = getOriginalInputPrecisions();
    auto dstTypes = getOriginalOutputPrecisions();
    // @todo graph optimizer should update original output precisions instead
    if (!fusedWith.empty()) {
        dstTypes = fusedWith.back()->getOriginalOutputPrecisions();
    }

    VecMemoryDescs srcDescs;
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < srcTypes.size(); i++) {
        if (srcTypes[i] == element::undefined) {
            srcDescs.push_back(MemoryDescUtils::makeEmptyDesc());
            continue;
        }
        const auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcTypes[i], getInputShapeAtPort(i));
        srcDescs.push_back(srcDesc);
    }

    VecMemoryDescs dstDescs;
    for (size_t i = 0; i < dstTypes.size(); i++) {
        const auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstTypes[i], getOutputShapeAtPort(i));
        dstDescs.push_back(dstDesc);
    }

    MemoryDescArgs descs{
        {ARG_SRC, srcDescs[DATA]},
        {ARG_WEI, srcDescs[WEIGHTS]},
        {ARG_BIAS, m_attrs.withBias ? srcDescs[BIAS] : MemoryDescUtils::makeEmptyDesc()},
        {ARG_DST, dstDescs[0]},
    };

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    factory =
        std::make_shared<ExecutorFactory<ConvAttrs>>(m_attrs, postOps, executionContext, descs, memoryFormatFilter);
    const std::vector<MemoryDescArgs> nodeDescriptorsList = factory->getProperMemoryDescriptors(descs);

    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(srcDescs.size());

        auto getBlockedMask = [](const std::shared_ptr<MemoryDesc>& memDesc, const bool isGrouped) {
            if (memDesc->getType() & MemoryDescType::Blocked && !isGrouped)
                return BlockedMemoryDesc::EMPTY_MASK;
            return BlockedMemoryDesc::FULL_MASK;
        };

        for (const auto& desc : nodeDescriptors) {
            if (m_atoi.count(desc.first)) {
                const auto& inputDesc = desc.second;
                nodeConfig.inConfs[m_atoi[desc.first]] = {inputDesc, getBlockedMask(inputDesc, m_attrs.isGrouped)};
            }
        }

        for (size_t i = 3; i < srcDescs.size(); i++) {
            nodeConfig.inConfs[i] = srcDescs[i];
        }

        const int inPlaceOutPort = withSum ? static_cast<int>(getParentEdges().size()) - 1 : -1;
        const auto& outputDesc = nodeDescriptors.at(ARG_DST);
        nodeConfig.outConfs.emplace_back(outputDesc, getBlockedMask(outputDesc, m_attrs.isGrouped), inPlaceOutPort);

        if (withDWConv) {
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
            nodeConfig.inConfs.emplace_back(
                getSumMemDesc(nodeDescriptors.at(ARG_DST), getInputShapeAtPort(getParentEdges().size() - 1)),
                BlockedMemoryDesc::FULL_MASK,
                -1);
        }

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }

    return;
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
    if (!fusedWith.empty())
        return false;
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

void Convolution::createPrimitive() {
    for (const auto& entry : m_atoi) {
        const auto argumentId = entry.first;
        const auto inputId = entry.second;
        memory[argumentId] = getSrcMemoryAtPort(inputId);
    }

    if (!m_attrs.withBias) {
        memory[ARG_BIAS] = MemoryDescUtils::makeEmptyMemory(context);
    }

    if (withDWConv) {
        memory[ARG_ATTR_POST_OP_DW | ARG_WEI] = getSrcMemoryAtPort(getOriginalInputsNumber() + 0);
        memory[ARG_ATTR_POST_OP_DW | ARG_BIAS] = getSrcMemoryAtPort(getOriginalInputsNumber() + 1);
    }

    if (!legacyInputZeroPoints.empty()) {
        memory[ARG_ATTR_ZERO_POINTS | ARG_SRC] = memoryViewToVector(legacyInputZeroPoints, getEngine());
    }

    if (!legacyWeightsZeroPoints.empty()) {
        memory[ARG_ATTR_ZERO_POINTS | ARG_WEI] = memoryViewToVector(legacyWeightsZeroPoints, getEngine());
    }

    if (!legacyOutputCompensation.empty()) {
        memory[ARG_ATTR_ZERO_POINTS | ARG_DST] = memoryViewToVector(legacyOutputCompensation, getEngine());
    }

    if (!inputZeroPoints.empty()) {
        // WA Pass different representation of zero points using different identifier ARG_SRC_2
        // which is normally not used by convolution
        memory[ARG_ATTR_ZERO_POINTS | ARG_SRC_2] = memoryViewToVector(inputZeroPoints, getEngine());
    }

    memory[ARG_DST] = getDstMemoryAtPort(0);

    executor = factory->make(memory);

    getSelectedPrimitiveDescriptor()->setImplementationType(executor->implType());

    Node::createPrimitive();
}

void Convolution::prepareParams() {
    executor->update(memory);
}

void Convolution::execute(const dnnl::stream& strm) {
    assert(executor);
    executor->execute();
}

void Convolution::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
    if (withSumBroadcast) {
        if (!subgraph) {
            OPENVINO_THROW("Unexpected: Fused ops subgraph has not been created in ",
                           getTypeStr(),
                           " with name ",
                           getName());
        }
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
        convOutMem->load(outMem);
    }
}

void Convolution::redefineOutputMemory(const std::vector<VectorDims>& newOutputShapes) {
    if (withSum) {
        const size_t sumPortNum = getParentEdges().size() - 1;
        const auto& sumInpMem = getParentEdgeAt(sumPortNum)->getMemory();
        if (newOutputShapes.front() != sumInpMem.getStaticDims()) {
            withSumBroadcast = true;
            if (!subgraph) {
                subgraph = std::make_shared<FusedSubgraph>(fusedWith, *this, context);
            }
            auto inp0 = subgraph->getInput(0);
            inp0->redefineOutputMemory(newOutputShapes);

            auto inp1 = subgraph->getInput(1);
            inp1->redefineOutputMemory({sumInpMem.getStaticDims()});
            // here we postpone output memory reallocation due to the fact that it is the same memory with the sum
            // second input
            return;
        } else {
            withSumBroadcast = false;
        }
    }
    Node::redefineOutputMemory(newOutputShapes);
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
        withDWConv = true;
        auto& inActivationDims = convolutionNode->inputShapes[0].getStaticDims();
        dw_conv_ih = inActivationDims[convolutionNode->inputShapes[0].getRank() - 2];
        dw_conv_iw = inActivationDims[convolutionNode->inputShapes[0].getRank() - 1];

        auto& outDims = convolutionNode->outputShapes[0].getStaticDims();
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
    }

    Node::addFusedNode(fusingNode);
}

void Convolution::initializeInputZeroPoints(const uint8_t* inputZpData, const size_t inputZpSize) {
    if (!inputZeroPoints.empty() || !legacyInputZeroPoints.empty())
        OPENVINO_THROW("input zero point is not empty '", getName(), "'");
    if (inputZpSize)
        inputZeroPointType = ZeroPointsType::PerTensor;
    for (size_t j = 0; j < inputZpSize; j++) {
        legacyInputZeroPoints.push_back(inputZpData[j]);
        if (inputZpData[j] != inputZpData[0])
            inputZeroPointType = ZeroPointsType::PerChannel;
    }
    // Only enable per-tensor zero point on avx512-amx and avx512-core-vnni, avx2_vnni_2.
    // avx2_vnni is not enabled per-tensor z because of perf regression brgconv with per-tensor zpcompared with jit
    // per-channel zp If zero point is pertensor, both legacy zp and stock zp would be passed into conv node. The conv
    // node would determine how to create post-ops attribute and prioritize to choose final onednn kernel.
    if (inputZeroPointType == ZeroPointsType::PerTensor && (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx) ||
                                                            impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_vnni) ||
                                                            impl::cpu::x64::mayiuse(impl::cpu::x64::avx2_vnni_2)))
        inputZeroPoints.push_back(static_cast<int32_t>(inputZpData[0]));
    else
        inputZeroPointType = ZeroPointsType::PerChannel;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
