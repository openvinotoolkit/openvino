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
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "onednn/dnnl.h"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "pooling.h"
#include "reorder.h"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

using namespace dnnl;

namespace ov::intel_cpu::node {
namespace {

struct ConvKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;

    std::vector<size_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;

    dnnl::primitive_attr attr;
    impl_desc_type implType;

    bool constWeight;

    [[nodiscard]] size_t hash() const;
    bool operator==(const ConvKey& rhs) const;
};

size_t ConvKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = get_vector_hash(seed, stride);
    seed = get_vector_hash(seed, dilation);
    seed = get_vector_hash(seed, paddingL);
    seed = get_vector_hash(seed, paddingR);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    seed = hash_combine(seed, constWeight);
    return seed;
}

bool ConvKey::operator==(const ConvKey& rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }

    retVal = retVal && stride == rhs.stride;
    retVal = retVal && dilation == rhs.dilation;
    retVal = retVal && paddingL == rhs.paddingL;
    retVal = retVal && paddingR == rhs.paddingR;

    retVal = retVal && *attr.get() == *rhs.attr.get() && implType == rhs.implType && constWeight == rhs.constWeight;
    return retVal;
}

}  // namespace

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
                    size_t inpPort = 1lu;
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
        if (idx < inputs.size()) {
            return inputs[idx];
        }
        OPENVINO_THROW("OutOfBounds: Unexpected input index in Convolution::fusedSubgraph::getInput idx=",
                       idx,
                       " inputs.size()=",
                       inputs.size());
    }

    [[nodiscard]] std::shared_ptr<Input> getOutput(size_t idx) const {
        if (idx < outputs.size()) {
            return outputs[idx];
        }
        OPENVINO_THROW("OutOfBounds: Unexpected output index in Convolution::fusedSubgraph::getInput idx=",
                       idx,
                       " inputs.size()=",
                       outputs.size());
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
    : Node(op, context, NgraphShapeInferFactory(op)),
      withBiases(false),
      withSum(false),
      withDWConv(false),
      isGrouped(false),
      dw_conv_oc(0),
      dw_conv_ih(0),
      dw_conv_iw(0),
      dw_conv_in_dt(memory::data_type::undef),
      groupNum(1lu),
      IC(1),
      groupIC(1),
      groupOC(1),
      eltwisePrecision(ov::element::f32) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto convolutionOp = ov::as_type_ptr<ov::op::v1::Convolution>(op);
    auto groupConvolutionOp = ov::as_type_ptr<ov::op::v1::GroupConvolution>(op);

    if (convolutionOp) {
        algorithm = Algorithm::ConvolutionCommon;

        groupNum = 1;
        isGrouped = false;

        weightDims = convolutionOp->input_value(1).get_shape();

        IC = weightDims[1];
        groupIC = IC;
        groupOC = weightDims[0];

        expectedBiasDims = {groupOC};

        for (size_t i : convolutionOp->get_strides()) {
            stride.push_back(i);
        }
        for (size_t i : convolutionOp->get_dilations()) {
            dilation.push_back(static_cast<ptrdiff_t>(i) - 1);
        }
        paddingL = convolutionOp->get_pads_begin();
        paddingR = convolutionOp->get_pads_end();
        autoPadding = one_of(convolutionOp->get_auto_pad(), ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER);
    } else if (groupConvolutionOp) {
        algorithm = Algorithm::ConvolutionGrouped;

        groupNum = groupConvolutionOp->input_value(1).get_shape()[0];
        isGrouped = true;

        weightDims = groupConvolutionOp->input_value(1).get_shape();

        groupIC = weightDims[2];
        IC = groupIC * groupNum;
        groupOC = weightDims[1];

        expectedBiasDims = {groupOC * groupNum};

        for (size_t i : groupConvolutionOp->get_strides()) {
            stride.push_back(i);
        }
        for (size_t i : groupConvolutionOp->get_dilations()) {
            dilation.push_back(static_cast<ptrdiff_t>(i) - 1);
        }
        paddingL = groupConvolutionOp->get_pads_begin();
        paddingR = groupConvolutionOp->get_pads_end();
        autoPadding =
            one_of(groupConvolutionOp->get_auto_pad(), ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER);
    }
    // Only apply this heuristic logic on FP32 IR. IC=1 ,OC=1 would disable brgconv on avx2.
    const bool isAvx2FP32 = !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
                            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !context->isGraphQuantized();
    useJitPlanar = ((IC == 1 && groupOC * groupNum == 1) && isAvx2FP32);
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

    return one_of(inputDataType, memory::data_type::u8, memory::data_type::s8) &&
           weightsDataType == memory::data_type::s8;
}

ov::element::Type Convolution::fusedEltwisePrecision(const NodePtr& fusingNode) const {
    if (sumPrc != ov::element::dynamic) {
        return sumPrc;
    }

    ov::element::Type eltwisePrecision;

    int fusingPort = fusingNode->getFusingPort();
    if (fusingPort == 0) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(1);
    } else if (fusingPort == 1) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(0);
    } else {
        THROW_CPU_NODE_ERR("Cannot determine Eltwise post op precision");
    }

    return eltwisePrecision;
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
    if (isBrgConvAvailable()) {
        return priorities;
    }

    static const std::vector<impl_desc_type> priorities_wo_brgemm = [&] {
        std::vector<impl_desc_type> result;
        std::copy_if(priorities.begin(), priorities.end(), std::back_inserter(result), [](impl_desc_type type) {
            return !(type & impl_desc_type::brgconv);
        });
        return result;
    }();
    return priorities_wo_brgemm;
}

const bool Convolution::isBrgConvAvailable() {
    // When avx2 brgconv heuristic case,  disable brgconv to WA the regression.
    const bool isBrgConvAvailable = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2) && !useJitPlanar;
    return isBrgConvAvailable;
}

void Convolution::getSupportedDescriptors() {
    if (!descs.empty()) {
        return;
    }
    if (!attrs.empty()) {
        THROW_CPU_NODE_ERR("has a non-empty attrs vector");
    }

    attrs.reserve(2);
    withBiases = getOriginalInputsNumber() == 3;

    auto expectedInputEdgesNum = static_cast<int>(getOriginalInputsNumber());
    for (auto& i : fusedWith) {
        if (i->getType() == Type::Convolution) {
            expectedInputEdgesNum += static_cast<int>(i->getOriginalInputsNumber()) - 1;
        }

        if (i->getAlgorithm() == Algorithm::EltwiseAdd) {
            auto* eltwiseNode = dynamic_cast<Eltwise*>(i.get());
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                expectedInputEdgesNum++;
            }
        }
    }

    auto inputDataType = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(0));
    if (!legacyInputZeroPoints.empty()) {
        inputDataType = memory::data_type::u8;
    }

    outputDataType = DnnlExtensionUtils::ElementTypeToDataType(getOriginalOutputPrecisionAtPort(0));
    eltwisePrecision = DnnlExtensionUtils::DataTypeToElementType(outputDataType);
    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::ElementTypeToDataType(
            fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
        eltwisePrecision = DnnlExtensionUtils::DataTypeToElementType(outputDataType);
    }

    // We need to make sure that convolution output and second input of fused Eltwise operation
    // have equal precision sizes since they use the same physical memory. In case precisions are different we upscale
    // to FP32.
    if (outputDataType != memory::data_type::f32 && outputDataType != memory::data_type::bf16 &&
        outputDataType != memory::data_type::f16 && withSum) {
        for (auto& i : fusedWith) {
            if (i->getAlgorithm() == Algorithm::EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<Eltwise*>(i.get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(i);
                    if (DnnlExtensionUtils::DataTypeToElementType(outputDataType).size() != eltwisePrecision.size()) {
                        eltwisePrecision = ov::element::f32;
                        outputDataType = memory::data_type::f32;
                    }
                    break;
                }
            }
        }
    }

    if (static_cast<int>(getParentEdges().size()) != expectedInputEdgesNum) {
        THROW_CPU_NODE_ERR("Incorrect number of input edges, expected: ",
                           expectedInputEdgesNum,
                           " actual: ",
                           getParentEdges().size());
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("Incorrect number of output edges");
    }

    int ndims = getInputShapeAtPort(0).getRank();

    withDWConv = isFusedWith(Type::Convolution);
    if (withDWConv && isDynamicNode()) {
        THROW_CPU_NODE_ERR("DW convolution is fused into the node with dynamic shape.");
    }

    for (size_t i = 0; i < fusedWith.size(); i++) {
        auto* convolutionNode = dynamic_cast<Convolution*>(fusedWith[i].get());
        if (convolutionNode) {
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
                if (i == 0) {
                    dw_conv_in_dt = DnnlExtensionUtils::ElementTypeToDataType(getOriginalOutputPrecisionAtPort(0));
                } else {
                    dw_conv_in_dt = DnnlExtensionUtils::ElementTypeToDataType(
                        fusedWith[i - 1]->getOriginalOutputPrecisionAtPort(0));
                }
            } else {
                dw_conv_in_dt = memory::data_type::f32;
            }

            for (size_t j = 0; j < paddingR.size(); j++) {
                int with_group = isGrouped ? 1 : 0;
                int krn = weightDims[with_group + 2 + j];
                int src = getInputShapeAtPort(0).getStaticDims()[2 + j];
                int dst = getOutputShapeAtPort(0).getStaticDims()[2 + j];

                krn = (krn - 1) * (dilation[j] + 1) + 1;
                int calc_dst = (src - krn + paddingL[j]) / stride[j] + 1;
                paddingR[j] = (dst - calc_dst) * stride[j];
            }
        }
    }

    MemoryDescPtr in_candidate, out_candidate;
    memory::format_tag nspc =
        ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc);
    memory::format_tag ncsp =
        ndims == 3 ? memory::format_tag::ncw : (ndims == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw);
    memory::format_tag nCsp8c = ndims == 3 ? memory::format_tag::nCw8c
                                           : (ndims == 4 ? memory::format_tag::nChw8c : memory::format_tag::nCdhw8c);
    memory::format_tag nCsp16c = ndims == 3 ? memory::format_tag::nCw16c
                                            : (ndims == 4 ? memory::format_tag::nChw16c : memory::format_tag::nCdhw16c);

    if (canBeExecutedInInt8()) {
        DEBUG_LOG(getName(), "Creating I8 descriptor");

        // so far oneDNN INT8 convolution only support s8,u8,s32,f32,bf16 output types
        if (outputDataType == memory::data_type::f16) {
            outputDataType = memory::data_type::f32;
            eltwisePrecision = ov::element::f32;
        }

        SetPostOpsAndZeroPoints(attrs);

        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0), inputDataType, nspc);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0), outputDataType, nspc);
        createDescriptor({in_candidate}, {out_candidate});
        return;
    }

    auto getSupportedDataType = [this, ndims](ov::element::Type originalPrec) {
        auto originalDT = DnnlExtensionUtils::ElementTypeToDataType(originalPrec);
        auto dt = memory::data_type::f32;

        // supported lower precisions: bf16, f16
        if (one_of(originalDT, memory::data_type::bf16, memory::data_type::f16) && hasHardwareSupport(originalPrec)) {
            dt = originalDT;
        }

        // fallback to f32 on special case for performance reasons
        if (isDepthWise() && ndims == 5) {
            dt = memory::data_type::f32;
        }
        return dt;
    };

    inputDataType = getSupportedDataType(getOriginalInputPrecisionAtPort(0));
    outputDataType = getSupportedDataType(getOriginalOutputPrecisionAtPort(0));

    eltwisePrecision = ov::element::f32;
    for (auto& i : fusedWith) {
        if (i->getAlgorithm() == Algorithm::EltwiseAdd) {
            auto* eltwiseNode = dynamic_cast<Eltwise*>(i.get());
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                eltwisePrecision = fusedEltwisePrecision(i);
                // TODO(amalyshe): there might be situation when convolution can be executed in BF16,
                // output is required in FP32 but eltwise inplace tensor would be in BF16
                // currently we forcedly change output to the BF16 that will add reoreder after the node
                // Another situation can be when we mark output as FP32 and Eltwise asPrecison (which stand
                // for input of inplace tensor precision) to FP32. This will add reorder for that in-place tensor
                // bofore the fused convolution. This behaviour might be more correct regarding expected markup
                // of the graph but performance of first and second approaches might be different. Need to verify
                outputDataType = getSupportedDataType(eltwisePrecision);
                eltwisePrecision = DnnlExtensionUtils::DataTypeToElementType(outputDataType);
            }
        }
    }
    // correction for cases of FP32 input - we do not have FP32 convolution supported BF16 output
    if (inputDataType == memory::data_type::f32 &&
        (outputDataType == memory::data_type::bf16 || eltwisePrecision == ov::element::bf16 ||
         outputDataType == memory::data_type::f16 || eltwisePrecision == ov::element::f16)) {
        outputDataType = memory::data_type::f32;
        eltwisePrecision = ov::element::f32;
    }
    SetPostOpsAndZeroPoints(attrs);

    if (!one_of(ndims, 3, 4, 5)) {
        return;
    }

    auto inputShape = getInputShapeAtPort(0);
    auto outputShape = getOutputShapeAtPort(0);

#if defined(OPENVINO_ARCH_X86_64)
    // nspc shows better performance only with brgconv implementation
    bool nspcFirst = isBrgConvAvailable() &&
                     one_of(inputDataType, memory::data_type::f16, memory::data_type::bf16, memory::data_type::f32);
    bool nspcAdded = false;
    if (nspcFirst) {
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
        createDescriptor({in_candidate}, {out_candidate});
        nspcAdded = true;
    }

    if (IC == 1 && groupOC == 1) {
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, ncsp);
        createDescriptor({in_candidate}, {out_candidate});
    } else if (IC < 4) {
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp16c);
        createDescriptor({in_candidate}, {out_candidate});
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp8c);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nCsp16c);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp16c);
        createDescriptor({in_candidate}, {out_candidate});
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nCsp8c);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp8c);
        createDescriptor({in_candidate}, {out_candidate});
    }

    in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
    out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, ncsp);
    createDescriptor({in_candidate}, {out_candidate});

    if (!nspcAdded &&
        (inputDataType != memory::data_type::bf16 && inputDataType != memory::data_type::f16 && isNspcAvailable())) {
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
        createDescriptor({in_candidate}, {out_candidate});
    }
#else
    (void)ncsp;
    (void)nCsp8c;
    (void)nCsp16c;

    in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
    out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
    createDescriptor({in_candidate}, {out_candidate});
#endif
}

void Convolution::setPostOps(dnnl::primitive_attr& attr,
                             const VectorDims& dims,
                             bool useLegacyPostOps,
                             bool initWeights) {
    dnnl::post_ops ops;
    auto& args = convPostOpsArgs[useLegacyPostOps];
    bool isINT8 = canBeExecutedInInt8();
    // Weight dims in NON-Group CONV: [OC, IC, KH, KW], perchannel weight scale applied on OC DIM,
    // weiScaleMaskPerChannel =  1 << 0 Weight dims in Group CONV:[Group, OC, IC, KH, KW], perchannel weight scale
    // applied on GROUP and OC DIM, weiScaleMaskPerChannel = ( 1 << 0 | 1<< 1) = 0x03
    DnnlPostOpsComposerLegacy
        dnnlpoc(getEngine(), attr, ops, args, dims, 1, isINT8, isGrouped ? 3 : 1 << 0, getDQScales(), withBiases);

    DEBUG_LOG(getName(), " useLegacyPostOps=", useLegacyPostOps, " initWeights=", initWeights);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (node->getType() == Type::Split || node->getType() == Type::Concatenation) {
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->isSpecialConvolutionAddFusing()) {
                if (withSumBroadcast) {
                    break;
                }
                DEBUG_LOG(getName(), ": Append ", node->getName(), " as sum post op");
                ops.append_sum(1.0, 0, DnnlExtensionUtils::ElementTypeToDataType(eltwisePrecision));
            } else {
                if (useLegacyPostOps) {
                    // try mapping with optimization w/o using binary postOps
                    if (eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType, false)) {
                        DEBUG_LOG(getName(), ": Append ", node->getName(), " as original post op without binary");
                        continue;
                    }
                    DEBUG_LOG(getName(), ": Append ", node->getName(), " as legacy post op");
                    int channelAxis = 1;
                    eltwiseNode->appendPostOps(ops, dims, args, channelAxis);
                } else {
                    DEBUG_LOG(getName(), ": Append ", node->getName(), " as original post op with binary");
                    eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
                }
            }
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            // drop rounding one special residual pattern
            // TODO: validate this unsafe optimization
            bool do_rounding = true;
            if (i == 0) {
                bool hasSubsequentSum = false;
                bool hasSubsequentFQ = false;
                for (size_t j = i + 1; j < fusedWith.size(); j++) {
                    auto& nextNode = fusedWith[j];

                    auto* nextEltwiseNode = dynamic_cast<Eltwise*>(nextNode.get());
                    if (nextEltwiseNode && nextEltwiseNode->isSpecialConvolutionAddFusing()) {
                        hasSubsequentSum = true;
                    }

                    auto* nextQuantizeNode = dynamic_cast<FakeQuantize*>(nextNode.get());
                    if (nextQuantizeNode) {
                        hasSubsequentFQ = true;
                    }
                }
                if (hasSubsequentSum && hasSubsequentFQ) {
                    do_rounding = false;
                }
            }

            if (useLegacyPostOps) {
                // can we implement it without binary postOps?
                if (fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType, false, do_rounding)) {
                    DEBUG_LOG(getName(), ": Append ", node->getName(), " as original post op without binary");
                    continue;
                }
                // fallback to legacy
                DEBUG_LOG(getName(), ": Append ", node->getName(), " as legacy post op");
                int channelAxis = 1;
                fakeQuantizeNode->appendPostOps(ops, dims, args, channelAxis);
            } else {
                DEBUG_LOG(getName(), ": Append ", node->getName(), " as original post op with binary");
                fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType, true, do_rounding);
            }
            continue;
        }

        auto* convolutionNode = dynamic_cast<Convolution*>(node.get());
        if (convolutionNode) {
            if (initWeights) {
                args[DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS] = getSrcMemoryAtPort(getOriginalInputsNumber() + 0);
                args[DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS] = getSrcMemoryAtPort(getOriginalInputsNumber() + 1);

                DEBUG_LOG(getName(), ": Append ", node->getName(), " as DW convolution");
                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih,
                                   dw_conv_iw,
                                   dw_conv_kernel[Y_AXIS],
                                   dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS],
                                   dw_conv_strides[X_AXIS],
                                   dnnl::memory::convert_to_c(dw_conv_in_dt));
            } else {
                DEBUG_LOG(getName(), ": Append ", node->getName(), " as DW convolution");
                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih,
                                   dw_conv_iw,
                                   dw_conv_kernel[Y_AXIS],
                                   dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS],
                                   dw_conv_strides[X_AXIS],
                                   dnnl::memory::convert_to_c(dw_conv_in_dt));
            }
            continue;
        }

        THROW_CPU_NODE_ERR("Fusing of ",
                           NameFromType(node->getType()),
                           " operation to ",
                           NameFromType(this->getType()),
                           " node is not implemented");
    }

    attr.set_post_ops(ops);
}

void Convolution::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto getBlockedMask = [](const std::shared_ptr<MemoryDesc>& memDesc, const bool isGrouped) {
        if (memDesc->getType() & MemoryDescType::Blocked && !isGrouped) {
            return BlockedMemoryDesc::EMPTY_MASK;
        }
        return BlockedMemoryDesc::FULL_MASK;
    };

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = withSum ? static_cast<int>(getParentEdges().size()) - 1 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);

            inConfs.emplace_back(desc, getBlockedMask(desc, isGrouped));
        }

        if (withDWConv) {
            const std::vector<size_t> dwWeightsDims{dw_conv_oc, 1, 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]};
            const std::vector<size_t> dwBiasesDims{dw_conv_oc};

            const auto dwWeightsPrc = DnnlExtensionUtils::ElementTypeToDataType(
                dw_conv_in_dt == dnnl_u8 ? ov::element::i8 : ov::element::f32);
            const auto dwWeightsDesc = std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwWeightsDims),
                                                                               dwWeightsPrc,
                                                                               memory::format_tag::Goihw8g);
            inConfs.emplace_back(dwWeightsDesc);

            const auto dwBiasPrc = memory::data_type::f32;
            const auto dwBiasDesc =
                std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwBiasesDims), dwBiasPrc, memory::format_tag::x);
            inConfs.emplace_back(dwBiasDesc);
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, getBlockedMask(desc, isGrouped), inPlaceOutPort);
        }

        if (withSum) {
            const auto outputPrecision = outConfs.back().getMemDesc()->getPrecision();
            const auto sumDesc = getSumMemDesc(prim_desc)->cloneWithNewPrecision(outputPrecision);
            inConfs.emplace_back(sumDesc);
        }

        NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };
#ifdef CPU_DEBUG_CAPS
    {
        if (!customImplPriorities.empty()) {
            DEBUG_LOG("#",
                      getName(),
                      " customImplPriorities [",
                      0,
                      "/",
                      customImplPriorities.size(),
                      "]: ",
                      impl_type_to_string(customImplPriorities[0]));
        }
    }
#endif
    for (size_t dIdx = 0; dIdx < descs.size(); dIdx++) {
        auto& desc = descs[dIdx];
        auto primitive_desc = desc.get(true);  // true mean allow empty
        if (primitive_desc == nullptr) {
            continue;
        }
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(primitive_desc));

        auto add_supported_desc = [&](dnnl::primitive_desc& desc) {
            addSupportedPrimitiveDescriptor(desc);
            descIdx.push_back(dIdx);
        };

        const bool first_match = customImplPriorities.empty();
        DEBUG_LOG("#",
                  getName(),
                  ",descIndex:",
                  dIdx + 1,
                  "/",
                  descs.size(),
                  ", itpd.impl_info_str(): ",
                  desc.impl_info_str(),
                  ", parsed imp_type: ",
                  impl_type_to_string(parse_impl_name(desc.impl_info_str())),
                  ", first_match: ",
                  first_match ? "true" : "false");
        DnnlExtensionUtils::for_each_implementation(
            desc,
            first_match,
            [&](impl_desc_type implType) {
                return contains(getImplPriority(), implType);
            },
            add_supported_desc);

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty()) {
            add_supported_desc(first_desc);
        }
    }
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

int Convolution::registerToAllocationContext(int offset, AllocationContext& context) {
    if (subgraph) {
        return subgraph->RegisterToAllocationContext(offset, context);
    }

    return Node::registerToAllocationContext(offset, context);
}

void Convolution::createPrimitive() {
    if (subgraph) {
        subgraph->Activate();
    }

    Node::createPrimitive();
}

bool Convolution::created() const {
    return getType() == Type::Convolution;
}

namespace {
dnnl::convolution_forward::primitive_desc createDescriptorInternal(const dnnl::engine& engine,
                                                                   const dnnl::memory::desc& inputDesc,
                                                                   const dnnl::memory::desc& weightDesc,
                                                                   const dnnl::memory::desc& biasDesc,
                                                                   const dnnl::memory::desc& outputDesc,
                                                                   bool withBiases,
                                                                   const std::vector<size_t>& stride,
                                                                   const std::vector<ptrdiff_t>& dilation,
                                                                   const std::vector<ptrdiff_t>& paddingL,
                                                                   const std::vector<ptrdiff_t>& paddingR,
                                                                   dnnl::algorithm alg,
                                                                   const dnnl::primitive_attr& attr) {
    if (withBiases) {
        return dnnl::convolution_forward::primitive_desc(engine,
                                                         prop_kind::forward_inference,
                                                         alg,
                                                         inputDesc,
                                                         weightDesc,
                                                         biasDesc,
                                                         outputDesc,
                                                         dnnl::memory::dims(stride.begin(), stride.end()),
                                                         dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                         dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                         dnnl::memory::dims(paddingR.begin(), paddingR.end()),
                                                         attr,
                                                         true);  // allow_empty
    }
    return dnnl::convolution_forward::primitive_desc(engine,
                                                     prop_kind::forward_inference,
                                                     alg,
                                                     inputDesc,
                                                     weightDesc,
                                                     outputDesc,
                                                     dnnl::memory::dims(stride.begin(), stride.end()),
                                                     dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                     dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                     dnnl::memory::dims(paddingR.begin(), paddingR.end()),
                                                     attr,
                                                     true);  // allow_empty
}
}  // namespace

static memory::data_type deriveWeightDataType(memory::data_type src_dt) {
    memory::data_type wdt = src_dt;
    if (one_of(src_dt, memory::data_type::s8, memory::data_type::u8)) {
        wdt = memory::data_type::s8;
    }
    return wdt;
}

void Convolution::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                   const std::vector<MemoryDescPtr>& outputDesc) {
    MemoryDescPtr inpDesc;
    if (inputDesc[0]->isDefined()) {
        inpDesc = inputDesc[0];
    } else {
        auto dummyInDims = makeInputDummyShape(inputDesc[0]->getShape());
        inpDesc = inputDesc[0]->cloneWithNewDims(dummyInDims);
    }
    DnnlMemoryDescPtr definedInpMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc);
    DnnlMemoryDescPtr definedOutMemDesc;

    if (outputDesc[0]->isDefined()) {
        definedOutMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outputDesc[0]);
    } else {
        std::vector<Shape> shapes = {definedInpMemDesc->getShape(), Shape(weightDims)};
        auto outDims = shapeInferGeneric(shapes);
        definedOutMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outputDesc[0]->cloneWithNewDims(outDims.front()));
    }

    const auto& inDnnlDesc = definedInpMemDesc->getDnnlDesc();
    const auto& outDnnlDesc = definedOutMemDesc->getDnnlDesc();

    memory::data_type wdt = deriveWeightDataType(inDnnlDesc.get_data_type());

    dnnl::memory::desc weightDnnlDesc(DnnlExtensionUtils::convertToDnnlDims(weightDims), wdt, memory::format_tag::any);
    dnnl::memory::desc biasDnnlDesc;

    if (withBiases) {
        // oneDNN ARM Convolution primitive supports only identical in/out data types
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
        memory::data_type bdt = outDnnlDesc.get_data_type();
#else
        memory::data_type bdt = memory::data_type::f32;
        /* brdgmm_dw_conv has more perf gain on bf16/fp16 inference.
        brdgmm_dw_conv supports only bia_type the same as src_type or dst_type.
        dw convolution support in onednn 3.5.
        BF16:
        kernel type | brgdconv | jit_uni_dw_convolution_fwd_t
        support impl type | native bf16 ISA without AMX | avx512_core_bf16 or avx512_core
        bias dt | oneof(src,dest) | oneof(src, dest, f32)
        FP16:
        kernel type | brgdconv | brgemm_convolution_fwd_t
        impl type | native FP16 ISA without AMX | native FP16 ISA
        bias type | oneof(src,dest) | oneof(src, dest, f32)
        @todo: this bias type changes may have minor accuracy impact on some models, so when upstream ONEDNN extend this
        kind of matrix support (ticket MFDNN-12936) we can continue use bdt = memory::data_type::f32 here;
        */
        auto out_dt = outDnnlDesc.get_data_type();
        if (!canBeExecutedInInt8() && isDepthWise()) {
            bool isF16BiasSupported = (out_dt == memory::data_type::f16) && hasHardwareSupport(ov::element::f16);
            bool isBF16BiasSupported = (out_dt == memory::data_type::bf16) &&
                                       (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16) ||
                                        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2));

            if (isF16BiasSupported || isBF16BiasSupported) {
                bdt = out_dt;
            }
        }
#endif
        biasDnnlDesc =
            dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(expectedBiasDims), bdt, memory::format_tag::any);
    }

    std::vector<dnnl::algorithm> algorithms;

    algorithms.push_back(baseConvAlgorithm);

    updatePadding();

    for (const auto alg : algorithms) {
        for (const auto& attr : attrs) {
            const auto desc = createDescriptorInternal(getEngine(),
                                                       inDnnlDesc,
                                                       weightDnnlDesc,
                                                       biasDnnlDesc,
                                                       outDnnlDesc,
                                                       withBiases,
                                                       stride,
                                                       dilation,
                                                       paddingL,
                                                       paddingR,
                                                       alg,
                                                       attr);
            descs.emplace_back(desc);
        }
    }
}

void Convolution::addZeroPoints(dnnl::primitive_attr& attr) {
    if (inputZeroPoints.empty()) {
        return;
    }
    DEBUG_LOG(getName(), ": Set original input zeropoints");
    attr.set_zero_points_mask(DNNL_ARG_SRC, 0);

    if (!stockInputZeroPointsMemPtr) {
        DnnlBlockedMemoryDesc memoryDesc(ov::element::i32, {inputZeroPoints.size()});
        stockInputZeroPointsMemPtr = std::make_shared<Memory>(getEngine(), memoryDesc, inputZeroPoints.data());
    }
}

void Convolution::addLegacyZeroPoints(dnnl::primitive_attr& attr) {
    if (!legacyInputZeroPoints.empty()) {
        DEBUG_LOG(getName(), ": Set legacy input zero points");
        attr.set_input_zero_points(legacyInputZeroPoints.size(), 1 << 1 /*through C dim*/);
        if (!legacyInputZeroPointsMemPtr) {
            DnnlBlockedMemoryDesc memoryDesc(ov::element::u8, {legacyInputZeroPoints.size()});
            legacyInputZeroPointsMemPtr =
                std::make_shared<Memory>(getEngine(), memoryDesc, legacyInputZeroPoints.data());
        }
    }

    if (!legacyWeightsZeroPoints.empty()) {
        DEBUG_LOG(getName(), ": Set legacy weights zero points");
        attr.set_weights_zero_points(legacyWeightsZeroPoints.size(), 1 << 1 /*through C dim*/);

        if (!legacyWeightsZeroPointsMemPtr) {
            DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, {legacyWeightsZeroPoints.size()});
            legacyWeightsZeroPointsMemPtr =
                std::make_shared<Memory>(getEngine(), memoryDesc, legacyWeightsZeroPoints.data());
        }
    }

    if (!legacyOutputCompensation.empty()) {
        DEBUG_LOG(getName(), ": Set legacy output compensationss");
        attr.set_output_compensations(legacyOutputCompensation.size(), 1 << 1 /*through C dim*/);

        if (!legacyOutputCompensationMemPtr) {
            DnnlBlockedMemoryDesc memoryDesc(ov::element::i32, {legacyOutputCompensation.size()});
            legacyOutputCompensationMemPtr =
                std::make_shared<Memory>(getEngine(), memoryDesc, legacyOutputCompensation.data());
        }
    }
}

static bool attrContainsPostOp(const dnnl::primitive_attr& attr, const dnnl::impl::primitive_kind_t kind) {
    const auto ops = attr.get_post_ops();
    return ops.get()->find(kind) != -1;
}

// See the src/plugins/intel_cpu/src/docs/convPostOps.md for details
void Convolution::SetPostOpsAndZeroPoints(std::vector<dnnl::primitive_attr>& attrs) {
    attrs.resize(1);
    auto outputShape = outputStaticShape();
    // attr[0] - Legacy post ops + Legacy zero points.
    DEBUG_LOG(getName(), ": set post ops, attr 0, useLegacyPostOps=true");
    setPostOps(attrs[0], outputShape, true);
    addLegacyZeroPoints(attrs[0]);

    // dw-conv would be fused into conv only on AVX2 platform. no need attr[1]. Avoid extra useless attribute.
    if (attrContainsPostOp(attrs[0], dnnl::impl::primitive_kind::convolution)) {
        return;
    }

    // no matter if brgconv is available, 1 attribute is enough. Avoid duplicated attribute
    if (inputZeroPointType == zpType::None && !attrContainsPostOp(attrs[0], dnnl::impl::primitive_kind::depthwise) &&
        !attrContainsPostOp(attrs[0], dnnl::impl::primitive_kind::quantization)) {
        return;
    }
    // Per channel zero point can only supported on attr[0].Avoid extra useless attribute.
    if (inputZeroPointType == zpType::PerChannel) {
        DEBUG_LOG(getName(), ": Per channel zero point can only supported on attr[0].Avoid extra useless attribute.");
        return;
    }
    if (!isBrgConvAvailable()) {
        DEBUG_LOG(getName(), ": brgconv is not available. Skip extra attribute");
        return;
    }
    // Try 2 attributes.
    attrs.resize(2);
    if (inputZeroPointType == zpType::PerTensor &&
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
        // WR to ONEDNN limitation. attr[1] - legacy post ops + stock zero point.
        //@todo:Unify to use binary postops+stock zero point when limitation is fixed.
        // For now, have to adapt to JIT_AMX kernel for performance.
        DEBUG_LOG(getName(), ": set post ops, attr 1, useLegacyPostOps=true");
        setPostOps(attrs[1], outputShape, true);
    } else {
        DEBUG_LOG(getName(), ": set post ops, attr 1, useLegacyPostOps=false");
        setPostOps(attrs[1], outputShape, false);
    }
    addZeroPoints(attrs[1]);
}

void Convolution::initDescriptor(const NodeConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();

    if (!selectedPD) {
        return;
    }

    // attr[0] for legacy post ops;
    // attr[1] is mostly for binaryPostops except when having per-tensor zp on AMX.
    const int descId = descIdx[selectedPrimitiveDescriptorIndex];
    int attrId = attrs.size() == 1 ? 0 : descId % 2 == 0 ? 0 : 1;

    preferLegacyPostOps = (attrId == 0 || (attrId == 1 && (inputZeroPointType == zpType::PerTensor) &&
                                           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)));
    // attr[0] for legacy zero point.
    // attr[1] for stock per-tensor zero point.
    preferLegacyZeroPoint = (attrId == 0);

    DEBUG_LOG(getName(),
              " selectedPrimitiveDescriptorIndex: ",
              selectedPrimitiveDescriptorIndex,
              " DescIdx: ",
              descId,
              " Selected impl type: ",
              selectedPD->getImplementationType(),
              " Desc impl type: ",
              parse_impl_name(descs[descId].impl_info_str()),
              " preferLegacyPostOps: ",
              preferLegacyPostOps,
              " preferLegacyZeroPoint: ",
              preferLegacyZeroPoint);

    auto updateNodeConfig = [&](const NodeConfig& cfg) {
        auto updatedConfig = cfg;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            PortConfig& dataConfig = updatedConfig.inConfs[i];
            dataConfig.inPlace(-1);
            dataConfig.setMemDesc(dataConfig.getMemDesc());
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            PortConfig& dataConfig = updatedConfig.outConfs[i];
            dataConfig.inPlace(-1);
            dataConfig.setMemDesc(dataConfig.getMemDesc());
            if (withSum) {
                auto& eltwiseConfig = updatedConfig.inConfs.back();
                eltwiseConfig.setMemDesc(eltwiseConfig.getMemDesc()->cloneWithNewPrecision(eltwisePrecision));
                dataConfig.inPlace(getParentEdges().size() - 1);
            }
        }

        return updatedConfig;
    };

    if (!canBeExecutedInInt8()) {  // strided blobs are suppoted only for FP32 convolutions
        descs.clear();
        createDescriptor({config.inConfs[0].getMemDesc()}, {config.outConfs[0].getMemDesc()});

        for (auto& desc : descs) {
            if (DnnlExtensionUtils::find_implementation(desc, selectedPD->getImplementationType())) {
                selectedPD->setConfig(config);
                return;
            }
        }
    }

    auto currentConfig = selectedPD->getConfig();
    const auto& updatedConfig = updateNodeConfig(currentConfig);

    selectedPD->setConfig(updatedConfig);
}

std::shared_ptr<MemoryDesc> Convolution::getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const {
    if (idx == 1) {
        // report original plain layout for weight since it needs to be reordered dynamically at runtime
        return std::make_shared<CpuBlockedMemoryDesc>(getOriginalInputPrecisionAtPort(idx),
                                                      Shape(getInputShapeAtPort(idx).getStaticDims()));
    }
    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1) : prim_desc.src_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

bool Convolution::canFuse(const NodePtr& node) const {
#if defined(OV_CPU_WITH_ACL)
    if (!fusedWith.empty())
        return false;
#endif
    return canFuseSimpleOperation(node);
}

dnnl::memory Convolution::getWeights() const {
    return getParentEdgeAt(1)->getMemory().getPrimitive();
}

dnnl::memory Convolution::getBias() const {
    return getParentEdgeAt(2)->getMemory().getPrimitive();
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

bool Convolution::isNspcAvailable() const {
    using impl::cpu::x64::mayiuse;

    // do not use in non-quantized networks until it is enforced externally
    if (!context->isGraphQuantized()) {
        auto predicate = [](memory::format_tag tag) {
            return one_of(tag, memory::format_tag::nwc, memory::format_tag::nhwc, memory::format_tag::ndhwc);
        };
        if (std::none_of(inputMemoryFormatsFilter.begin(), inputMemoryFormatsFilter.end(), predicate)) {
            return false;
        }
    }
    // AVX2 heuristic
    if (useJitPlanar) {
        return false;
    }
    // A bunch of heuristics are designed to cut off not optimal nspc convolution applications
    auto inpDims = getInputShapeAtPort(0).getDims();
    auto outDims = getOutputShapeAtPort(0).getDims();
    auto ndims = inpDims.size();
    if (isDepthWise()) {
        // 1d equivalent cases are painfully slow
        if (inpDims.size() == 3 || 1 == inpDims[inpDims.size() - 2]) {
            return false;
        }
    } else {
        // it was empirically observed that the nspc convolutions perform much slower than the blocked ones if the
        // channels number more than the specific value
        size_t spatialRank = ndims - 2;  // two means batch dim plus channels dim

        bool is1x1 = false;

        if (!isGrouped) {
            auto weightDimsReversItr = weightDims.crbegin();
            auto strideReversItr = stride.crbegin();
            auto paddingLreversItr = paddingL.crbegin();
            auto paddingRreversItr = paddingR.crbegin();

            for (size_t i = 0; i < spatialRank; ++i) {
                is1x1 = true && *(weightDimsReversItr++) == 1 && *(strideReversItr++) == 1 &&
                        *(paddingLreversItr++) == 0 && *(paddingRreversItr++) == 0;
            }
        }

        // if the activation field size is 1x1 the avx512 1x1 nspc convolution pollutes caches so that the layer after
        // the convolution performs slow
        if (mayiuse(impl::cpu::x64::avx512_core) && is1x1) {
            auto end = inpDims.rbegin();
            std::advance(end, spatialRank);
            if (std::all_of(inpDims.rbegin(), end, [](size_t x) {
                    return dimsEqualStrong(1, x);
                })) {
                return false;
            }
        }

        unsigned thresholdNumChannels = 128u;  // for avx and below
        if (is1x1) {
            thresholdNumChannels = 2048u;
        } else if (mayiuse(impl::cpu::x64::avx512_core)) {
            thresholdNumChannels = 512u;
        }

        size_t OC = outDims[1];
        if (std::max(IC, OC) >= thresholdNumChannels) {
            return false;
        }
        if (!mayiuse(impl::cpu::x64::avx)) {
            // SSE41 nspc convolutions do not support ic and oc tails yet and the blocked implementation will be much
            // better than gemm
            if ((IC % 8) || (OC % 8)) {
                return false;
            }
        }
    }

    return true;
}

void Convolution::prepareParams() {
    auto srcMemPtr = getSrcMemoryAtPort(0);
    auto wghMemPtr = getSrcMemoryAtPort(1);
    auto dstMemPtr = getOutputMemory();
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Destination memory was undefined.");
    }
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Input memory was undefined.");
    }
    if (!wghMemPtr || !wghMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Weight memory was undefined.");
    }
    MemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getSrcMemoryAtPort(2);
        if (!biasMemPtr || !biasMemPtr->isDefined()) {
            THROW_CPU_NODE_ERR("Input memory is undefined.");
        }
    }

    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }

    DnnlMemoryDescCPtr inMemoryDesc = srcMemPtr->getDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr weightMemoryDesc = wghMemPtr->getDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outMemoryDesc = dstMemPtr->getDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr biasDesc;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->getDescWithType<DnnlMemoryDesc>();
    }

    auto initPrimitiveAttr = [&]() {
        dnnl::primitive_attr attr;
        if (preferLegacyZeroPoint) {
            addLegacyZeroPoints(attr);
        } else {
            addZeroPoints(attr);
        }
        setPostOps(attr, outMemoryDesc->getShape().getStaticDims(), preferLegacyPostOps, true);
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        return std::make_shared<dnnl::primitive_attr>(std::move(attr));
    };

    AttrPtr pAttrLocal;

    if (isDynamicNode()) {
        if (!pAttr || withSum) {
            pAttr = initPrimitiveAttr();
        }
        pAttrLocal = pAttr;
    } else {
        pAttrLocal = initPrimitiveAttr();
    }

    updatePadding();
    ConvKey key = {inMemoryDesc,
                   weightMemoryDesc,
                   biasDesc,
                   outMemoryDesc,
                   stride,
                   dilation,
                   paddingL,
                   paddingR,
                   *pAttrLocal,
                   selected_pd->getImplementationType(),
                   getParentEdgeAt(1)->getParent()->isConstant()};

    auto engine = getEngine();
    auto convAlg = baseConvAlgorithm;
    auto builder = [&engine, convAlg](const ConvKey& key) -> executorPtr {
        // remove the requirement on weight memory layout to let primitive
        // report the best layout for weight to be reordered dynamically at runtime
        auto wghDescAny =
            dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp1->getShape().getStaticDims()),
                               deriveWeightDataType(key.inp0->getDataType()),
                               memory::format_tag::any);
        auto createDnnlConvDesc = [](const dnnl::engine& engine,
                                     const dnnl::memory::desc& srcDesc,
                                     const dnnl::memory::desc& wghDesc,
                                     const dnnl::memory::desc& dstDesc,
                                     const DnnlMemoryDescCPtr& biasDescPtr,
                                     const std::vector<size_t>& stride,
                                     const std::vector<ptrdiff_t>& dilation,
                                     const std::vector<ptrdiff_t>& paddingL,
                                     const std::vector<ptrdiff_t>& paddingR,
                                     dnnl::algorithm alg,
                                     const dnnl::primitive_attr& attr) -> dnnl::primitive_desc {
            dnnl::memory::desc dnnlBiasDesc;
            if (biasDescPtr) {
                dnnlBiasDesc = biasDescPtr->getDnnlDesc();
            }

            return createDescriptorInternal(engine,
                                            srcDesc,
                                            wghDesc,
                                            dnnlBiasDesc,
                                            dstDesc,
                                            (biasDescPtr != nullptr),
                                            stride,
                                            dilation,
                                            paddingL,
                                            paddingR,
                                            alg,
                                            attr);
        };

        dnnl::primitive_desc prim_desc = createDnnlConvDesc(engine,
                                                            key.inp0->getDnnlDesc(),
                                                            wghDescAny,
                                                            key.out->getDnnlDesc(),
                                                            key.bias,
                                                            key.stride,
                                                            key.dilation,
                                                            key.paddingL,
                                                            key.paddingR,
                                                            convAlg,
                                                            key.attr);

        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, key.implType);

        if (found) {
            return std::make_shared<ConvolutionExecutor>(prim_desc,
                                                         key.inp0->getDnnlDesc(),
                                                         key.inp1->getDnnlDesc(),
                                                         key.out->getDnnlDesc(),
                                                         engine,
                                                         key.constWeight);
        }

        // primitive desc with proper implementation type not found, use the first available
        auto inDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp0->getShape().getStaticDims()),
                                         key.inp0->getDataType(),
                                         memory::format_tag::any);
        auto outDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.out->getShape().getStaticDims()),
                                          key.out->getDataType(),
                                          memory::format_tag::any);

        auto reorderConvDesc = createDnnlConvDesc(engine,
                                                  inDesc,
                                                  wghDescAny,
                                                  outDesc,
                                                  key.bias,
                                                  key.stride,
                                                  key.dilation,
                                                  key.paddingL,
                                                  key.paddingR,
                                                  convAlg,
                                                  key.attr);

        // unable to create a primitive desc
        if (!reorderConvDesc) {
            return nullptr;
        }

        if (key.attr.get()->post_ops_.count(dnnl::impl::primitive_kind::sum)) {
            return std::make_shared<ConvolutionSumExecutor>(reorderConvDesc,
                                                            key.inp0->getDnnlDesc(),
                                                            key.inp1->getDnnlDesc(),
                                                            key.out->getDnnlDesc(),
                                                            engine,
                                                            key.constWeight);
        }

        return std::make_shared<ConvolutionExecutor>(reorderConvDesc,
                                                     key.inp0->getDnnlDesc(),
                                                     key.inp1->getDnnlDesc(),
                                                     key.out->getDnnlDesc(),
                                                     engine,
                                                     key.constWeight);
    };

    auto prevExecPtr = execPtr;
    execPtr = nullptr;
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;

    if (!execPtr) {
        THROW_CPU_NODE_ERR("Primitive descriptor was not found");
    }

    primArgs[DNNL_ARG_SRC] = srcMemPtr->getPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->getPrimitive();

    if (key.constWeight) {
        // const weight preparation/reordering needs to be done once at next execution
        // when the input weight data is guaranteed to be ready (considering possible const-folding
        // subgraphs inserted between constant weight node and conv)
        auto it = primArgs.find(DNNL_ARG_WEIGHTS);
        if (it == primArgs.end() || !prevExecPtr ||
            !execPtr->getWeightDesc()->isCompatible(*(prevExecPtr->getWeightDesc()))) {
            primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(execPtr->getWeightDesc())->getPrimitive();
        }
    } else {
        // non-const weight will be reordered by executor on every exec
        primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->getPrimitive();
    }

    if (withBiases) {
        primArgs[DNNL_ARG_BIAS] = biasMemPtr->getPrimitive();
    }

    if (preferLegacyZeroPoint) {
        appendLegacyZeroPointsArgs();
    } else {
        appendZeroPointsArgs();
    }

    Node::appendPostOpArgs(*pAttrLocal, primArgs, convPostOpsArgs[preferLegacyPostOps]);

    auto scratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();

#ifdef CPU_DEBUG_CAPS
    auto pd = execPtr->getPrimitiveDesc();
    DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
#endif
}

Convolution::ConvolutionExecutor::ConvolutionExecutor(const dnnl::primitive_desc& pd,
                                                      const dnnl::memory::desc& inMemDesc,
                                                      const dnnl::memory::desc& weightMemDesc,
                                                      const dnnl::memory::desc& outMemDesc,
                                                      const dnnl::engine& engine,
                                                      bool constWeight)
    : DnnlExecutor(pd) {
    if (inMemDesc != getDnnlSrcDesc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, getDnnlSrcDesc(), engine)});
    }

    if (!constWeight && weightMemDesc != getDnnlWeightDesc()) {
        // const weight will be reordered at first execution
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, getDnnlWeightDesc(), engine)});
    }

    if (outMemDesc != getDnnlDstDesc()) {
        outputReorders.insert({DNNL_ARG_DST, IntermReorder(getDnnlDstDesc(), outMemDesc, engine)});
    }
}

Convolution::ConvolutionSumExecutor::ConvolutionSumExecutor(const dnnl::primitive_desc& pd,
                                                            const dnnl::memory::desc& inMemDesc,
                                                            const dnnl::memory::desc& weightMemDesc,
                                                            const dnnl::memory::desc& outMemDesc,
                                                            const dnnl::engine& engine,
                                                            bool constWeight)
    : DnnlExecutor(pd) {
    if (inMemDesc != getDnnlSrcDesc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, getDnnlSrcDesc(), engine)});
    }

    if (!constWeight && weightMemDesc != getDnnlWeightDesc()) {
        // const weight will be reordered at first execution
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, getDnnlWeightDesc(), engine)});
    }

    if (outMemDesc != getDnnlDstDesc()) {
        // In the case of fusing sum, we have to reorder the output data before executing the primitive,
        // since the output data are used as an accumulator for the covolution computations.
        inputReorders.insert({DNNL_ARG_DST, IntermReorder(outMemDesc, getDnnlDstDesc(), engine)});
        outputReorders.insert({DNNL_ARG_DST, IntermReorder(getDnnlDstDesc(), outMemDesc, engine)});
    }
}

void Convolution::ConvolutionSumExecutor::reorder_exec(std::unordered_map<int, dnnl::memory> primArgs,
                                                       const dnnl::stream& strm) {
    auto outputMem = primArgs.at(DNNL_ARG_DST);
    for (auto& inReorder : inputReorders) {
        if (primArgs.count(inReorder.first)) {
            dnnl::memory memDst(inReorder.second.getDstDesc(), strm.get_engine());
            inReorder.second.exec(primArgs[inReorder.first], memDst, strm);
            primArgs[inReorder.first] = memDst;
        } else {
            OPENVINO_THROW("DnnlExecutor has reorder for input ", inReorder.first, ", but doesn't have source memory");
        }
    }
    execPrim.execute(strm, primArgs);
    if (!outputReorders.empty()) {
        outputReorders.at(DNNL_ARG_DST).exec(primArgs.at(DNNL_ARG_DST), outputMem, strm);
    }
}

void Convolution::execute(const dnnl::stream& strm) {
    if (!execPtr) {
        THROW_CPU_NODE_ERR("executor is not compiled");
    }

    execPtr->exec(primArgs, strm);
}

void Convolution::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
    if (withSumBroadcast) {
        if (!subgraph) {
            THROW_CPU_NODE_ERR("Fused ops subgraph has not been created");
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
        convOutMem->load(outMem, true, false);
    }
}

void Convolution::updatePadding() {
    // update padding.
    if (isDynamicNode() && autoPadding) {
        paddingL = shapeInference->get_pads_begin();
        paddingR = shapeInference->get_pads_end();
    }
}

void Convolution::redefineOutputMemory(const std::vector<VectorDims>& newOutputShapes) {
    if (withSum) {
        const size_t sumPortNum = getParentEdges().size() - 1;
        const auto& sumInpMem = getParentEdgeAt(sumPortNum)->getMemory();
        if (newOutputShapes.front() != sumInpMem.getStaticDims()) {
            withSumBroadcast = true;

            auto inp0 = subgraph->getInput(0);
            inp0->redefineOutputMemory(newOutputShapes);

            auto inp1 = subgraph->getInput(1);
            inp1->redefineOutputMemory({sumInpMem.getStaticDims()});
            // here we postpone output memory reallocation due to the fact that it is the same memory with the sum
            // second input
            return;
        }
        withSumBroadcast = false;
    }
    Node::redefineOutputMemory(newOutputShapes);
}

MemoryDescPtr Convolution::getSumMemDesc(const primitive_desc& primitive_desc_it) {
    if (getOutputShapeAtPort(0).isDynamic()) {
        // When we set input shape with ranged dims, sum node input shape maybe mismatch with output shape, we just
        // change ranged min value to 1 to meet this case. For example: Output shape = {1, 160, {128, 256}, {128, 256}}
        // Sum input shape = {1, 160, 1, 1}
        // Update sum shape to {1, 160, {1, 256}, {1, 256}}
        auto shape = getOutputShapeAtPort(0);
        auto sumShape = getInputShapeAtPort(getParentEdges().size() - 1);
        Shape finalShape = shape;
        if (shape.getRank() == sumShape.getRank()) {
            auto sumDims = sumShape.getDims();
            auto minDims = shape.getMinDims();
            auto maxDims = shape.getMaxDims();
            for (size_t i = 0; i < maxDims.size(); i++) {
                if ((maxDims[i] > minDims[i]) && sumDims[i] == 1) {
                    minDims[i] = 1;
                }
            }
            finalShape = Shape(minDims, maxDims);
        }

        return DnnlExtensionUtils::makeUndefinedDesc(primitive_desc_it.dst_desc(0), finalShape);
    }
    return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.dst_desc(0));
}

MemoryPtr Convolution::getOutputMemory() const {
    if (withSumBroadcast) {
        if (!subgraph) {
            THROW_CPU_NODE_ERR("Fused ops subgraph has not been created");
        }
        auto inp0 = subgraph->getInput(0);
        return inp0->getDstMemoryAtPort(0);
    }
    return getDstMemoryAtPort(0);
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
    Node::addFusedNode(fusingNode);
}

void Convolution::appendLegacyZeroPointsArgs() {
    if (legacyInputZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = legacyInputZeroPointsMemPtr->getPrimitive();
    }
    if (legacyWeightsZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = legacyWeightsZeroPointsMemPtr->getPrimitive();
    }
    if (legacyOutputCompensationMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST] = legacyOutputCompensationMemPtr->getPrimitive();
    }
}

void Convolution::appendZeroPointsArgs() {
    if (stockInputZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = stockInputZeroPointsMemPtr->getPrimitive();
    }
}

void Convolution::initializeInputZeroPoints(const uint8_t* inputZpData, const size_t inputZpSize) {
    if (!inputZeroPoints.empty() || !legacyInputZeroPoints.empty()) {
        THROW_CPU_NODE_ERR("input zero point is not empty");
    }
    if (inputZpSize) {
        inputZeroPointType = zpType::PerTensor;
    }
    for (size_t j = 0; j < inputZpSize; j++) {
        legacyInputZeroPoints.push_back(inputZpData[j]);
        if (inputZpData[j] != inputZpData[0]) {
            inputZeroPointType = zpType::PerChannel;
        }
    }
    // Only enable per-tensor zero point on avx512-amx and avx512-core-vnni, avx2_vnni_2.
    // avx2_vnni is not enabled per-tensor z because of perf regression brgconv with per-tensor zpcompared with jit
    // per-channel zp If zero point is pertensor, both legacy zp and stock zp would be passed into conv node. The conv
    // node would determine how to create post-ops attribute and prioritize to choose final onednn kernel.
    if (inputZeroPointType == zpType::PerTensor && (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx) ||
                                                    impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_vnni) ||
                                                    impl::cpu::x64::mayiuse(impl::cpu::x64::avx2_vnni_2))) {
        inputZeroPoints.push_back(static_cast<int32_t>(inputZpData[0]));
    } else {
        inputZeroPointType = zpType::PerChannel;
    }
}

VectorDims Convolution::makeInputDummyShape(const Shape& inpShape) const {
    // There are a bunch of heuristics mostly aimed to guess the most appropriate oneDNN implementation, to reduce the
    // amount of the implementation mismatch and the internal reordering as a consequence.
    constexpr Dim dummyInputDim = 64;

    const size_t spatialRank = stride.size();
    const size_t filterStartIndx = weightDims.size() - spatialRank;

    VectorDims dummyInputShapeVals(inpShape.getRank(), dummyInputDim);
    dummyInputShapeVals[1] = IC;  // channels

    for (size_t i = 0; i < spatialRank; i++) {
        if (weightDims[filterStartIndx + i] > dummyInputShapeVals[2 + i]) {
            constexpr Dim dummyOutputDim = 16;
            dummyInputShapeVals[2 + i] = (dummyOutputDim - 1) * stride[i] - (paddingL[i] + paddingR[i]) +
                                         weightDims[filterStartIndx + i] +
                                         (weightDims[filterStartIndx + i] - 1) * (dilation[i]);
        }
    }
    return MemoryDescUtils::makeDummyShape(inpShape, dummyInputShapeVals).getStaticDims();
}

VectorDims Convolution::outputStaticShape() const {
    auto& outputShape = getOutputShapeAtPort(0);
    if (outputShape.isDynamic()) {
        auto inpDummyShape = makeInputDummyShape(getInputShapeAtPort(0));
        auto outputDims = shapeInferGeneric({Shape(inpDummyShape), Shape(weightDims)});
        return Shape(outputDims.front()).getStaticDims();
    }
    return outputShape.getStaticDims();
}

}  // namespace ov::intel_cpu::node
