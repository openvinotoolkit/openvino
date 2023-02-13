// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv.h"
#include "reorder.h"
#include "input.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "pooling.h"
#include "concat.h"
#include <graph.h>
#include "cpu/x64/cpu_isa_traits.hpp"
#include <string>
#include <vector>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "common/cpu_convert.h"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"
#include <common/primitive_hashing_utils.hpp>
#include <cpu/cpu_primitive.hpp>
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
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

    size_t hash() const;
    bool operator==(const ConvKey& rhs) const;
};

size_t ConvKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(ptr->getDnnlDesc().data));
        }
    }

    seed = get_vector_hash(seed, stride);
    seed = get_vector_hash(seed, dilation);
    seed = get_vector_hash(seed, paddingL);
    seed = get_vector_hash(seed, paddingR);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool ConvKey::operator==(const ConvKey &rhs) const {
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

    retVal = retVal && *attr.get() == *rhs.attr.get() && implType == rhs.implType;
    return retVal;
}

} // namespace

class Convolution::FusedSubgraph {
public:
    FusedSubgraph(const std::vector<NodePtr> &opList, const Convolution &conv, const GraphContext::CPtr context) {
        _graph = std::unique_ptr<Graph>(new Graph());

        std::unordered_set<NodePtr> nodesSet;
        std::vector<EdgePtr> edges;

        auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
            auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
            child->addEdge(edge);
            edges.push_back(edge);
            nodesSet.insert(parent);
            nodesSet.insert(child);
        };

        //Make inputs
        const auto &inpMemDesc1 = conv.getBaseMemDescAtOutputPort(0);
        auto inp0 = std::make_shared<Input>(inpMemDesc1, "inp0", "Parameter", context);
        inputs.push_back(inp0);
        const size_t sumPortNum = conv.getParentEdges().size() - 1;
        const auto &inpMemDesc2 = conv.getBaseMemDescAtInputPort(sumPortNum);
        auto inp1 = std::make_shared<Input>(inpMemDesc2, "inp1", "Parameter", context);
        inputs.push_back(inp1);

        auto itr = std::find_if(opList.begin(), opList.end(), [](const NodePtr &node) {
            if (auto eltwise = std::dynamic_pointer_cast<Eltwise>(node)) {
                return eltwise->isSpecialConvolutionAddFusing();
            }
            return false;
        });

        if (itr == opList.end())
            return;

        auto sumNode = *itr;
        addEdge(inp0, sumNode, 0, 0);
        addEdge(inp1, sumNode, 0, 1);

        //Replicate the rest of the subgraph
        auto parentItr = itr;
        while (++itr != opList.end()) {
            auto parentNode = *parentItr;
            auto currentNode = *itr;
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

        //Make output
        const auto &outMemDesc = conv.getBaseMemDescAtOutputPort(0);
        auto out = std::make_shared<Input>(outMemDesc, "out", "Result", context);
        addEdge(*parentItr, out, 0, 0);
        outputs.push_back(out);

        std::vector<NodePtr> nodes(nodesSet.begin(), nodesSet.end());

        _graph->CreateGraph(nodes, edges, context, "fused_subgraph");
    }

    std::shared_ptr<Input> getInput(size_t idx) const {
        if (idx < inputs.size()) {
            return inputs[idx];
        } else {
            IE_THROW(OutOfBounds) << "Unexpected input index in Convolution::fusedSubgraph::getInput idx=" << idx
                                  << " inputs.size()=" << inputs.size();
        }
    }

    std::shared_ptr<Input> getOutput(size_t idx) const {
        if (idx < outputs.size()) {
            return outputs[idx];
        } else {
            IE_THROW(OutOfBounds) << "Unexpected output index in Convolution::fusedSubgraph::getInput idx=" << idx
                                  << " inputs.size()=" << outputs.size();
        }
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

bool Convolution::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ngraph::is_type<ngraph::op::v1::Convolution>(op) && !ngraph::is_type<ngraph::op::v1::GroupConvolution>(op)) {
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

Convolution::Convolution(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)), withBiases(false), withSum(false), withDWConv(false),
          isGrouped(false), dw_conv_oc(0), dw_conv_ih(0), dw_conv_iw(0), dw_conv_in_dt(memory::data_type::undef),
          groupNum(1lu), IC(1), groupIC(1), groupOC(1), eltwisePrecision(Precision::FP32) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto convolutionOp = ngraph::as_type_ptr<ngraph::op::v1::Convolution>(op);
    auto groupConvolutionOp = ngraph::as_type_ptr<ngraph::op::v1::GroupConvolution>(op);

    if (convolutionOp) {
        algorithm = Algorithm::ConvolutionCommon;

        groupNum = 1;
        isGrouped = false;

        weightDims = convolutionOp->input_value(1).get_shape();

        IC = weightDims[1];
        groupIC = IC;
        groupOC = weightDims[0];

        biasesDims = { groupOC };

        for (int i = 0; i < convolutionOp->get_strides().size(); i++) {
            stride.push_back(convolutionOp->get_strides()[i]);
        }
        for (int i = 0; i < convolutionOp->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(convolutionOp->get_dilations()[i]) - 1);
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

        biasesDims = {groupOC * groupNum};

        for (int i = 0; i < groupConvolutionOp->get_strides().size(); i++) {
            stride.push_back(groupConvolutionOp->get_strides()[i]);
        }
        for (int i = 0; i < groupConvolutionOp->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(groupConvolutionOp->get_dilations()[i]) - 1);
        }
        paddingL = groupConvolutionOp->get_pads_begin();
        paddingR = groupConvolutionOp->get_pads_end();
        autoPadding = one_of(groupConvolutionOp->get_auto_pad(), ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER);
    }
}

bool Convolution::canBeExecutedInInt8() const {
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    if (!legacyInputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(1));
    if (!legacyWeightsZeroPoints.empty())
        weightsDataType = memory::data_type::s8;

    return one_of(inputDataType, memory::data_type::u8, memory::data_type::s8) && weightsDataType == memory::data_type::s8;
}

InferenceEngine::Precision Convolution::fusedEltwisePrecision(const NodePtr& fusingNode) const {
    if (sumPrc != Precision::UNSPECIFIED)
        return sumPrc;

    InferenceEngine::Precision eltwisePrecision;

    int fusingPort = fusingNode->getFusingPort();
    if (fusingPort == 0) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(1);
    } else if (fusingPort == 1) {
        eltwisePrecision = fusingNode->getOriginalInputPrecisionAtPort(0);
    } else {
        IE_THROW() << "Cannot determine Eltwise post op precision for Convolution node with name '" << getName() << "'";
    }

    return eltwisePrecision;
}

const std::vector<impl_desc_type>& Convolution::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::brgconv_avx512_amx_1x1,
        impl_desc_type::brgconv_avx512_amx,
        impl_desc_type::jit_avx512_amx_dw,
        impl_desc_type::jit_avx512_amx_1x1,
        impl_desc_type::jit_avx512_amx,
        impl_desc_type::brgconv_avx512_1x1,
        impl_desc_type::brgconv_avx512,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
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

    if (!shouldTryBrgconv) {
        // remove brgconv_avx512_amx_1x1/brgconv_avx512_amx/brgconv_avx512/brgconv_avx512_1x1
        for (auto it = priorities.begin(); it != priorities.end(); ) {
            if (((*it) & brgconv_avx512) == brgconv_avx512)
                it = priorities.erase(it);
            else
                ++it;
        }
    }

    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

void Convolution::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (!attrs.empty())
        IE_THROW() << "attrs vector is not empty '" << getName() << "'";
    bool enforceBrgconv = false;
    attrs.reserve(2);
    withBiases = getOriginalInputsNumber() == 3;

    if (!implPriorities.empty()) {
        isPrimitivesPriorityDefined = true;
        // winograd support only constant weights and bias
        isWino = std::find(implPriorities.begin(), implPriorities.end(), impl_desc_type::jit_avx512_winograd) != implPriorities.end() &&
                 dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) && !canBeExecutedInInt8() &&
                 getParentEdgeAt(1)->getParent()->isConstant() && getParentEdgeAt(1)->getParent()->getType() == Type::Input &&
                 (withBiases ? (getParentEdgeAt(2)->getParent()->isConstant() && getParentEdgeAt(2)->getParent()->getType() == Type::Input) : true);

        // AVX512 brconv may be disabled by heuristics due to performance issues. User can force it via Primitives priority mechanism.
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
            std::any_of(implPriorities.begin(), implPriorities.end(), [](const impl_desc_type& desc_type) {
                return static_cast<bool>(desc_type & impl_desc_type::brgconv_avx512);
            })) {
            shouldTryBrgconv = true;
            enforceBrgconv = true;
        }
    }

    int expectedInputEdgesNum = static_cast<int>(getOriginalInputsNumber());
    for (int i = 0; i < fusedWith.size(); i++) {
        if (fusedWith[i]->getType() == Type::Convolution) {
            expectedInputEdgesNum += static_cast<int>(fusedWith[i]->getOriginalInputsNumber()) - 1;
        }

        if (fusedWith[i]->getAlgorithm() == Algorithm::EltwiseAdd) {
            auto* eltwiseNode = dynamic_cast<Eltwise *>(fusedWith[i].get());
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                expectedInputEdgesNum++;
            }
        }
    }

    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    if (!legacyInputZeroPoints.empty())
        inputDataType = memory::data_type::u8;

    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));
    eltwisePrecision = DnnlExtensionUtils::DataTypeToIEPrecision(outputDataType);
    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
        eltwisePrecision = DnnlExtensionUtils::DataTypeToIEPrecision(outputDataType);
    }

    // We need to make sure that convolution output and second input of fused Eltwise operation
    // have equal precision sizes since they use the same physical memory. In case precisions are different we upscale to FP32.
    if (outputDataType != memory::data_type::f32 && outputDataType != memory::data_type::bf16 && withSum) {
        for (int i = 0; i < fusedWith.size(); i++) {
            if (fusedWith[i]->getAlgorithm() == Algorithm::EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<Eltwise *>(fusedWith[i].get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(fusedWith[i]);
                    if (DnnlExtensionUtils::DataTypeToIEPrecision(outputDataType).size() != eltwisePrecision.size()) {
                        eltwisePrecision = Precision::FP32;
                        outputDataType = memory::data_type::f32;
                    }
                    break;
                }
            }
        }
    }

    if (getParentEdges().size() != expectedInputEdgesNum)
        IE_THROW() << "Incorrect number of input edges for layer " << getName() << ", expected: " << expectedInputEdgesNum
                   << " actual: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    int ndims = getInputShapeAtPort(0).getRank();

    withDWConv = isFusedWith(Type::Convolution);
    if (withDWConv && isDynamicNode()) {
        IE_THROW() << "DW convolution is fused into convolution node " << getName() << " with dynamic shape.";
    }

    for (int i = 0; i < fusedWith.size(); i++) {
        auto *convolutionNode = dynamic_cast<Convolution *>(fusedWith[i].get());
        if (convolutionNode) {
            auto& inActivationDims = convolutionNode->inputShapes[0].getStaticDims();
            dw_conv_ih = inActivationDims[convolutionNode->inputShapes[0].getRank() - 2];
            dw_conv_iw = inActivationDims[convolutionNode->inputShapes[0].getRank() - 1];

            auto& outDims = convolutionNode->outputShapes[0].getStaticDims();
            dw_conv_oc = outDims[1];

            const auto &dwWeightsDims = convolutionNode->inputShapes[1].getStaticDims();
            dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 1]);
            dw_conv_kernel.push_back(dwWeightsDims[dwWeightsDims.size() - 2]);
            dw_conv_strides = convolutionNode->getStride();

            if (canBeExecutedInInt8()) {
                if (i == 0) {
                    dw_conv_in_dt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));
                } else {
                    dw_conv_in_dt = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[i - 1]->getOriginalOutputPrecisionAtPort(0));
                }
            } else {
                dw_conv_in_dt = memory::data_type::f32;
            }

            for (int j = 0; j < paddingR.size(); j++) {
                int with_group = isGrouped ? 1 : 0;
                int krn = weightDims[with_group + 2 + j];
                int src = getInputShapeAtPort(0).getStaticDims()[2 + j];
                int dst = getOutputShapeAtPort(0).getStaticDims()[2 + j];

                krn = (krn - 1)*(dilation[j] + 1) + 1;
                int calc_dst = (src - krn + paddingL[j]) / stride[j] + 1;
                paddingR[j] = (dst - calc_dst) * stride[j];
            }
        }
    }

    MemoryDescPtr in_candidate, out_candidate;
    if (canBeExecutedInInt8()) {
        //  We have to extend convolution_x8s8s32x from oneDNN to support BF16 output data type
        if (outputDataType == memory::data_type::bf16)
            outputDataType = memory::data_type::f32;
        if (eltwisePrecision == Precision::BF16)
            eltwisePrecision = Precision::FP32;
        // initTryBrgconvFlag depends on outputDataType, should be after outputDataType computed
        if (!enforceBrgconv)
            initTryBrgconvFlag();
        in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0), inputDataType,
            ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0), outputDataType,
            ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc));
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        inputDataType = (getOriginalInputPrecisionAtPort(0) == Precision::BF16
                && !(isDepthWise() && ndims == 5)) ? memory::data_type::bf16 : memory::data_type::f32;
        outputDataType = (getOriginalOutputPrecisionAtPort(0) == Precision::BF16
                && !(isDepthWise() && ndims == 5)) ? memory::data_type::bf16 : memory::data_type::f32;
        eltwisePrecision = Precision::FP32;
        for (int i = 0; i < fusedWith.size(); i++) {
            if (fusedWith[i]->getAlgorithm() == Algorithm::EltwiseAdd) {
                auto* eltwiseNode = dynamic_cast<Eltwise *>(fusedWith[i].get());
                if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                    eltwisePrecision = fusedEltwisePrecision(fusedWith[i]);
                    // TODO(amalyshe): there might be situation when convolution can be executed in BF16,
                    // output is required in FP32 but eltwise inplace tensor would be in BF16
                    // currently we forcedly change output to the BF16 that will add reoreder after the node
                    // Another situation can be when we mark output as FP32 and Eltwise asPrecison (which stand
                    // for input of inplace tensor precision) to FP32. This will add reorder for that in-place tensor
                    // bofore the fused convolution. This behaviour might be more correct regarding expected markup
                    // of the graph but performance of first and second approaches might be different. Need to verify
                    outputDataType = eltwisePrecision == Precision::BF16 ? memory::data_type::bf16 : memory::data_type::f32;
                    eltwisePrecision = DnnlExtensionUtils::DataTypeToIEPrecision(outputDataType);
                }
            }
        }
        // correction for cases of FP32 input - we do not have FP32 convolution supported BF16 output
        if (inputDataType == memory::data_type::f32
            && (outputDataType == memory::data_type::bf16 || eltwisePrecision == Precision::BF16)) {
            outputDataType = memory::data_type::f32;
            eltwisePrecision = Precision::FP32;
        }
        // initTryBrgconvFlag depends on outputDataType and eltwisePrecision.
        if (!enforceBrgconv)
            initTryBrgconvFlag();
        if (one_of(ndims, 3, 4, 5)) {
            memory::format_tag nspc = ndims == 3 ? memory::format_tag::nwc : (ndims == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc);
            memory::format_tag ncsp = ndims == 3 ? memory::format_tag::ncw : (ndims == 4 ? memory::format_tag::nchw : memory::format_tag::ncdhw);
            memory::format_tag nCsp8c = ndims == 3 ? memory::format_tag::nCw8c : (ndims == 4 ? memory::format_tag::nChw8c : memory::format_tag::nCdhw8c);
            memory::format_tag nCsp16c = ndims == 3 ? memory::format_tag::nCw16c : (ndims == 4 ? memory::format_tag::nChw16c : memory::format_tag::nCdhw16c);

            auto inputShape = getInputShapeAtPort(0);
            auto outputShape = getOutputShapeAtPort(0);

            bool acceptedFormat = inputDataType == memory::data_type::bf16;
            bool nspcAdded = false;
            acceptedFormat |= (shouldTryBrgconv && inputDataType == memory::data_type::f32);
            if (acceptedFormat && impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
                createDescriptor({ in_candidate }, { out_candidate });
                nspcAdded = true;
            }

            if (IC == 1 && groupOC == 1) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, ncsp);
                createDescriptor({ in_candidate }, { out_candidate });
            } else if (IC < 4) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp16c);
                createDescriptor({ in_candidate }, { out_candidate });
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp8c);
                createDescriptor({ in_candidate }, { out_candidate });
            } else {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nCsp16c);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp16c);
                createDescriptor({ in_candidate }, { out_candidate });
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nCsp8c);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nCsp8c);
                createDescriptor({ in_candidate }, { out_candidate });
            }

            in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, ncsp);
            out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, ncsp);
            createDescriptor({ in_candidate }, { out_candidate });

            if (!nspcAdded && (inputDataType != memory::data_type::bf16 && isNspcAvailable())) {
                in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inputShape, inputDataType, nspc);
                out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(outputShape, outputDataType, nspc);
                createDescriptor({ in_candidate }, { out_candidate });
            }
        }
    }
}

void Convolution::setPostOps(dnnl::primitive_attr& attr,
                             const VectorDims& dims,
                             bool useLegacyPostOps,
                             bool initWeights) {
    dnnl::post_ops ops;
    auto& args = convPostOpsArgs[useLegacyPostOps];
    bool isINT8 = canBeExecutedInInt8();

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, args, dims, 1, isINT8);

    DEBUG_LOG(getName(), " useLegacyPostOps=", useLegacyPostOps, " initWeights=", initWeights);

    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (node->getType() == Type::Split || node->getType() == Type::Concatenation)
            continue;

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->isSpecialConvolutionAddFusing()) {
                if (withSumBroadcast) {
                    break;
                }
                ops.append_sum(1.0, DnnlExtensionUtils::IEPrecisionToDataType(eltwisePrecision));
            } else {
                if (useLegacyPostOps) {
                    // try mapping with optimization w/o using binary postOps
                    if (eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType, false)) {
                        continue;
                    }
                    eltwiseNode->appendPostOps(ops, dims, args);
                } else {
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
                for (int j = i + 1; j < fusedWith.size(); j++) {
                    auto &nextNode = fusedWith[j];

                    auto *nextEltwiseNode = dynamic_cast<Eltwise *>(nextNode.get());
                    if (nextEltwiseNode && nextEltwiseNode->isSpecialConvolutionAddFusing()) {
                        hasSubsequentSum = true;
                    }

                    auto *nextQuantizeNode = dynamic_cast<FakeQuantize *>(nextNode.get());
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
                    continue;
                }
                // fallback to legacy
                fakeQuantizeNode->appendPostOps(ops, dims, args);
            } else {
                fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType, true, do_rounding);
            }
            continue;
        }

        auto* convolutionNode = dynamic_cast<Convolution*>(node.get());
        if (convolutionNode) {
            if (initWeights) {
                args[DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS] = getParentEdgeAt(getOriginalInputsNumber() + 0)->getMemoryPtr();
                args[DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS] = getParentEdgeAt(getOriginalInputsNumber() + 1)->getMemoryPtr();

                // todo: rewrite onto append_dw_k3s2p1
                ops.append_dw_conv(dw_conv_ih,
                                   dw_conv_iw,
                                   dw_conv_kernel[Y_AXIS],
                                   dw_conv_kernel[X_AXIS],
                                   dw_conv_strides[Y_AXIS],
                                   dw_conv_strides[X_AXIS],
                                   dnnl::memory::convert_to_c(dw_conv_in_dt));
            } else {
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

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void Convolution::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}

void Convolution::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    SetPostOpsAndZeroPoints(attrs);
    bool containJitImpl = false;

    for (auto& desc : descs) {
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        for (int i = 0; i < attrs.size(); i++) {
            auto &attr = attrs[i];
            auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
            while (static_cast<bool>(itpd)) {
                NodeConfig config;
                config.dynBatchSupport = true;
                for (size_t i = 0; i < descInputNumbers(desc); i++) {
                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    auto desc = getSrcMemDesc(itpd, i);
                    if (desc->getType() & MemoryDescType::Blocked && !isGrouped) {
                        dataConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                    } else {
                        dataConfig.setMemDesc(std::move(desc));
                    }

                    config.inConfs.push_back(dataConfig);
                }

                if (withDWConv) {
                    auto weightsPrc = DnnlExtensionUtils::IEPrecisionToDataType(dw_conv_in_dt == dnnl_u8 ? Precision::I8 : Precision::FP32);
                    auto biasPrc = memory::data_type::f32;

                    std::vector<size_t> dwWeightsDims({dw_conv_oc, 1, 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                    std::vector<size_t> dwBiasesDims({dw_conv_oc});

                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwWeightsDims), weightsPrc, memory::format_tag::Goihw8g));
                    config.inConfs.push_back(dataConfig);

                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwBiasesDims), biasPrc, memory::format_tag::x));
                    config.inConfs.push_back(dataConfig);
                 }

                for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                    PortConfig dataConfig;
                    if (withSum) {
                        dataConfig.inPlace(getParentEdges().size() - 1);
                    }

                    dataConfig.constant(false);
                    auto desc = getDstMemDesc(itpd, i);
                    if (desc->getType() & MemoryDescType::Blocked && !isGrouped) {
                        dataConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                    } else {
                        dataConfig.setMemDesc(std::move(desc));
                    }

                    config.outConfs.push_back(dataConfig);

                    if (withSum) {
                        dataConfig.inPlace(-1);
                        dataConfig.setMemDesc(getSumMemDesc(itpd)->cloneWithNewPrecision(dataConfig.getMemDesc()->getPrecision()));
                        config.inConfs.push_back(dataConfig);
                    }
                }
                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
                if (impl_type & jit)
                    containJitImpl = true;

                supportedPrimitiveDescriptors.emplace_back(config, impl_type);
                if (!itpd.next_impl())
                    break;
            }
        }
    }
}

bool Convolution::created() const {
    return getType() == Type::Convolution;
}

namespace {
std::shared_ptr<dnnl::convolution_forward::desc>
createDescriptorInternal(const dnnl::memory::desc& inputDesc,
                         const dnnl::memory::desc& weightDesc,
                         const dnnl::memory::desc& biasDesc,
                         const dnnl::memory::desc& outputDesc,
                         bool withBiases,
                         const std::vector<size_t>& stride,
                         const std::vector<ptrdiff_t>& dilation,
                         const std::vector<ptrdiff_t>& paddingL,
                         const std::vector<ptrdiff_t>& paddingR,
                         dnnl::algorithm alg) {
    std::shared_ptr<dnnl::convolution_forward::desc> conv_desc;
    try {
        if (withBiases) {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                                                          inputDesc, weightDesc, biasDesc, outputDesc,
                                                          dnnl::memory::dims(stride.begin(), stride.end()),
                                                          dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                          dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                          dnnl::memory::dims(paddingR.begin(), paddingR.end())));
        } else {
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_scoring, alg,
                                                          inputDesc, weightDesc, outputDesc,
                                                          dnnl::memory::dims(stride.begin(), stride.end()),
                                                          dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                          dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                          dnnl::memory::dims(paddingR.begin(), paddingR.end())));
        }
    } catch (...) {
        IE_THROW() << "Cannot create convolution forward descriptor";
    }
    return conv_desc;
}
} // namespace

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
        std::vector<Shape> shapes = { definedInpMemDesc->getShape(), Shape(weightDims) };
        auto outDims = shapeInferGeneric(shapes);
        definedOutMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outputDesc[0]->cloneWithNewDims(outDims.front()));
    }

    const auto& inDnnlDesc = definedInpMemDesc->getDnnlDesc();
    const auto& outDnnlDesc = definedOutMemDesc->getDnnlDesc();

    memory::data_type wdt = static_cast<memory::data_type>(inDnnlDesc.data.data_type);

    if (inDnnlDesc.data.data_type == dnnl_s8 || inDnnlDesc.data.data_type == dnnl_u8) {
        wdt = memory::data_type::s8;
    }

    dnnl::memory::desc weightDnnlDesc(DnnlExtensionUtils::convertToDnnlDims(weightDims), wdt, memory::format_tag::any);
    dnnl::memory::desc biasDnnlDesc;

    if (withBiases) {
        memory::data_type bdt = memory::data_type::f32;
        biasDnnlDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasesDims), bdt, memory::format_tag::any);
    }

    std::vector<dnnl::algorithm> algorithms;

    if (isWinograd())
        algorithms.push_back(dnnl::algorithm::convolution_winograd);
    algorithms.push_back(dnnl::algorithm::convolution_direct);

    updatePadding();
    for (auto alg : algorithms) {
        descs.emplace_back(createDescriptorInternal(inDnnlDesc, weightDnnlDesc, biasDnnlDesc, outDnnlDesc, withBiases,
                                                    stride, dilation, paddingL, paddingR, alg));
    }
}

void Convolution::addZeroPoints(dnnl::primitive_attr& attr) {
    if (inputZeroPoints.empty())
        return;
    attr.set_zero_points(DNNL_ARG_SRC, 0, inputZeroPoints);
    if (!stockInputZeroPointsMemPtr) {
        stockInputZeroPointsMemPtr.reset(new Memory(getEngine()));
        DnnlBlockedMemoryDesc memoryDesc(Precision::I32, {inputZeroPoints.size()});
        stockInputZeroPointsMemPtr->Create(memoryDesc, inputZeroPoints.data());
    }
}

void Convolution::addLegacyZeroPoints(dnnl::primitive_attr& attr) {
    if (!legacyInputZeroPoints.empty()) {
        attr.set_input_zero_points(legacyInputZeroPoints.size(), 1 << 1 /*through C dim*/);
        if (!legacyInputZeroPointsMemPtr) {
            legacyInputZeroPointsMemPtr.reset(new Memory(getEngine()));
            DnnlBlockedMemoryDesc memoryDesc(Precision::U8, {legacyInputZeroPoints.size()});
            legacyInputZeroPointsMemPtr->Create(memoryDesc, legacyInputZeroPoints.data());
        }
    }

    if (!legacyWeightsZeroPoints.empty()) {
        attr.set_weights_zero_points(legacyWeightsZeroPoints.size(), 1 << 1 /*through C dim*/);

        if (!legacyWeightsZeroPointsMemPtr) {
            legacyWeightsZeroPointsMemPtr.reset(new Memory(getEngine()));
            DnnlBlockedMemoryDesc memoryDesc(Precision::FP32, {legacyWeightsZeroPoints.size()});
            legacyWeightsZeroPointsMemPtr->Create(memoryDesc, legacyWeightsZeroPoints.data());
        }
    }

    if (!legacyOutputCompensation.empty()) {
        attr.set_output_compensations(legacyOutputCompensation.size(), 1 << 1 /*through C dim*/);

        if (!legacyOutputCompensationMemPtr) {
            legacyOutputCompensationMemPtr.reset(new Memory(getEngine()));
            DnnlBlockedMemoryDesc memoryDesc(Precision::I32, {legacyOutputCompensation.size()});
            legacyOutputCompensationMemPtr->Create(memoryDesc, legacyOutputCompensation.data());
        }
    }
}

//See the src/plugins/intel_cpu/src/docs/convPostOps.md for details
void Convolution::SetPostOpsAndZeroPoints(std::vector<dnnl::primitive_attr> &attrs) {
    // attr[0] - Legacy post ops + Legacy zero points.
    attrs.resize(1);
    auto outputShape = outputStaticShape();
    setPostOps(attrs[0], outputShape, true);
    addLegacyZeroPoints(attrs[0]);
    //dw-conv would be fused into conv only on AVX2 platform. no need attr[1]. Avoid extra useless attribute.
    if (attrs[0].get_post_ops().get()->find(dnnl::impl::primitive_kind::convolution) != -1) {
        return;
    }
    //no matter whether shouldTryBrgconv is true, 1 attribute is enough. Avoid duplicated attribute
    if (inputZeroPointType == zpType::None && attrs[0].get_post_ops().get()->find(dnnl::impl::primitive_kind::depthwise) == -1 &&
        attrs[0].get_post_ops().get()->find(dnnl::impl::primitive_kind::quantization) == -1) {
        return;
    }
    // Per channel zero point can only supported on attr[0].Avoid extra useless attribute.
    if (inputZeroPointType == zpType::PerChannel)
        return;
    if (!shouldTryBrgconv)
        return;
    // Try 2 attributes. Consider the shouldTRyBrgconv could be set via RTinfo to enforce brgconv.
    attrs.resize(2);
    if (inputZeroPointType == zpType::PerTensor && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx))
        //WR to ONEDNN limitation. attr[1] - legacy post ops + stock zero point.
        //@todo:Unify to use binary postops+stock zero point when limitation is fixed.
        //For now, have to adapt to JIT_AMX kernel for performance.
        setPostOps(attrs[1], outputShape, true);
    else
        setPostOps(attrs[1], outputShape, false);
    addZeroPoints(attrs[1]);
}

void Convolution::initDescriptor(const NodeConfig& config) {
    auto *selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }

    // Strided blobs feature support.
    // Works only for FP32 convolutions for now.
    bool isStridedBlobsSupported = true;

    // TODO [NM]: refactor via using global executionPrecision.
    if (canBeExecutedInInt8()) {
        isStridedBlobsSupported = false;
    }

    if (isStridedBlobsSupported) {
        createDescriptor({config.inConfs[0].getMemDesc()}, {config.outConfs[0].getMemDesc()});
    }

    auto rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;

    bool containJitImpl = false;

    for (size_t i = 0; i < descs.size(); i++) {
        auto& desc = descs[i];
        if (containJitImpl && isPossibleToSkipInitConfig(desc))
            continue;
        //Attributes support level differs in fork onednn.
        for (int n = 0; n < attrs.size(); n++) {
            auto &attr = attrs[n];
            auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
            while (static_cast<bool>(itpd)) {
                NodeConfig cfg;
                cfg.dynBatchSupport = true;
                for (size_t j = 0; j < descInputNumbers(desc); j++) {
                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(getSrcMemDesc(itpd, j));
                    cfg.inConfs.push_back(dataConfig);
                }

                if (withDWConv) {
                    auto weightsPrc = DnnlExtensionUtils::IEPrecisionToDataType(dw_conv_in_dt == dnnl_u8 ? Precision::I8 : Precision::FP32);
                    auto biasPrc = memory::data_type::f32;

                    std::vector <size_t> dwWeightsDims({dw_conv_oc, 1, 1, dw_conv_kernel[Y_AXIS], dw_conv_kernel[X_AXIS]});
                    std::vector <size_t> dwBiasesDims({dw_conv_oc});

                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwWeightsDims), weightsPrc, memory::format_tag::Goihw8g));
                    cfg.inConfs.push_back(dataConfig);

                    dataConfig.setMemDesc(std::make_shared<DnnlBlockedMemoryDesc>(Shape(dwBiasesDims), biasPrc, memory::format_tag::x));
                    cfg.inConfs.push_back(dataConfig);
                }

                for (size_t j = 0; j < descOutputNumbers(desc); j++) {
                    PortConfig dataConfig;
                    dataConfig.inPlace(-1);
                    dataConfig.constant(false);
                    dataConfig.setMemDesc(getDstMemDesc(itpd, j));
                    if (withSum) {
                        auto eltwiseConfig = dataConfig;
                        eltwiseConfig.setMemDesc(eltwiseConfig.getMemDesc()->cloneWithNewPrecision(eltwisePrecision));
                        cfg.inConfs.push_back(eltwiseConfig);
                        dataConfig.inPlace(getParentEdges().size() - 1);
                    }

                    cfg.outConfs.push_back(dataConfig);
                }
                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
                if (impl_type & jit)
                    containJitImpl = true;

                if (selected_count == selectedPrimitiveDescriptorIndex) {
                    if (impl_type != selectedPD->getImplementationType()) {
                        IE_THROW() << getName() << " expected selectedPD impl_type: " << impl_type_to_string(selectedPD->getImplementationType())
                                    << " but got: " << impl_type_to_string(impl_type);
                    }
                    rightConfig = cfg;
                    //attr[0] for legacy post ops;
                    //attr[1] is mostly for binaryPostops except when having per-tensor zp on AMX.
                    preferLegacyPostOps = (n == 0 || (n == 1 && (inputZeroPointType == zpType::PerTensor) &&
                                                    dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)));
                    //attr[0] for legacy zero point.
                    //attr[1] for stock per-tensor zero point.
                    preferLegacyZeroPoint = (n == 0);
                }
                if (i == descs.size() - 1 && isStridedBlobsSupported) {
                    if (impl_type == selectedPD->getImplementationType()) {
                        rightConfig = config;
                    }
                }
                selected_count++;
                if (!itpd.next_impl())
                    break;
            }
        }
    }
    selectedPD->setConfig(rightConfig);
}

void Convolution::filterSupportedPrimitiveDescriptors() {
    Node::filterSupportedPrimitiveDescriptors();
    // We also need to filter descs in Convolution node
    filterSupportedDescriptors();
}

void Convolution::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            IE_THROW() << "Incorrect number of input or output memory formats for Convolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                auto src_tdesc = DnnlExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::convolution_forward::desc>(*itd)->data.src_desc);
                isSuitableDesc &= src_tdesc->isSame(inputMemoryFormatsFilter[0]);
            }
            if (!outputMemoryFormatsFilter.empty()) {
                auto dst_tdesc = DnnlExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::convolution_forward::desc>(*itd)->data.dst_desc);
                isSuitableDesc &= dst_tdesc->isSame(outputMemoryFormatsFilter[0]);
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool Convolution::isPossibleToSkipInitConfig(DnnlDesriptor &desc) const {
    //  WA: In some cases, we can predict in advance the type of primitive that will be called in the future.
    //  In particular, isPossibleToSkipInitConfig() checks whether we can skip the creation of primitives with
    //  gemm implementation, which significantly increase the network load time.
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty())
        return false;

    if (isPrimitivesPriorityDefined)
        return false;

    //  Here we check that we will not delete jit_planar_conv primitive by mistake.
    //  It requires:
    //      1) strides equal 1;
    //      2) not grouped;
    //      3) first dim of weights is not 1.
    bool isPossibleJitPlanar = true;
    if (isGrouped || weightDims[0] != 1)
        isPossibleJitPlanar = false;
    for (int i = 0; i < stride.size(); i++)
        if (stride[i] != 1)
            isPossibleJitPlanar = false;

    std::shared_ptr<dnnl::convolution_forward::desc> convDesc(desc);
    auto srcMemDesc = DnnlExtensionUtils::makeDescriptor(convDesc->data.src_desc);
    auto dstMemDesc = DnnlExtensionUtils::makeDescriptor(convDesc->data.dst_desc);
    auto srcDataType = convDesc->data.src_desc.data_type;
    auto dstDataType = convDesc->data.dst_desc.data_type;
    bool isPlanarFloatConv = srcMemDesc->hasLayoutType(LayoutType::ncsp)
                             && dstMemDesc->hasLayoutType(LayoutType::ncsp)
                             && srcDataType == memory::data_type::f32
                             && dstDataType == memory::data_type::f32;

    return !isPossibleJitPlanar && isPlanarFloatConv;
}

std::shared_ptr<MemoryDesc> Convolution::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : primitive_desc_it.src_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

bool Convolution::canFuse(const NodePtr& node) const {
    return canFuseSimpleOperation(node);
}

dnnl::memory Convolution::getWeights() const {
    return getParentEdgeAt(1)->getMemory().GetPrimitive();
}

void Convolution::setDynamicBatchLim(int lim) {
    if (!execPtr) {
        IE_THROW() << "Can't set dynamic batch for Convolution node with name: " << getName() << ", because executor is not compiled";
    }
    if (execPtr->needReordering()) {
        IE_THROW() << "Can't execute Convolution node with dynamic batch via executor with reorders";
    }
    Node::setDynamicBatchLim(lim);
}

dnnl::memory Convolution::getBias() const {
    return getParentEdgeAt(2)->getMemory().GetPrimitive();
}

InferenceEngine::Precision Convolution::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
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
        // it was empirically observed that the nspc convolutions perform much slower than the blocked ones if the channels number more than the specific value
        size_t spatialRank = ndims - 2; //two means batch dim plus channels dim

        bool is1x1 = false;

        if (!isGrouped) {
            auto weightDimsReversItr = weightDims.crbegin();
            auto strideReversItr = stride.crbegin();
            auto paddingLreversItr = paddingL.crbegin();
            auto paddingRreversItr = paddingR.crbegin();

            for (size_t i = 0; i < spatialRank; ++i) {
                is1x1 = true
                        && *(weightDimsReversItr++) == 1
                        && *(strideReversItr++) == 1
                        && *(paddingLreversItr++) == 0
                        && *(paddingRreversItr++) == 0;
            }
        }

        // if the activation field size is 1x1 the avx512 1x1 nspc convolution pollutes caches so that the layer after the convolution performs slow
        if (mayiuse(impl::cpu::x64::avx512_core) && is1x1) {
            auto end = inpDims.rbegin();
            std::advance(end, spatialRank);
            if (std::all_of(inpDims.rbegin(), end, [](size_t x) { return dimsEqualStrong(1, x); })) {
                return false;
            }
        }

        unsigned thresholdNumChannels = 128u; // for avx and below
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
            // SSE41 nspc convolutions do not support ic and oc tails yet and the blocked implementation will be much better than gemm
            if ((IC % 8) || (OC % 8)) {
                return false;
            }
        }
    }

    return true;
}

InferenceEngine::Blob::Ptr Convolution::createInternalBlob(InferenceEngine::SizeVector dims, size_t edgeNum, bool isGrouped) {
    const auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(edgeNum)->getParent());
    if (!constNode) {
        IE_THROW() << "Cannot cast " << edgeNum << " input to Input node for " << getName() << ".";
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto const elementsCount = blb->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, dims, getWeightsLayoutByDims(dims, isGrouped));

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
    internalBlob->allocate();

    if (internalBlob->size() != elementsCount) {
        IE_THROW() << "Created internal blob and const blob has different size for node: " << getName() << ".";
    }

    cpu_convert(blb->GetPtr(),
                internalBlob->buffer(),
                DnnlExtensionUtils::DataTypeToIEPrecision(blb->GetDataType()),
                internalBlob->getTensorDesc().getPrecision(),
                elementsCount);

    return internalBlob;
}

void Convolution::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto wghMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
    auto dstMemPtr = getOutputMemory();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory was not allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory was not allocated.";
    if (!wghMemPtr || !wghMemPtr->isAllocated())
        IE_THROW() << "Weight memory was not allocated.";
    MemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory didn't allocate.";
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    DnnlMemoryDescCPtr inMemoryDesc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr weightMemoryDesc = wghMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outMemoryDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr biasDesc;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
    }

    auto initPrimitiveAttr = [&]() {
        dnnl::primitive_attr attr;
        if (preferLegacyZeroPoint)
            addLegacyZeroPoints(attr);
        else
            addZeroPoints(attr);
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
                   selected_pd->getImplementationType()};

    auto engine = getEngine();
    auto builder = [&engine](const ConvKey& key) -> executorPtr {
        auto createDnnlConvDesc = [](const dnnl::memory::desc& srcDesc,
                                     const dnnl::memory::desc& wghDesc,
                                     const dnnl::memory::desc& dstDesc,
                                     DnnlMemoryDescCPtr biasDescPtr,
                                     const std::vector<size_t>& stride,
                                     const std::vector<ptrdiff_t>& dilation,
                                     const std::vector<ptrdiff_t>& paddingL,
                                     const std::vector<ptrdiff_t>& paddingR,
                                     dnnl::algorithm alg) -> std::shared_ptr<DnnlDesriptor> {
            dnnl::memory::desc dnnlBiasDesc;
            if (biasDescPtr) {
                // WA to align IR bias representation (3 to 5 rank tensors) to oneDNN representation (1 rank tensor)
                dnnlBiasDesc = biasDescPtr->getDnnlDesc().reshape({dstDesc.dims()[1]});
            }

            return std::make_shared<DnnlDesriptor>(createDescriptorInternal(srcDesc,
                                                                            wghDesc,
                                                                            dnnlBiasDesc,
                                                                            dstDesc,
                                                                            (biasDescPtr != nullptr),
                                                                            stride,
                                                                            dilation,
                                                                            paddingL,
                                                                            paddingR,
                                                                            alg));
        };

        const auto alg = (key.implType & impl_desc_type::winograd) ? dnnl::algorithm::convolution_winograd : dnnl::algorithm::convolution_direct;
        std::shared_ptr<DnnlDesriptor> desc = createDnnlConvDesc(key.inp0->getDnnlDesc(),
                                                                      key.inp1->getDnnlDesc(),
                                                                      key.out->getDnnlDesc(),
                                                                      key.bias,
                                                                      key.stride,
                                                                      key.dilation,
                                                                      key.paddingL,
                                                                      key.paddingR,
                                                                      alg);

        auto itpd = desc->createPrimitiveDescriptorIterator(engine, key.attr);

        executorPtr execPtr = nullptr;
        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                auto prim_desc = convolution_forward::primitive_desc(itpd.get());
                execPtr = std::make_shared<ConvolutionExecutor>(prim_desc,
                                                                key.inp0->getDnnlDesc(),
                                                                key.inp1->getDnnlDesc(),
                                                                key.out->getDnnlDesc(),
                                                                engine);
                break;
            }

            if (!itpd.next_impl()) {
                break;
            }
        }

        if (!execPtr) {
            auto inDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp0->getShape().getStaticDims()),
                                                                                           key.inp0->getDataType(),
                                                                                           memory::format_tag::any);
            auto wghDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp1->getShape().getStaticDims()),
                                                                                        key.inp1->getDataType(),
                                                                                        memory::format_tag::any);
            auto outDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.out->getShape().getStaticDims()),
                                                                                        key.out->getDataType(),
                                                                                        memory::format_tag::any);

            std::shared_ptr<DnnlDesriptor> reorderConvDesc = createDnnlConvDesc(inDesc,
                                                                                  wghDesc,
                                                                                  outDesc,
                                                                                  key.bias,
                                                                                  key.stride,
                                                                                  key.dilation,
                                                                                  key.paddingL,
                                                                                  key.paddingR,
                                                                                  dnnl::algorithm::convolution_direct);

            auto reordItpd = reorderConvDesc->createPrimitiveDescriptorIterator(engine, key.attr);
            if (static_cast<bool>(reordItpd)) {
                auto prim_desc = convolution_forward::primitive_desc(reordItpd.get());
                execPtr = std::make_shared<ConvolutionExecutor>(prim_desc,
                                                                key.inp0->getDnnlDesc(),
                                                                key.inp1->getDnnlDesc(),
                                                                key.out->getDnnlDesc(),
                                                                engine);
            }
        }

        return execPtr;
    };

    execPtr = nullptr;
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;

    if (execPtr) {
        primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
        primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->GetPrimitive();
        primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();

        if (withBiases) {
            primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
        }

        if (preferLegacyZeroPoint)
            appendLegacyZeroPointsArgs();
        else
            appendZeroPointsArgs();

        Node::appendPostOpArgs(*pAttrLocal, primArgs, convPostOpsArgs[preferLegacyPostOps]);

        auto pd = execPtr->getPrimitiveDesc();
        auto scratchpadMem = getScratchPadMem(pd);
        primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            DEBUG_LOG("verbose##", getName(), "##", pd->info(), "\n");
        }
#endif
    } else {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }
}

Convolution::ConvolutionExecutor::ConvolutionExecutor(const dnnl::convolution_forward::primitive_desc& pd,
                                                                const dnnl::memory::desc& inMemDesc,
                                                                const dnnl::memory::desc& weightMemDesc,
                                                                const dnnl::memory::desc& outMemDesc,
                                                                const dnnl::engine& engine) {
    execPrim = dnnl::convolution_forward(pd);

    if (inMemDesc != pd.src_desc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, pd.src_desc(), engine)});
    }

    if (weightMemDesc != pd.weights_desc()) {
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, pd.weights_desc(), engine)});
    }

    if (outMemDesc != pd.dst_desc()) {
        outputReorders.insert({DNNL_ARG_DST, IntermReorder(pd.dst_desc(), outMemDesc, engine)});
    }
}

void Convolution::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute Convolution node with name: " << getName() << ", because executor is not compiled";
    }

    execPtr->exec(primArgs, strm);
}

void Convolution::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
    if (withSumBroadcast) {
        if (!subgraph) {
            IE_THROW(Unexpected) << "Fused ops subgraph has not been created in " << getTypeStr() << " with name " << getName();
        }
        const size_t sumPortNum = getParentEdges().size() - 1;
        const auto& sumInpMem = getParentEdgesAtPort(sumPortNum).front()->getMemory();
        auto inp1 = subgraph->getInput(1);
        inp1->getChildEdgesAtPort(0).front()->getMemoryPtr()->setDataHandle(sumInpMem.GetData());

        subgraph->infer();

        auto out = subgraph->getOutput(0);
        const auto& outMem = out->getParentEdgesAtPort(0).front()->getMemory();
        auto convOutMem = getChildEdgesAtPort(0).front()->getMemoryPtr();
        Node::redefineOutputMemory({outMem.getStaticDims()});
        convOutMem->SetData(outMem);
    }
}

void Convolution::updatePadding() {
    //update padding.
    if (isDynamicNode() && autoPadding) {
        paddingL = shapeInference->get_pads_begin();
        paddingR = shapeInference->get_pads_end();
    }
}

void Convolution::redefineOutputMemory(const std::vector<VectorDims> &newOutputShapes) {
    if (withSum) {
        const size_t sumPortNum = getParentEdges().size() - 1;
        const auto& sumInpMem = getParentEdgesAtPort(sumPortNum).front()->getMemory();
        if (newOutputShapes.front() != sumInpMem.getStaticDims()) {
            withSumBroadcast = true;
            if (!subgraph) {
                subgraph = std::make_shared<FusedSubgraph>(fusedWith, *this, context);
            }
            auto inp0 = subgraph->getInput(0);
            inp0->redefineOutputMemory(newOutputShapes);

            auto inp1 = subgraph->getInput(1);
            inp1->redefineOutputMemory({sumInpMem.getStaticDims()});
            // here we postpone output memory reallocation due to the fact that it is the same memory with the sum second input
            return;
        } else {
            withSumBroadcast = false;
        }
    }
    Node::redefineOutputMemory(newOutputShapes);
}

MemoryDescPtr Convolution::getSumMemDesc(primitive_desc_iterator &primitive_desc_it) {
    if (getOutputShapeAtPort(0).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(primitive_desc_it.dst_desc(0), getOutputShapeAtPort(0));
    }
    return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.dst_desc(0));
}

MemoryPtr Convolution::getOutputMemory() const {
    if (withSumBroadcast) {
        if (!subgraph) {
            IE_THROW(Unexpected) << "Fused ops subgraph has not been created in " << getTypeStr() << " with name " << getName();
        }
        auto inp0 = subgraph->getInput(0);
        return inp0->getChildEdgesAtPort(0).front()->getMemoryPtr();
    } else {
        return getChildEdgesAtPort(0).front()->getMemoryPtr();
    }
}

void Convolution::addFusedNode(const NodePtr &fusingNode) {
    if (Type::Eltwise == fusingNode->getType()) {
        if (fusingNode->getAlgorithm() == Algorithm::EltwiseAdd) {
            auto eltwiseNode = std::dynamic_pointer_cast<Eltwise>(fusingNode);
            if (eltwiseNode && eltwiseNode->isSpecialConvolutionAddFusing()) {
                withSum = true;
            }
        }
        if (withSum && isDynamicNode()) {
            for (size_t i = 0; i < fusingNode->getParentEdges().size(); ++i) {
                auto edge = fusingNode->getParentEdgesAtPort(i).front();
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
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = legacyInputZeroPointsMemPtr->GetPrimitive();
    }
    if (legacyWeightsZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = legacyWeightsZeroPointsMemPtr->GetPrimitive();
    }
    if (legacyOutputCompensationMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST] = legacyOutputCompensationMemPtr->GetPrimitive();
    }
}


void Convolution::appendZeroPointsArgs() {
    if (stockInputZeroPointsMemPtr != nullptr) {
        primArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = stockInputZeroPointsMemPtr->GetPrimitive();
    }
}


void Convolution::initTryBrgconvFlag() {
    // Due to performance issue, brgconv will only be enabled by default:
    // 1, static shape(dynamic shape may change weights layout if the input shape changes and cause performance issue: 86948)
    // 2, support amx except having input zero point.
    // 3, support avx512 without legacy postops/per channel zero point when avx512
    if (!isDynamicNode()) {
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
            shouldTryBrgconv = true;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            shouldTryBrgconv = true;
            // should remove after binary postops performance issue resolved
            // heuristics: if it's  avx512 ISA model && it doesn't have binary post ops or per channel zero point.
            dnnl::primitive_attr attr;
            setPostOps(attr, outputStaticShape(), false);
            const auto& ops = attr.get_post_ops();
            if (ops.get()->find(dnnl::impl::primitive_kind::binary) != -1) {
                shouldTryBrgconv = false;
            }
        }
    }
}

void Convolution::initializeInputZeroPoints(const uint8_t* inputZpData, const size_t inputZpSize) {
    if (!inputZeroPoints.empty() || !legacyInputZeroPoints.empty())
        IE_THROW() << "input zero point is not empty '" << getName() << "'";
    if (inputZpSize)
        inputZeroPointType = zpType::PerTensor;
    for (int j = 0; j < inputZpSize; j++) {
        legacyInputZeroPoints.push_back(inputZpData[j]);
        if (inputZpData[j] != inputZpData[0])
            inputZeroPointType = zpType::PerChannel;
    }
    //Only enable per-tensor zero point on avx512-amx and avx512-core.
    //If zero point is pertensor, both legacy zp and stock zp
    //would be passed into conv node. The conv node would determine how to create
    //post-ops attribute and prioritize to choose final onednn kernel.
    if (inputZeroPointType == zpType::PerTensor &&
        (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx) || impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_vnni)))
        inputZeroPoints.push_back(static_cast<int32_t>(inputZpData[0]));
}

VectorDims Convolution::makeInputDummyShape(const Shape& inpShape) const {
    // There are a bunch of heuristics mostly aimed to guess the most appropriate oneDNN implementation, to reduce the
    // amount of the implementation mismatch and the internal reordering as a consequence.
    constexpr Dim dummyInputDim = 64;

    const size_t spatialRank = stride.size();
    const size_t filterStartIndx = weightDims.size() - spatialRank;

    VectorDims dummyInputShapeVals(inpShape.getRank(), dummyInputDim);
    dummyInputShapeVals[1] = IC; //channels

    for (size_t i = 0; i < spatialRank; i++) {
        if (weightDims[filterStartIndx + i] > dummyInputShapeVals[2 + i]) {
            constexpr Dim dummyOutputDim = 16;
            dummyInputShapeVals[2 + i] = (dummyOutputDim - 1) * stride[i] -
                                (paddingL[i] + paddingR[i]) +
                                weightDims[filterStartIndx + i] +
                                (weightDims[filterStartIndx + i]- 1) * (dilation[i]);
        }
    }
    return MemoryDescUtils::makeDummyShape(inpShape, dummyInputShapeVals).getStaticDims();
}

VectorDims Convolution::outputStaticShape() const {
    auto& outputShape = getOutputShapeAtPort(0);
    if (outputShape.isDynamic()) {
        auto inpDummyShape = makeInputDummyShape(getInputShapeAtPort(0));
        auto outputDims = shapeInferGeneric({ Shape(inpDummyShape), Shape(weightDims) });
        return Shape(outputDims.front()).getStaticDims();
    }
    return outputShape.getStaticDims();
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
