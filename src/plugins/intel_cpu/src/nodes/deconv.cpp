// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv.h"

#include "eltwise.h"
#include "fake_quantize.h"
#include "input.h"
#include <dnnl_extension_utils.h>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>
#include <nodes/common/cpu_memcpy.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include <utils/shape_inference/shape_inference_ngraph.hpp>

#include <string>
#include <vector>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

using Int8DeconvDesc = dnnl::deconvolution_forward::primitive_desc;

namespace {

/**
 * Deconvolution shape inference factory. It defines the input mask depending on the existence of the `output_shape` input.
 * Since in case it exists, plugin should pass the input data to the shape inference function.
 *
 */
class DeconfolutionShapeInferFactory : public ShapeInferFactory {
public:
    DeconfolutionShapeInferFactory(std::shared_ptr<ngraph::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (m_op->get_input_size() > 2) {
            return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), PortMask(2));
        }
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), EMPTY_PORT_MASK);
    }
private:
    std::shared_ptr<ngraph::Node> m_op;
};
} // namespace

bool Deconvolution::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op) == nullptr &&
                std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op) == nullptr) {
            errorMessage = "Only opset1 ConvolutionBackpropData and GroupConvolutionBackpropData operations are supported";
            return false;
        }
        size_t ndims = op->get_input_partial_shape(0).rank().get_length();
        if ((ndims < 3) || (ndims > 5)) {
            errorMessage = "Only 3D, 4D and 5D blobs are supported as input";
            return false;
        }
        if (op->get_input_partial_shape(1).is_dynamic() || (op->get_input_size() > 2 && op->get_input_partial_shape(2).is_dynamic())) {
            errorMessage = "Doesn't support dynamic shapes for 'weights' and 'output_shape' inputs";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Deconvolution::Deconvolution(const std::shared_ptr<ngraph::Node>& op,
                             const GraphContext::CPtr context) : Node(op, context, DeconfolutionShapeInferFactory(op)) {
    deconvAttrs.layerName = getName();
    std::string errorMessage;
    errorPrefix = "Deconvolution node with name '" + deconvAttrs.layerName + "' ";
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorPrefix + errorMessage;

    const auto& weightDims = getWeightDims();

    if (auto convBackprop = std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op)) {
        algorithm = Algorithm::DeconvolutionCommon;

        deconvAttrs.IC = weightDims[0];
        deconvAttrs.OC = weightDims[1];
        deconvAttrs.expectedBiasDims  = {deconvAttrs.OC};

        deconvAttrs.groupNum = 1;
        deconvAttrs.withGroups = false;

        for (size_t i = 0; i < convBackprop->get_strides().size(); i++) {
            deconvAttrs.stride.push_back(static_cast<ptrdiff_t>(convBackprop->get_strides()[i]));
        }
        for (size_t i = 0; i < convBackprop->get_dilations().size(); i++) {
            deconvAttrs.dilation.push_back(static_cast<ptrdiff_t>(convBackprop->get_dilations()[i]) - 1);
        }
        deconvAttrs.paddingL = convBackprop->get_pads_begin();
        deconvAttrs.paddingR = convBackprop->get_pads_end();

        deconvAttrs.outputPadding = convBackprop->get_output_padding();

        deconvAttrs.autoPad = one_of(convBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
    } else if (auto groupConvBackprop = std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op)) {
        algorithm = Algorithm::DeconvolutionGrouped;

        deconvAttrs.groupNum = weightDims[0];
        deconvAttrs.IC = deconvAttrs.groupNum * weightDims[1];
        deconvAttrs.OC = deconvAttrs.groupNum * weightDims[2];
        deconvAttrs.expectedBiasDims  = {deconvAttrs.OC * deconvAttrs.groupNum};
        deconvAttrs.withGroups = deconvAttrs.groupNum > 1;
        deconvAttrs.isDW = deconvAttrs.withGroups && deconvAttrs.groupNum == deconvAttrs.OC && deconvAttrs.groupNum == deconvAttrs.IC;

        for (size_t i = 0; i < groupConvBackprop->get_strides().size(); i++) {
            deconvAttrs.stride.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_strides()[i]));
        }
        for (size_t i = 0; i < groupConvBackprop->get_dilations().size(); i++) {
            deconvAttrs.dilation.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_dilations()[i]) - 1);
        }
        deconvAttrs.paddingL = groupConvBackprop->get_pads_begin();
        deconvAttrs.paddingR = groupConvBackprop->get_pads_end();

        deconvAttrs.outputPadding = groupConvBackprop->get_output_padding();

        deconvAttrs.autoPad = one_of(groupConvBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
    }
    for (size_t i = 0; i < deconvAttrs.dilation.size(); i++) {
        deconvAttrs.kernel.push_back(weightDims[deconvAttrs.withGroups + 2 + i]);
    }

    deconvAttrs.externOutShape = inputShapes.size() == 3;
    biasPort = deconvAttrs.externOutShape ? 3 : 2;
    if (deconvAttrs.externOutShape && isDynamicNode()) {
        bool isConstOutShape = ngraph::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        if (isConstOutShape) {
            deconvAttrs.lastOutputSpatialDims = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(2))->cast_vector<int32_t>();
        }
        const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
        if (getInputShapeAtPort(2).getStaticDims()[0] != spDimsNum || (isConstOutShape && deconvAttrs.lastOutputSpatialDims.size() != spDimsNum)) {
            IE_THROW() << errorPrefix << "'output_shape' input has incorrect number of elements. Expected = " << spDimsNum;
        }
    }
    attr = std::make_shared<dnnl::primitive_attr>();
}

InferenceEngine::Blob::Ptr Deconvolution::createWeiBlobAsIO(InferenceEngine::SizeVector dims) {
    auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent());
    if (!constNode)
        IE_THROW() << "Cannot cast const input node for node " << deconvAttrs.layerName << ".";
    auto blb = constNode->getMemoryPtr();
    if (!blb)
        IE_THROW() << "Cannot get const weights blob for node " << deconvAttrs.layerName << ".";

    auto const blbSize = blb->GetSize();

    // WA: In int8 case, we are processing weights using internal blob.
    InferenceEngine::SizeVector dimsForBlockedDesc{dims};
    std::swap(dimsForBlockedDesc[deconvAttrs.withGroups + 0], dimsForBlockedDesc[deconvAttrs.withGroups + 1]);

    InferenceEngine::SizeVector orderForBlockedDesc;
    if (deconvAttrs.withGroups) {
        orderForBlockedDesc = {0, 2, 1};
    } else {
        orderForBlockedDesc = {1, 0};
    }
    for (size_t i = 2 + deconvAttrs.withGroups; i < dimsForBlockedDesc.size(); i++)
        orderForBlockedDesc.push_back(i);

    BlockingDesc blkDesc(dimsForBlockedDesc, orderForBlockedDesc);
    InferenceEngine::TensorDesc tensorDesc(DnnlExtensionUtils::DataTypeToIEPrecision(blb->GetDataType()), dims, blkDesc);

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<int8_t>(tensorDesc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    if (data == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << deconvAttrs.layerName << ".";
    size_t intBuffSize = internalBlob->byteSize();

    size_t offset = blbSize;
    if (intBuffSize < offset) {
        IE_THROW() << "Cannot create internal buffer. Buffer can be overrun.";
    }
    cpu_memcpy_s(data, intBuffSize, blb->GetPtr(), blbSize);

    return internalBlob;
}

bool Deconvolution::canBeExecutedInInt8() const {
    if (std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent()) == nullptr) {
        return false;
    }

    if (!deconvAttrs.withGroups && deconvAttrs.stride.back() > 3)
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        const auto& inMaxDims = getOutputShapeAtPort(0).getMaxDims();
        if (std::any_of(inMaxDims.begin(), inMaxDims.end(), [](Dim dim) { return dim == Shape::UNDEFINED_DIM; })) {
            return false;
        }
        // heuristicConst = 2^26
        // heuristicParam = deconvAttrs.IC^2 * SP
        size_t heuristicConst = 67108864;
        auto heuristicParam = deconvAttrs.IC * deconvAttrs.IC;
        for (size_t i = 2; i < inMaxDims.size(); i++)
            heuristicParam *= inMaxDims[i];
        if (heuristicParam > heuristicConst)
            return false;
    }

    for (size_t i = 0; i < deconvAttrs.kernel.size(); i++) {
        if (deconvAttrs.kernel[i] < deconvAttrs.stride[i])
            return false;
    }

    // not supported in oneDNN
    int channelBlock = impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core) ? 16
            : impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) ? 8 : 4;
    if (deconvAttrs.withGroups && !deconvAttrs.isDW && (deconvAttrs.IC % channelBlock != 0 || deconvAttrs.OC % channelBlock != 0))
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core) && deconvAttrs.stride.back() > 3)
        return false;

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inPrecision);

    InferenceEngine::Precision weiPrecision = getOriginalInputPrecisionAtPort(1);
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(weiPrecision);

    if (deconvAttrs.isDW && (inputDataType == dnnl_s8 || deconvAttrs.dilation.size() == 3))
        return false;

    return (inputDataType == dnnl_s8 || inputDataType == dnnl_u8) && weightsDataType == dnnl_s8;
}

bool Deconvolution::canFuse(const NodePtr& node) const {
    if (canBeExecutedInInt8())
        return canFuseSimpleOperation(node);

    return (fusedWith.empty() && node->canBePerformedAsScaleShift(this));
}

std::pair<VectorDims, VectorDims> Deconvolution::makeDummyInOutShape() {
    auto inShape = MemoryDescUtils::makeDummyShape(getInputShapeAtPort(0));
    auto outShape = getOutputShapeAtPort(0);

    if (isDynamicNode()) {
        auto inputDims = inShape.getStaticDims();
        inputDims[1] = deconvAttrs.IC;

        if (deconvAttrs.externOutShape) {
            if (deconvAttrs.lastOutputSpatialDims.empty()) {
                const auto& shape = getOutputShapeAtPort(0);
                deconvAttrs.lastOutputSpatialDims.resize(shape.getRank() - 2);

                const auto& minDims = shape.getMinDims();
                const auto& maxDims = shape.getMaxDims();
                const auto& dims = shape.getDims();
                for (size_t i = 0; i < dims.size() - 2; ++i) {
                    deconvAttrs.lastOutputSpatialDims[i] = dims[i + 2] == Shape::UNDEFINED_DIM ? std::min(maxDims[i + 2],
                                                                                              std::max(minDims[i + 2], static_cast<Dim>(64))) : dims[i + 2];
                }
            }

            const auto& origInDims = getInputShapeAtPort(0).getDims();
            const auto& origInMinDims = getInputShapeAtPort(0).getMinDims();
            const auto& origInMaxDims = getInputShapeAtPort(0).getMaxDims();
            const auto& weightDims = getWeightDims();
            const size_t wghOffset = getAlgorithm() == Algorithm::DeconvolutionGrouped ? 1 : 0;

            VectorDims paddings(deconvAttrs.paddingL.size());
            if (!deconvAttrs.autoPad) {
                for (size_t i = 0; i < paddings.size(); ++i) {
                    paddings[i] = deconvAttrs.paddingL[i] + deconvAttrs.paddingR[i];
                }
            } else {
                for (size_t i = 0; i < origInDims.size() - 2; i++) {
                    if (origInDims[i + 2] == Shape::UNDEFINED_DIM &&
                        (origInMinDims[i + 2] != 0 || origInMaxDims[i + 2] != Shape::UNDEFINED_DIM)) {
                        // if input shape is dynamic and bounded, paddings should be computed basing on the following limitations:
                        // 1. paddings must not be negative
                        // 2. the result padding must have such a value to keep the dummy dimensions inside the predefined interval
                        auto c1 = deconvAttrs.lastOutputSpatialDims[i] - deconvAttrs.outputPadding[i] - 1 -
                                    (deconvAttrs.dilation[i] + 1) * static_cast<int32_t>(weightDims[wghOffset + 2 + i] - 1);

                        if (origInMaxDims[i + 2] != Shape::UNDEFINED_DIM) {
                            auto upper_bound = deconvAttrs.stride[i] * static_cast<int32_t>(origInMaxDims[i + 2] - 1) - c1;
                            if (upper_bound < 0) {
                                IE_THROW() << errorPrefix << ": paddings for dummy shapes can't be computed";
                            }
                        }

                        auto lower_bound = deconvAttrs.stride[i] * static_cast<int32_t>(origInMinDims[i + 2] - 1) - c1;
                        if (lower_bound > 0) {
                            paddings[i] = lower_bound;
                        }
                    }
                }
            }

            for (size_t i = 0; i < inputDims.size() - 2; i++) {
                if (origInDims[2 + i] == Shape::UNDEFINED_DIM) {
                    inputDims[2 + i] = (deconvAttrs.lastOutputSpatialDims[i] - (deconvAttrs.dilation[i] + 1) *
                                        (weightDims[wghOffset + 2 + i] - 1) - 1 + paddings[i] - deconvAttrs.outputPadding[i]) /
                                        deconvAttrs.stride[i] + 1;
                }
            }
        }
        inShape = Shape(inputDims);
        outShape = Shape(shapeInferInternal(inShape.getStaticDims(), deconvAttrs.lastOutputSpatialDims));
        deconvAttrs.paddingL = shapeInference->get_pads_begin();
        deconvAttrs.paddingR = shapeInference->get_pads_end();
    }
    return {inShape.getStaticDims(), outShape.getStaticDims()};
}

std::vector<memory::format_tag> Deconvolution::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc, memory::format_tag::ntc,
                memory::format_tag::ncw, memory::format_tag::nCw8c, memory::format_tag::nCw16c };
    else if (dims.getRank() == 4)
        return {memory::format_tag::nchw, memory::format_tag::nChw8c,
                memory::format_tag::nChw16c, memory::format_tag::nhwc };
    else if (dims.getRank() == 5)
        return {memory::format_tag::ncdhw, memory::format_tag::nCdhw8c,
                memory::format_tag::nCdhw16c, dnnl::memory::format_tag::ndhwc };
    return {memory::format_tag::any};
}

void Deconvolution::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    deconvAttrs.isInt8 = canBeExecutedInInt8();
    deconvAttrs.withBiases = deconvAttrs.externOutShape ? getOriginalInputsNumber() == 4 : getOriginalInputsNumber() == 3;

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision weiPrecision = getOriginalInputPrecisionAtPort(1);
    InferenceEngine::Precision outPrecision = getOriginalOutputPrecisionAtPort(0);

    VectorDims inDims, outDims;
    std::tie(inDims, outDims) = makeDummyInOutShape();
    inShape = Shape(inDims);
    Shape outShape(outDims);
    initPaddingR(inShape, outShape);

    //ONEDNN deconvolution_fwd_t primitive can support bias fusing.
    //ONEDNN convolution_data_bwd_t can't support bias fusing.
    //Current only int8 precision choose deconvolution_fwd_t.
    if (deconvAttrs.withBiases && !deconvAttrs.isInt8) {
        IE_THROW() << errorPrefix << " supports bias fusing only for int8 execution precision";
    }

    if (deconvAttrs.isInt8) {
        // TODO: We have to extend jit_avx512_core_x8s8s32x_deconv_fwd_kernel from oneDNN to support BF16 output data type
        if (InferenceEngine::Precision::BF16 == inPrecision)
            inPrecision = InferenceEngine::Precision::FP32;
        if (InferenceEngine::Precision::BF16 == outPrecision)
            outPrecision = InferenceEngine::Precision::FP32;
    } else {
        if (!inPrecision.is_float())
            inPrecision = InferenceEngine::Precision::FP32;
        if (!outPrecision.is_float())
            outPrecision = InferenceEngine::Precision::FP32;
    }
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inPrecision);
    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outPrecision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;
    if (inputDataType == memory::data_type::f16 || outputDataType == memory::data_type::f16)
       inputDataType = outputDataType = memory::data_type::f16;
    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    if (getParentEdges().size() != (deconvAttrs.withBiases ? (biasPort + 1) : biasPort)) {
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    }
    if (getChildEdges().empty()) {
        IE_THROW() << errorPrefix << " has incorrect number of output edges";
    }

    setPostOps(*attr, outShape.getStaticDims());

    if (deconvAttrs.isInt8) {
        deconvAttrs.int8WeightDims = getWeightDims();
        //  WA: if int8 deconvolution is supported, we create internal weights blob in IO format
        std::swap(deconvAttrs.int8WeightDims[deconvAttrs.withGroups + 0], deconvAttrs.int8WeightDims[deconvAttrs.withGroups + 1]);
        internalBlobs.push_back(createWeiBlobAsIO(deconvAttrs.int8WeightDims));
        auto format = getInputShapeAtPort(0).getRank() == 5 ? dnnl::memory::format_tag::ndhwc : dnnl::memory::format_tag::nhwc;
        MemoryDescPtr in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0), inputDataType, format);
        MemoryDescPtr out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0), outputDataType, format);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
            MemoryDescPtr in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getInputShapeAtPort(0), inputDataType, format);
            MemoryDescPtr out_candidate = std::make_shared<DnnlBlockedMemoryDesc>(getOutputShapeAtPort(0), outputDataType, format);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void Deconvolution::initPaddingR(const Shape &inShape, const Shape &outShape) {
    for (size_t i = 0; i < deconvAttrs.paddingR.size(); i++) {
        int with_group = getAlgorithm() == Algorithm::DeconvolutionGrouped ? 1 : 0;
        const auto& weightDims = getWeightDims();
        int krn = weightDims[with_group + 2 + i];
        int src = outShape.getStaticDims()[2 + i];
        int dst = inShape.getStaticDims()[2 + i];

        krn = (krn - 1)*(deconvAttrs.dilation[i] + 1) + 1;
        deconvAttrs.paddingR[i] = (dst - 1) * deconvAttrs.stride[i] - (src - krn + deconvAttrs.paddingL[i]);
    }
}

void Deconvolution::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims) {
    dnnl::post_ops ops;

    // ONEDNN define the convolution forward as :
    //  [N, deconvAttrs.OC, OH, OW] = [N, deconvAttrs.IC, IH, IW]* [deconvAttrs.OC, deconvAttrs.IC, KH, KW]
    // ONEDNN define the convolution data backward as:
    //  [N, deconvAttrs.IC, OH, OW] = [N, deconvAttrs.OC, IH, IW]* [deconvAttrs.OC, deconvAttrs.IC, KH, KW]
    // So for the backward and forward convolutions, the weights dimensions definition in ONEDNN is the same.
    // deconvAttrs.OC is the conv forward output channel, deconvAttrs.IC is conv forward input channel.

    // But for the deconvolution, OC and IC are the deconv output and input channels respectively
    // ONEDNN defines the deconv OP as:
    // [N, OC, OH, OW] = [N, IC, IH, IW] * [OC, IC, KH, KW]
    // For deconv OP,  OC = deconvAttrs.IC, IC = deconvAttrs.OC.
    // Openvino per-channel weight scales are applied on deconvAttrs.IC/OC dimension.
    // So for deconvolution,
    // Weight dims in NON-Group deconv: [OC, IC, KH, KW], perchannel weight scale is applied on OC DIM
    //                                  weiScaleMaskPerChannel =  1 << 0
    // Weight dims in Group deconv:     [Group, OC, IC, KH, KW], perchannel weight scale is applied on GROUP and OC,
    //                                   weiScaleMaskPerChannel = ( 1 << 0 | 1 << 1) = 0x03
    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, 1, deconvAttrs.isInt8,
                                deconvAttrs.withGroups ? 3 : 1 << 0,  getDQScales(), deconvAttrs.withBiases);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            // TODO [DS]: change to shape from memory
            if (deconvAttrs.isInt8) {
                // deconvolution support output scales and binary postOps
                eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            } else {
                // use legacy depthwise since backprop convolution does not support binary post ops
                eltwiseNode->appendPostOps(ops, dims, postOpsArgs);
            }
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

bool Deconvolution::created() const {
    return getType() == Type::Deconvolution;
}

bool Deconvolution::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    if (deconvAttrs.externOutShape) {
        if (deconvAttrs.lastOutputSpatialDims != readOutputSpatialDims()) {
            return true;
        }
    }

    return false;
}

VectorDims Deconvolution::shapeInferInternal(const VectorDims &inDims, std::vector<int32_t> outSpDims) const {
    std::vector<std::reference_wrapper<const VectorDims>> inputShapesRefs{std::ref(inDims), std::ref(getWeightDims())};
    std::unordered_map<size_t, MemoryPtr> inputValues;
    VectorDims outSpDimsVecShape;

    auto port_mask = shapeInference->get_port_mask();
    if (port_mask) {
        for (size_t i = 0; i < inputShapes.size(); ++i) {
            if (port_mask & 1 << i) {
                if (outSpDims.size() != getInputShapeAtPort(i).getStaticDims()[0]) {
                    IE_THROW() << "Can't compute output shape for node with name: " << deconvAttrs.layerName
                            << ", because the node has 'output_shape' input, but provided output spatial dims number is incorrect";
                }
                outSpDimsVecShape = {outSpDims.size()};
                inputShapesRefs.push_back(std::cref(outSpDimsVecShape));
                CpuBlockedMemoryDesc desc(Precision::I32, Shape(outSpDimsVecShape));
                auto mem = std::make_shared<Memory>(getEngine());
                mem->Create(desc, outSpDims.data());
                inputValues[i] = mem;
                break;
            }
        }
    }

    auto result = shapeInference->infer(inputShapesRefs, inputValues);
    if (ShapeInferStatus::success != result.status) {
        IE_THROW(Unexpected) << "Unexpected shape inference result status in node of type " << getTypeStr() << " with name " << deconvAttrs.layerName;
    }
    return std::move(result.dims.back());
}

void Deconvolution::execute(dnnl::stream strm) {
    std::vector<MemoryCPtr> srcMemory;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgesAtPort(i)[0]->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgesAtPort(i)[0]->getMemoryPtr());
    }

    //TODO: need to pass post ops data
    execPtrDeconv->exec(srcMemory, dstMemory, nullptr, strm);

    if (deconvAttrs.externOutShape) {
        deconvAttrs.lastOutputSpatialDims = readOutputSpatialDims();
    }
}

Node::AttrPtr Deconvolution::makePrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims);

    return attr;
}

Node::AttrPtr Deconvolution::initPrimitiveAttr() {
    return attr;
}

void Deconvolution::createPrimitive() {
    if (deconvAttrs.isInt8) {
        VectorDims inDims, outDims;
        DnnlMemoryDescPtr inDesc;
        auto wgh_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(deconvAttrs.int8WeightDims),
                                                memory::data_type::s8, memory::format_tag::any);
        DnnlMemoryDescPtr outDesc;

        auto selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr) {
            IE_THROW() << "Preferable primitive descriptor is not set for node " << deconvAttrs.layerName << ".";
        }

        const auto selectedImpl = selected_pd->getImplementationType();

        if (isDynamicNode()) {
            std::tie(inDims, outDims) = makeDummyInOutShape();
            initPaddingR(Shape(inDims), Shape(outDims));
            auto inDummyDsc = getBaseMemDescAtInputPort(0)->cloneWithNewDims(inDims);
            auto outDummyDsc = getBaseMemDescAtOutputPort(0)->cloneWithNewDims(outDims);
            inDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inDummyDsc);
            outDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outDummyDsc);
        } else {
            inDims = getInputShapeAtPort(0).getStaticDims();
            outDims = getOutputShapeAtPort(0).getStaticDims();
            inDesc = getParentEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
            outDesc = getChildEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
        }

        dnnl::memory::desc dnnlBiasDesc;
        if (deconvAttrs.withBiases) {
            DnnlMemoryDescPtr biasDesc = getParentEdgesAtPort(biasPort).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
            dnnlBiasDesc = biasDesc->getDnnlDesc();
        }

        const AttrPtr pAttrConst = makePrimitiveAttr(outDims);
        auto prim_desc = createInt8MkldnnDeconvDesc(inDesc->getDnnlDesc(), wgh_candidate, dnnlBiasDesc, outDesc->getDnnlDesc(), deconvAttrs.withBiases,
                                               deconvAttrs.stride, deconvAttrs.dilation, deconvAttrs.paddingL, deconvAttrs.paddingR, *pAttrConst, getEngine());

        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, selectedImpl);

        if (found) {
            prepareMemory({DnnlExtensionUtils::makeDescriptor(prim_desc.weights_desc(0))});
        } else {
            prepareMemory({std::make_shared<DnnlBlockedMemoryDesc>(
                        MemoryDescUtils::convertToDnnlBlockedMemoryDesc(internalBlobs.front()->getTensorDesc()))});
        }
    }

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void Deconvolution::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto wghMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory has not been allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory has not been allocated.";
    if (!wghMemPtr || !wghMemPtr->isAllocated())
        IE_THROW() << "Weight memory has not been allocated.";
    auto *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << deconvAttrs.layerName << ".";
    auto inMemoryDesc = getParentEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
    auto outMemoryDesc = getChildEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();

    AttrPtr pAttrLocal;
    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = makePrimitiveAttr(dstMemPtr->getStaticDims());
        }
        pAttrLocal = pAttr;
        if (deconvAttrs.autoPad || deconvAttrs.externOutShape) {
            deconvAttrs.paddingL = shapeInference->get_pads_begin();
            deconvAttrs.paddingR = shapeInference->get_pads_end();
        }
        initPaddingR(inMemoryDesc->getShape(), outMemoryDesc->getShape());
    } else {
        pAttrLocal = makePrimitiveAttr(dstMemPtr->getStaticDims());
    }
    (*pAttrLocal).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    DnnlMemoryDescCPtr wghDesc;
    MemoryPtr biasMemPtr = nullptr;
    DnnlMemoryDescCPtr biasDesc;

    if (deconvAttrs.isInt8) {
        wghDesc = internalBlobMemory.front()->GetDescWithType<DnnlMemoryDesc>();
        if (deconvAttrs.withBiases) {
            biasMemPtr = getParentEdgesAtPort(biasPort)[0]->getMemoryPtr();
            if (!biasMemPtr || !biasMemPtr->isAllocated())
                IE_THROW() << "Bias memory  memory didn't allocate.";
            biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
        }
    } else {
        wghDesc = getParentEdgesAtPort(1).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
    }

    deconvAttrs.key = {inMemoryDesc,
                       wghDesc,
                       biasDesc,
                       outMemoryDesc,
                       deconvAttrs.stride,
                       deconvAttrs.dilation,
                       deconvAttrs.paddingL,
                       deconvAttrs.paddingR,
                       deconvAttrs.isInt8,
                       *pAttrLocal,
                       selected_pd->getImplementationType()};

    deconvAttrs.engine = getEngine();
    deconvAttrs.cache = context->getParamsCache();

    deconvAttrs.initPrimArgs = [this, &srcMemPtr, &dstMemPtr, &biasMemPtr, &wghMemPtr, &pAttrLocal](
            std::shared_ptr<std::unordered_map<int, dnnl::memory>> primArgsPtr,
            std::shared_ptr<DnnlExecutor> dnnlExecPtr, CacheEntryBase::LookUpStatus lookUpStatus) {
        if (dnnlExecPtr) {
            if (deconvAttrs.key.isInt8) {
                (*primArgsPtr)[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
                (*primArgsPtr)[DNNL_ARG_WEIGHTS] = internalBlobMemory.front()->GetPrimitive();
                (*primArgsPtr)[DNNL_ARG_DST]=  dstMemPtr->GetPrimitive();
                if (deconvAttrs.withBiases)
                    (*primArgsPtr)[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
            } else {
                (*primArgsPtr)[DNNL_ARG_DIFF_DST] = srcMemPtr->GetPrimitive();
                (*primArgsPtr)[DNNL_ARG_WEIGHTS] = wghMemPtr->GetPrimitive();
                (*primArgsPtr)[DNNL_ARG_DIFF_SRC] = dstMemPtr->GetPrimitive();
            }
            Node::appendPostOpArgs(*pAttrLocal, (*primArgsPtr), postOpsArgs);

            auto scratchpadMem = getScratchPadMem(dnnlExecPtr->getScratchPadDesc());
            (*primArgsPtr)[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
#ifdef CPU_DEBUG_CAPS
            if (lookUpStatus == CacheEntryBase::LookUpStatus::Miss) {
                auto pd = dnnlExecPtr->getPrimitiveDesc();
                DEBUG_LOG("verbose##", deconvAttrs.layerName, "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
            }
#endif
        } else {
            IE_THROW() << "Primitive descriptor was not found for node " << deconvAttrs.layerName << ".";
        }
    };

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgesAtPort(i).front()->getMemory().GetDescWithType<DnnlMemoryDesc>());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgesAtPort(i).front()->getMemory().GetDescWithType<DnnlMemoryDesc>());
    }

    execPtrDeconv = selected_pd->getExecutorFactoryAs<DeconvExecutorFactory>()->makeExecutor(deconvAttrs, srcMemoryDescs,
                                                                                             dstMemoryDescs, *attr);
    selected_pd->setImplementationType(execPtrDeconv->getImplType());
}

void Deconvolution::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);

            inConfs.emplace_back(desc, BlockedMemoryDesc::EMPTY_MASK);
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, BlockedMemoryDesc::EMPTY_MASK, inPlaceOutPort);
        }

        const NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }

        auto factory = std::make_shared<DeconvExecutorFactory>(deconvAttrs, srcMemoryDescs, dstMemoryDescs,
                                                               std::make_shared<ExecutorContext>(context, getImplPriority()));

        supportedPrimitiveDescriptors.emplace_back(config, impl_type, factory);
    };

    /* When custom implementation priorities are NOT defined it is enough to
     * just use the first implementation from the priority list.
     * When custom implementation priorities are defined, all the implementations should be considered,
     * since custom implementations can be not available at all, so a fallback to the default ones must happen
     * To achive the fallback, it is necessary to create a supported primitive descriptor for each implementation
     * since oneDNN primitive is mutating while iterating */

    for (auto& desc : descs) {
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get()));
        const bool first_match = customImplPriorities.empty();
        DnnlExtensionUtils::for_each_implementation(desc,
                                                    first_match,
                                                    [&](impl_desc_type implType) {
                                                        return contains(getImplPriority(), implType);
                                                    },
                                                    [&](dnnl::primitive_desc& desc) {
                                                        addSupportedPrimitiveDescriptor(desc);
                                                    });

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty())
            addSupportedPrimitiveDescriptor(first_desc);
    }
}

void Deconvolution::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                     const std::vector<MemoryDescPtr> &outputDesc) {
    auto inDesc = inputDesc[0]->isDefined() ? inputDesc[0] : inputDesc[0]->cloneWithNewDims(inShape.getStaticDims());
    auto dnnlInDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*inDesc);
    const auto& in_candidate = dnnlInDesc.getDnnlDesc();

    auto outDesc = outputDesc[0];
    if (!outDesc->isDefined()) {
        const auto outShape = shapeInferInternal(inDesc->getShape().getStaticDims(), deconvAttrs.lastOutputSpatialDims);
        outDesc = outDesc->cloneWithNewDims(outShape);
    }
    auto dnnlOutDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*outDesc);
    const auto& out_candidate = dnnlOutDesc.getDnnlDesc();
    dnnl::memory::desc bias_candidate;

    // grouping and autoblocking is not compatible
    if ((deconvAttrs.withGroups && !deconvAttrs.isDW) && (dnnlInDesc.blocksExtended() || dnnlOutDesc.blocksExtended()))
        return;

    AttrPtr attr = initPrimitiveAttr();
    if (deconvAttrs.isInt8) {
        if (deconvAttrs.withBiases) {
            memory::data_type bdt = memory::data_type::f32;
            bias_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(deconvAttrs.expectedBiasDims), bdt, memory::format_tag::any);
        }
        dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(deconvAttrs.int8WeightDims), memory::data_type::s8, memory::format_tag::any);
        descs.emplace_back(createDescriptorInternalInt8(in_candidate, wgh_candidate, bias_candidate,
                                                        out_candidate, deconvAttrs.withBiases, deconvAttrs.stride,
                                                        deconvAttrs.dilation, deconvAttrs.paddingL, deconvAttrs.paddingR, *attr, getEngine()));
    } else {
        dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(getWeightDims()),
                                           dnnlInDesc.getDataType(), memory::format_tag::any);
        convolution_backward_data::primitive_desc deconv_desc;
        convolution_forward::primitive_desc fwd_conv_pd;
        std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(in_candidate, wgh_candidate, out_candidate, dnnl::algorithm::convolution_direct,
                                                                             deconvAttrs.stride, deconvAttrs.dilation, deconvAttrs.paddingL,
                                                                             deconvAttrs.paddingR, *attr, getEngine());
        IE_ASSERT(fwd_conv_pd &&  deconv_desc && deconv_desc.get(true) != nullptr)
                << "Failed to create convolution_backward_data::primitive_desc: " << "Node: ##" << getName();
        fwdConvPD.push_back(fwd_conv_pd); // oneDNN requires forward pd to exists until primitive is created
        descs.push_back(deconv_desc);
    }
}

std::shared_ptr<MemoryDesc> Deconvolution::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    if (idx == 2 && !deconvAttrs.withBiases) {
        return std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::I32, Shape(getInputShapeAtPort(2).getStaticDims()));
    } else if (idx > 0 && deconvAttrs.isInt8) {
        // we need to store 'weight' input as edge,
        // because at this moment we can't simple replace internal blob with input, since we need to save weight data as is, but with different order
        return std::make_shared<CpuBlockedMemoryDesc>(getOriginalInputPrecisionAtPort(idx), Shape(getInputShapeAtPort(idx).getStaticDims()));
    }

    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1) : deconvAttrs.isInt8 ? prim_desc.src_desc(idx) : prim_desc.diff_dst_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> Deconvolution::getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc =  deconvAttrs.isInt8 ? prim_desc.dst_desc(idx) : prim_desc.diff_src_desc(idx);
    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision Deconvolution::getRuntimePrecision() const {
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

std::vector<int32_t> Deconvolution::readOutputSpatialDims() const {
    if (getParentEdges().size() < 3) {
        IE_THROW() << "Can't get output spatial dims. Inputs number = " << getParentEdges().size();
    }
    const auto &shapeMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
    if (!shapeMemPtr || !shapeMemPtr->isAllocated()) {
        IE_THROW() << "'output_shape' input memory is not allocated.";
    }
    const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
    if (shapeMemPtr->getStaticDims()[0] != spDimsNum) {
        IE_THROW() << "Can't read output spatial dims, beause 'output_shape' input has incorrect number of elements";
    }
    const int32_t *outShapePtr = reinterpret_cast<const int32_t *>(shapeMemPtr->GetPtr());
    std::vector<int32_t> outSpDims(outShapePtr, outShapePtr + shapeMemPtr->getStaticDims()[0]);
    return outSpDims;
}

bool Deconvolution::canFuseBias() const {
    //ONEDNN deconvolution_fwd_t primitive can support bias fusing.
    //ONEDNN convolution_data_bwd_t can't support bias fusing.
    //Current only int8 precision choose deconvolution_fwd_t.
    return  (canBeExecutedInInt8() &&
            (deconvAttrs.externOutShape ? getParentEdges().size() == 3 : getParentEdges().size() == 2));
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
