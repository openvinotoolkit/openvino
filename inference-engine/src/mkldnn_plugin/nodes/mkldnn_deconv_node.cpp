// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_input_node.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include <ngraph/opsets/opset1.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <nodes/common/cpu_memcpy.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNDeconvolutionNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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
            errorMessage = "Doesn't support dynamic 'weights' and 'output_shape' shapes";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const std::shared_ptr<ngraph::Node>& op,
                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return MKLDNNExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(0));
    });
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Deconvolution node with name '" + getName() + "'";

        auto convBackprop = std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op);
        auto groupConvBackprop = std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op);
        weightDims = getInputShapeAtPort(1).getStaticDims();

        if (convBackprop) {
            algorithm = DeconvolutionCommon;

            IC = weightDims[0];
            OC = weightDims[1];

            groupNum = 1;
            withGroups = false;

            for (int i = 0; i < convBackprop->get_strides().size(); i++) {
                stride.push_back(static_cast<ptrdiff_t>(convBackprop->get_strides()[i]));
            }
            for (int i = 0; i < convBackprop->get_dilations().size(); i++) {
                dilation.push_back(static_cast<ptrdiff_t>(convBackprop->get_dilations()[i]) - 1);
            }
            paddingL = convBackprop->get_pads_begin();
            paddingR = convBackprop->get_pads_end();

            autoPad = one_of(convBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
        } else if (groupConvBackprop) {
            algorithm = DeconvolutionGrouped;

            IC = weightDims[1];
            OC = weightDims[2];

            groupNum = weightDims[0];
            withGroups = groupNum > 1;
            isDW = withGroups && groupNum == OC && groupNum == IC;

            for (int i = 0; i < groupConvBackprop->get_strides().size(); i++) {
                stride.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_strides()[i]));
            }
            for (int i = 0; i < groupConvBackprop->get_dilations().size(); i++) {
                dilation.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_dilations()[i]) - 1);
            }
            paddingL = groupConvBackprop->get_pads_begin();
            paddingR = groupConvBackprop->get_pads_end();

            autoPad = one_of(groupConvBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
        }
        for (int i = 0; i < dilation.size(); i++) {
            kernel.push_back(weightDims[withGroups + 2 + i]);
        }

        withOutputShape = inputShapes.size() == 3;
        if (isDynamicNode() && withOutputShape &&
                op->get_input_node_shared_ptr(2)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
            outSpatialDims = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(2))->cast_vector<int32_t>();
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

InferenceEngine::Blob::Ptr MKLDNNDeconvolutionNode::createWeiBlobAsIO(InferenceEngine::SizeVector dims) {
    auto constNode = std::dynamic_pointer_cast<MKLDNNInputNode>(getParentEdgeAt(1)->getParent());
    if (!constNode)
        IE_THROW() << "Cannot cast const input node for node " << getName() << ".";
    auto blb = constNode->getMemoryPtr();
    if (!blb)
        IE_THROW() << "Cannot get const weights blob for node " << getName() << ".";

    auto const blbSize = blb->GetSize();

    // WA: In int8 case, we are processing weights using internal blob.
    InferenceEngine::SizeVector dimsForBlockedDesc{dims};
    std::swap(dimsForBlockedDesc[withGroups + 0], dimsForBlockedDesc[withGroups + 1]);

    InferenceEngine::SizeVector orderForBlockedDesc;
    if (withGroups) {
        orderForBlockedDesc = {0, 2, 1};
    } else {
        orderForBlockedDesc = {1, 0};
    }
    for (int i = 2 + withGroups; i < dimsForBlockedDesc.size(); i++)
        orderForBlockedDesc.push_back(i);

    BlockingDesc blkDesc(dimsForBlockedDesc, orderForBlockedDesc);
    InferenceEngine::TensorDesc tensorDesc(MKLDNNExtensionUtils::DataTypeToIEPrecision(blb->GetDataType()), dims, blkDesc);

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<int8_t>(tensorDesc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    if (data == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";
    size_t intBuffSize = internalBlob->byteSize();

    size_t offset = blbSize;
    if (intBuffSize < offset) {
        IE_THROW() << "Cannot create internal buffer. Buffer can be overrun.";
    }
    cpu_memcpy_s(data, intBuffSize, blb->GetPtr(), blbSize);

    return internalBlob;
}

bool MKLDNNDeconvolutionNode::canBeExecutedInInt8() const {
    if (std::dynamic_pointer_cast<MKLDNNInputNode>(getParentEdgeAt(1)->getParent()) == nullptr) {
        return false;
    }

    if (!withGroups && stride.back() > 3)
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_common)) {
        auto inDims = getOutputShapeAtPort(0).getMaxDims();
        if (std::any_of(inDims.begin(), inDims.end(), [](Dim dim) { return dim == Shape::UNDEFINED_DIM; })) {
            return false;
        }
        // heuristicConst = 2^26
        // heuristicParam = IC^2 * SP
        auto heuristicConst = 67108864;
        auto heuristicParam = IC * IC;
        for (int i = 2; i < inDims.size(); i++)
            heuristicParam *= inDims[i];
        if (heuristicParam > heuristicConst)
            return false;
    }

    for (int i = 0; i < kernel.size(); i++) {
        if (kernel[i] < stride[i])
            return false;
    }

    // not supported in oneDNN
    int channelBlock = impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_common) ? 16
            : impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) ? 8 : 4;
    if (withGroups && !isDW && (IC % channelBlock != 0 || OC % channelBlock != 0))
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_common) && stride.back() > 3)
        return false;

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrecision);

    InferenceEngine::Precision weiPrecision = getOriginalInputPrecisionAtPort(1);
    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(weiPrecision);

    if (isDW && (inputDataType == dnnl_s8 || dilation.size() == 3))
        return false;

    return (inputDataType == dnnl_s8 || inputDataType == dnnl_u8) && weightsDataType == dnnl_s8;
}

bool MKLDNNDeconvolutionNode::canFuse(const MKLDNNNodePtr& node) const {
    if (canBeExecutedInInt8())
        return canFuseSimpleOperation(node);

    return (fusedWith.empty() && node->canBePerformedAsScaleShift(this));
}

void MKLDNNDeconvolutionNode::initPadding(const std::shared_ptr<ngraph::Node> op) {
    if (getAlgorithm() == DeconvolutionCommon) {
        const auto deconv = ngraph::as_type_ptr<const ngraph::op::v1::ConvolutionBackpropData>(op);
        paddingL = deconv->get_pads_begin();
        paddingR = deconv->get_pads_end();
    } else if (getAlgorithm() == DeconvolutionGrouped) {
        const auto deconv = ngraph::as_type_ptr<const ngraph::op::v1::GroupConvolutionBackpropData>(op);
        paddingL = deconv->get_pads_begin();
        paddingR = deconv->get_pads_end();
    }
}

void MKLDNNDeconvolutionNode::getSupportedDescriptors() {
    isInt8 = canBeExecutedInInt8();

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrecision = getOriginalOutputPrecisionAtPort(0);
    if (isInt8) {
        // TODO: We have to extend jit_avx512_core_x8s8s32x_deconv_fwd_kernel from oneDNN to support BF16 output data type
        if (InferenceEngine::Precision::BF16 == inPrecision)
            inPrecision = InferenceEngine::Precision::FP32;
        if (InferenceEngine::Precision::BF16 == outPrecision)
            outPrecision = InferenceEngine::Precision::FP32;
    } else {
        if (!one_of(inPrecision, InferenceEngine::Precision::FP32, InferenceEngine::Precision::BF16))
            inPrecision = InferenceEngine::Precision::FP32;
        if (!one_of(outPrecision, InferenceEngine::Precision::FP32, InferenceEngine::Precision::BF16))
            outPrecision = InferenceEngine::Precision::FP32;
    }
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outPrecision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;
    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }

    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";

    auto dummyInShape = MemoryDescUtils::makeDummyShape(getInputShapeAtPort(0));
    auto dummyOutShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    if (isDynamicNode()) {
        if (withOutputShape && outSpatialDims.empty()) {
            outSpatialDims.resize((getInputShapeAtPort(0).getRank() - 2), 64);
        }
        dummyOutShape = Shape(deconvShapeInfer(dummyInShape.getStaticDims()));
        initPadding(opToShapeInfer);
    }
    initPaddingR(dummyInShape, dummyOutShape);

    if (isInt8) {
        //  WA: if int8 deconvolution is supported, we create internal weights blob in IO format
        std::swap(weightDims[withGroups + 0], weightDims[withGroups + 1]);
        internalBlobs.push_back(createWeiBlobAsIO(weightDims));
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
    setPostOps(attr, dummyOutShape.getStaticDims());
}

void MKLDNNDeconvolutionNode::initPaddingR(const Shape &inShape, const Shape &outShape) {
    for (int i = 0; i < paddingR.size(); i++) {
        int with_group = getAlgorithm() == DeconvolutionGrouped ? 1 : 0;
        int krn = weightDims[with_group + 2 + i];
        int src = outShape.getStaticDims()[2 + i];
        int dst = inShape.getStaticDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }
}

void MKLDNNDeconvolutionNode::setPostOps(mkldnn::primitive_attr &attr, const VectorDims &dims) {
    mkldnn::post_ops ops;

    auto getBinPostOpShape = [&](){
        const auto outShape = getOutputShapeAtPort(0).getStaticDims();
        const auto outShapeRank = getOutputShapeAtPort(0).getRank();
        const auto chIdx = getFusingAxis();
        std::vector<size_t> binaryShape(outShapeRank, 1);
        binaryShape[chIdx] = outShape[chIdx];
        return binaryShape;
    };

    for (auto &node : fusedWith) {
        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            // TODO [DS]: change to shape from memory
            constexpr int align = 16;
            // use legacy depthwise since backprop convolution does not support binary post ops
            eltwiseNode->appendPostOps(ops, getOutputShapeAtPort(0).getStaticDims(), align);
            continue;
        }
        if (auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get())) {
            fakeQuantizeNode->appendBinPostOps(ops, getBinPostOpShape(), binaryPostOpsArgs);
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNDeconvolutionNode::filterSupportedPrimitiveDescriptors() {
    MKLDNNNode::filterSupportedPrimitiveDescriptors();
    filterSupportedDescriptors();
}

void MKLDNNDeconvolutionNode::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            IE_THROW() << "Incorrect number of input or output memory formats for Deconvolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto src_tdesc = MKLDNNExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::deconvolution_forward::desc>(*itd)->data.src_desc);
                    isSuitableDesc &= src_tdesc->isSame(inputMemoryFormatsFilter[0]);
                } else {
                    auto src_tdesc = MKLDNNExtensionUtils::makeDescriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_src_desc);
                    isSuitableDesc &= src_tdesc->isSame(inputMemoryFormatsFilter[0]);
                }
            }
            if (!outputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto dst_tdesc = MKLDNNExtensionUtils::makeDescriptor(std::shared_ptr<mkldnn::deconvolution_forward::desc>(*itd)->data.dst_desc);
                    isSuitableDesc &= dst_tdesc->isSame(outputMemoryFormatsFilter[0]);
                } else {
                    auto dst_tdesc = MKLDNNExtensionUtils::makeDescriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_dst_desc);
                    isSuitableDesc &= dst_tdesc->isSame(outputMemoryFormatsFilter[0]);
                }
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNDeconvolutionNode::created() const {
    return getType() == Deconvolution;
}

bool MKLDNNDeconvolutionNode::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    if (withOutputShape) {
        if (outSpatialDims.empty()) {
            return true;
        }
        const int32_t *shapeMemPtr = reinterpret_cast<const int32_t *>(getParentEdgesAtPort(2)[0]->getMemory().GetPtr());
        for (size_t i = 0; i < outSpatialDims.size(); i++) {
            if (outSpatialDims[i] != shapeMemPtr[i])
                return true;
        }
    }

    return false;
}

std::vector<VectorDims> MKLDNNDeconvolutionNode::shapeInfer() const {
    if (withOutputShape) {
        const auto &shapeMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        const int32_t *outShapePtr = reinterpret_cast<const int32_t *>(shapeMemPtr->GetPtr());
        if (outSpatialDims.empty())
            outSpatialDims.resize(shapeMemPtr->getStaticDims()[0]);
        for (size_t i = 0; i < outSpatialDims.size(); i++) {
            outSpatialDims[i] = outShapePtr[i];
        }
    }
    const auto &dataMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    return {deconvShapeInfer(dataMemPtr->getStaticDims())};
}

VectorDims MKLDNNDeconvolutionNode::deconvShapeInfer(const VectorDims &inDims) const {
    ngraph::OutputVector inputsForShapeInfer;
    inputsForShapeInfer.push_back(std::make_shared<ngraph::opset1::Parameter>(opToShapeInfer->get_input_element_type(0), ov::PartialShape(inDims)));
    inputsForShapeInfer.push_back(opToShapeInfer->get_input_node_shared_ptr(1));

    if (!outSpatialDims.empty()) {
        if (opToShapeInfer->get_input_size() != 3) {
            IE_THROW() << "Can't compute output shape for node with name: " << getName()
                       << ", because node has no 'output_shape' input, but output spatial dims provided";
        }
        inputsForShapeInfer.push_back(ngraph::opset1::Constant::create(ngraph::element::Type_t::i32, {outSpatialDims.size()}, outSpatialDims));
    } else if (opToShapeInfer->get_input_size() == 3) {
        IE_THROW() << "Can't compute output shape for node with name: " << getName()
                   << ", because node has 'output_shape' input, but output spatial dims not provided";
    }

    opToShapeInfer = opToShapeInfer->clone_with_new_inputs(inputsForShapeInfer);
    opToShapeInfer->validate_and_infer_types();

    IE_ASSERT(opToShapeInfer->get_output_size() == 1);

    const auto &partShape = opToShapeInfer->get_output_partial_shape(0);
    if (partShape.is_dynamic())
        IE_THROW(NotImplemented) << "CPU plug-in doesn't support default shape infer for nodes with internal dynamism";
    return partShape.get_shape();
}

void MKLDNNDeconvolutionNode::executeDynamicImpl(mkldnn::stream strm) {
    if (primArgs.at(DNNL_ARG_DIFF_DST).get_desc() != getParentEdgesAtPort(0)[0]->getMemory().GetPrimitive().get_desc()) {
        auto src = getParentEdgesAtPort(0)[0]->getMemory().GetPrimitive();
        auto dst = srcPlanarMemPtr->GetPrimitive();
        reorderSrc = {src, dst};
        reorderSrc.execute(strm, src, dst);
    }
    if (primArgs.at(DNNL_ARG_WEIGHTS).get_desc() != getParentEdgesAtPort(1)[0]->getMemory().GetPrimitive().get_desc()) {
        auto src = getParentEdgesAtPort(1)[0]->getMemory().GetPrimitive();
        auto dst = wghPlanarMemPtr->GetPrimitive();
        reorderWgh = {src, dst};
        reorderWgh.execute(strm, src, dst);
    }
    execute(strm);
    if (primArgs.at(DNNL_ARG_DIFF_SRC).get_desc() != getChildEdgesAtPort(0)[0]->getMemory().GetPrimitive().get_desc()) {
        auto src = dstPlanarMemPtr->GetPrimitive();
        auto dst = getChildEdgesAtPort(0)[0]->getMemory().GetPrimitive();
        reorderDst = {src, dst};
        reorderDst.execute(strm, src, dst);
    }
}

std::shared_ptr<MKLDNNDescriptor> MKLDNNDeconvolutionNode::createMkldnnDeconvDesc(const mkldnn::memory::desc& srcDesc,
                                                                                  const mkldnn::memory::desc& wghDesc,
                                                                                  const mkldnn::memory::desc& dstDesc,
                                                                                  bool isWinograd) const {
    std::shared_ptr<MKLDNNDescriptor> desc;
    if (isInt8) {
        desc = std::make_shared<MKLDNNDescriptor>(createDescriptorInternalInt8(srcDesc, wghDesc, dstDesc,
                                                  mkldnn::algorithm::deconvolution_direct));
    } else {
        mkldnn::algorithm alg = isWinograd ? mkldnn::algorithm::convolution_winograd : mkldnn::algorithm::convolution_direct;
        std::shared_ptr<convolution_backward_data::desc> deconv_desc;
        std::shared_ptr<convolution_forward::primitive_desc> fwd_conv_pd;
        std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(srcDesc, wghDesc, dstDesc, alg);
        if (fwd_conv_pd->get(true) == nullptr) {
            IE_THROW() << "Forward convolution primitive descriptor is nullable for node with name: " << getName();
        }
        desc = std::make_shared<MKLDNNDescriptor>(deconv_desc, fwd_conv_pd);
    }
    return desc;
}

void MKLDNNDeconvolutionNode::createDeconvPrim(std::shared_ptr<MKLDNNDescriptor> desc,
                                               MKLDNNMemoryPtr srcMemPtr,
                                               MKLDNNMemoryPtr wghMemPtr,
                                               MKLDNNMemoryPtr dstMemPtr,
                                               AttrPtr attr,
                                               impl_desc_type selectedImpl,
                                               bool forceGemm) {
    auto itpd = desc->createPrimitiveDescriptorIterator(getEngine(), *attr);

    while (static_cast<bool>(itpd)) {
        impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

        if (forceGemm) {
            if (impl_type == impl_desc_type::jit_gemm) {
                auto prim_desc = convolution_backward_data::primitive_desc(itpd.get());
                prim.reset(new convolution_backward_data(prim_desc));
                primArgs = {{DNNL_ARG_DIFF_DST, srcMemPtr->GetPrimitive()}, {DNNL_ARG_WEIGHTS, wghMemPtr->GetPrimitive()},
                            {DNNL_ARG_DIFF_SRC, dstMemPtr->GetPrimitive()}};
                break;
            }
        } else {
            if (isInt8) {
                if (impl_type == selectedImpl) {
                    if (internalBlobMemory.empty()) {
                        prepareMemory(itpd);
                    }
                    auto prim_desc = deconvolution_forward::primitive_desc(itpd.get());
                    prim.reset(new deconvolution_forward(prim_desc));
                    primArgs = {{DNNL_ARG_SRC, srcMemPtr->GetPrimitive()}, {DNNL_ARG_WEIGHTS, internalBlobMemory.front()->GetPrimitive()},
                                {DNNL_ARG_DST, dstMemPtr->GetPrimitive()}};
                    break;
                }
            } else {
                if (impl_type == selectedImpl) {
                    auto prim_desc = convolution_backward_data::primitive_desc(itpd.get());
                    prim.reset(new convolution_backward_data(prim_desc));
                    primArgs = {{DNNL_ARG_DIFF_DST, srcMemPtr->GetPrimitive()}, {DNNL_ARG_WEIGHTS, wghMemPtr->GetPrimitive()},
                                {DNNL_ARG_DIFF_SRC, dstMemPtr->GetPrimitive()}};
                    break;
                }
            }
        }

        if (!itpd.next_impl()) {
            if (!forceGemm) {
                auto in_planar_desc = DnnlBlockedMemoryDesc(InferenceEngine::Precision::FP32, Shape(srcMemPtr->getStaticDims()));
                if (!srcPlanarMemPtr) {
                    srcPlanarMemPtr = std::make_shared<MKLDNNMemory>(getEngine());
                    srcPlanarMemPtr->Create(in_planar_desc);
                } else {
                    srcPlanarMemPtr->redefineDesc(in_planar_desc);
                }

                auto wgh_planar_desc = isInt8 ? *getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetDescWithType<DnnlMemoryDesc>() :
                                                 DnnlBlockedMemoryDesc(InferenceEngine::Precision::FP32, Shape(wghMemPtr->getStaticDims()));
                if (!wghPlanarMemPtr) {
                    wghPlanarMemPtr = std::make_shared<MKLDNNMemory>(getEngine());
                    wghPlanarMemPtr->Create(wgh_planar_desc);
                } else {
                    wghPlanarMemPtr->redefineDesc(wgh_planar_desc);
                }
                auto out_planar_desc = DnnlBlockedMemoryDesc(InferenceEngine::Precision::FP32, Shape(dstMemPtr->getStaticDims()));
                if (!dstPlanarMemPtr) {
                    dstPlanarMemPtr = std::make_shared<MKLDNNMemory>(getEngine());
                    dstPlanarMemPtr->Create(out_planar_desc);
                } else {
                    dstPlanarMemPtr->redefineDesc(out_planar_desc);
                }

                std::shared_ptr<MKLDNNDescriptor> desc = createMkldnnDeconvDesc(in_planar_desc.getDnnlDesc(), wgh_planar_desc.getDnnlDesc(),
                                                                                out_planar_desc.getDnnlDesc(),
                                                                                false);

                createDeconvPrim(desc, srcPlanarMemPtr, wghPlanarMemPtr, dstPlanarMemPtr, attr, impl_desc_type::jit_gemm, true);
                break;
            } else {
                IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
            }
        }
    }
}

void MKLDNNDeconvolutionNode::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    auto inMemoryDesc = getParentEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
    auto outMemoryDesc = getChildEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();

    auto initPrimitiveAttr = [&]() {
        mkldnn::primitive_attr attr;
        setPostOps(attr, dstMemPtr->getStaticDims());
        return std::make_shared<mkldnn::primitive_attr>(std::move(attr));
    };

    AttrPtr pAttrLocal;

    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = initPrimitiveAttr();
        }
        pAttrLocal = pAttr;
        if (autoPad || withOutputShape) {
            initPadding(opToShapeInfer);
        }
        initPaddingR(inMemoryDesc->getShape(), outMemoryDesc->getShape());
    } else {
        pAttrLocal = initPrimitiveAttr();
    }

    const auto in_candidate = inMemoryDesc->getDnnlDesc();
    const auto out_candidate = outMemoryDesc->getDnnlDesc();

    mkldnn::memory::desc wgh_candidate;
    if (isInt8) {
        if (internalBlobMemory.empty()) {
            wgh_candidate = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(weightDims), memory::data_type::s8, memory::format_tag::any);
        } else {
            wgh_candidate = internalBlobMemory.front()->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        }
    } else {
        wgh_candidate = getParentEdgesAtPort(1).front()->getMemory().GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    }

    std::shared_ptr<MKLDNNDescriptor> desc = createMkldnnDeconvDesc(in_candidate, wgh_candidate, out_candidate,
                                                                    selected_pd->getImplementationType() == MKLDNNPlugin::impl_desc_type::jit_avx512_winograd);

    createDeconvPrim(desc, srcMemPtr, getParentEdgesAtPort(1)[0]->getMemoryPtr(), dstMemPtr, pAttrLocal, selected_pd->getImplementationType());
}

const mkldnn::memory& MKLDNNDeconvolutionNode::getWeights() const {
    return isInt8 ? internalBlobMemory[0]->GetPrimitive() : getParentEdgeAt(1)->getMemory().GetPrimitive();
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

MKLDNNDeconvolutionNode::DefaultDeconvDescs MKLDNNDeconvolutionNode::createDescriptorInternalDefault(const mkldnn::memory::desc& in_candidate,
                                                                                                     const mkldnn::memory::desc& wgh_candidate,
                                                                                                     const mkldnn::memory::desc& out_candidate,
                                                                                                     mkldnn::algorithm alg) const {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
    conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, alg,
                                                  out_candidate, wgh_candidate, in_candidate,
                                                  convertDims(stride),
                                                  convertDims(dilation),
                                                  convertDims(paddingL),
                                                  convertDims(paddingR)));

    std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
    deconv_desc.reset(new convolution_backward_data::desc(alg, out_candidate, wgh_candidate,
                                                          in_candidate,
                                                          convertDims(stride),
                                                          convertDims(dilation),
                                                          convertDims(paddingL),
                                                          convertDims(paddingR)));

    auto fwd_conv_pd = std::make_shared<convolution_forward::primitive_desc>(*conv_desc, getEngine(), true);

    return {deconv_desc, fwd_conv_pd};
}

std::shared_ptr<deconvolution_forward::desc> MKLDNNDeconvolutionNode::createDescriptorInternalInt8(const mkldnn::memory::desc& in_candidate,
                                                                                                   const mkldnn::memory::desc& wgh_candidate,
                                                                                                   const mkldnn::memory::desc& out_candidate,
                                                                                                   mkldnn::algorithm alg) const {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    std::shared_ptr<mkldnn::deconvolution_forward::desc> deconv_desc;
    deconv_desc.reset(new deconvolution_forward::desc(prop_kind::forward_inference, mkldnn::algorithm::deconvolution_direct,
                                                      in_candidate, wgh_candidate, out_candidate,
                                                      convertDims(stride), convertDims(dilation),
                                                      convertDims(paddingL), convertDims(paddingR)));
    return deconv_desc;
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                               const std::vector<MemoryDescPtr> &outputDesc) {
    auto inDesc = inputDesc[0]->isDefined() ? inputDesc[0] : MemoryDescUtils::makeDummyDesc(*inputDesc[0]);
    auto dnnlInDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*inDesc);
    auto in_candidate = dnnlInDesc.getDnnlDesc();

    auto outDesc = outputDesc[0];
    if (!outDesc->isDefined()) {
        const auto dummyOutShape = deconvShapeInfer(inDesc->getShape().getStaticDims());
        outDesc = outDesc->cloneWithNewDims(dummyOutShape);
    }
    auto dnnlOutDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*outDesc);
    auto out_candidate = dnnlOutDesc.getDnnlDesc();

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (dnnlInDesc.blocksExtended() || dnnlOutDesc.blocksExtended()))
        return;

    if (isInt8) {
        mkldnn::memory::desc wgh_candidate(MKLDNNExtensionUtils::convertToDnnlDims(weightDims), memory::data_type::s8, memory::format_tag::any);
        descs.emplace_back(createDescriptorInternalInt8(in_candidate, wgh_candidate, out_candidate,
                                                        mkldnn::algorithm::deconvolution_direct));
    } else {
        mkldnn::memory::desc wgh_candidate(MKLDNNExtensionUtils::convertToDnnlDims(weightDims), dnnlInDesc.getDataType(), memory::format_tag::any);
        for (auto alg : {mkldnn::algorithm::convolution_winograd, mkldnn::algorithm::convolution_direct}) {
            std::shared_ptr<convolution_backward_data::desc> deconv_desc;
            std::shared_ptr<convolution_forward::primitive_desc> fwd_conv_pd;
            std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(in_candidate, wgh_candidate, out_candidate, alg);
            if (fwd_conv_pd->get(true) == nullptr)
                continue;
            descs.emplace_back(deconv_desc, fwd_conv_pd);
        }
    }
}

std::shared_ptr<MemoryDesc> MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (idx > 0) {
        if (isInt8) {
            return std::make_shared<CpuBlockedMemoryDesc>(getOriginalInputPrecisionAtPort(idx), Shape(getInputShapeAtPort(idx).getStaticDims()));
        } else if (idx == 2) {
            return std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::I32, Shape(getInputShapeAtPort(2).getStaticDims()));
        }
    }

    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : isInt8 ? primitive_desc_it.src_desc(idx) : primitive_desc_it.diff_dst_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return MKLDNNExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return MKLDNNExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> MKLDNNDeconvolutionNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc =  isInt8 ? primitive_desc_it.dst_desc(idx) : primitive_desc_it.diff_src_desc(idx);
    if (getOutputShapeAtPort(idx).isDynamic()) {
        return MKLDNNExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }
    return MKLDNNExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision MKLDNNDeconvolutionNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNDeconvolutionNode, Deconvolution);
