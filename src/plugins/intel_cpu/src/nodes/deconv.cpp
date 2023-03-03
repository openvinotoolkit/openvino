// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "input.h"
#include <string>
#include <vector>
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
#include <common/primitive_hashing_utils.hpp>
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include <utils/shape_inference/shape_inference_ngraph.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

using DefaultDeconvDescs = std::pair<std::shared_ptr<dnnl::convolution_backward_data::desc>,
                                         std::shared_ptr<dnnl::convolution_forward::primitive_desc>>;
using Int8DeconvDesc = std::shared_ptr<dnnl::deconvolution_forward::desc>;

namespace {

struct DeconvKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;

    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    ov::CoordinateDiff paddingL;
    ov::CoordinateDiff paddingR;

    bool isInt8;

    dnnl::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const DeconvKey& rhs) const;
};

size_t DeconvKey::hash() const {
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

    seed = hash_combine(seed, isInt8);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool DeconvKey::operator==(const DeconvKey &rhs) const {
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

    retVal = retVal && isInt8 == rhs.isInt8;

    retVal = retVal && *attr.get() == *rhs.attr.get() && implType == rhs.implType;
    return retVal;
}

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
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Deconvolution node with name '" + getName() + "'";

        auto convBackprop = std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op);
        auto groupConvBackprop = std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op);
        const auto& weightDims = getWeightDims();

        if (convBackprop) {
            algorithm = Algorithm::DeconvolutionCommon;

            IC = weightDims[0];
            OC = weightDims[1];
            biasesDims = {OC};

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

            outputPadding = convBackprop->get_output_padding();

            autoPad = one_of(convBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
        } else if (groupConvBackprop) {
            algorithm = Algorithm::DeconvolutionGrouped;

            groupNum = weightDims[0];
            IC = groupNum * weightDims[1];
            OC = groupNum * weightDims[2];
            biasesDims = {OC * groupNum};
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

            outputPadding = groupConvBackprop->get_output_padding();

            autoPad = one_of(groupConvBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
        }
        for (int i = 0; i < dilation.size(); i++) {
            kernel.push_back(weightDims[withGroups + 2 + i]);
        }

        externOutShape = inputShapes.size() == 3;
        biasPort = externOutShape ? 3 : 2;
        if (externOutShape && isDynamicNode()) {
            bool isConstOutShape = ngraph::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (isConstOutShape) {
                lastOutputSpatialDims = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(2))->cast_vector<int32_t>();
            }
            const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
            if (getInputShapeAtPort(2).getStaticDims()[0] != spDimsNum || (isConstOutShape && lastOutputSpatialDims.size() != spDimsNum)) {
                IE_THROW() << "'output_shape' input has incorrect number of elements. Expected = " << spDimsNum;
            }
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }

    attr = std::make_shared<dnnl::primitive_attr>();
}

InferenceEngine::Blob::Ptr Deconvolution::createWeiBlobAsIO(InferenceEngine::SizeVector dims) {
    auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent());
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
    InferenceEngine::TensorDesc tensorDesc(DnnlExtensionUtils::DataTypeToIEPrecision(blb->GetDataType()), dims, blkDesc);

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

bool Deconvolution::canBeExecutedInInt8() const {
    if (std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent()) == nullptr) {
        return false;
    }

    if (!withGroups && stride.back() > 3)
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        const auto& inMaxDims = getOutputShapeAtPort(0).getMaxDims();
        if (std::any_of(inMaxDims.begin(), inMaxDims.end(), [](Dim dim) { return dim == Shape::UNDEFINED_DIM; })) {
            return false;
        }
        // heuristicConst = 2^26
        // heuristicParam = IC^2 * SP
        auto heuristicConst = 67108864;
        auto heuristicParam = IC * IC;
        for (int i = 2; i < inMaxDims.size(); i++)
            heuristicParam *= inMaxDims[i];
        if (heuristicParam > heuristicConst)
            return false;
    }

    for (int i = 0; i < kernel.size(); i++) {
        if (kernel[i] < stride[i])
            return false;
    }

    // not supported in oneDNN
    int channelBlock = impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core) ? 16
            : impl::cpu::x64::mayiuse(impl::cpu::x64::avx2) ? 8 : 4;
    if (withGroups && !isDW && (IC % channelBlock != 0 || OC % channelBlock != 0))
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core) && stride.back() > 3)
        return false;

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inPrecision);

    InferenceEngine::Precision weiPrecision = getOriginalInputPrecisionAtPort(1);
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(weiPrecision);

    if (isDW && (inputDataType == dnnl_s8 || dilation.size() == 3))
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
        inputDims[1] = IC;

        if (externOutShape) {
            if (lastOutputSpatialDims.empty()) {
                const auto& shape = getOutputShapeAtPort(0);
                lastOutputSpatialDims.resize(shape.getRank() - 2);

                const auto& minDims = shape.getMinDims();
                const auto& maxDims = shape.getMaxDims();
                const auto& dims = shape.getDims();
                for (size_t i = 0; i < dims.size() - 2; ++i) {
                    lastOutputSpatialDims[i] = dims[i + 2] == Shape::UNDEFINED_DIM ? std::min(maxDims[i + 2],
                                                                                              std::max(minDims[i + 2], static_cast<Dim>(64))) : dims[i + 2];
                }
            }
            ov::CoordinateDiff pb = autoPad ? ov::CoordinateDiff(paddingL.size(), 0) : paddingL;
            ov::CoordinateDiff pe = autoPad ? ov::CoordinateDiff(paddingR.size(), 0) : paddingR;

            const auto& origInDims = getInputShapeAtPort(0).getDims();
            const auto& weightDims = getWeightDims();
            const size_t wghOffset = getAlgorithm() == Algorithm::DeconvolutionGrouped ? 1 : 0;
            for (size_t i = 0; i < inputDims.size() - 2; i++) {
                if (origInDims[2 + i] == Shape::UNDEFINED_DIM) {
                    inputDims[2 + i] = ((lastOutputSpatialDims[i] - (dilation[i] + 1) *
                                        (weightDims[wghOffset + 2 + i] - 1) - 1 + pb[i] + pe[i] - outputPadding[i])) /
                                        stride[i] + 1;
                }
            }
        }
        inShape = Shape(inputDims);
        outShape = Shape(shapeInferInternal(inShape.getStaticDims(), lastOutputSpatialDims));
        paddingL = shapeInference->get_pads_begin();
        paddingR = shapeInference->get_pads_end();
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
    isInt8 = canBeExecutedInInt8();
    withBiases = externOutShape ? getOriginalInputsNumber() == 4 : getOriginalInputsNumber() == 3;
    //ONEDNN deconvolution_fwd_t primitive can support bias fusing.
    //ONEDNN convolution_data_bwd_t can't support bias fusing.
    //Current only int8 precision choose deconvolution_fwd_t.
    if (withBiases && !isInt8) {
        IE_THROW() << errorPrefix << " supports bias fusing only for int8 execution precision";
    }

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
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inPrecision);
    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outPrecision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;
    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    if (getParentEdges().size() != (withBiases ? (biasPort + 1) : biasPort)) {
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    }
    if (getChildEdges().empty()) {
        IE_THROW() << errorPrefix << " has incorrect number of output edges";
    }
    VectorDims inDims, outDims;
    std::tie(inDims, outDims) = makeDummyInOutShape();
    inShape = Shape(inDims);
    Shape outShape(outDims);
    initPaddingR(inShape, outShape);

    if (isInt8) {
        int8WeightDims = getWeightDims();
        //  WA: if int8 deconvolution is supported, we create internal weights blob in IO format
        std::swap(int8WeightDims[withGroups + 0], int8WeightDims[withGroups + 1]);
        internalBlobs.push_back(createWeiBlobAsIO(int8WeightDims));
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
    setPostOps(*attr, outShape.getStaticDims());
}

void Deconvolution::initPaddingR(const Shape &inShape, const Shape &outShape) {
    for (int i = 0; i < paddingR.size(); i++) {
        int with_group = getAlgorithm() == Algorithm::DeconvolutionGrouped ? 1 : 0;
        const auto& weightDims = getWeightDims();
        int krn = weightDims[with_group + 2 + i];
        int src = outShape.getStaticDims()[2 + i];
        int dst = inShape.getStaticDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        paddingR[i] = (dst - 1) * stride[i] - (src - krn + paddingL[i]);
    }
}

void Deconvolution::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims) {
    dnnl::post_ops ops;

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, 1, isInt8);

    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            // TODO [DS]: change to shape from memory
            if (isInt8) {
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

void Deconvolution::filterSupportedPrimitiveDescriptors() {
    Node::filterSupportedPrimitiveDescriptors();
    filterSupportedDescriptors();
}

void Deconvolution::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            IE_THROW() << "Incorrect number of input or output memory formats for Deconvolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto src_tdesc = DnnlExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::deconvolution_forward::desc>(*itd)->data.src_desc);
                    isSuitableDesc &= src_tdesc->isSame(inputMemoryFormatsFilter[0]);
                } else {
                    auto src_tdesc = DnnlExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::convolution_backward_data::desc>(*itd)->data.diff_src_desc);
                    isSuitableDesc &= src_tdesc->isSame(inputMemoryFormatsFilter[0]);
                }
            }
            if (!outputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto dst_tdesc = DnnlExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::deconvolution_forward::desc>(*itd)->data.dst_desc);
                    isSuitableDesc &= dst_tdesc->isSame(outputMemoryFormatsFilter[0]);
                } else {
                    auto dst_tdesc = DnnlExtensionUtils::makeDescriptor(std::shared_ptr<dnnl::convolution_backward_data::desc>(*itd)->data.diff_dst_desc);
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

bool Deconvolution::created() const {
    return getType() == Type::Deconvolution;
}

bool Deconvolution::needShapeInfer() const {
    if (inputShapesModified()) {
        return true;
    }
    if (externOutShape) {
        if (lastOutputSpatialDims != readOutputSpatialDims()) {
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
                    IE_THROW() << "Can't compute output shape for node with name: " << getName()
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
        IE_THROW(Unexpected) << "Unexpected shape inference result status in node of type " << getTypeStr() << " with name " << getName();
    }
    return std::move(result.dims.back());
}

void Deconvolution::setDynamicBatchLim(int lim) {
    if (!execPtr) {
        IE_THROW() << "Can't set dynamic batch for Deconvolution node with name: " << getName() << ", because executor is not compiled";
    }
    if (execPtr->needReordering()) {
        IE_THROW() << "Can't execute Deconvolution node with dynamic batch via executor with reorders";
    }
    Node::setDynamicBatchLim(lim);
}

void Deconvolution::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute Deconvolution node with name: " << getName() << ", because executor is not compiled";
    }

    execPtr->exec(primArgs, strm);

    if (externOutShape) {
        lastOutputSpatialDims = readOutputSpatialDims();
    }
}

namespace {
DefaultDeconvDescs createDescriptorInternalDefault(const dnnl::memory::desc& in_candidate,
                                                   const dnnl::memory::desc& wgh_candidate,
                                                   const dnnl::memory::desc& out_candidate,
                                                   dnnl::algorithm alg,
                                                   const std::vector<ptrdiff_t>& stride,
                                                   const std::vector<ptrdiff_t>& dilation,
                                                   const ov::CoordinateDiff& paddingL,
                                                   const ov::CoordinateDiff& paddingR,
                                                   const dnnl::engine& engine) {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    std::shared_ptr<dnnl::convolution_forward::desc> conv_desc;
    conv_desc = std::make_shared<convolution_forward::desc>(prop_kind::forward_inference, alg,
                                                            out_candidate, wgh_candidate, in_candidate,
                                                            convertDims(stride),
                                                            convertDims(dilation),
                                                            convertDims(paddingL),
                                                            convertDims(paddingR));

    std::shared_ptr<dnnl::convolution_backward_data::desc> deconv_desc;
    deconv_desc = std::make_shared<convolution_backward_data::desc>(alg, out_candidate, wgh_candidate,
                                                                    in_candidate,
                                                                    convertDims(stride),
                                                                    convertDims(dilation),
                                                                    convertDims(paddingL),
                                                                    convertDims(paddingR));

    auto fwd_conv_pd = std::make_shared<convolution_forward::primitive_desc>(*conv_desc, engine, true);

    return {deconv_desc, fwd_conv_pd};
}

Int8DeconvDesc createDescriptorInternalInt8(const dnnl::memory::desc& in_candidate,
                                            const dnnl::memory::desc& wgh_candidate,
                                            const dnnl::memory::desc& bias_candidate,
                                            const dnnl::memory::desc& out_candidate,
                                            const bool with_bias,
                                            const std::vector<ptrdiff_t>& stride,
                                            const std::vector<ptrdiff_t>& dilation,
                                            const ov::CoordinateDiff& paddingL,
                                            const ov::CoordinateDiff& paddingR) {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };
    Int8DeconvDesc deconv_desc;
    if (with_bias)
        deconv_desc = std::make_shared<dnnl::deconvolution_forward::desc>(prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                                                                                        in_candidate, wgh_candidate, bias_candidate, out_candidate,
                                                                                        convertDims(stride), convertDims(dilation),
                                                                                        convertDims(paddingL), convertDims(paddingR));
    else
        deconv_desc = std::make_shared<dnnl::deconvolution_forward::desc>(prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                                                                                in_candidate, wgh_candidate, out_candidate,
                                                                                convertDims(stride), convertDims(dilation),
                                                                                convertDims(paddingL), convertDims(paddingR));
    return deconv_desc;
}

std::shared_ptr<DnnlDesriptor> createDefaultMkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                                const dnnl::memory::desc& wghDesc,
                                                                const dnnl::memory::desc& dstDesc,
                                                                bool isWinograd,
                                                                const std::vector<ptrdiff_t>& stride,
                                                                const std::vector<ptrdiff_t>& dilation,
                                                                const ov::CoordinateDiff& paddingL,
                                                                const ov::CoordinateDiff& paddingR,
                                                                const dnnl::engine& engine) {
    dnnl::algorithm alg = isWinograd ? dnnl::algorithm::convolution_winograd : dnnl::algorithm::convolution_direct;
    std::shared_ptr<convolution_backward_data::desc> deconv_desc;
    std::shared_ptr<convolution_forward::primitive_desc> fwd_conv_pd;
    std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(srcDesc, wghDesc, dstDesc, alg, stride, dilation, paddingL, paddingR, engine);
    if (fwd_conv_pd->get(true) == nullptr) {
        IE_THROW() << "Forward convolution primitive descriptor is nullable";
    }
    return std::make_shared<DnnlDesriptor>(deconv_desc, fwd_conv_pd);
}

std::shared_ptr<DnnlDesriptor> createInt8MkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                             const dnnl::memory::desc& wghDesc,
                                                             const dnnl::memory::desc& biasDesc,
                                                             const dnnl::memory::desc& dstDesc,
                                                             const bool withBias,
                                                             const std::vector<ptrdiff_t>& stride,
                                                             const std::vector<ptrdiff_t>& dilation,
                                                             const ov::CoordinateDiff& paddingL,
                                                             const ov::CoordinateDiff& paddingR) {
    return std::make_shared<DnnlDesriptor>(createDescriptorInternalInt8(srcDesc, wghDesc, biasDesc, dstDesc, withBias, stride, dilation, paddingL, paddingR));
}
} // namespace

Node::AttrPtr Deconvolution::makePrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims);

    return attr;
}

Node::AttrPtr Deconvolution::initPrimitiveAttr() {
    return attr;
}

void Deconvolution::createPrimitive() {
    if (isInt8) {
        VectorDims inDims, outDims;
        DnnlMemoryDescPtr inDesc;
        auto wgh_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(int8WeightDims), memory::data_type::s8, memory::format_tag::any);
        DnnlMemoryDescPtr outDesc;
        DnnlMemoryDescPtr biasDesc;

        const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr) {
            IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
        }

        const auto selectedImpl = selected_pd->getImplementationType();

        if (isDynamicNode()) {
            std::tie(inDims, outDims) = makeDummyInOutShape();
            initPaddingR(Shape(inDims), Shape(outDims));
            auto inDummyDsc = getBaseMemDescAtInputPort(0)->cloneWithNewDims(inDims);
            auto outDummyDsc = getBaseMemDescAtOutputPort(0)->cloneWithNewDims(outDims);
            inDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inDummyDsc);
            outDesc = MemoryDescUtils::convertToDnnlMemoryDesc(outDummyDsc);
            if (withBiases) {
                const VectorDims biasVecDims = getInputShapeAtPort(biasPort).getStaticDims();
                auto biasDummyDsc = getBaseMemDescAtInputPort(biasPort)->cloneWithNewDims(biasVecDims);
                biasDesc = MemoryDescUtils::convertToDnnlMemoryDesc(biasDummyDsc);
            }
        } else {
            inDims = getInputShapeAtPort(0).getStaticDims();
            outDims = getOutputShapeAtPort(0).getStaticDims();

            inDesc = getParentEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
            outDesc = getChildEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
            if (withBiases)
                biasDesc = getParentEdgesAtPort(biasPort).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
        }

        dnnl::memory::desc dnnlBiasDesc;
        if (biasDesc != nullptr)
            // WA to align IR bias representation (3 to 5 rank tensors) to oneDNN representation (1 rank tensor)
            dnnlBiasDesc = biasDesc->getDnnlDesc().reshape({DnnlExtensionUtils::convertToDnnlDim(biasesDims[0])});
        auto desc = createInt8MkldnnDeconvDesc(inDesc->getDnnlDesc(), wgh_candidate, dnnlBiasDesc, outDesc->getDnnlDesc(), withBiases,
                                               stride, dilation, paddingL, paddingR);
        AttrPtr pAttr = makePrimitiveAttr(outDims);
        auto itpd = desc->createPrimitiveDescriptorIterator(getEngine(), *pAttr);

        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == selectedImpl) {
                prepareMemory({DnnlExtensionUtils::makeDescriptor(itpd.weights_desc(0))});
                break;
            }

            if (!itpd.next_impl()) {
                prepareMemory({std::make_shared<DnnlBlockedMemoryDesc>(
                    MemoryDescUtils::convertToDnnlBlockedMemoryDesc(internalBlobs.front()->getTensorDesc()))});
                break;
            }
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
    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    auto inMemoryDesc = getParentEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
    auto outMemoryDesc = getChildEdgesAtPort(0).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();

    AttrPtr pAttrLocal;
    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = makePrimitiveAttr(dstMemPtr->getStaticDims());
        }
        pAttrLocal = pAttr;
        if (autoPad || externOutShape) {
            paddingL = shapeInference->get_pads_begin();
            paddingR = shapeInference->get_pads_end();
        }
        initPaddingR(inMemoryDesc->getShape(), outMemoryDesc->getShape());
    } else {
        pAttrLocal = makePrimitiveAttr(dstMemPtr->getStaticDims());
    }
    (*pAttrLocal).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    DnnlMemoryDescCPtr wghDesc;
    MemoryPtr biasMemPtr = nullptr;
    DnnlMemoryDescCPtr biasDesc;

    if (isInt8) {
        wghDesc = internalBlobMemory.front()->GetDescWithType<DnnlMemoryDesc>();
        if (withBiases) {
            biasMemPtr = getParentEdgesAtPort(biasPort)[0]->getMemoryPtr();
            if (!biasMemPtr || !biasMemPtr->isAllocated())
                IE_THROW() << "Bias memory  memory didn't allocate.";
            biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
        }
    } else {
        wghDesc = getParentEdgesAtPort(1).front()->getMemory().GetDescWithType<DnnlMemoryDesc>();
    }

    DeconvKey key = {inMemoryDesc,
                     wghDesc,
                     biasDesc,
                     outMemoryDesc,
                     stride,
                     dilation,
                     paddingL,
                     paddingR,
                     isInt8,
                     *pAttrLocal,
                     selected_pd->getImplementationType()};

    auto engine = getEngine();
    auto builder = [&engine](const DeconvKey& key) -> executorPtr {
        std::shared_ptr<DnnlDesriptor> desc;
        dnnl::memory::desc dnnlBiasDesc;
        if (key.isInt8) {
            if (key.bias)
                // WA to align IR bias representation (3 to 5 rank tensors) to oneDNN representation (1 rank tensor)
                dnnlBiasDesc = key.bias->getDnnlDesc().reshape({static_cast<dnnl::memory::dim>(key.out->getShape().getStaticDims()[1])});

            desc = createInt8MkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), dnnlBiasDesc, key.out->getDnnlDesc(),
                                                key.bias != nullptr, key.stride, key.dilation, key.paddingL, key.paddingR);
        } else {
            desc = createDefaultMkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), key.out->getDnnlDesc(),
                                                 (key.implType & impl_desc_type::winograd),
                                                 key.stride, key.dilation, key.paddingL, key.paddingR, engine);
        }

        auto itpd = desc->createPrimitiveDescriptorIterator(engine, key.attr);

        executorPtr execPtr = nullptr;
        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                if (key.isInt8) {
                    auto prim_desc = deconvolution_forward::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
                                                                   key.inp0->getDnnlDesc(),
                                                                   key.inp1->getDnnlDesc(),
                                                                   key.out->getDnnlDesc(),
                                                                   engine);
                } else {
                    auto prim_desc = convolution_backward_data::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorDefault>(prim_desc,
                                                                      key.inp0->getDnnlDesc(),
                                                                      key.inp1->getDnnlDesc(),
                                                                      key.out->getDnnlDesc(),
                                                                      engine);
                }
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

            std::shared_ptr<DnnlDesriptor> anyDeconvDesc;
            if (key.isInt8) {
                anyDeconvDesc = createInt8MkldnnDeconvDesc(inDesc, wghDesc, dnnlBiasDesc, outDesc, key.bias != nullptr,
                                                           key.stride, key.dilation, key.paddingL, key.paddingR);
            } else {
                anyDeconvDesc = createDefaultMkldnnDeconvDesc(inDesc, wghDesc, outDesc, (key.implType & impl_desc_type::winograd),
                                                              key.stride, key.dilation, key.paddingL, key.paddingR, engine);
            }
            auto anyDeconvItpd = anyDeconvDesc->createPrimitiveDescriptorIterator(engine, key.attr);
            if (static_cast<bool>(anyDeconvItpd)) {
                if (key.isInt8) {
                    auto prim_desc = deconvolution_forward::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
                                                                   key.inp0->getDnnlDesc(),
                                                                   key.inp1->getDnnlDesc(),
                                                                   key.out->getDnnlDesc(),
                                                                   engine);
                } else {
                    auto prim_desc = convolution_backward_data::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorDefault>(prim_desc,
                                                                      key.inp0->getDnnlDesc(),
                                                                      key.inp1->getDnnlDesc(),
                                                                      key.out->getDnnlDesc(),
                                                                      engine);
                }
            }
        }

        return execPtr;
    };

    execPtr = nullptr;
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;

    if (execPtr) {
        if (key.isInt8) {
            primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
            primArgs[DNNL_ARG_WEIGHTS] = internalBlobMemory.front()->GetPrimitive();
            primArgs[DNNL_ARG_DST]=  dstMemPtr->GetPrimitive();
            if (withBiases)
                primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
        } else {
            primArgs[DNNL_ARG_DIFF_DST] = srcMemPtr->GetPrimitive();
            primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->GetPrimitive();
            primArgs[DNNL_ARG_DIFF_SRC] = dstMemPtr->GetPrimitive();
        }
        Node::appendPostOpArgs(*pAttrLocal, primArgs, postOpsArgs);

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

void Deconvolution::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                               const std::vector<MemoryDescPtr> &outputDesc) {
    auto inDesc = inputDesc[0]->isDefined() ? inputDesc[0] : inputDesc[0]->cloneWithNewDims(inShape.getStaticDims());
    auto dnnlInDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*inDesc);
    auto in_candidate = dnnlInDesc.getDnnlDesc();

    auto outDesc = outputDesc[0];
    if (!outDesc->isDefined()) {
        const auto outShape = shapeInferInternal(inDesc->getShape().getStaticDims(), lastOutputSpatialDims);
        outDesc = outDesc->cloneWithNewDims(outShape);
    }
    auto dnnlOutDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*outDesc);
    auto out_candidate = dnnlOutDesc.getDnnlDesc();
    dnnl::memory::desc bias_candidate;

    // grouping and autoblocking is not compatible
    if ((withGroups && !isDW) && (dnnlInDesc.blocksExtended() || dnnlOutDesc.blocksExtended()))
        return;

    if (isInt8) {
        if (withBiases) {
            memory::data_type bdt = memory::data_type::f32;
            bias_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasesDims), bdt, memory::format_tag::any);
        }
        dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(int8WeightDims), memory::data_type::s8, memory::format_tag::any);
        descs.emplace_back(createDescriptorInternalInt8(in_candidate, wgh_candidate, bias_candidate,
                                                            out_candidate, withBiases, stride, dilation, paddingL, paddingR));
    } else {
        dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(getWeightDims()),
                                           dnnlInDesc.getDataType(), memory::format_tag::any);
        for (auto alg : {dnnl::algorithm::convolution_winograd, dnnl::algorithm::convolution_direct}) {
            std::shared_ptr<convolution_backward_data::desc> deconv_desc;
            std::shared_ptr<convolution_forward::primitive_desc> fwd_conv_pd;
            std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(in_candidate, wgh_candidate, out_candidate, alg,
                                                                                 stride, dilation, paddingL, paddingR, getEngine());
            if (fwd_conv_pd->get(true) == nullptr)
                continue;
            descs.emplace_back(deconv_desc, fwd_conv_pd);
        }
    }
}

std::shared_ptr<MemoryDesc> Deconvolution::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (idx == 2 && !withBiases) {
        return std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::I32, Shape(getInputShapeAtPort(2).getStaticDims()));
    } else if (idx > 0 && isInt8) {
        // we need to store 'weight' input as edge,
        // because at this moment we can't simple replace internal blob with input, since we need to save weight data as is, but with different order
        return std::make_shared<CpuBlockedMemoryDesc>(getOriginalInputPrecisionAtPort(idx), Shape(getInputShapeAtPort(idx).getStaticDims()));
    }

    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : isInt8 ? primitive_desc_it.src_desc(idx) : primitive_desc_it.diff_dst_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> Deconvolution::getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc =  isInt8 ? primitive_desc_it.dst_desc(idx) : primitive_desc_it.diff_src_desc(idx);
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

Deconvolution::DeconvExecutorDefault::DeconvExecutorDefault(const dnnl::convolution_backward_data::primitive_desc& pd,
                                                                      const dnnl::memory::desc& inMemDesc,
                                                                      const dnnl::memory::desc& weightMemDesc,
                                                                      const dnnl::memory::desc& outMemDesc,
                                                                      const dnnl::engine& engine) {
    execPrim = dnnl::convolution_backward_data(pd);

    if (inMemDesc != pd.diff_dst_desc()) {
        inputReorders.insert({DNNL_ARG_DIFF_DST, IntermReorder(inMemDesc, pd.diff_dst_desc(), engine)});
    }

    if (weightMemDesc != pd.weights_desc()) {
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, pd.weights_desc(), engine)});
    }

    if (outMemDesc != pd.diff_src_desc()) {
        outputReorders.insert({DNNL_ARG_DIFF_SRC, IntermReorder(pd.diff_src_desc(), outMemDesc, engine)});
    }
}

Deconvolution::DeconvExecutorInt8::DeconvExecutorInt8(const dnnl::deconvolution_forward::primitive_desc& pd,
                                                                const dnnl::memory::desc& inMemDesc,
                                                                const dnnl::memory::desc& weightMemDesc,
                                                                const dnnl::memory::desc& outMemDesc,
                                                                const dnnl::engine& engine) {
    execPrim = dnnl::deconvolution_forward(pd);

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
            (externOutShape ? getParentEdges().size() == 3 : getParentEdges().size() == 2));
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
