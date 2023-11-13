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
#include <common/primitive_hashing_utils.hpp>
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include <shape_inference/shape_inference_ngraph.hpp>

#if defined(OV_CPU_WITH_ACL)
#include "executors/acl/acl_utils.hpp"
#include "utils/debug_capabilities.h"
#endif

#include <oneapi/dnnl/dnnl.hpp>

#include <string>
#include <vector>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

using DefaultDeconvDescs = std::pair<dnnl::convolution_backward_data::primitive_desc,
                                     dnnl::convolution_forward::primitive_desc>;
using Int8DeconvDesc = dnnl::deconvolution_forward::primitive_desc;

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
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
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
    DeconfolutionShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (m_op->get_input_size() > 2) {
            return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), PortMask(2));
        }
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), EMPTY_PORT_MASK);
    }
private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace

bool Deconvolution::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
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

Deconvolution::Deconvolution(const std::shared_ptr<ov::Node>& op,
                             const GraphContext::CPtr context) : Node(op, context, DeconfolutionShapeInferFactory(op)) {
    std::string errorMessage;
    errorPrefix = "Deconvolution node with name '" + getName() + "' ";
    if (!isSupportedOperation(op, errorMessage))
        OPENVINO_THROW_NOT_IMPLEMENTED(errorPrefix + errorMessage);

    const auto& weightDims = getWeightDims();

    if (auto convBackprop = std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op)) {
        algorithm = Algorithm::DeconvolutionCommon;

        IC = weightDims[0];
        OC = weightDims[1];
        expectedBiasDims  = {OC};

        groupNum = 1;
        withGroups = false;

        for (size_t i = 0; i < convBackprop->get_strides().size(); i++) {
            deconvAttrs.stride.push_back(static_cast<ptrdiff_t>(convBackprop->get_strides()[i]));
        }
        for (size_t i = 0; i < convBackprop->get_dilations().size(); i++) {
            deconvAttrs.dilation.push_back(static_cast<ptrdiff_t>(convBackprop->get_dilations()[i]) - 1);
        }
        deconvAttrs.paddingL = convBackprop->get_pads_begin();
        deconvAttrs.paddingR = convBackprop->get_pads_end();

        deconvAttrs.outputPadding = convBackprop->get_output_padding();

        autoPad = one_of(convBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
    } else if (auto groupConvBackprop = std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op)) {
        algorithm = Algorithm::DeconvolutionGrouped;

        groupNum = weightDims[0];
        IC = groupNum * weightDims[1];
        OC = groupNum * weightDims[2];
        expectedBiasDims  = {OC * groupNum};
        withGroups = groupNum > 1;
        isDW = withGroups && groupNum == OC && groupNum == IC;

        for (size_t i = 0; i < groupConvBackprop->get_strides().size(); i++) {
            deconvAttrs.stride.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_strides()[i]));
        }
        for (size_t i = 0; i < groupConvBackprop->get_dilations().size(); i++) {
            deconvAttrs.dilation.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_dilations()[i]) - 1);
        }
        deconvAttrs.paddingL = groupConvBackprop->get_pads_begin();
        deconvAttrs.paddingR = groupConvBackprop->get_pads_end();

        deconvAttrs.outputPadding = groupConvBackprop->get_output_padding();

        autoPad = one_of(groupConvBackprop->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER);
    }
    for (size_t i = 0; i < deconvAttrs.dilation.size(); i++) {
        deconvAttrs.kernel.push_back(weightDims[withGroups + 2 + i]);
    }

    externOutShape = inputShapes.size() == 3;
    biasPort = externOutShape ? 3 : 2;
    if (externOutShape && isDynamicNode()) {
        bool isConstOutShape = ov::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        if (isConstOutShape) {
            lastOutputSpatialDims = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(2))->cast_vector<int32_t>();
        }
        const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
        if (getInputShapeAtPort(2).getStaticDims()[0] != spDimsNum || (isConstOutShape && lastOutputSpatialDims.size() != spDimsNum)) {
            OPENVINO_THROW(errorPrefix, "'output_shape' input has incorrect number of elements. Expected = ", spDimsNum);
        }
    }
    attr = std::make_shared<dnnl::primitive_attr>();
}

InferenceEngine::Blob::Ptr Deconvolution::createWeiBlobAsIO(InferenceEngine::SizeVector dims) {
    auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent());
    if (!constNode)
        OPENVINO_THROW("Cannot cast const input node for node ", getName(), ".");
    auto blb = constNode->getMemoryPtr();
    if (!blb)
        OPENVINO_THROW("Cannot get const weights blob for node ", getName(), ".");

    auto const blbSize = blb->getSize();

    // WA: In int8 case, we are processing weights using internal blob.
    InferenceEngine::SizeVector dimsForBlockedDesc{dims};
    std::swap(dimsForBlockedDesc[withGroups + 0], dimsForBlockedDesc[withGroups + 1]);

    VectorDims orderForBlockedDesc;
    if (withGroups) {
        orderForBlockedDesc = {0, 2, 1};
    } else {
        orderForBlockedDesc = {1, 0};
    }
    for (size_t i = 2 + withGroups; i < dimsForBlockedDesc.size(); i++)
        orderForBlockedDesc.push_back(i);

    BlockingDesc blkDesc(dimsForBlockedDesc, orderForBlockedDesc);
    InferenceEngine::TensorDesc tensorDesc(DnnlExtensionUtils::DataTypeToIEPrecision(blb->getDataType()), dims, blkDesc);

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<int8_t>(tensorDesc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    if (data == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";
    size_t intBuffSize = internalBlob->byteSize();

    size_t offset = blbSize;
    if (intBuffSize < offset) {
        OPENVINO_THROW("Cannot create internal buffer. Buffer can be overrun.");
    }
    cpu_memcpy_s(data, intBuffSize, blb->getData(), blbSize);

    return internalBlob;
}

bool Deconvolution::canBeExecutedInInt8() const {
    if (std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent()) == nullptr) {
        return false;
    }

    if (!withGroups && deconvAttrs.stride.back() > 3)
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        const auto& inMaxDims = getOutputShapeAtPort(0).getMaxDims();
        if (std::any_of(inMaxDims.begin(), inMaxDims.end(), [](Dim dim) { return dim == Shape::UNDEFINED_DIM; })) {
            return false;
        }
        // heuristicConst = 2^26
        // heuristicParam = IC^2 * SP
        size_t heuristicConst = 67108864;
        auto heuristicParam = IC * IC;
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
    if (withGroups && !isDW && (IC % channelBlock != 0 || OC % channelBlock != 0))
        return false;
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core) && deconvAttrs.stride.back() > 3)
        return false;

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(inPrecision);

    InferenceEngine::Precision weiPrecision = getOriginalInputPrecisionAtPort(1);
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(weiPrecision);

    if (isDW && (inputDataType == dnnl_s8 || deconvAttrs.dilation.size() == 3))
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

            const auto& origInDims = getInputShapeAtPort(0).getDims();
            const auto& origInMinDims = getInputShapeAtPort(0).getMinDims();
            const auto& origInMaxDims = getInputShapeAtPort(0).getMaxDims();
            const auto& weightDims = getWeightDims();
            const size_t wghOffset = getAlgorithm() == Algorithm::DeconvolutionGrouped ? 1 : 0;

            VectorDims paddings(deconvAttrs.paddingL.size());
            if (!autoPad) {
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
                        auto c1 = lastOutputSpatialDims[i] - deconvAttrs.outputPadding[i] - 1 -
                                    (deconvAttrs.dilation[i] + 1) * static_cast<int32_t>(weightDims[wghOffset + 2 + i] - 1);

                        if (origInMaxDims[i + 2] != Shape::UNDEFINED_DIM) {
                            auto upper_bound = deconvAttrs.stride[i] * static_cast<int32_t>(origInMaxDims[i + 2] - 1) - c1;
                            if (upper_bound < 0) {
                                OPENVINO_THROW(errorPrefix, ": paddings for dummy shapes can't be computed");
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
                    inputDims[2 + i] = (lastOutputSpatialDims[i] - (deconvAttrs.dilation[i] + 1) *
                                        (weightDims[wghOffset + 2 + i] - 1) - 1 + paddings[i] - deconvAttrs.outputPadding[i]) /
                                        deconvAttrs.stride[i] + 1;
                }
            }
        }
        inShape = Shape(inputDims);
        outShape = Shape(shapeInferInternal(inShape.getStaticDims(), lastOutputSpatialDims));
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
    isInt8 = canBeExecutedInInt8();
    deconvAttrs.withBiasesParam = withBiases = externOutShape ? getOriginalInputsNumber() == 4 : getOriginalInputsNumber() == 3;
    //ONEDNN deconvolution_fwd_t primitive can support bias fusing.
    //ONEDNN convolution_data_bwd_t can't support bias fusing.
    //Current only int8 precision choose deconvolution_fwd_t.
    if (withBiases && !isInt8) {
        OPENVINO_THROW(errorPrefix, " supports bias fusing only for int8 execution precision");
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
    if (getParentEdges().size() != (withBiases ? (biasPort + 1) : biasPort)) {
        OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
    }
    if (getChildEdges().empty()) {
        OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");
    }
    VectorDims inDims, outDims;
    std::tie(inDims, outDims) = makeDummyInOutShape();
    inShape = Shape(inDims);
    Shape outShape(outDims);
    initPaddingR(inShape, outShape);

#if defined(OV_CPU_WITH_ACL)
    NodeConfig config;
    config.inConfs.resize(getParentEdges().size());
    config.outConfs.resize(getOriginalOutputsNumber());

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        auto checkDesc = [&](LayoutType format) -> bool {
            NodeConfig config;
            config.inConfs.resize(getParentEdges().size());
            config.outConfs.resize(getOriginalOutputsNumber());

            for (size_t i = 0; i < getParentEdges().size(); ++i) {
                config.inConfs[i].setMemDesc(
                        creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i)));
            }
            config.outConfs[0].setMemDesc(
                    creatorsMap.at(format)->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0)));

            std::vector<MemoryDescPtr> srcMemoryDescs;
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            for (size_t i = 0; i < config.outConfs.size(); i++) {
                dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
            }

            return AclDeconvExecutorBuilder::customIsSupported(deconvAttrs, srcMemoryDescs, dstMemoryDescs);
        };
        useACL = checkDesc(LayoutType::nspc) || checkDesc(LayoutType::ncsp);
    }
    if (useACL) return;
#endif

    setPostOps(*attr, outShape.getStaticDims());

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
    //  [N, OC, OH, OW] = [N, IC, IH, IW]* [OC, IC, KH, KW]
    // ONEDNN define the convolution data backward as:
    //  [N, IC, OH, OW] = [N, OC, IH, IW]* [OC, IC, KH, KW]
    // So for the backward and forward convolutions, the weights dimensions definition in ONEDNN is the same.
    // OC is the conv forward output channel, IC is conv forward input channel.

    // But for the deconvolution, Deconv_OC and Deconv_IC are the deconv output and input channels respectively
    // ONEDNN defines the deconv OP as:
    // [N, Deconv_OC, OH, OW] = [N, Deconv_IC, IH, IW] * [Deconv_OC, Deconv_IC, KH, KW]
    // For deconv OP,  Deconv_OC = IC, Deconv_IC = OC.
    // Openvino per-channel weight scales are applied on IC/Deconv_OC dimension.
    // So for deconvolution,
    // Weight dims in NON-Group deconv: [Deconv_OC, Deconv_IC, KH, KW], perchannel weight scale is applied on Deconv_OC DIM
    //                                  weiScaleMaskPerChannel =  1 << 0
    // Weight dims in Group deconv:     [Group, Deconv_OC, Deconv_IC, KH, KW], perchannel weight scale is applied on GROUP and Deconv_OC,
    //                                   weiScaleMaskPerChannel = ( 1 << 0 | 1 << 1) = 0x03
    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, 1, isInt8, withGroups ? 3 : 1 << 0,  getDQScales(), withBiases);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
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

        OPENVINO_THROW("Fusing of ",
                       NameFromType(node->getType()),
                       " operation to ",
                       NameFromType(this->getType()),
                       " node is not implemented");
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
                    OPENVINO_THROW("Can't compute output shape for node with name: ",
                                   getName(),
                                   ", because the node has 'output_shape' input, but provided output spatial dims "
                                   "number is incorrect");
                }
                outSpDimsVecShape = {outSpDims.size()};
                inputShapesRefs.push_back(std::cref(outSpDimsVecShape));
                CpuBlockedMemoryDesc desc(Precision::I32, Shape(outSpDimsVecShape));
                auto mem = std::make_shared<Memory>(getEngine(), desc, outSpDims.data());
                inputValues[i] = mem;
                break;
            }
        }
    }

    auto result = shapeInference->infer(inputShapesRefs, inputValues);
    if (ShapeInferStatus::success != result.status) {
        OPENVINO_THROW("Unexpected: Unexpected shape inference result status in node of type ",
                       getTypeStr(),
                       " with name ",
                       getName());
    }
    return std::move(result.dims.back());
}

void Deconvolution::execute(dnnl::stream strm) {
    if (useACL) {
        std::vector<MemoryCPtr> srcMemory;
        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
        }
        std::vector<MemoryPtr> dstMemory;
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
        }
        //TODO: need to pass post ops data
        execPtrDeconv->exec(srcMemory, dstMemory, nullptr);
        return;
    }

    if (!execPtr) {
        OPENVINO_THROW("Can't execute Deconvolution node with name: ", getName(), ", because executor is not compiled");
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
                                                   const dnnl::algorithm alg,
                                                   const std::vector<ptrdiff_t>& stride,
                                                   const std::vector<ptrdiff_t>& dilation,
                                                   const ov::CoordinateDiff& paddingL,
                                                   const ov::CoordinateDiff& paddingR,
                                                   const dnnl::primitive_attr& attr,
                                                   const dnnl::engine& engine) {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    const dnnl::primitive_attr emptyAttr;

    auto conv_desc = convolution_forward::primitive_desc(
        engine,
        prop_kind::forward_inference,
        alg,
        out_candidate, wgh_candidate, in_candidate,
        convertDims(stride),
        convertDims(dilation),
        convertDims(paddingL),
        convertDims(paddingR),
        emptyAttr,
        true);

    if (!conv_desc.get(true)) {
        return {nullptr, nullptr};
    }

    auto deconv_desc = convolution_backward_data::primitive_desc(
        engine,
        alg,
        out_candidate, wgh_candidate, in_candidate,
        convertDims(stride),
        convertDims(dilation),
        convertDims(paddingL),
        convertDims(paddingR),
        conv_desc,
        attr,
        true);

    return {deconv_desc, conv_desc};
}

dnnl::primitive_desc createDescriptorInternalInt8(const dnnl::memory::desc& in_candidate,
                                                  const dnnl::memory::desc& wgh_candidate,
                                                  const dnnl::memory::desc& bias_candidate,
                                                  const dnnl::memory::desc& out_candidate,
                                                  const bool with_bias,
                                                  const std::vector<ptrdiff_t>& stride,
                                                  const std::vector<ptrdiff_t>& dilation,
                                                  const ov::CoordinateDiff& paddingL,
                                                  const ov::CoordinateDiff& paddingR,
                                                  const dnnl::primitive_attr& attr,
                                                  const dnnl::engine& engine) {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    if (with_bias) {
        return dnnl::deconvolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference,
            dnnl::algorithm::deconvolution_direct,
            in_candidate, wgh_candidate, bias_candidate, out_candidate,
            convertDims(stride), convertDims(dilation),
            convertDims(paddingL), convertDims(paddingR),
            attr);
    } else {
        return dnnl::deconvolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference,
            dnnl::algorithm::deconvolution_direct,
            in_candidate, wgh_candidate, out_candidate,
            convertDims(stride), convertDims(dilation),
            convertDims(paddingL), convertDims(paddingR),
            attr);
    }
}

DefaultDeconvDescs createDefaultMkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                                    const dnnl::memory::desc& wghDesc,
                                                                    const dnnl::memory::desc& dstDesc,
                                                                    const std::vector<ptrdiff_t>& stride,
                                                                    const std::vector<ptrdiff_t>& dilation,
                                                                    const ov::CoordinateDiff& paddingL,
                                                                    const ov::CoordinateDiff& paddingR,
                                                                    const dnnl::primitive_attr& attr,
                                                                    const dnnl::engine& engine) {
    dnnl::algorithm alg = dnnl::algorithm::convolution_direct;
    convolution_backward_data::primitive_desc deconv_desc;
    convolution_forward::primitive_desc fwd_conv_pd;
    std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(srcDesc, wghDesc, dstDesc, alg, stride, dilation, paddingL, paddingR, attr, engine);
    if (fwd_conv_pd.get(true) == nullptr) {
        OPENVINO_THROW("Forward convolution primitive descriptor is nullable");
    }

    return {deconv_desc, fwd_conv_pd};
}

dnnl::primitive_desc createInt8MkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                const dnnl::memory::desc& wghDesc,
                                                const dnnl::memory::desc& biasDesc,
                                                const dnnl::memory::desc& dstDesc,
                                                const bool withBias,
                                                const std::vector<ptrdiff_t>& stride,
                                                const std::vector<ptrdiff_t>& dilation,
                                                const ov::CoordinateDiff& paddingL,
                                                const ov::CoordinateDiff& paddingR,
                                                const dnnl::primitive_attr& attr,
                                                const dnnl::engine& engine) {
    return createDescriptorInternalInt8(
        srcDesc, wghDesc, biasDesc, dstDesc, withBias, stride, dilation, paddingL, paddingR, attr, engine);
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

        const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr) {
            OPENVINO_THROW("Preferable primitive descriptor is not set for node ", getName(), ".");
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

            inDesc = getParentEdgesAtPort(0).front()->getMemory().getDescWithType<DnnlMemoryDesc>();
            outDesc = getChildEdgesAtPort(0).front()->getMemory().getDescWithType<DnnlMemoryDesc>();
        }

        dnnl::memory::desc dnnlBiasDesc;
        if (withBiases) {
            DnnlMemoryDescPtr biasDesc = getParentEdgesAtPort(biasPort).front()->getMemory().getDescWithType<DnnlMemoryDesc>();
            dnnlBiasDesc = biasDesc->getDnnlDesc();
        }

        const AttrPtr pAttr = makePrimitiveAttr(outDims);
        auto prim_desc = createInt8MkldnnDeconvDesc(inDesc->getDnnlDesc(), wgh_candidate, dnnlBiasDesc, outDesc->getDnnlDesc(), withBiases,
                                               deconvAttrs.stride, deconvAttrs.dilation, deconvAttrs.paddingL, deconvAttrs.paddingR, *pAttr, getEngine());

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
        OPENVINO_THROW("Destination memory has not been allocated.");
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        OPENVINO_THROW("Input memory has not been allocated.");
    if (!wghMemPtr || !wghMemPtr->isAllocated())
        OPENVINO_THROW("Weight memory has not been allocated.");
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        OPENVINO_THROW("Preferable primitive descriptor is not set for node ", getName(), ".");

    if (useACL) {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemoryDescs.push_back(getParentEdgesAtPort(i).front()->getMemory().getDescWithType<DnnlMemoryDesc>());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemoryDescs.push_back(getChildEdgesAtPort(i).front()->getMemory().getDescWithType<DnnlMemoryDesc>());
        }

        execPtrDeconv = selected_pd->getExecutorFactoryAs<DeconvExecutorFactory>()->makeExecutor(deconvAttrs, srcMemoryDescs,
                                                                                                 dstMemoryDescs, *attr);
        selected_pd->setImplementationType(execPtrDeconv->getImplType());
        return;
    }

    auto inMemoryDesc = getParentEdgesAtPort(0).front()->getMemory().getDescWithType<DnnlMemoryDesc>();
    auto outMemoryDesc = getChildEdgesAtPort(0).front()->getMemory().getDescWithType<DnnlMemoryDesc>();

    AttrPtr pAttrLocal;
    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = makePrimitiveAttr(dstMemPtr->getStaticDims());
        }
        pAttrLocal = pAttr;
        if (autoPad || externOutShape) {
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

    if (isInt8) {
        wghDesc = internalBlobMemory.front()->getDescWithType<DnnlMemoryDesc>();
        if (withBiases) {
            biasMemPtr = getParentEdgesAtPort(biasPort)[0]->getMemoryPtr();
            if (!biasMemPtr || !biasMemPtr->isAllocated())
                OPENVINO_THROW("Bias memory  memory didn't allocate.");
            biasDesc = biasMemPtr->getDescWithType<DnnlMemoryDesc>();
        }
    } else {
        wghDesc = getParentEdgesAtPort(1).front()->getMemory().getDescWithType<DnnlMemoryDesc>();
    }

    DeconvKey key = {inMemoryDesc,
                     wghDesc,
                     biasDesc,
                     outMemoryDesc,
                     deconvAttrs.stride,
                     deconvAttrs.dilation,
                     deconvAttrs.paddingL,
                     deconvAttrs.paddingR,
                     isInt8,
                     *pAttrLocal,
                     selected_pd->getImplementationType()};

    auto engine = getEngine();
    auto builder = [&engine](const DeconvKey& key) -> executorPtr {
        dnnl::primitive_desc desc;
        convolution_forward::primitive_desc fwd_conv_pd;
        dnnl::memory::desc dnnlBiasDesc;
        if (key.isInt8) {
            if (key.bias)
                dnnlBiasDesc = key.bias->getDnnlDesc();

            desc = createInt8MkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), dnnlBiasDesc, key.out->getDnnlDesc(),
                                              key.bias != nullptr, key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, engine);
        } else {
            std::tie(desc, fwd_conv_pd) = createDefaultMkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), key.out->getDnnlDesc(),
                                                                        key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, engine);
#if defined(SELECTIVE_BUILD_ANALYZER)
            // Create dummy primitive to WA CC issue.
            OPENVINO_ASSERT(dnnl::primitive(fwd_conv_pd));
#endif
        }

        primitive_desc_iterator itpd = desc;
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

            dnnl::primitive_desc anyDeconvDesc;
            convolution_forward::primitive_desc fwdConvPd;

            if (key.isInt8) {
                anyDeconvDesc = createInt8MkldnnDeconvDesc(inDesc, wghDesc, dnnlBiasDesc, outDesc, key.bias != nullptr,
                                                           key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, engine);
            } else {
                std::tie(anyDeconvDesc, fwdConvPd) = createDefaultMkldnnDeconvDesc(inDesc, wghDesc, outDesc,
                                                              key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, engine);
#if defined(SELECTIVE_BUILD_ANALYZER)
                // Create dummy primitive to WA CC issue.
                OPENVINO_ASSERT(dnnl::primitive(fwd_conv_pd));
#endif
            }

            if (anyDeconvDesc) {
                if (key.isInt8) {
                    auto prim_desc = deconvolution_forward::primitive_desc(anyDeconvDesc.get());
                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
                                                                   key.inp0->getDnnlDesc(),
                                                                   key.inp1->getDnnlDesc(),
                                                                   key.out->getDnnlDesc(),
                                                                   engine);
                } else {
                    auto prim_desc = convolution_backward_data::primitive_desc(anyDeconvDesc.get());
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
            primArgs[DNNL_ARG_SRC] = srcMemPtr->getPrimitive();
            primArgs[DNNL_ARG_WEIGHTS] = internalBlobMemory.front()->getPrimitive();
            primArgs[DNNL_ARG_DST]=  dstMemPtr->getPrimitive();
            if (withBiases)
                primArgs[DNNL_ARG_BIAS] = biasMemPtr->getPrimitive();
        } else {
            primArgs[DNNL_ARG_DIFF_DST] = srcMemPtr->getPrimitive();
            primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->getPrimitive();
            primArgs[DNNL_ARG_DIFF_SRC] = dstMemPtr->getPrimitive();
        }
        Node::appendPostOpArgs(*pAttrLocal, primArgs, postOpsArgs);

        auto scratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
        primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            auto pd = execPtr->getPrimitiveDesc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
    } else {
        OPENVINO_THROW("Primitive descriptor was not found for node ", getName(), ".");
    }
}

void Deconvolution::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                     const std::vector<MemoryDescPtr> &outputDesc) {
    auto inDesc = inputDesc[0]->isDefined() ? inputDesc[0] : inputDesc[0]->cloneWithNewDims(inShape.getStaticDims());
    auto dnnlInDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*inDesc);
    const auto& in_candidate = dnnlInDesc.getDnnlDesc();

    auto outDesc = outputDesc[0];
    if (!outDesc->isDefined()) {
        const auto outShape = shapeInferInternal(inDesc->getShape().getStaticDims(), lastOutputSpatialDims);
        outDesc = outDesc->cloneWithNewDims(outShape);
    }
    auto dnnlOutDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(*outDesc);
    const auto& out_candidate = dnnlOutDesc.getDnnlDesc();
    dnnl::memory::desc bias_candidate;

    // grouping and autoblocking is not compatible
    if ((withGroups && !isDW) && (dnnlInDesc.blocksExtended() || dnnlOutDesc.blocksExtended()))
        return;

    AttrPtr attr = initPrimitiveAttr();
    if (isInt8) {
        if (withBiases) {
            memory::data_type bdt = memory::data_type::f32;
            bias_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(expectedBiasDims), bdt, memory::format_tag::any);
        }
        dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(int8WeightDims), memory::data_type::s8, memory::format_tag::any);
        descs.emplace_back(createDescriptorInternalInt8(in_candidate, wgh_candidate, bias_candidate,
                                                        out_candidate, withBiases, deconvAttrs.stride, deconvAttrs.dilation,
                                                        deconvAttrs.paddingL, deconvAttrs.paddingR, *attr, getEngine()));
    } else {
        dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(getWeightDims()),
                                           dnnlInDesc.getDataType(), memory::format_tag::any);
        convolution_backward_data::primitive_desc deconv_desc;
        convolution_forward::primitive_desc fwd_conv_pd;
        std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(in_candidate, wgh_candidate, out_candidate, dnnl::algorithm::convolution_direct,
                                                                                deconvAttrs.stride, deconvAttrs.dilation, deconvAttrs.paddingL,
                                                                                deconvAttrs.paddingR, *attr, getEngine());
        if (fwd_conv_pd && deconv_desc && deconv_desc.get(true) != nullptr) {
            fwdConvPD.push_back(fwd_conv_pd);  // oneDNN requires forward pd to exists until primitive is created
            descs.push_back(deconv_desc);
        }
    }
}

std::shared_ptr<MemoryDesc> Deconvolution::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    if (idx == 2 && !withBiases) {
        return std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::I32, Shape(getInputShapeAtPort(2).getStaticDims()));
    } else if (idx > 0 && isInt8) {
        // we need to store 'weight' input as edge,
        // because at this moment we can't simple replace internal blob with input, since we need to save weight data as is, but with different order
        return std::make_shared<CpuBlockedMemoryDesc>(getOriginalInputPrecisionAtPort(idx), Shape(getInputShapeAtPort(idx).getStaticDims()));
    }

    auto desc = idx > 0 ? prim_desc.weights_desc(idx - 1) : isInt8 ? prim_desc.src_desc(idx) : prim_desc.diff_dst_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> Deconvolution::getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc =  isInt8 ? prim_desc.dst_desc(idx) : prim_desc.diff_src_desc(idx);
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
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

Deconvolution::DeconvExecutorDefault::DeconvExecutorDefault(const dnnl::convolution_backward_data::primitive_desc& pd,
                                                                      const dnnl::memory::desc& inMemDesc,
                                                                      const dnnl::memory::desc& weightMemDesc,
                                                                      const dnnl::memory::desc& outMemDesc,
                                                                      const dnnl::engine& engine) : DnnlExecutor(pd) {
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
                                                                const dnnl::engine& engine) : DnnlExecutor(pd) {
    if (inMemDesc != getDnnlSrcDesc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, getDnnlSrcDesc(), engine)});
    }

    if (weightMemDesc != getDnnlWeightDesc()) {
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, getDnnlWeightDesc(), engine)});
    }

    if (outMemDesc != getDnnlDstDesc()) {
        outputReorders.insert({DNNL_ARG_DST, IntermReorder(getDnnlDstDesc(), outMemDesc, engine)});
    }
}

std::vector<int32_t> Deconvolution::readOutputSpatialDims() const {
    if (getParentEdges().size() < 3) {
        OPENVINO_THROW("Can't get output spatial dims. Inputs number = ", getParentEdges().size());
    }
    const auto &shapeMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
    if (!shapeMemPtr || !shapeMemPtr->isAllocated()) {
        OPENVINO_THROW("'output_shape' input memory is not allocated.");
    }
    const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
    if (shapeMemPtr->getStaticDims()[0] != spDimsNum) {
        OPENVINO_THROW("Can't read output spatial dims, beause 'output_shape' input has incorrect number of elements");
    }
    const int32_t *outShapePtr = reinterpret_cast<const int32_t *>(shapeMemPtr->getData());
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

void Deconvolution::initSupportedPrimitiveDescriptors() {
    if (!useACL) {
        Node::initSupportedPrimitiveDescriptors();
        return;
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format) {
        NodeConfig config;
        config.inConfs.resize(getParentEdges().size());
        config.outConfs.resize(getOriginalOutputsNumber());

        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            config.inConfs[i].setMemDesc(
                // ACL expected equal precision
                creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(i)));
        }
        config.outConfs[0].setMemDesc(
                // ACL expected equal precision
                creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getOutputShapeAtPort(0)));

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

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::acl, factory);
    };
    pushDesc(LayoutType::ncsp);
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
