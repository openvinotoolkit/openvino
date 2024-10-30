// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv.h"

#include "dnnl_extension_utils.h"
#include <memory_desc/cpu_memory_desc_utils.h>
#include <nodes/common/cpu_memcpy.h>

#include "common/primitive_hashing_utils.hpp"
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include "cpu/x64/cpu_isa_traits.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"

#include "eltwise.h"
#include "fake_quantize.h"
#include "input.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "executors/acl/acl_utils.hpp"
#include "utils/debug_capabilities.h"
#endif

#include <oneapi/dnnl/dnnl.hpp>

#include <string>
#include <vector>

using namespace dnnl;

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

    bool constWeight;
    bool isImplicit1x1PaddingAsymmetric;

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

    seed = hash_combine(seed, constWeight);
    seed = hash_combine(seed, isImplicit1x1PaddingAsymmetric);

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

    retVal = retVal && constWeight == rhs.constWeight;
    retVal = retVal && isImplicit1x1PaddingAsymmetric == rhs.isImplicit1x1PaddingAsymmetric;

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
        if (std::dynamic_pointer_cast<const ov::op::v1::ConvolutionBackpropData>(op) == nullptr &&
                std::dynamic_pointer_cast<const ov::op::v1::GroupConvolutionBackpropData>(op) == nullptr) {
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

    if (auto convBackprop = std::dynamic_pointer_cast<const ov::op::v1::ConvolutionBackpropData>(op)) {
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
    } else if (auto groupConvBackprop = std::dynamic_pointer_cast<const ov::op::v1::GroupConvolutionBackpropData>(op)) {
        algorithm = Algorithm::DeconvolutionGrouped;

        groupNum = weightDims[0];
        IC = groupNum * weightDims[1];
        OC = groupNum * weightDims[2];
        expectedBiasDims  = {OC};
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
#if defined(OV_CPU_WITH_ACL)
    deconvAttrs.aclFastMath = context->getConfig().aclFastMath;
#endif

    externOutShape = inputShapes.size() == 3;
    biasPort = externOutShape ? 3 : 2;
    if (externOutShape && (isConstOutShape = ov::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2))))
        lastOutputSpatialDims = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(2))->cast_vector<int32_t>();
    if (externOutShape && isDynamicNode()) {
        const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
        if (getInputShapeAtPort(2).getStaticDims()[0] != spDimsNum || (isConstOutShape && lastOutputSpatialDims.size() != spDimsNum)) {
            OPENVINO_THROW(errorPrefix, "'output_shape' input has incorrect number of elements. Expected = ", spDimsNum);
        }
    }

    size_t spatialRank = getInputShapeAtPort(0).getRank() - 2;
    auto weightDimsReversItr = weightDims.crbegin();
    is1x1 = true;
    for (size_t i = 0; i < spatialRank; ++i)
        is1x1 = is1x1 && *(weightDimsReversItr++) == 1;
    // 1x1 deconv has some test case failed. The cause is upstream ONEDNN unsupported brgemm implementation cases are
    // enabled in forked ONEDNNN https://github.com/openvinotoolkit/oneDNN/blob/117e287000b48a34a7218fcaa274a91571141728/src/common/convolution.cpp#L138.
    // Some test cases on 1x1 kernel failed on accuracy check, current WA is disabling brgemm deconv implementation for such cases.
    if (is1x1 && deconvAttrs.paddingL != deconvAttrs.paddingR) {
        // case1: Specify asymmetric padding explicitly
        asymmetricPaddingAnd1x1 = true;
    } else if (isConstOutShape && !isDynamicNode()) {
        // case2: Implicit asymmetric padding cases.
        asymmetricPaddingAnd1x1 = isImplicit1x1PaddingAsymmetric(getInputShapeAtPort(0).getStaticDims());
    }
    attr = std::make_shared<dnnl::primitive_attr>();
}

void Deconvolution::createDnnlCompatibleWeights() {
    MemoryPtr blob = getSrcMemoryAtPort(1);

    if (!blob)
        OPENVINO_THROW("Cannot get const weights blob for node ", getName(), ".");

    weightIsConst = getParentEdgeAt(1)->getParent()->isConstant();
    auto blockedDims = getWeightDims();
    VectorDims order;
    if (withGroups) {
        order = {0, 2, 1};
    } else {
        order = {1, 0};
    }
    for (size_t i = 2 + withGroups; i < blockedDims.size(); i++)
        order.push_back(i);

    auto desc = CpuBlockedMemoryDesc(DnnlExtensionUtils::DataTypeToElementType(blob->getDataType()),
                                     Shape(dnnlCompatibleWeiDims),
                                     blockedDims,
                                     order);
    // Create the memory with the edge memory block. In the case of the weight memory changes when inference,
    // dnnlCompatibleWeights memory would be updated automatically via update inform mechanism.
    dnnlCompatibleWeights = std::make_shared<Memory>(getEngine(), desc, blob->getMemoryBlock());
}

bool Deconvolution::canBeExecutedInInt8() const {
    if (std::dynamic_pointer_cast<Input>(getParentEdgeAt(1)->getParent()) == nullptr) {
        return false;
    }
    if (!one_of(getInputShapeAtPort(0).getRank(), 3ul, 4ul, 5ul)) {
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

    ov::element::Type inPrecision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = DnnlExtensionUtils::ElementTypeToDataType(inPrecision);

    ov::element::Type weiPrecision = getOriginalInputPrecisionAtPort(1);
    auto weightsDataType = DnnlExtensionUtils::ElementTypeToDataType(weiPrecision);

    if (isDW && (inputDataType == dnnl_s8 || deconvAttrs.dilation.size() == 3))
        return false;

    return (inputDataType == dnnl_s8 || inputDataType == dnnl_u8) && weightsDataType == dnnl_s8;
}

bool Deconvolution::canFuse(const NodePtr& node) const {
    if (canBeExecutedInInt8())
        return canFuseSimpleOperation(node);
    // Upstream ONEDNN conv_backward_data primitive can't support any post-ops, fork onednn added depthwise support in conv_backward_data JIT implementation.
    // ONEDNN deconv primitive can support most of post-ops, but the post-ops implementation details are different.
    // So current deconv implementation list in onednn has 2 kinds of implements:
    //    1. deconv implementation with JIT post-ops supported in the kernel (such as brgdeconv)
    //    2. forked conv_data_backwards implementation with JIT depthwise post-ops + reference implementation for other post ops.
    // Considering that some deconv fallback on the JIT implementation, we limit the post ops fusing to avoid regressions.
    // Regression with stylegan2 int8 model pattern:
    // none-quantzied deconv(with none-const weight) + FQ pattern fall back on JIT because of onednn limitation. (fall back ticket MFDNN-11577).
    // If FQ is fused, it runs with the ref post-ops implementation.
    // @todo: if onednn can ensure all the deconv run with the brgemm implementation, we can unify the fuse criteria between int8 and fp32 use cases.
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
        return {memory::format_tag::ncw,
                memory::format_tag::nCw8c,
                memory::format_tag::nCw16c,
                memory::format_tag::nwc};
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

    ov::element::Type inPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outPrecision = getOriginalOutputPrecisionAtPort(0);
    if (isInt8) {
        // TODO: We have to extend jit_avx512_core_x8s8s32x_deconv_fwd_kernel from oneDNN to support BF16 output data type
        if (ov::element::bf16 == inPrecision)
            inPrecision = ov::element::f32;
        if (ov::element::bf16 == outPrecision)
            outPrecision = ov::element::f32;
    } else {
        if (!inPrecision.is_real())
            inPrecision = ov::element::f32;
        if (!outPrecision.is_real())
            outPrecision = ov::element::f32;
    }
    auto inputDataType = DnnlExtensionUtils::ElementTypeToDataType(inPrecision);
    outputDataType = DnnlExtensionUtils::ElementTypeToDataType(outPrecision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;
    if (inputDataType == memory::data_type::f16 || outputDataType == memory::data_type::f16)
       inputDataType = outputDataType = memory::data_type::f16;
    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::ElementTypeToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    if (getParentEdges().size() != (withBiases ? (biasPort + 1) : biasPort)) {
        OPENVINO_THROW(errorPrefix, " has incorrect number of input edges");
    }
    if (getChildEdges().empty()) {
        OPENVINO_THROW(errorPrefix, " has incorrect number of output edges");
    }
    VectorDims inDims, outDims;
    std::tie(inDims, outDims) = makeDummyInOutShape();
    inShape  = Shape(inDims);
    outShape = Shape(outDims);
    initPaddingR(inShape, outShape);

#if defined(OV_CPU_WITH_ACL)
    NodeConfig config;
    config.inConfs.resize(getParentEdges().size());
    config.outConfs.resize(getOriginalOutputsNumber());

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto checkDesc = [&](LayoutType format, LayoutType weights_format = LayoutType::ncsp) -> bool {
        NodeConfig config;
        config.inConfs.resize(getParentEdges().size());
        config.outConfs.resize(getOriginalOutputsNumber());
        // ACL use same precision for all inputs
        config.inConfs[0].setMemDesc(
                creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(0)));
        config.inConfs[1].setMemDesc(
                creatorsMap.at(weights_format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(1)));
        for (size_t i = 2; i < getParentEdges().size(); ++i) {
            config.inConfs[i].setMemDesc(
                    creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(i)));
        }

        for (size_t i = 0; i < config.outConfs.size(); ++i) {
            config.outConfs[i].setMemDesc(
                    creatorsMap.at(format)->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(i)));
        }

        std::vector<MemoryDescPtr> srcMemoryDescs;
        srcMemoryDescs.push_back(config.inConfs[0].getMemDesc()->cloneWithNewDims(inDims));
        for (size_t i = 1; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc()->clone());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(config.outConfs[0].getMemDesc()->cloneWithNewDims(outDims));
        for (size_t i = 1; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc()->clone());
        }

        return AclDeconvExecutorBuilder::customIsSupported(deconvAttrs, srcMemoryDescs, dstMemoryDescs);
    };
    useACL = checkDesc(LayoutType::nspc) || checkDesc(LayoutType::ncsp);
    if (useACL) return;
#endif
    dnnlCompatibleWeiDims = getWeightDims();
    // Construct the ONEDNN deconv OP weight shape.
    // OV ConvBackWardData defines weight shape as [Conv_OC, Conv_IC, ....].
    // ONEDNN Deconv define weight shape as [Deconv_OC, Deconv_IC,...],
    // Deconv_OC = Conv_IC , Deconv_IC = Conv_OC
    std::swap(dnnlCompatibleWeiDims[withGroups + 0], dnnlCompatibleWeiDims[withGroups + 1]);
    setPostOps(*attr, outShape.getStaticDims());

    if (isInt8) {
        const auto& rank = getInputShapeAtPort(0).getRank();
        auto format = rank == 5   ? dnnl::memory::format_tag::ndhwc
                      : rank == 4 ? dnnl::memory::format_tag::nhwc
                                  : dnnl::memory::format_tag::nwc;
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
    DnnlPostOpsComposerLegacy dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, 1, isInt8, withGroups ? 3 : 1 << 0,  getDQScales(), withBiases);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));
        // Abandon legacy post-ops invocation in deconv.
        // @todo: Remove the legacy post-ops implementations for conv_backward_data in the fork onednn
        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            DEBUG_LOG(getName(), ": Append ", node->getName(), " as original post op with binary");
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
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
                CpuBlockedMemoryDesc desc(ov::element::i32, Shape(outSpDimsVecShape));
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
            srcMemory.push_back(getSrcMemoryAtPort(i));
        }
        std::vector<MemoryPtr> dstMemory;
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemory.push_back(getDstMemoryAtPort(i));
        }
        //TODO: need to pass post ops data
        execPtrDeconvACL->exec(srcMemory, dstMemory, nullptr);
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
dnnl::primitive_desc createDescriptorInternal(const dnnl::memory::desc& in_candidate,
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
} // namespace

Node::AttrPtr Deconvolution::makePrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims);

    return attr;
}

Node::AttrPtr Deconvolution::initPrimitiveAttr() {
    return attr;
}

const std::vector<impl_desc_type>& Deconvolution::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities {
        impl_desc_type::unknown,
        // Undef impl type is used to express use-cases there real type is unkown during compilation
        // Undef has higher priority than defined types in order to force primitive selection logic to make decision based on other properties
        impl_desc_type::undef,
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
#if defined(OPENVINO_ARCH_ARM64)
        impl_desc_type::jit_asimd,
#endif
        impl_desc_type::gemm_any,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_acl,
        impl_desc_type::acl,
        impl_desc_type::jit_gemm,
        impl_desc_type::ref_any,
        impl_desc_type::ref,
    };

    if (!asymmetricPaddingAnd1x1)
        return priorities;

    static const std::vector<impl_desc_type> priorities_wo_brgemm = [&] {
        std::vector<impl_desc_type>result;
        std::copy_if(priorities.begin(), priorities.end(), std::back_inserter(result),
            [](impl_desc_type type) { return !(type & impl_desc_type::brgconv); });
        return result;}();
    return priorities_wo_brgemm;
}

bool Deconvolution::isImplicit1x1PaddingAsymmetric(const VectorDims& inputDims) {
    auto isZero = [](std::ptrdiff_t i) { return i == 0; };
    size_t spatialRank = getInputShapeAtPort(0).getRank() - 2;
    if (is1x1 && std::all_of(deconvAttrs.paddingR.begin(), deconvAttrs.paddingR.end(), isZero)
                        && std::all_of(deconvAttrs.paddingL.begin(), deconvAttrs.paddingL.end(), isZero)
                        && std::all_of(deconvAttrs.outputPadding.begin(), deconvAttrs.outputPadding.end(), isZero)
                       ) {
            auto calPaddingEnd =  [](int64_t i, int64_t o,  int64_t s) -> int64_t {
                // Accoriding to https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html,
                // output[i] = (input[i] -1) * stride[i] - 2 x padding[i] + dilation[i] x (kernel_size[i] - 1) + output_padding[i] + 1.
                // When kernel_size[i] = 1, output_padding = 0, output[i] = (input[i] -1) * stride[i]  - 2 x padding[i] + 1.
                // implicit padding end =  2 x padding[i]  = (input[i] -1) * stride[i] + 1 - output[i]
                return (i - 1) * s + 1 - o;};
            for (size_t i = 0; i < spatialRank; i++) {
                int64_t inputDim = static_cast<int64_t>(inputDims[i + 2]);
                int64_t outputDim = static_cast<int64_t>(lastOutputSpatialDims[i]);
                int64_t stride = static_cast<int64_t>(deconvAttrs.stride[i]);
                if (calPaddingEnd(inputDim, outputDim, stride) > 0) {
                    return true;
                }
            }
    }
    return false;
}

void Deconvolution::prepareParams() {
    auto srcMemPtr = getSrcMemoryAtPort(0);
    auto wghMemPtr = getSrcMemoryAtPort(1);
    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined())
        OPENVINO_THROW("Destination memory is undefined.");
    if (!srcMemPtr || !srcMemPtr->isDefined())
        OPENVINO_THROW("Input memory is undefined.");
    if (!wghMemPtr || !wghMemPtr->isDefined())
        OPENVINO_THROW("Weight memory is undefined.");
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        OPENVINO_THROW("Preferable primitive descriptor is not set for node ", getName(), ".");

    if (useACL) {
        if (isDynamicNode()) {
            initPaddingR(getParentEdgeAt(0)->getMemory().getDescPtr()->getShape(),
                         getChildEdgeAt(0)->getMemory().getDescPtr()->getShape());
        }
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemory().getDescWithType<DnnlMemoryDesc>());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemory().getDescWithType<DnnlMemoryDesc>());
        }

        execPtrDeconvACL = selected_pd->getExecutorFactoryAs<DeconvExecutorFactory>()->makeExecutor(deconvAttrs, srcMemoryDescs,
                                                                                                 dstMemoryDescs, *attr);
        selected_pd->setImplementationType(execPtrDeconvACL->getImplType());
        return;
    }
    auto inMemoryDesc = getParentEdgeAt(0)->getMemory().getDescWithType<DnnlMemoryDesc>();
    auto outMemoryDesc = getChildEdgeAt(0)->getMemory().getDescWithType<DnnlMemoryDesc>();

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

    MemoryPtr biasMemPtr = nullptr;
    DnnlMemoryDescCPtr biasDesc;

    if (!dnnlCompatibleWeights)
        createDnnlCompatibleWeights();
    DnnlMemoryDescPtr wghDesc = dnnlCompatibleWeights->getDescWithType<DnnlMemoryDesc>();

    if (withBiases) {
        biasMemPtr = getSrcMemoryAtPort(biasPort);
        if (!biasMemPtr || !biasMemPtr->isDefined())
            OPENVINO_THROW("Bias memory  memory is undefined.");
        biasDesc = biasMemPtr->getDescWithType<DnnlMemoryDesc>();
    }
    bool is1x1PaddingAsymmetric  = false;
    if (externOutShape && (!isConstOutShape || isDynamicNode())) {
        // Check implicit asymmetric padding case for dynamic case and runtime output shape.
        is1x1PaddingAsymmetric = isImplicit1x1PaddingAsymmetric(getSrcMemoryAtPort(0)->getShape().getStaticDims());
    }
    DeconvKey key = {inMemoryDesc,
                     wghDesc,
                     biasDesc,
                     outMemoryDesc,
                     deconvAttrs.stride,
                     deconvAttrs.dilation,
                     deconvAttrs.paddingL,
                     deconvAttrs.paddingR,
                     weightIsConst,
                     is1x1PaddingAsymmetric,
                     *pAttrLocal,
                     selected_pd->getImplementationType()};

    auto engine = getEngine();

    auto builder = [&engine](const DeconvKey& key) -> executorPtr {
        dnnl::primitive_desc desc;
        convolution_forward::primitive_desc fwd_conv_pd;
        dnnl::memory::desc dnnlBiasDesc;
        const auto& weiDims = key.inp1->getShape().getStaticDims();
        const auto srcDataType = key.inp0->getDataType();
        const auto weiDataType = (one_of(srcDataType, memory::data_type::s8, memory::data_type::u8)) ?
                                    memory::data_type::s8 : srcDataType;
        auto wghDescAny =
            dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(weiDims),
                               weiDataType,
                               memory::format_tag::any);
        if (key.bias)
            dnnlBiasDesc = key.bias->getDnnlDesc();

        desc = createDescriptorInternal(key.inp0->getDnnlDesc(), wghDescAny, dnnlBiasDesc, key.out->getDnnlDesc(),
                                            key.bias != nullptr, key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, engine);

        primitive_desc_iterator itpd = desc;
        executorPtr execPtr = nullptr;

        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            //Skip the brgemm implemenation for asymmetric padding case because of the accuracy issue.
            if (key.isImplicit1x1PaddingAsymmetric && (impl_type & impl_desc_type::brgconv))
                continue;
            if (impl_type == key.implType) {
                auto prim_desc = deconvolution_forward::primitive_desc(itpd.get());
                execPtr = std::make_shared<DeconvDNNLExecutor>(prim_desc,
                                                            key.inp0->getDnnlDesc(),
                                                            key.inp1->getDnnlDesc(),
                                                            key.out->getDnnlDesc(),
                                                            engine,
                                                            key.constWeight);
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
            auto outDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.out->getShape().getStaticDims()),
                                                                                        key.out->getDataType(),
                                                                                        memory::format_tag::any);

            dnnl::primitive_desc anyDeconvDesc;
            anyDeconvDesc = createDescriptorInternal(inDesc, wghDescAny, dnnlBiasDesc, outDesc, key.bias != nullptr,
                                                        key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, engine);
            if (anyDeconvDesc) {
                auto prim_desc = deconvolution_forward::primitive_desc(anyDeconvDesc.get());
                execPtr = std::make_shared<DeconvDNNLExecutor>(prim_desc,
                                                               key.inp0->getDnnlDesc(),
                                                               key.inp1->getDnnlDesc(),
                                                               key.out->getDnnlDesc(),
                                                               engine,
                                                               key.constWeight);
            }
        }

        return execPtr;
    };

    auto prevExecPtr = execPtr;
    execPtr = nullptr;
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);


    execPtr = result.first;
    if (!execPtr)
        OPENVINO_THROW("Primitive descriptor was not found for node ", getName(), ".");

    primArgs[DNNL_ARG_SRC] = srcMemPtr->getPrimitive();
    primArgs[DNNL_ARG_DST]=  dstMemPtr->getPrimitive();
    if (weightIsConst) {
        // const weight preparation/reordering needs to be done once at next execution
        // when the input weight data is guaranteed to be ready (considering possible const-folding
        // subgraphs inserted between constant weight node and conv)
        auto it = primArgs.find(DNNL_ARG_WEIGHTS);
        if (it == primArgs.end() || !prevExecPtr ||
            !execPtr->getWeightDesc()->isCompatible(*(prevExecPtr->getWeightDesc()))) {
            primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(execPtr->getWeightDesc(), wghDesc)->getPrimitive();
        }
    } else {
        // non-const weight will be reordered by executor on every exec
        primArgs[DNNL_ARG_WEIGHTS] = dnnlCompatibleWeights->getPrimitive();
    }

    if (withBiases)
        primArgs[DNNL_ARG_BIAS] = biasMemPtr->getPrimitive();

    Node::appendPostOpArgs(*pAttrLocal, primArgs, postOpsArgs);

    auto scratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
#ifdef CPU_DEBUG_CAPS
    auto pd = execPtr->getPrimitiveDesc();
    DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
#endif
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
    if (withBiases) {
        memory::data_type bdt = memory::data_type::f32;
        bias_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(expectedBiasDims), bdt, memory::format_tag::any);
    }
    dnnl::memory::desc wgh_candidate(DnnlExtensionUtils::convertToDnnlDims(dnnlCompatibleWeiDims), isInt8 ? memory::data_type::s8 : dnnlInDesc.getDataType(),
                                    memory::format_tag::any);
    descs.emplace_back(createDescriptorInternal(in_candidate, wgh_candidate, bias_candidate,
                                                    out_candidate, withBiases, deconvAttrs.stride, deconvAttrs.dilation,
                                                    deconvAttrs.paddingL, deconvAttrs.paddingR, *attr, getEngine()));
}

std::shared_ptr<MemoryDesc> Deconvolution::getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    if (idx == 2 && !withBiases) {
        //Expected dest shape;
        return std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, Shape(getInputShapeAtPort(2).getStaticDims()));
    } else if (idx > 0) {
        // weight and bias are exposed with the planar layout.
        // we need to store 'weight' input as edge,
        // because at this moment we can't simple replace internal blob with input, since we need to save weight data as is, but with different order
        return std::make_shared<CpuBlockedMemoryDesc>(getOriginalInputPrecisionAtPort(idx), Shape(getInputShapeAtPort(idx).getStaticDims()));
    }
    //idx =0 case
    auto desc = prim_desc.src_desc(idx);
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> Deconvolution::getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const {
    auto desc =  prim_desc.dst_desc(idx);
    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(desc);
}

ov::element::Type Deconvolution::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToElementType((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

Deconvolution::DeconvDNNLExecutor::DeconvDNNLExecutor(const dnnl::deconvolution_forward::primitive_desc& pd,
                                                                const dnnl::memory::desc& inMemDesc,
                                                                const dnnl::memory::desc& weightMemDesc,
                                                                const dnnl::memory::desc& outMemDesc,
                                                                const dnnl::engine& engine,
                                                                bool constWeight) : DnnlExecutor(pd) {
    if (inMemDesc != getDnnlSrcDesc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, getDnnlSrcDesc(), engine)});
    }

    if (!constWeight && weightMemDesc != getDnnlWeightDesc()) {
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
    const auto &shapeMemPtr = getSrcMemoryAtPort(2);
    if (!shapeMemPtr || !shapeMemPtr->isDefined()) {
        OPENVINO_THROW("'output_shape' input memory is undefined.");
    }
    const auto spDimsNum = getInputShapeAtPort(0).getRank() - 2;
    if (shapeMemPtr->getStaticDims()[0] != spDimsNum) {
        OPENVINO_THROW("Can't read output spatial dims, beause 'output_shape' input has incorrect number of elements");
    }
    const int32_t *outShapePtr = shapeMemPtr->getDataAs<const int32_t>();
    std::vector<int32_t> outSpDims(outShapePtr, outShapePtr + shapeMemPtr->getStaticDims()[0]);
    return outSpDims;
}

bool Deconvolution::canFuseBias() const {
    //ONEDNN deconvolution_fwd_t primitive can support bias fusing. but has different implementations.
    //For the brgdeconv implementation in the deconv list, bias is implemented via JIT kernel.
    //For the fall back ref implementation entry(previous conv_backward_data), bias is implemented via reference post-ops.
    //It is difficult to recognize  whether the deconv will run with brg or fall back to backwards data implementation on the fusing
    //transformation stage. In the end, all the deconv should run with brg implement.
    //And in model zoo only limited deconv has bias or other post-ops in IR.
    //Based on above, enable the bias fusing for all deconv implementations.
    return  (externOutShape ? getParentEdges().size() == 3 : getParentEdges().size() == 2);
}

void Deconvolution::initSupportedPrimitiveDescriptors() {
    if (!useACL) {
        Node::initSupportedPrimitiveDescriptors();
        return;
    }

    VectorDims inDims, outDims;
    std::tie(inDims, outDims) = makeDummyInOutShape();
    auto tmpInShape  = Shape(inDims);
    auto tmpOutShape = Shape(outDims);
    initPaddingR(tmpInShape, tmpOutShape);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format, LayoutType weights_format = LayoutType::ncsp) {
        NodeConfig config;
        config.inConfs.resize(getParentEdges().size());
        config.outConfs.resize(getOriginalOutputsNumber());

        config.inConfs[0].setMemDesc(
                creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(0)));
        config.inConfs[1].setMemDesc(
                creatorsMap.at(weights_format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(1)));

        for (size_t i = 2; i < getParentEdges().size(); ++i) {
            config.inConfs[i].setMemDesc(
                    creatorsMap.at(format)->createSharedDesc(getOriginalInputPrecisionAtPort(0), getInputShapeAtPort(i)));
        }

        for (size_t i = 0; i < config.outConfs.size(); ++i) {
            config.outConfs[i].setMemDesc(
                    creatorsMap.at(format)->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(i)));
        }

        std::vector<MemoryDescPtr> srcMemoryDescs;
        srcMemoryDescs.push_back(config.inConfs[0].getMemDesc()->cloneWithNewDims(tmpInShape.getDims()));
        for (size_t i = 1; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc()->clone());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(config.outConfs[0].getMemDesc()->cloneWithNewDims(tmpOutShape.getDims()));
        for (size_t i = 1; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc()->clone());
        }

        auto factory = std::make_shared<DeconvExecutorFactory>(deconvAttrs, srcMemoryDescs, dstMemoryDescs,
                                                               std::make_shared<ExecutorContext>(context, getImplPriority()));

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::gemm_acl, factory);
    };
    pushDesc(LayoutType::nspc);
    pushDesc(LayoutType::ncsp);
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
