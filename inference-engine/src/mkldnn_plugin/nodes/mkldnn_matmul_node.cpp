// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "mkldnn_eltwise_node.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <mkldnn_types.h>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "mkldnn_extension_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNMatMulNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
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

MKLDNNMatMulNode::MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
    MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    errorPrefix = "MatMul node with name '" + getName() + "'";

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();

    // if (getName() == "292") {
    //     std::cout << "MATMUL 0: " << getName() << std::endl;
    //     std::cout << inputShapes[0].toString() << std::endl;
    //     std::cout << inputShapes[1].toString() << std::endl;
    // }
}

bool MKLDNNMatMulNode::canFuse(const MKLDNNNodePtr& node) const {
    return one_of(node->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                  EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                  EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu);
}

void MKLDNNMatMulNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) const {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}


std::shared_ptr<mkldnn::primitive_attr> MKLDNNMatMulNode::initPrimitiveAttr() const {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, true);

    return attr;
}

/* Example MatMul:
 * 2x128x512(T) * 2x128x512 = 2x512x512
 * First input 2x128x512(T) should be transposed
 * oneDNN requires memory::desc for this input to:
 * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
 * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
 */
static VectorDims getStridesAndModifyShape(Shape& shape, const bool transpose) {
    const auto getRank = shape.getRank();

    VectorDims strides(getRank, 1);
    for (size_t i = 1; i < getRank; i++) {
        strides[getRank - i - 1 ] = strides[getRank - i] * shape.getStaticDims()[getRank - i];
    }

    if (transpose && getRank > 1) {
        // form new shape
        auto dims = shape.getStaticDims();
        std::swap(dims[getRank - 2], dims[getRank - 1]);
        shape = Shape{dims};
        // update strides
        strides[getRank - 1] = shape.getStaticDims()[getRank - 2];
        strides[getRank - 2] = 1;
    }

    return strides;
}

void MKLDNNMatMulNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    auto firstInPortPrec = getOriginalInputPrecisionAtPort(0);
    auto secondInPortPrec = getOriginalInputPrecisionAtPort(1);
    auto outPortPrec = getOriginalOutputPrecisionAtPort(0);

    if (firstInPortPrec.size() != secondInPortPrec.size())
        firstInPortPrec = secondInPortPrec = getMaxPrecision(getOriginalInputPrecisions());

    if (!fusedWith.empty()) {
        outPortPrec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (inputShapes[0].getRank() != inputShapes[1].getRank() || inputShapes[0].getRank() != outputShapes[0].getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShapes[0].getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = getInputShapeAtPort(0).getDims();
    const auto& inDims1 = getInputShapeAtPort(1).getDims();
    const auto& outDims = getOutputShapeAtPort(0).getDims();

    // coverity[copy_paste_error]
    if (!dimsEqualWeak(inDims0[xAxis0], inDims1[yAxis1]) ||
        !dimsEqualWeak(inDims0[yAxis0], outDims[yAxis]) ||
        !dimsEqualWeak(inDims1[xAxis1], outDims[xAxis]))
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((!dimsEqualWeak(inDims0[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims0[dim_idx], 1)) ||
            (!dimsEqualWeak(inDims1[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims1[dim_idx], 1))) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    // if (getName() == "292") {
    //     std::cout << "MATMUL 1: " << getName() << std::endl;
    //     std::cout << inputShapes[0].toString() << std::endl;
    //     std::cout << inputShapes[1].toString() << std::endl;
    // }
// std::cout << "MM: 0" << std::endl;
    std::vector<Shape> staticInputShapes(2);
    staticInputShapes[0] = inputShapes[0].isStatic() ? inputShapes[0] : MemoryDescUtils::makeDummyShape(inputShapes[0]);
    staticInputShapes[1] = inputShapes[1].isStatic() ? inputShapes[1] : MemoryDescUtils::makeDummyShape(inputShapes[1]);

    auto staticOutputShape = outputShapes[0].isStatic() ? outputShapes[0] : Shape(shapeInferGeneric(staticInputShapes).front());

    const VectorDims inStrides0 = getStridesAndModifyShape(staticInputShapes[0], transposeIn[0]);
    const VectorDims inStrides1 = getStridesAndModifyShape(staticInputShapes[1], transposeIn[1]);
// std::cout << "MM: 1" << std::endl;
    inDataDesc[0] = std::make_shared<DnnlBlockedMemoryDesc>(firstInPortPrec, staticInputShapes[0], inStrides0);
    inDataDesc[1] = std::make_shared<DnnlBlockedMemoryDesc>(secondInPortPrec, staticInputShapes[1], inStrides1);
    outDataDesc   = std::make_shared<DnnlBlockedMemoryDesc>(outPortPrec, staticOutputShape);
// std::cout << "MM: 2" << std::endl;
    createDescriptor({inDataDesc[0], inDataDesc[1]}, {outDataDesc});
// std::cout << "MM: 3" << std::endl;
}

void MKLDNNMatMulNode::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                        const std::vector<MemoryDescPtr>& outputDesc) {

    const auto a = inDataDesc[0]->getDnnlDesc();
    const auto b = inDataDesc[1]->getDnnlDesc();
    const auto c = outDataDesc->getDnnlDesc();

    MKLDNNDescriptor desc{
        std::make_shared<matmul::desc>(inDataDesc[0]->getDnnlDesc(),
                                       inDataDesc[1]->getDnnlDesc(),
                                       outDataDesc->getDnnlDesc())};

    descs.push_back(desc);
}

void MKLDNNMatMulNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto attr = initPrimitiveAttr();

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = -1;
                portConfig.constant = false;

                auto src_desc = getSrcMemDesc(itpd, i);
                if (src_desc->getType() & MemoryDescType::Blocked) {
                    portConfig.desc = src_desc->as<BlockedMemoryDesc>()->cloneWithUndefStridesAndOffset();
                } else {
                    portConfig.desc = std::move(src_desc);
                }

                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = canBeInPlace() ? 0 : -1;
                portConfig.constant = false;

                auto dst_desc = getDstMemDesc(itpd, i);
                if (dst_desc->getType() & MemoryDescType::Blocked) {
                    portConfig.desc = dst_desc->as<BlockedMemoryDesc>()->cloneWithUndefStridesAndOffset();
                } else {
                    portConfig.desc = std::move(dst_desc);
                }

                config.outConfs.push_back(portConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNMatMulNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

MemoryDescPtr MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);

    return std::make_shared<CpuBlockedMemoryDesc>(
        MKLDNNExtensionUtils::DataTypeToIEPrecision(static_cast<mkldnn::memory::data_type>(desc.data.data_type)),
        inputShapes[idx]); /* provide initial shapes, so hide transpose effect */
}

bool MKLDNNMatMulNode::created() const {
    return getType() == MatMul;
}

size_t MKLDNNMatMulNode::getMaxBatch() const {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MKLDNNMatMulNode::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MKLDNNMatMulNode::prepareParams() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate input memory";

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    DnnlMemoryDescPtr src0TransposedDesc;
    DnnlMemoryDescPtr src1TransposedDesc;

    AttrPtr attr;

    if (isDynamicNode()) {
        if (!pAttr) {
            pAttr = initPrimitiveAttr();
        }
        attr = pAttr;

        const auto& src0Desc = src0MemPtr->getDesc();
        const auto& src1Desc = src1MemPtr->getDesc();

        auto src0Shape = src0Desc.getShape();
        auto src0Strides = getStridesAndModifyShape(src0Shape, transposeIn[0]);
        src0TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src0Desc.getPrecision(), src0Shape, src0Strides);

        auto src1Shape = src1Desc.getShape();
        auto src1Strides = getStridesAndModifyShape(src1Shape, transposeIn[1]);
        src1TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src1Desc.getPrecision(), src1Shape, src1Strides);
    } else {
        attr = initPrimitiveAttr();
        src0TransposedDesc = inDataDesc[0];
        src1TransposedDesc = inDataDesc[1];
    }

    auto dstDnnlDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    MKLDNNDescriptor desc{
            std::make_shared<matmul::desc>(src0TransposedDesc->getDnnlDesc(),
                                           src1TransposedDesc->getDnnlDesc(),
                                           dstDnnlDesc->getDnnlDesc())};

    matmul::primitive_desc prim_desc;
    primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);

    while (static_cast<bool>(itpd))  {
        impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

        if (impl_type == selected_pd->getImplementationType()) {
            prim_desc = itpd.get();
            break;
        }
        if (!itpd.next_impl())
            IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim.reset(new matmul(prim_desc));

    const auto& src0 = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    const auto& src1 = getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPrimitive();
    const auto& dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

    primArgs = {{DNNL_ARG_SRC_0, src0}, {DNNL_ARG_WEIGHTS_0, src1}, {DNNL_ARG_DST, dst}};
}

void MKLDNNMatMulNode::executeDynamicImpl(dnnl::stream strm) {
    MKLDNNNode::execute(strm);
}

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
