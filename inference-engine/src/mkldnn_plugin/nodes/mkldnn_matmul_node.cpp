// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_matmul_node.h"

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "mkldnn_eltwise_node.h"

#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
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
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }

        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        const auto shapeA = matMul->get_input_shape(0);
        const auto shapeB = matMul->get_input_shape(1);

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_shape(i).size();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_shape().size();
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

    const auto& inDims0 = getInputShapeAtPort(0).getStaticDims();
    const auto& inDims1 = getInputShapeAtPort(1).getStaticDims();
    const auto& outDims = getOutputShapeAtPort(0).getStaticDims();

    // coverity[copy_paste_error]
    if (inDims0[xAxis0] != inDims1[yAxis1] ||
        inDims0[yAxis0] != outDims[yAxis] ||
        inDims1[xAxis1] != outDims[xAxis])
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((inDims0[dim_idx] != outDims[dim_idx] &&
             inDims0[dim_idx] != 1) ||
            (inDims1[dim_idx] != outDims[dim_idx] &&
             inDims1[dim_idx] != 1)) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    /* Example MatMul:
     * 2x128x512(T) * 2x128x512 = 2x512x512
     * First input 2x128x512(T) should be transposed
     * oneDNN requires memory::desc for this input to:
     * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
     * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
     */
    auto getStridesAndDims = [](Shape& shape, const bool transpose) {
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
    };

    initialInShapes[0] = inputShapes[0];
    initialInShapes[1] = inputShapes[1];

    const VectorDims inStrides0 = getStridesAndDims(inputShapes[0], transposeIn[0]);
    const VectorDims inStrides1 = getStridesAndDims(inputShapes[1], transposeIn[1]);
    const VectorDims outStrides = getStridesAndDims(outputShapes[0], false);

    inDataDesc[0] = std::make_shared<DnnlBlockedMemoryDesc>(firstInPortPrec, inputShapes[0], inStrides0);
    inDataDesc[1] = std::make_shared<DnnlBlockedMemoryDesc>(secondInPortPrec, inputShapes[1], inStrides1);
    outDataDesc   = std::make_shared<DnnlBlockedMemoryDesc>(outPortPrec, getOutputShapeAtPort(0), outStrides);

    createDescriptor({inDataDesc[0], inDataDesc[1]}, {outDataDesc});
}

void MKLDNNMatMulNode::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                        const std::vector<MemoryDescPtr>& outputDesc) {
    MKLDNNDescriptor desc{
        std::shared_ptr<matmul::desc>(
            new matmul::desc(MemoryDescUtils::convertToDnnlMemoryDesc(inDataDesc[0])->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(inDataDesc[1])->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outDataDesc)->getDnnlDesc()))};

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
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->GetPrimitivePtr() || !src1MemPtr || !src1MemPtr->GetPrimitivePtr())
        IE_THROW()  << errorPrefix << " did not allocate input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    if (prim)
        return;

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();
    std::shared_ptr<matmul::primitive_desc> prim_desc;
    prim_desc = std::make_shared<matmul::primitive_desc>(
            createPrimitiveDescriptor<matmul::primitive_desc, matmul::desc>(*attr));

    prim.reset(new matmul(*prim_desc));

    auto src0 = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto src1 = getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

    primArgs = {{DNNL_ARG_SRC_0, src0}, {DNNL_ARG_WEIGHTS_0, src1}, {DNNL_ARG_DST, dst}};
}

MemoryDescPtr MKLDNNMatMulNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);

    return std::make_shared<CpuBlockedMemoryDesc>(
        MKLDNNExtensionUtils::DataTypeToIEPrecision(static_cast<mkldnn::memory::data_type>(desc.data.data_type)),
        initialInShapes[idx]); /* provide initial shapes, so hide transpose effect */
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

REG_MKLDNN_PRIM_FOR(MKLDNNMatMulNode, MatMul);
