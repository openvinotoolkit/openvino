// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reshape_node.h"

#include <mkldnn_extension_utils.h>
#include <mkldnn_types.h>

#include <ngraph/opsets/opset1.hpp>
#include <string>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNReshapeNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
                                             std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        if (!std::dynamic_pointer_cast<const ngraph::opset1::Reshape>(op) &&
            !std::dynamic_pointer_cast<const ngraph::opset1::Squeeze>(op) &&
            !std::dynamic_pointer_cast<const ngraph::opset1::Unsqueeze>(op)) {
            errorMessage = "Only opset1 Reshape, Squeeze, Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNReshapeNode::MKLDNNReshapeNode(const std::shared_ptr<ngraph::Node>& op,
                                     const mkldnn::engine& eng,
                                     MKLDNNWeightsSharing::Ptr& cache)
    : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

MKLDNNReshapeNode::MKLDNNReshapeNode(const std::string& name,
                                     const Shape& inDims,
                                     const Shape& outDims,
                                     Precision precision,
                                     const mkldnn::engine& eng,
                                     MKLDNNWeightsSharing::Ptr& wCache)
    : MKLDNNNode("Reshape", name, eng, wCache) {
    this->inputShapes.push_back(inDims);
    this->outputShapes.push_back(outDims);
    addOriginalInputPrecision(precision);
    addOriginalOutputPrecision(precision);
}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision inPrec = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrec = getOriginalOutputPrecisionAtPort(0);

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec)
        inPrec = outPrec;

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace = -1;
        config.inConfs[i].constant = false;
        config.inConfs[i].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inPrec, getInputShapeAtPort(i));
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace = 0;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outPrec, getOutputShapeAtPort(0));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
}

bool MKLDNNReshapeNode::created() const {
    return getType() == Reshape;
}
REG_MKLDNN_PRIM_FOR(MKLDNNReshapeNode, Reshape);
