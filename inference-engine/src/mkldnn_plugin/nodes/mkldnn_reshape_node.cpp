// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reshape_node.h"
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNReshapeNode::MKLDNNReshapeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getOriginalOutputPrecisionAtPort(0);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inputDataType != outputDataType)
        inputDataType = outputDataType;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    for (size_t i = 0; i <getParentEdges().size(); i++) {
        config.inConfs[i].inPlace = -1;
        config.inConfs[i].constant = false;
        config.inConfs[i].desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType);
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace = 0;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
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
