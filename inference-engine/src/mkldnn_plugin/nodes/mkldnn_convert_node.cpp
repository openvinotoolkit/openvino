// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_extension_utils.h>
#include "mkldnn_convert_node.h"
#include "common/cpu_convert.h"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNConvertNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto convert = std::dynamic_pointer_cast<const ngraph::opset1::Convert>(op);
        if (!convert) {
            errorMessage = "Only opset1 Convert operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNConvertNode::MKLDNNConvertNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Convert node with name '" + getName() + "'";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

MKLDNNConvertNode::MKLDNNConvertNode(const InferenceEngine::SizeVector &dims, const InferenceEngine::Precision &inPrc, const InferenceEngine::Precision &outPrc,
                                     const std::string &nodeName, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode("Convert", nodeName, eng, cache) {
    inputShapes.emplace_back(dims);
    addOriginalInputPrecision(inPrc);
    outputShapes.emplace_back(dims);
    addOriginalOutputPrecision(outPrc);

    errorPrefix = "Convert node with name '" + getName() + "'";
}

void MKLDNNConvertNode::getSupportedDescriptors() {
    // if tensor descriptors are set via setDescs method we need to update the inDims/outDims data
    // from correspond tensor descriptors.
    if (outputShapes.empty())
        outputShapes.push_back(output->getShape());
    if (inputShapes.empty())
        inputShapes.push_back(input->getShape());
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";
}

void MKLDNNConvertNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    NodeConfig config;
    PortConfig dataIn;
    PortConfig dataConfigOut;

    config.dynBatchSupport = false;

    // if input and output pointers are not null, then the inp/output tensor descriptors were set using setDescs method, so
    // they should be used as the actual descriptors.
    if (input && output) {
        dataIn.desc = input->clone();
        config.inConfs.push_back(dataIn);

        // inp/out layouts must be the same
        dataConfigOut.desc = config.inConfs[0].desc->clone();
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    } else if (getOriginalInputsNumber() == 1 && getOriginalOutputsNumber() == 1) {
        const Shape& insShape = getParentEdgeAt(0)->getShape();
        auto insPrecision = getOriginalInputPrecisionAtPort(0);
        const Shape& outputShape = getChildEdgeAt(0)->getShape();
        auto outPrecision = getOriginalOutputPrecisionAtPort(0);

        config.inConfs.push_back(dataIn);
        config.outConfs.push_back(dataConfigOut);

        auto creators = BlockedDescCreator::getCommonCreators();
        auto range = BlockedDescCreator::makeFilteredRange(creators, insShape.getRank());

        for (auto itr = range.first; itr != range.second; ++itr) {
            config.inConfs[0].desc = MKLDNNPlugin::make_unique<BlockedMemoryDesc>(itr->second->createDesc(insPrecision, insShape.getDims()));
            config.outConfs[0].desc = MKLDNNPlugin::make_unique<BlockedMemoryDesc>(itr->second->createDesc(outPrecision, outputShape.getDims()));

            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    } else {
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges";
    }
}

void MKLDNNConvertNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has nullable preferable primitive descriptor";
}

void MKLDNNConvertNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();
    if (parentMem.GetElementsCount() != childMem.GetElementsCount())
        IE_THROW() << errorPrefix << " has different elements number in input and output buffers";

    void* srcPtr = parentMem.GetPtr();
    void* dstPtr = childMem.GetPtr();
    cpu_convert(srcPtr, dstPtr, parentMem.GetDesc().getPrecision(), childMem.GetDesc().getPrecision(), parentMem.GetElementsCount());
}

bool MKLDNNConvertNode::created() const {
    return getType() == Convert;
}

REG_MKLDNN_PRIM_FOR(MKLDNNConvertNode, Convert);
