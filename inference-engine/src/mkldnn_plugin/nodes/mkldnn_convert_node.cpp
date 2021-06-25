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
    inDims.emplace_back(dims);
    addOriginalInputPrecision(inPrc);
    outDims.emplace_back(dims);
    addOriginalOutputPrecision(outPrc);

    errorPrefix = "Convert node with name '" + getName() + "'";
}

void MKLDNNConvertNode::getSupportedDescriptors() {
    // if tensor descriptors are set via setDescs method we need to update the inDims/outDims data
    // from correspond tensor descriptors.
    if (outDims.empty() && output && output->getLayout() != InferenceEngine::Layout::ANY)
        outDims.push_back(MKLDNNDims(output->getDims()));
    if (inDims.empty() && input && input->getLayout() != InferenceEngine::Layout::ANY)
        inDims.push_back(MKLDNNDims(input->getDims()));
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";
}

void MKLDNNConvertNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    LayerConfig config;
    DataConfig dataIn;
    DataConfig dataConfigOut;

    config.dynBatchSupport = false;

    // if input and output pointers are not null, then the inp/output tensor descriptors were set using setDescs method, so
    // they should be used as the actual descriptors.
    if (input && input->getLayout() != InferenceEngine::Layout::ANY && output && output->getLayout() != InferenceEngine::Layout::ANY) {
        dataIn.desc = *input;
        config.inConfs.push_back(dataIn);

        const auto& blockingDesc = config.inConfs[0].desc.getBlockingDesc(); // inp/out layouts must be the same
        dataConfigOut.desc = TensorDesc(output->getPrecision(), input->getDims(), blockingDesc);
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, MKLDNNMemoryDesc(config.outConfs.front().desc).getFormat());
    } else if (getOriginalInputsNumber() == 1 && getOriginalOutputsNumber() == 1) {
        const SizeVector& insDims = getParentEdgeAt(0)->getDims().ToSizeVector();
        auto insPrecision = getOriginalInputPrecisionAtPort(0);
        const SizeVector& outputDims = getChildEdgeAt(0)->getDims().ToSizeVector();
        auto outPrecision = getOriginalOutputPrecisionAtPort(0);

        config.inConfs.push_back(dataIn);
        config.outConfs.push_back(dataConfigOut);

        auto creators = TensorDescCreator::getCommonCreators();
        auto range = TensorDescCreator::makeFilteredRange(creators, insDims.size());

        for (auto itr = range.first; itr != range.second; ++itr) {
            config.inConfs[0].desc = itr->second->createDesc(insPrecision, insDims);
            config.outConfs[0].desc = itr->second->createDesc(outPrecision, outputDims);

            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, MKLDNNMemoryDesc(config.outConfs.front().desc).getFormat());
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
    cpu_convert(srcPtr, dstPtr, getParentEdgeAt(0)->getDesc().getPrecision(), getChildEdgeAt(0)->getDesc().getPrecision(), parentMem.GetElementsCount());
}

bool MKLDNNConvertNode::created() const {
    return getType() == Convert;
}

REG_MKLDNN_PRIM_FOR(MKLDNNConvertNode, Convert);
