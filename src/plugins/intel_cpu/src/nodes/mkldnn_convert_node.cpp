// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_extension_utils.h>
#include "mkldnn_convert_node.h"
#include "common/cpu_convert.h"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include <utils/ngraph_utils.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNConvertNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

MKLDNNConvertNode::MKLDNNConvertNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Convert node with name '" + getName() + "'";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto convert = ov::as_type_ptr<const ngraph::opset1::Convert>(op);
    origPrc = details::convertPrecision(convert->get_destination_type());
}

std::vector<VectorDims> MKLDNNConvertNode::shapeInfer() const {
    return std::vector<VectorDims>{getParentEdgesAtPort(0)[0]->getMemory().getStaticDims()};
}

MKLDNNConvertNode::MKLDNNConvertNode(const Shape &shape, const InferenceEngine::Precision &inPrc, const InferenceEngine::Precision &outPrc,
                                     const std::string &nodeName, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode("Convert", nodeName, eng, cache)
        , origPrc(outPrc) {
    inputShapes.push_back(shape);
    addOriginalInputPrecision(inPrc);
    outputShapes.push_back(shape);
    addOriginalOutputPrecision(outPrc);

    isDynamic = shape.isDynamic();

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

bool MKLDNNConvertNode::isSupportedDesc(const MemoryDesc &desc) {
    bool isSupported = desc.getType() & MemoryDescType::Blocked;
    if (desc.getType() == MemoryDescType::DnnlBlocked)
        isSupported &= desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData();
    return isSupported;
}

void MKLDNNConvertNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    NodeConfig config;
    PortConfig dataIn;
    PortConfig dataConfigOut;

    config.dynBatchSupport = false;

    bool canInitExternalDesc = false;
    if (input && output) {
        canInitExternalDesc = true;
        canInitExternalDesc &= isSupportedDesc(*input);
        canInitExternalDesc &= isSupportedDesc(*output);
    }

    // if input and output pointers are not null and not contain extra data, then the inp/output tensor descriptors were set using setDescs method, so
    // they should be used as the actual descriptors.
    if (canInitExternalDesc) {
        dataIn.desc = input;
        config.inConfs.push_back(dataIn);

        // inp/out layouts must be the same
        dataConfigOut.desc = config.inConfs[0].desc;
        dataConfigOut.desc = dataConfigOut.desc->cloneWithNewPrecision(output->getPrecision());
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    } else if (inputShapes.size() == 1 && outputShapes.size() == 1) {
        const Shape& insShape = getInputShapeAtPort(0);
        auto insPrecision = getOriginalInputPrecisionAtPort(0);
        const Shape& outputShape = getOutputShapeAtPort(0);
        auto outPrecision = getOriginalOutputPrecisionAtPort(0);

        config.inConfs.push_back(dataIn);
        config.outConfs.push_back(dataConfigOut);

        auto creators = BlockedDescCreator::getCommonCreators();
        auto range = BlockedDescCreator::makeFilteredRange(creators, insShape.getRank());

        for (auto itr = range.first; itr != range.second; ++itr) {
            config.inConfs[0].desc = std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(insPrecision, insShape));
            config.outConfs[0].desc = std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(outPrecision, outputShape));

            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    } else {
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges";
    }
}

void MKLDNNConvertNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

void MKLDNNConvertNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();

    const auto parentPaddElemCount = parentMem.GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    const auto childPaddElemCount = childMem.GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    if (parentPaddElemCount != childPaddElemCount)
        IE_THROW() << errorPrefix << " has different elements number in input and output buffers";

    void* srcPtr = parentMem.GetPtr();
    void* dstPtr = childMem.GetPtr();

    cpu_convert(srcPtr,
                dstPtr,
                parentMem.getDesc().getPrecision(),
                origPrc,
                childMem.getDesc().getPrecision(),
                parentPaddElemCount);
}

bool MKLDNNConvertNode::created() const {
    return getType() == Convert;
}

REG_MKLDNN_PRIM_FOR(MKLDNNConvertNode, Convert);
