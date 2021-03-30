// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_broadcast_node.h"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

MKLDNNBroadcastNode::MKLDNNBroadcastNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache), broadcast(getCnnLayer()->getNode()) {}

void MKLDNNBroadcastNode::getSupportedDescriptors() {
    auto broadcastNode = std::dynamic_pointer_cast<ngraph::op::v1::Broadcast>(broadcast);
    if (broadcastNode == nullptr)
        IE_THROW() << "Cannot convert Ngraph node for Broadcast node with name: " << getName();
    broadcastType = broadcastNode->get_broadcast_spec().m_type;

    if (getParentEdges().size() != 3)
        IE_THROW() << "Broadcast node with name " << getName() << " has incorrect number of input edges. "
                "Expected: 3, Actual: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << "Broadcast node with name " << getName() << " has no output edges.";
    auto dstDims0 = getChildEdgeAt(0)->getDims();
    for (int i = 1; i < getChildEdges().size(); i++) {
        auto DstDims = getChildEdgeAt(i)->getDims();
        if (DstDims.ndims() != dstDims0.ndims())
            IE_THROW() << "Output edges 0 and " << i << " have different dims for Broadcast node with name " << getName();
        for (int j = 0; j < dstDims0.ndims(); j++) {
            if (dstDims0[j] != DstDims[j]) {
                IE_THROW() << "Output edges 0 and " << i << " have different dims for Broadcast node with name " << getName();
            }
        }
    }
    if (getParentEdgeAt(0)->getDims().ndims() > getChildEdgeAt(0)->getDims().ndims())
        IE_THROW() << "Broadcast node with name " << getName() << " has incorrect input shape. Input shape cannot be more than output shape. "
                "Actual input shape size: " << getParentEdgeAt(0)->getDims().ndims() << ", output shape size: " << getChildEdgeAt(0)->getDims().ndims();

    if (getParentEdgeAt(1)->getDims().ndims() != 1)
        IE_THROW() << "TargetShape must be 1D tensor for Broadcast node with name " << getName();
    if (!getParentEdgeAt(1)->getParent()->isConstant())
        IE_THROW() << "Broadcast node with name " << getName() << " has non constant parent Node on 1-st input. "
                "This case is currently not supported in CPU plug-in.";

    auto blobPrecision = getParentEdgeAt(1)->getParent()->getCnnLayer()->blobs["custom"]->getTensorDesc().getPrecision();
    if (blobPrecision == Precision::I32 || blobPrecision == Precision::I64) {
        auto targetShapeBlob = getParentEdgeAt(1)->getParent()->getCnnLayer()->blobs["custom"].get();
        if (targetShapeBlob == nullptr)
            IE_THROW() << "Cannot get targetShapeBlob for Broadcast node with name " << getName();
    } else {
        IE_THROW() << "TargetShapeBlob has unsupported precision " << blobPrecision.name() << " for Broadcast node with name " << getName();
    }

    auto ndims = getChildEdgeAt(0)->getDims().ndims();
    auto srcDims = getParentEdgeAt(0)->getDims().ToSizeVector();
    auto dstDims = getChildEdgeAt(0)->getDims().ToSizeVector();

    if (broadcastType == ngraph::op::AutoBroadcastType::NUMPY) {
        repeats = dstDims;
        for (int i = 0; i < srcDims.size(); i++) {
            repeats[ndims - 1 - i] /= srcDims[srcDims.size() - 1 - i];
        }
    } else if (broadcastType == ngraph::op::AutoBroadcastType::EXPLICIT) {
        if (getParentEdgeAt(2)->getDims().ndims() != 1)
            IE_THROW() << "AxesMapping must be 1D tensor for Broadcast node with name " << getName();
        if (!getParentEdgeAt(2)->getParent()->isConstant())
            IE_THROW() << "Broadcast node with name " << getName() << " has non constant parent Node on 2-nd input. "
                    "This case is currently not supported in CPU plug-in.";

        auto axesMappingBlob = getParentEdgeAt(2)->getParent()->getCnnLayer()->blobs["custom"].get();
        if (axesMappingBlob == nullptr)
            IE_THROW() << "Cannot get targetShapeBlob for Broadcast node with name " << getName();

        blobPrecision = getParentEdgeAt(2)->getParent()->getCnnLayer()->blobs["custom"]->getTensorDesc().getPrecision();
        std::vector<int> axesMappingData(axesMappingBlob->size());
        if (blobPrecision == Precision::I32) {
            for (int i = 0; i < axesMappingBlob->size(); i++) {
                axesMappingData[i] = axesMappingBlob->buffer().as<int *>()[i];
            }
        } else if (blobPrecision == Precision::I64) {
            for (int i = 0; i < axesMappingBlob->size(); i++) {
                axesMappingData[i] = axesMappingBlob->buffer().as<int64_t *>()[i];
            }
        } else {
            IE_THROW() << "AxesMappingBlob has unsupported precision " << blobPrecision.name() << " for Broadcast node with name " << getName();
        }

        repeats = dstDims;
        for (int i = 0; i < getParentEdgeAt(2)->getDims()[0]; i++) {
            repeats[axesMappingData[i]] /= srcDims[i];
            axesMapping.push_back(axesMappingData[i]);
        }
    } else {
        IE_THROW() << "Broadcast node with name " << getName() << " has unsupported broadcast type.";
    }
}

void MKLDNNBroadcastNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors = getSupportedConfigs(this);
}

void MKLDNNBroadcastNode::createPrimitive() {
    for (int i = 0; i < getChildEdges().size(); i++) {
        auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory " << i << "didn't allocate for Broadcast node with name " << getName();
    }
    for (int i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Input memory " << i << "didn't allocate for Broadcast node with name " << getName();
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for Broadcast node with name " << getName();

    SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
    SizeVector dstBlockedDims = getChildEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();

    if (broadcastType == ngraph::op::AutoBroadcastType::EXPLICIT) {
        SizeVector newSrcBlockedDims = SizeVector(dstBlockedDims.size(), 1);

        for (int i = 0; i < getParentEdgeAt(2)->getDims()[0]; i++) {
            newSrcBlockedDims[axesMapping[i]] = srcBlockedDims[i];
        }

        srcBlockedDims = newSrcBlockedDims;
    }

    optimizedCase = prepareOptimizedParams(this, srcBlockedDims, dstBlockedDims);
}

void MKLDNNBroadcastNode::execute(mkldnn::stream strm) {
    if (optimizedCase) {
        optimizedExecute(this);
    } else {
        ngraphExecute(this, broadcast);
    }
}

bool MKLDNNBroadcastNode::created() const {
    return getType() == Broadcast;
}
REG_MKLDNN_PRIM_FOR(MKLDNNBroadcastNode, Broadcast);
