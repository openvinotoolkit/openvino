// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tile_node.h"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

MKLDNNTileNode::MKLDNNTileNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache), tile(getCnnLayer()->getNode()) {}

void MKLDNNTileNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW() << "Tile node with name " << getName() << " has incorrect number of input edges. "
                "Expected: 2, Actual: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << "Tile node with name " << getName() << " has no output edges.";
    auto dstDims0 = getChildEdgeAt(0)->getDims();
    for (int i = 1; i < getChildEdges().size(); i++) {
        auto DstDims = getChildEdgeAt(i)->getDims();
        if (DstDims.ndims() != dstDims0.ndims())
            IE_THROW() << "Output edges 0 and " << i << " have different dims for Tile node with name " << getName();
        for (int j = 0; j < dstDims0.ndims(); j++) {
            if (dstDims0[j] != DstDims[j]) {
                IE_THROW() << "Output edges 0 and " << i << " have different dims for Tile node with name " << getName();
            }
        }
    }
    if (getParentEdgeAt(0)->getDims().ndims() > getChildEdgeAt(0)->getDims().ndims())
        IE_THROW() << "Tile node with name " << getName() << " has incorrect input shape. Input shape cannot be more than output shape. "
                "Actual input shape size: " << getParentEdgeAt(0)->getDims().ndims() << ", output shape size: " << getChildEdgeAt(0)->getDims().ndims();

    if (getParentEdgeAt(1)->getDims().ndims() != 1)
        IE_THROW() << "Repeats must be 1D tensor for Tile node with name " << getName();
    if (!getParentEdgeAt(1)->getParent()->isConstant())
        IE_THROW() << "Tile node with name " << getName() << " has non constant parent Node on 1-st input. "
                "This case is currently not supported in CPU plug-in.";

    auto repeatsBlob = getParentEdgeAt(1)->getParent()->getCnnLayer()->blobs["custom"].get();
    if (repeatsBlob == nullptr)
        IE_THROW() << "Cannot get repeatsBlob for Tile node with name " << getName();

    auto blobPrecision = getParentEdgeAt(1)->getParent()->getCnnLayer()->blobs["custom"]->getTensorDesc().getPrecision();
    std::vector<int> repeatsData(repeatsBlob->size());
    if (blobPrecision == Precision::I32) {
        for (int i = 0; i < repeatsBlob->size(); i++) {
            repeatsData[i] = repeatsBlob->buffer().as<int *>()[i];
        }
    } else if (blobPrecision == Precision::I64) {
        for (int i = 0; i < repeatsBlob->size(); i++) {
            repeatsData[i] = repeatsBlob->buffer().as<int64_t *>()[i];
        }
    } else {
        IE_THROW() << "RepeatsBlob has unsupported precision " << blobPrecision.name() << " for Tile node with name " << getName();
    }

    for (int i = 0; i < getParentEdgeAt(1)->getDims().ToSizeVector()[0]; i++) {
        repeats.push_back(repeatsData[i]);
    }
    while (repeats.size() < getChildEdgeAt(0)->getDims().ndims()) {
        repeats.insert(repeats.begin(), 1);
    }
}

void MKLDNNTileNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors = getSupportedConfigs(this);
}

void MKLDNNTileNode::createPrimitive() {
    for (int i = 0; i < getChildEdges().size(); i++) {
        auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory " << i << "didn't allocate for Tile node with name " << getName();
    }
    for (int i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Input memory " << i << "didn't allocate for Tile node with name " << getName();
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for Tile node with name " << getName();

    SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
    SizeVector dstBlockedDims = getChildEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
    optimizedCase = prepareOptimizedParams(this, srcBlockedDims, dstBlockedDims);
}

void MKLDNNTileNode::execute(mkldnn::stream strm) {
    if (optimizedCase) {
        optimizedExecute(this);
    } else {
        ngraphExecute(this, tile);
    }
}

bool MKLDNNTileNode::created() const {
    return getType() == Tile;
}
REG_MKLDNN_PRIM_FOR(MKLDNNTileNode, Tile);
