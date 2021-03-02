// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mkldnn_extension_utils.h>
#include "mkldnn_convert_node.h"
#include "common/cpu_convert.h"
#include "common/tensor_desc_creator.h"

#define THROW_ERROR THROW_IE_EXCEPTION << getTypeStr() << " layer with name '" << getName() <<"' ERROR: "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConvertNode::MKLDNNConvertNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNConvertNode::getSupportedDescriptors() {
    // if tensor descriptors are set via setDescs method we need to update the inDims/outDims data
    // from correspond tensor descriptors.
    if (outDims.empty() && output && output->getLayout() != InferenceEngine::Layout::ANY)
        outDims.push_back(MKLDNNDims(output->getDims()));
    if (inDims.empty() && input && input->getLayout() != InferenceEngine::Layout::ANY)
        inDims.push_back(MKLDNNDims(input->getDims()));
    if (getParentEdges().size() != 1)
        THROW_ERROR << "Incorrect number of input edges";
    if (getChildEdges().empty())
        THROW_ERROR << "Incorrect number of output edges";
}

void MKLDNNConvertNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto layer = getCnnLayer();
    if (layer == nullptr) {
        THROW_ERROR << "Cannot get CNN layer";
    }

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
    } else if (layer->insData.size() == 1 && layer->outData.size() == 1) {
        auto insData = layer->insData[0].lock();
        if (nullptr == insData) {
            THROW_ERROR << "Input data is empty";
        }

        const SizeVector& insDims = insData->getTensorDesc().getDims();
        auto insPrecision = insData->getTensorDesc().getPrecision();
        const SizeVector& outputDims = layer->outData[0]->getTensorDesc().getDims();
        auto outPrecision = layer->outData[0]->getTensorDesc().getPrecision();

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
        THROW_ERROR << "Incorrect number of input/output edges";
    }
}

void MKLDNNConvertNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "Preferable primitive descriptor is not set.";
}

void MKLDNNConvertNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();
    if (parentMem.GetElementsCount() != childMem.GetElementsCount())
        THROW_ERROR << "Input and output buffers have different elements count";

    void* srcPtr = parentMem.GetPtr();
    void* dstPtr = childMem.GetPtr();
    cpu_convert(srcPtr, dstPtr, getParentEdgeAt(0)->getDesc().getPrecision(), getChildEdgeAt(0)->getDesc().getPrecision(), parentMem.GetElementsCount());
}

bool MKLDNNConvertNode::created() const {
    return getType() == Convert;
}
REG_MKLDNN_PRIM_FOR(MKLDNNConvertNode, Convert);
