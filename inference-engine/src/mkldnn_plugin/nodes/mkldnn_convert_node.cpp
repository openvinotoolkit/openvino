// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_convert_node.h"
#include "common/cpu_convert.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConvertNode::MKLDNNConvertNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNConvertNode::getSupportedDescriptors() {
    if (outDims.empty() && output.getLayout() != InferenceEngine::Layout::ANY)
        outDims.push_back(MKLDNNDims(output.getDims()));
    if (inDims.empty() && input.getLayout() != InferenceEngine::Layout::ANY)
        inDims.push_back(MKLDNNDims(input.getDims()));
    if (getParentEdges().size() != 1 || getChildEdges().size() != 1)
        THROW_IE_EXCEPTION << "Convert layer with name '" << getName() << "' has incorrect number of input/output edges";
}

void MKLDNNConvertNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto layer = getCnnLayer();
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "Cannot get CNN layer for layer name " << getName();
    }

    LayerConfig config;
    DataConfig dataIn;
    DataConfig dataConfigOut;

    memory::format fmt;

    if (input.getLayout() != InferenceEngine::Layout::ANY && output.getLayout() != InferenceEngine::Layout::ANY) {
        dataIn.desc = input;
        config.inConfs.push_back(dataIn);

        const auto layout = config.inConfs[0].desc.getLayout(); // inp/out layouts must be the same
        auto outTensorDesc = output;
        dataConfigOut.desc = TensorDesc(output.getPrecision(), output.getDims(), layout);
        config.outConfs.push_back(dataConfigOut);
        fmt = MKLDNNMemory::Convert(layout);
    } else if (!layer->insData.empty() && !layer->outData.empty()) {
        const SizeVector& ins_dims = layer->insData[0].lock()->getTensorDesc().getDims();
        const auto layout = layer->insData[0].lock()->getTensorDesc().getLayout(); // inp/out layouts must be the same
        dataIn.desc = TensorDesc(layer->insData[0].lock()->getTensorDesc().getPrecision(), ins_dims, layout);
        config.inConfs.push_back(dataIn);

        const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
        dataConfigOut.desc = TensorDesc(layer->outData[0]->getTensorDesc().getPrecision(), out_dims, layout);
        config.outConfs.push_back(dataConfigOut);
        fmt = MKLDNNMemory::Convert(layout);
    }

    config.dynBatchSupport = false;
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);
}

void MKLDNNConvertNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

void MKLDNNConvertNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();
    if (parentMem.GetElementsCount() != childMem.GetElementsCount())
        THROW_IE_EXCEPTION << "Convert layer with name '" << getName() << "' has input and output buffers with different elements count";

    void *srcPtr = getParentEdgeAt(0)->getMemory().GetData();
    void *dstPtr = getChildEdgeAt(0)->getMemory().GetData();
    cpu_convert(srcPtr, dstPtr, getParentEdgeAt(0)->getDesc().getPrecision(), getChildEdgeAt(0)->getDesc().getPrecision(), parentMem.GetElementsCount());
}

bool MKLDNNConvertNode::created() const {
    return getType() == Convert;
}
REG_MKLDNN_PRIM_FOR(MKLDNNConvertNode, Convert);
