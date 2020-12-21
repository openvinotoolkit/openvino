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

    if (input && input->getLayout() != InferenceEngine::Layout::ANY && output && output->getLayout() != InferenceEngine::Layout::ANY) {
        dataIn.desc = *input;
        config.inConfs.push_back(dataIn);

        const auto& BlockingDesc = config.inConfs[0].desc.getBlockingDesc(); // inp/out layouts must be the same
        dataConfigOut.desc = TensorDesc(output->getPrecision(), input->getDims(), BlockingDesc);
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, MKLDNNMemoryDesc(config.outConfs.front().desc).getFormat());
    } else if (layer->insData.size() == 1 && layer->outData.size() == 1) {
        auto insData = layer->insData[0].lock();
        if (nullptr == insData) {
            THROW_ERROR << "Input data is empty";
        }

        const SizeVector& ins_dims = insData->getTensorDesc().getDims();
        auto insPrecision = insData->getTensorDesc().getPrecision();
        const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
        auto outPrecision = layer->outData[0]->getTensorDesc().getPrecision();

        config.inConfs.push_back(dataIn);
        config.outConfs.push_back(dataConfigOut);

        auto commonCreators = TensorDescCreator::getCommonCreators();

        if (ins_dims.size() > 1) {
            for (auto item : TensorDescCreator::getCommonCreators()) {
                config.inConfs[0].desc = item.second->createDesc(insPrecision, ins_dims);
                config.outConfs[0].desc = item.second->createDesc(outPrecision, out_dims);

                supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, MKLDNNMemoryDesc(config.outConfs.front().desc).getFormat());
            }
        } else {
            config.inConfs[0].desc = commonCreators[TensorDescCreatorTypes::plain]->createDesc(insPrecision, ins_dims);
            config.outConfs[0].desc = commonCreators[TensorDescCreatorTypes::plain]->createDesc(outPrecision, out_dims);

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

static inline uint8_t* getDataPtr(const MKLDNNMemory& memoryPtr) {
    return reinterpret_cast<uint8_t*>(memoryPtr.GetData()) + memoryPtr.GetDescriptor().data.layout_desc.blocking.offset_padding *
        MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(memoryPtr.GetDescriptor().data.data_type));
}

void MKLDNNConvertNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();
    auto& childMem = getChildEdgeAt(0)->getMemory();
    if (parentMem.GetElementsCount() != childMem.GetElementsCount())
        THROW_ERROR << "Input and output buffers have different elements count";

    void* srcPtr = getDataPtr(parentMem);
    void* dstPtr = getDataPtr(childMem);
    cpu_convert(srcPtr, dstPtr, getParentEdgeAt(0)->getDesc().getPrecision(), getChildEdgeAt(0)->getDesc().getPrecision(), parentMem.GetElementsCount());
}

bool MKLDNNConvertNode::created() const {
    return getType() == Convert;
}
REG_MKLDNN_PRIM_FOR(MKLDNNConvertNode, Convert);
