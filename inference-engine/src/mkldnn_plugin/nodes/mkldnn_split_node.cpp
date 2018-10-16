// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <map>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSplitNode::MKLDNNSplitNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNSplitNode::getSupportedDescriptors() {
    auto * splitLayer = dynamic_cast<SplitLayer*>(getCnnLayer().get());

    if (splitLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert split layer.";

    axis = splitLayer->_axis;

    if (axis != 1)
        THROW_IE_EXCEPTION << "Split support only axis 1.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input nodes.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output nodes.";

    // WA. Check applicability and limitations
    for (size_t i = 1; i < getCnnLayer()->outData.size(); i++) {
        int num_port_connection = getCnnLayer()->outData[i]->inputTo.size();
        // limitation. If num of edges more than num of ports,
        // we connect it to first port. So check that all ports [1:]
        // have only one connection.
        if (num_port_connection > 1)
            THROW_IE_EXCEPTION << "Unsupported topology. Split layer \"" << getCnnLayer()->name << "\" "
                               << "has output edges more than output ports.";
    }
}

void MKLDNNSplitNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(srcDims, inputDataType, memory::format::any);
    config.outConfs.resize(getChildEdges().size());

    if (srcDims.ndims() < 2)
        THROW_IE_EXCEPTION << "Split " << getName() << " isn't supported 1d blobs";

    auto num_chanels = 0;
    auto dstFirstDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto o_Dims = outDims[i];
        if (dstFirstDims.ndims() != o_Dims.ndims()) {
            THROW_IE_EXCEPTION << "Split " << getName() << " supports only output blob with equal number of dimensions";
        }

        config.outConfs[i].inPlace = -1;
        config.outConfs[i].constant = false;
        config.outConfs[i].desc = MKLDNNMemoryDesc(o_Dims, outputDataType, memory::format::any);
        num_chanels += o_Dims[1];
        for (size_t j = 0; j < dstFirstDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (o_Dims[j] != dstFirstDims[j])
                THROW_IE_EXCEPTION << "Split " << getName() << "has incorrect output dimensions";
        }
    }
    dstFirstDims[1] = num_chanels;
    if (dstFirstDims.size() != srcDims.size())
        THROW_IE_EXCEPTION << "The sizes of input blob and sum of output blobs are not equal.";
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);

    auto numOfDim = static_cast<size_t>(srcDims.ndims());

    SizeVector order;
    SizeVector offsets;
    size_t offset = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < numOfDim; i++) {
        order.push_back(i);
        offsets.push_back(0);
    }

    SizeVector strides(numOfDim);
    strides[numOfDim - 1] = 1;
    for (size_t i = 2; i <= numOfDim; i++) {
        if (numOfDim - i < axis) {
            strides[numOfDim - i] = std::numeric_limits<size_t>::max();
        } else {
            strides[numOfDim - i] = strides[numOfDim - i + 1] * srcDims[numOfDim - i + 1];
        }
    }

    config.inConfs[0].desc = TensorDesc(Precision::FP32, srcDims.ToSizeVector(), {srcDims.ToSizeVector(), order, offset, offsets, strides});
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto outDims = getChildEdgeAt(i)->getDims();
        config.outConfs[i].inPlace = 0;
        config.outConfs[i].desc = TensorDesc(Precision::FP32, outDims.ToSizeVector(),
                                            {outDims.ToSizeVector(), order, offset, offsets, strides});
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);

    if (numOfDim != 4)
        return;

    order = {0, 1, 2, 3, 1};
    offsets = {0, 0, 0, 0, 0};
    numOfDim = 5;

    // nChw8c and nChw16c
    for (int sizeS : {8, 16}) {
        SizeVector blkDims = srcDims.ToSizeVector();
        if (blkDims[1] % sizeS)
            continue;
        blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1 : 0);
        blkDims.push_back(sizeS);

        strides.resize(numOfDim);
        strides[numOfDim - 1] = 1;
        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = std::numeric_limits<size_t>::max();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }
        config.inConfs[0].desc = TensorDesc(Precision::FP32, srcDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

        bool canInplace = true;
        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto outDims = getChildEdgeAt(i)->getDims().ToSizeVector();
            blkDims = outDims;

            if (blkDims[1] % sizeS) {
                canInplace = false;
                break;
            }
            blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1 : 0);
            blkDims.push_back(sizeS);
            config.outConfs[i].desc =  TensorDesc(Precision::FP32, outDims, {blkDims, order, offset, offsets, strides});
        }
        if (canInplace)
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }
}

void MKLDNNSplitNode::createPrimitive() {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (!getChildEdgeAt(i)->getMemoryPtr() || !getChildEdgeAt(i)->getMemory().GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNSplitNode::execute(mkldnn::stream strm) {
    if (isOptimized())
        return;

    // FIXME: add more optimal implementation
    MKLDNNDims par_dims = getParentEdgeAt(0)->getDims();
    int MB = batchToProcess();
    auto srcBlob = getParentEdgeAt(0)->getBlob();
    const auto *srcData = srcBlob->cbuffer().as<const float *>();
    size_t srcSize = getParentEdgeAt(0)->getMemory().GetSize();
    size_t src_batch_off = srcBlob->getTensorDesc().offset(srcBlob->size() / srcBlob->getTensorDesc().getDims()[0])
            - srcBlob->getTensorDesc().offset(0);

    for (size_t i = 0, sIdx = 0; i < getChildEdges().size(); i++) {
        auto dstBlob = getChildEdgeAt(i)->getBlob();
        auto *dstData = dstBlob->buffer().as<float *>();
        size_t dst_slice_size = dstBlob->size() / dstBlob->getTensorDesc().getDims()[0];
        size_t dst_batch_off = dstBlob->getTensorDesc().offset(dst_slice_size) - dstBlob->getTensorDesc().offset(0);

        for (size_t dIdx = 0; dIdx < dst_slice_size; dIdx++, sIdx++) {
            for (unsigned b = 0; b < MB; b++) {
                if (sIdx + b*src_batch_off >= srcSize)
                    THROW_IE_EXCEPTION << "Incorrect configuration of split layer " << getName() << "!";
                dstData[b * dst_batch_off + dstBlob->getTensorDesc().offset(dIdx)] =
                        srcData[b * src_batch_off + srcBlob->getTensorDesc().offset(sIdx)];
            }
        }
    }
}

bool MKLDNNSplitNode::created() const {
    return getType() == Split;
}

void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {
    if (implPriorities.size() > 0 && implPriorities[0] == impl_desc_type::ref) {
        selectPrimitiveDescriptorByIndex(0);
        return;
    }
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    bool hasUnknown = false;
    std::vector<size_t> canSelectPrimitive;
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        bool hasAny = true;
        auto &primDescInfo = supportedPrimitiveDescriptors[i];
        if (primDescInfo.getImplementationType() != impl_desc_type::unknown ||
            primDescInfo.getConfig().outConfs[0].inPlace < 0)
            continue;
        hasUnknown = true;
        for (auto iInfo : primDescInfo.getConfig().inConfs) {
            if (iInfo.desc.getLayout() != InferenceEngine::Layout::ANY) {
                hasAny = false;
                break;
            }
        }

        if (hasAny) {
            for (auto oInfo : primDescInfo.getConfig().outConfs) {
                if (oInfo.desc.getLayout() != InferenceEngine::Layout::ANY) {
                    hasAny = false;
                    break;
                }
            }
        }

        if (!hasAny) {
            canSelectPrimitive.push_back(i);
        }
    }

    bool canOptimize = false;
    if (hasUnknown) {
        canOptimize = true;

        if (canSelectPrimitive.size() == 1) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
            return;
        }
    }

    std::map<mkldnn::memory::format, size_t> formatFrequency;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        if (parent->getSelectedPrimitiveDescriptor() == nullptr)
            continue;

        int outputIndex = parentEdge->getOutputNum();
        if (outputIndex < 0)
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        if (outputIndex >= parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
            outputIndex = 0;
        auto outDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIndex].desc);
        if (!outDesc)
            continue;
        if (formatFrequency.find(outDesc.getFormat()) != formatFrequency.end())
            formatFrequency[outDesc.getFormat()] += 1;
        else
            formatFrequency[outDesc.getFormat()] = 1;
    }
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto child = childEdge->getChild();
        if (child->getSelectedPrimitiveDescriptor() == nullptr)
            continue;
        int inputIndex = childEdge->getOutputNum();
        if (inputIndex < 0)
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        if (inputIndex >= child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size())
            inputIndex = 0;
        auto outDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIndex].desc);
        if (!outDesc)
            continue;
        if (formatFrequency.find(outDesc.getFormat()) != formatFrequency.end())
            formatFrequency[outDesc.getFormat()] += 1;
        else
            formatFrequency[outDesc.getFormat()] = 1;
    }

    size_t maxCount = 0;
    mkldnn::memory::format convertTo = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
    for (auto &it : formatFrequency) {
        if (it.second > maxCount && !MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, it.first).blocksExtended()) {
            maxCount = it.second;
            convertTo = it.first;
        }
    }

    if (canOptimize && MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, convertTo).blocksExtended())
        canOptimize = false;
    for (size_t i = 0; canOptimize && i < getChildEdges().size(); i++) {
        if (MKLDNNMemoryDesc(getChildEdgeAt(i)->getDims(), outputDataType, convertTo).blocksExtended())
            canOptimize = false;
    }

    if (canOptimize) {
        for (auto supportedPdIndex : canSelectPrimitive) {
            if (MKLDNNMemoryDesc(supportedPrimitiveDescriptors[supportedPdIndex].getConfig().inConfs[0].desc).getFormat() == convertTo) {
                selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPdIndex));
                return;
            }
        }
    }

    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        auto &primDescInfo = supportedPrimitiveDescriptors[i];
        if (primDescInfo.getImplementationType() == impl_desc_type::unknown)
            continue;
        if (convertTo == MKLDNNMemoryDesc(supportedPrimitiveDescriptors[i].getConfig().outConfs[0].desc).getFormat()) {
            size_t num = 0;
            for (num = 0; num < getParentEdges().size(); num++) {
                if (MKLDNNMemoryDesc(getParentEdgeAt(num)->getDims(), inputDataType, convertTo).blocksExtended())
                    break;
            }
            if (num == getParentEdges().size()) {
                selectPrimitiveDescriptorByIndex(i);
                return;
            }
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNSplitNode::isOptimized() {
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].inPlace >= 0;
}

void MKLDNNSplitNode::initOptimalPrimitiveDescriptor() {
    if (!isOptimized()) {
        MKLDNNNode::initOptimalPrimitiveDescriptor();
        return;
    }

    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.inConfs.size(); i++) {
        if (config.inConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
            !isUninitTensorDesc(config.inConfs[i].desc))
            continue;

        int num = getParentEdgeAt(i)->getOutputNum();
        if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()) {
            if (num >= 0) {
                if (isUninitTensorDesc(getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc) &&
                        getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].inPlace >= 0)
                    getParentEdgeAt(i)->getParent()->initOptimalPrimitiveDescriptor();
                if (!isUninitTensorDesc(getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc) &&
                    MKLDNNExtensionUtils::initTensorsAreEqual(
                            getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc,
                            config.inConfs[i].desc)) {
                    config.inConfs[i].desc = getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc;
                    continue;
                }
            }
        }
        config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                              config.inConfs[i].desc.getDims(), {
                                                                      config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.inConfs[i].desc.getBlockingDesc().getOrder()
                                                              });
    }
    const auto& cnnLayer = getCnnLayer();
    if (!cnnLayer)
        THROW_IE_EXCEPTION << "Cannot create Split layer " << getName() << " without CNNLayer!";
    if (config.outConfs.size() != getChildEdges().size())
        THROW_IE_EXCEPTION << "Invalid config for Split layer " << getName();
    size_t offset = 0;
    for (size_t i = 0; i < cnnLayer->outData.size(); i++) {
        size_t confNum(0);
        bool found = false;
        for (size_t j = i; j < getChildEdges().size(); j++) {
            if (cnnLayer->outData[i]->inputTo.find(getChildEdgeAt(j)->getChild()->getName()) == cnnLayer->outData[i]->inputTo.end())
                continue;
            confNum = j;
            config.outConfs[j].desc = InferenceEngine::TensorDesc(config.outConfs[j].desc.getPrecision(),
                                                                  config.outConfs[j].desc.getDims(), {
                                                                          config.outConfs[j].desc.getBlockingDesc().getBlockDims(),
                                                                          config.outConfs[j].desc.getBlockingDesc().getOrder(),
                                                                          config.inConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                          config.inConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                          config.inConfs[0].desc.getBlockingDesc().getStrides()
                                                                  });
            found = true;
        }
        if (!found) {
            confNum = i;
            config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                                  config.outConfs[i].desc.getDims(), {
                                                                          config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                          config.outConfs[i].desc.getBlockingDesc().getOrder(),
                                                                          config.inConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                          config.inConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                          config.inConfs[0].desc.getBlockingDesc().getStrides()
                                                                  });
        }
        size_t axisSize = 1;
        for (size_t j = axis; j < config.outConfs[confNum].desc.getBlockingDesc().getBlockDims().size(); j++) {
            axisSize *= config.outConfs[confNum].desc.getBlockingDesc().getBlockDims()[j];
        }
        offset += axisSize;
    }
    initDescriptor(config);
}
