// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include "common/cpu_memcpy.h"
#include <legacy/ie_layers.h>
#include <string>
#include <vector>
#include <map>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>
#include <ie_parallel.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSplitNode::MKLDNNSplitNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNSplitNode::getSupportedDescriptors() {
    auto * splitLayer = dynamic_cast<SplitLayer*>(getCnnLayer().get());

    if (splitLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert split layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input nodes.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output nodes.";

    axis = splitLayer->_axis;
    if (axis >= getParentEdgeAt(0)->getDims().ndims())
        THROW_IE_EXCEPTION << "Invalid value of axis parameter in split layer";
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
    auto memoryFormat = MKLDNNMemory::GetPlainFormat(srcDims);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(srcDims, inputDataType, memoryFormat);
    config.outConfs.resize(outDims.size());

    std::vector<memory::format> outFormats;

    auto axis_size = 0;
    auto dstFirstDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto o_Dims = outDims[i];
        if (dstFirstDims.ndims() != o_Dims.ndims()) {
            THROW_IE_EXCEPTION << "Split " << getName() << " supports only output blob with equal number of dimensions";
        }

        config.outConfs[i].inPlace = -1;
        config.outConfs[i].constant = false;
        config.outConfs[i].desc = MKLDNNMemoryDesc(o_Dims, outputDataType, memoryFormat);
        outFormats.push_back(memoryFormat);

        axis_size += o_Dims[axis];
        for (size_t j = 0; j < dstFirstDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (o_Dims[j] != dstFirstDims[j])
                THROW_IE_EXCEPTION << "Split " << getName() << " has incorrect output dimensions";
        }
    }
    dstFirstDims[axis] = axis_size;
    if (dstFirstDims.size() != srcDims.size())
        THROW_IE_EXCEPTION << "The sizes of input blob and sum of output blobs are not equal.";
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFormats);

    auto numOfDim = static_cast<size_t>(srcDims.ndims());

    SizeVector order;
    SizeVector offsets(numOfDim, 0lu);
    size_t offset = (std::numeric_limits<size_t>::max)();
    for (size_t i = 0; i < numOfDim; i++) {
        order.push_back(i);
    }

    SizeVector strides(numOfDim);
    strides[numOfDim - 1] = 1;
    for (size_t i = 2; i <= numOfDim; i++) {
        if (numOfDim - i < axis) {
            strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
        } else {
            strides[numOfDim - i] = strides[numOfDim - i + 1] * srcDims[numOfDim - i + 1];
        }
    }

    config.inConfs[0].desc = TensorDesc(Precision::FP32, srcDims.ToSizeVector(), {srcDims.ToSizeVector(), order, offset, offsets, strides});
    outFormats.clear();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto dims = outDims[i].ToSizeVector();
        config.outConfs[i].inPlace = 0;
        config.outConfs[i].desc = TensorDesc(Precision::FP32, dims,
                                            {dims, order, offset, offsets, strides});
        outFormats.push_back(MKLDNNMemory::Convert(config.outConfs[i].desc.getLayout()));
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);

    if ((numOfDim != 4 && numOfDim != 5) || axis != 1)
        return;

    order.push_back(1);
    numOfDim = order.size();
    offsets = SizeVector(numOfDim, 0lu);

    // nChw8c and nChw16c
    for (size_t sizeS : {8lu, 16lu}) {
        SizeVector blkDims = srcDims.ToSizeVector();
        if (blkDims[1] % sizeS)
            continue;
        blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1lu : 0lu);
        blkDims.push_back(sizeS);

        strides.resize(numOfDim);
        strides[numOfDim - 1] = 1lu;
        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }
        config.inConfs[0].desc = TensorDesc(Precision::FP32, srcDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

        outFormats.clear();
        bool canInplace = true;
        for (size_t i = 0; i < outDims.size(); i++) {
            auto dims = outDims[i].ToSizeVector();
            blkDims = dims;

            if (blkDims[1] % sizeS) {
                canInplace = false;
                break;
            }
            blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1lu : 0lu);
            blkDims.push_back(sizeS);
            config.outConfs[i].desc = TensorDesc(Precision::FP32, dims, {blkDims, order, offset, offsets, strides});

            outFormats.emplace_back(MKLDNNMemory::Convert(config.outConfs[i].desc.getLayout()));
        }
        if (canInplace)
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);
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
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    if (!isOptimized())
        prepareOptimizedParams();
}

void MKLDNNSplitNode::execute(mkldnn::stream strm) {
    if (isOptimized())
        return;

    int MB = batchToProcess();
    uint8_t* srcData = getDataPtr(this->getParentEdgeAt(0)->getMemory());
    size_t batch = this->getParentEdgeAt(0)->getDims()[0];

    if (batch != MB)
        optimizedParams.countStrides = optimizedParams.countStrides / batch * MB;

    parallel_for2d(this->getChildEdges().size(), optimizedParams.countStrides, [&](size_t i, size_t j) {
        uint8_t* dstData = getDataPtr(this->getChildEdgeAt(i)->getMemory());

        cpu_memcpy(&dstData[j * optimizedParams.sizeData[i]],
                   &srcData[optimizedParams.srcShifts[i] + j * optimizedParams.srcStride],
                   optimizedParams.sizeData[i]);
    });
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

    // This logic is needed to cover cases when Split node cannot be optimized out for particular block size
    // In general it is significantly better to have additional reorders in graph than to use reference Split implementation
    if (convertTo == memory::nChw16c || convertTo == memory::nCdhw16c ||
        convertTo == memory::nChw8c || convertTo == memory::nCdhw8c) {
        int blockSize = convertTo == memory::nChw16c || convertTo == memory::nCdhw16c ? 16 : 8;
        bool shouldDecreaseBlockSize = false;
        for (auto& parentEdge : getParentEdges()) {
            if (parentEdge.lock()->getDims()[1] % blockSize != 0)
                shouldDecreaseBlockSize = true;
        }

        for (auto& childEdge : getChildEdges()) {
            if (childEdge.lock()->getDims()[1] % blockSize != 0)
                shouldDecreaseBlockSize = true;
        }

        if (shouldDecreaseBlockSize) {
            int decreasedBlockSize = 8;
            bool canDecreaseBlockSize = true;
            for (auto &parentEdge : getParentEdges()) {
                if (parentEdge.lock()->getDims()[1] % decreasedBlockSize != 0)
                    canDecreaseBlockSize = false;
            }

            for (auto &childEdge : getChildEdges()) {
                if (childEdge.lock()->getDims()[1] % decreasedBlockSize != 0)
                    canDecreaseBlockSize = false;
            }

            if (canDecreaseBlockSize)
                convertTo = getParentEdgeAt(0)->getDims().ndims() == 5 ? memory::nCdhw8c : memory::nChw8c;
            else
                convertTo = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
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

    bool convertToIsBlocked = (convertTo == mkldnn::memory::nChw8c) || (convertTo == mkldnn::memory::nCdhw8c);
    if (canOptimize && !convertToIsBlocked) {
        for (auto supportedPdIndex : canSelectPrimitive) {
            auto memoryDesc = MKLDNNMemoryDesc(supportedPrimitiveDescriptors[supportedPdIndex].getConfig().inConfs[0].desc);
            bool memoryDescFormatIsBlocked = (memoryDesc.getFormat() == mkldnn::memory::nChw8c) ||
                    (memoryDesc.getFormat() == mkldnn::memory::nCdhw8c);

            if (!memoryDescFormatIsBlocked) {
                selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPdIndex));
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

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
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
    if (config.outConfs.size() != outDims.size())
        THROW_IE_EXCEPTION << "Invalid config for Split layer " << getName();
    size_t offset = 0;
    for (size_t i = 0; i < cnnLayer->outData.size(); i++) {
        size_t confNum = i;
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getStrides()
                                                              });
        size_t axisSize = 1;
        for (size_t j = axis; j < config.outConfs[confNum].desc.getBlockingDesc().getBlockDims().size(); j++) {
            axisSize *= config.outConfs[confNum].desc.getBlockingDesc().getBlockDims()[j];
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNSplitNode::setDynamicBatchLim(int lim) {
    if (axis == 0)
        THROW_IE_EXCEPTION << "Dynamic batch is not supported by split layer with axis == 0 parameter";

    dynBatchLim = lim;
    if (prim) {
        prim.setBatchLimit(batchToProcess(), getParentEdges().size(), getChildEdges().size());
    }
}

inline uint8_t* MKLDNNSplitNode::getDataPtr(const MKLDNNMemory& memoryPtr) {
    return reinterpret_cast<uint8_t*>(memoryPtr.GetData()) + memoryPtr.GetDescriptor().data.layout_desc.blocking.offset_padding *
            MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(memoryPtr.GetDescriptor().data.data_type));
}

void MKLDNNSplitNode::prepareOptimizedParams() {
    auto srcDims = this->getParentEdgeAt(0)->getDims();
    int nDims = srcDims.ndims();
    uint8_t srcSizeData = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();

    optimizedParams.countStrides = 1;
    for (int i = 0; i < axis; i++)
            optimizedParams.countStrides *= srcDims[i];

    optimizedParams.srcStride = 0;
    optimizedParams.sizeData.resize(this->getChildEdges().size());
    for (int i = 0; i < this->getChildEdges().size(); i++) {
        optimizedParams.sizeData[i] = srcSizeData;

        for (int j = axis; j < nDims; j++)
            optimizedParams.sizeData[i] *= this->getChildEdgeAt(i)->getDims()[j];

        optimizedParams.srcStride += optimizedParams.sizeData[i];
    }

    optimizedParams.srcShifts.resize(this->getChildEdges().size());
    optimizedParams.srcShifts[0] = 0;
    for (int i = 1; i < this->getChildEdges().size(); i++) {
        optimizedParams.srcShifts[i] = optimizedParams.srcShifts[i - 1] + optimizedParams.sizeData[i - 1];
    }
}
REG_MKLDNN_PRIM_FOR(MKLDNNSplitNode, Split);
