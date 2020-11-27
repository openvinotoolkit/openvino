// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include "common/cpu_memcpy.h"
#include <legacy/ie_layers.h>
#include <vector>
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

    InferenceEngine::Precision inpPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto outPrecision = inpPrecision; // the split layer doesn't convert precisions
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inpPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outPrecision);

    //Set plain format
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

    //Support per channel format
    outFormats.clear();
    auto perChannelFormat = MKLDNNMemory::GetPerChannelFormat(srcDims);
    config.inConfs[0].desc = MKLDNNMemoryDesc(srcDims, inputDataType, perChannelFormat);
    for (size_t i = 0; i < outDims.size(); ++i) {
        auto o_Dims = outDims[i];
        config.outConfs[i].desc = MKLDNNMemoryDesc(o_Dims, outputDataType, perChannelFormat);
        outFormats.push_back(perChannelFormat);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFormats);

    //Support channel blocked format
    SizeVector order(srcDims.ndims());
    std::iota(order.begin(), order.end(), 0);
    order.push_back(1);
    std::vector<unsigned> blockedPdIndexes;

    for (size_t sizeS : {8lu, 16lu}) {
        SizeVector blkDims = srcDims.ToSizeVector();
        if (blkDims[1] % sizeS)
            continue;
        blkDims[1] = blkDims[1] / sizeS;
        blkDims.push_back(sizeS);

        config.inConfs[0].desc = TensorDesc(inpPrecision, srcDims.ToSizeVector(), {blkDims, order});

        outFormats.clear();
        bool blocked = true;
        for (size_t i = 0; i < outDims.size(); i++) {
            auto dims = outDims[i].ToSizeVector();
            blkDims = dims;

            if (blkDims[1] % sizeS) {
                blocked = false;
                break;
            }
            blkDims[1] = blkDims[1] / sizeS;
            blkDims.push_back(sizeS);
            config.outConfs[i].desc = TensorDesc(outPrecision, dims, {blkDims, order});
            outFormats.emplace_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
        if (blocked) {
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFormats);
            blockedPdIndexes.push_back(supportedPrimitiveDescriptors.size() - 1);
        }
    }

    // Optimized inplace case
    auto numOfDim = static_cast<size_t>(srcDims.ndims());

    order.clear();
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

    config.inConfs[0].desc = TensorDesc(inpPrecision, srcDims.ToSizeVector(), {srcDims.ToSizeVector(), order, offset, offsets, strides});
    outFormats.clear();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto dims = outDims[i].ToSizeVector();
        config.outConfs[i].inPlace = 0;
        config.outConfs[i].desc = TensorDesc(outPrecision, dims,
                                            {dims, order, offset, offsets, strides});
        outFormats.push_back(MKLDNNMemory::Convert(config.outConfs[i].desc.getLayout()));
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);

    if (axis > 1) return;

    // Optimized inplace case for blocked layout
    for (auto index : blockedPdIndexes) {
        const auto& refConfig = supportedPrimitiveDescriptors[index].getConfig();

        order = refConfig.inConfs[0].desc.getBlockingDesc().getOrder();
        SizeVector blkDims = refConfig.inConfs[0].desc.getBlockingDesc().getBlockDims();
        numOfDim = blkDims.size();
        offsets = SizeVector(numOfDim, 0lu);

        strides.resize(numOfDim);
        strides[numOfDim - 1] = 1lu;
        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }
        config.inConfs[0].desc = TensorDesc(inpPrecision, srcDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

        outFormats.clear();
        for (size_t i = 0; i < outDims.size(); i++) {
            blkDims = refConfig.outConfs[i].desc.getBlockingDesc().getBlockDims();
            auto dims = refConfig.outConfs[i].desc.getDims();

            config.outConfs[i].desc = TensorDesc(outPrecision, dims, {blkDims, order, offset, offsets, strides});
            outFormats.emplace_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
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

        cpu_memcpy(&dstData[j * optimizedParams.dataSize[i]],
                   &srcData[optimizedParams.srcDataOffsets[i] + j * optimizedParams.srcDataStride],
                   optimizedParams.dataSize[i]);
    });
}

bool MKLDNNSplitNode::created() const {
    return getType() == Split;
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
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getStrides()
                                                              });
        size_t axisSize = 1;
        for (size_t j = axis; j < config.outConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
            axisSize *= config.outConfs[i].desc.getBlockingDesc().getBlockDims()[j];
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
    auto inpTensorDesc = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc;

    //find axis order position
    auto order = inpTensorDesc.getBlockingDesc().getOrder();
    unsigned axisOrderPos = UINT_MAX;
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] == axis) {
            axisOrderPos = i;
            break;
        }
    }
    if (UINT_MAX == axisOrderPos) {
        THROW_IE_EXCEPTION << "Can't find the axis in the input tensor order list";
    }

    uint8_t srcDataSize = inpTensorDesc.getPrecision().size();
    auto srcDims = inpTensorDesc.getBlockingDesc().getBlockDims();
    int nDims = srcDims.size();

    optimizedParams.countStrides = 1;
    for (int i = 0; i < axisOrderPos; i++)
            optimizedParams.countStrides *= srcDims[i];

    optimizedParams.srcDataStride = 0;
    optimizedParams.dataSize.resize(this->getChildEdges().size());
    for (int i = 0; i < this->getChildEdges().size(); i++) {
        optimizedParams.dataSize[i] = srcDataSize;

        for (int j = axisOrderPos; j < nDims; j++)
            optimizedParams.dataSize[i] *= this->getChildEdgeAt(i)->getDesc().getBlockingDesc().getBlockDims()[j];

        optimizedParams.srcDataStride += optimizedParams.dataSize[i];
    }

    optimizedParams.srcDataOffsets.resize(this->getChildEdges().size());
    optimizedParams.srcDataOffsets[0] = 0;
    for (int i = 1; i < this->getChildEdges().size(); i++) {
        optimizedParams.srcDataOffsets[i] = optimizedParams.srcDataOffsets[i - 1] + optimizedParams.dataSize[i - 1];
    }
}
REG_MKLDNN_PRIM_FOR(MKLDNNSplitNode, Split);
