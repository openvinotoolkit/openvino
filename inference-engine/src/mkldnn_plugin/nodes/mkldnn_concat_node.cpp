// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_concat_node.h"

#include <map>
#include <utility>
#include <vector>
#include <mkldnn_extension_utils.h>

#include "details/ie_exception.hpp"
#include "ie_layers.h"
#include "mkldnn.hpp"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_dims.h"
#include "mkldnn_edge.h"
#include "mkldnn_memory.h"
#include <limits>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConcatNode::MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNConcatNode::getSupportedDescriptors() {
    auto * conLayer = dynamic_cast<ConcatLayer*>(getCnnLayer().get());

    if (conLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert concat layer.";

    axis = conLayer->_axis;

    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    auto& firstParentDims = getParentEdgeAt(0)->getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto& dims = getParentEdgeAt(i)->getDims();
        bool incorrectDims = false;
        for (size_t j = 0; j < firstParentDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (dims.ndims() != firstParentDims.ndims() || firstParentDims[j] != dims[j]) {
                incorrectDims = true;
                break;
            }
        }
        if (incorrectDims || firstParentDims.ndims() == 0) {
            THROW_IE_EXCEPTION << "Incorrect input dimensions for concat node " << getName();
        }
    }
}

void MKLDNNConcatNode::initSupportedPrimitiveDescriptors() {
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

    MKLDNNDims dstDims = getChildEdgeAt(0)->getDims();
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    bool hasEltwise = false;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge->getParent()->getType() == Eltwise)
            hasEltwise = true;

        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(parentEdge->getDims(), inputDataType, memory::format::any));
        config.inConfs.push_back(dataConfig);
    }

    auto dims = getChildEdgeAt(0)->getDims();

    config.outConfs.resize(1);
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(dims, outputDataType, MKLDNNMemory::GetPlainFormat(dims)));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
    if (dims.ndims() == 4) {
        if (dims[1] % 8 == 0) {
            config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(dims, outputDataType, mkldnn::memory::nChw8c));
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);

            if (dims[1] % 16 == 0) {
                config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(dims, outputDataType, mkldnn::memory::nChw16c));
                supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
            }
        }
    }

    if (axis != 1 || hasEltwise)
        return;

    auto numOfDim = static_cast<size_t>(dstDims.ndims());

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
            strides[numOfDim - i] = strides[numOfDim - i + 1] * dstDims[numOfDim - i + 1];
        }
    }

    config.outConfs[0].desc = TensorDesc(Precision::FP32, dstDims.ToSizeVector(), {dstDims.ToSizeVector(), order, offset, offsets, strides});
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        config.inConfs[i].inPlace = 0;
        config.inConfs[i].desc = TensorDesc(Precision::FP32, parentEdge->getDims().ToSizeVector(),
                                            {parentEdge->getDims().ToSizeVector(), order, offset, offsets, strides});
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);

    if (numOfDim == 4) {
        order = {0, 1, 2, 3, 1};
        offsets = {0, 0, 0, 0, 0};
        numOfDim = 5;

        // nChw8c and nChw16c
        for (int sizeS : {8, 16}) {
            SizeVector blkDims = dstDims.ToSizeVector();
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
            config.outConfs[0].desc = TensorDesc(Precision::FP32, dstDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

            bool canInplace = true;
            for (size_t i = 0; canInplace && i < getParentEdges().size(); i++) {
                auto parentEdge = getParentEdgeAt(i);
                blkDims = parentEdge->getDims().ToSizeVector();
                if (blkDims[1] % sizeS)
                    canInplace = false;

                blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1 : 0);
                blkDims.push_back(sizeS);
                config.inConfs[i].desc =  TensorDesc(Precision::FP32, parentEdge->getDims().ToSizeVector(),
                                                     {blkDims, order, offset, offsets, strides});
            }
            if (canInplace)
                supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    }
}

void MKLDNNConcatNode::selectOptimalPrimitiveDescriptor() {
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
                primDescInfo.getConfig().inConfs[0].inPlace < 0)
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

    bool canOptimize = true;
    for (size_t i = 0; canOptimize && i < getParentEdges().size(); i++) {
        const auto& parent = getParentEdgeAt(i)->getParent();
        for (size_t j = 0; canOptimize && j < parent->getChildEdges().size(); j++) {
            const auto& child = parent->getChildEdgeAt(j)->getChild();
            const auto* childConcat = dynamic_cast<MKLDNNConcatNode *>(child.get());
            if (!childConcat || childConcat == this)
                continue;
            if (childConcat->isOptimized())
                canOptimize = false;
        }
    }
    if (hasUnknown && axis == 1) {
        if (canSelectPrimitive.size() == 1) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
            return;
        }
    } else {
        canOptimize = false;
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
    mkldnn::memory::format convertTo = MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims());
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        }
    }

    if (canOptimize && MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, convertTo).blocksExtended())
        convertTo = MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims());
    for (size_t i = 0; canOptimize && i < getParentEdges().size(); i++) {
        if (MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, convertTo).blocksExtended())
            convertTo = MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims());
    }

    if (!canOptimize) {
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
        return;
    }

    for (auto supportedPdIndex : canSelectPrimitive) {
        if (MKLDNNMemoryDesc(supportedPrimitiveDescriptors[supportedPdIndex].getConfig().inConfs[0].desc).getFormat() == convertTo) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPdIndex));
            return;
        }
    }

    THROW_IE_EXCEPTION << "Cannot find optimal primitive desc for node: " << getName();
}

bool MKLDNNConcatNode::created() const {
    return getType() == Concatenation;
}

bool MKLDNNConcatNode::isOptimized() const {
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].inPlace >= 0;
}

void MKLDNNConcatNode::createPrimitive() {
    if (prim || isOptimized())
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<primitive::at> srcs_p;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate for node "
                               << getName() << ".";
        }

        auto desc = srcMemPtr->GetDescriptor();
        auto dims = getParentEdgeAt(i)->getDims();
        for (size_t j = 0; j < dims.ndims(); j++) {
            desc.data.dims[j] = dims[j];
        }

        srcs_pd.emplace_back(desc, srcMemPtr->GetPrimitiveDescriptor().get_engine());
        srcs_p.emplace_back(srcMemPtr->GetPrimitive());
    }

    auto desc = getChildEdgeAt(0)->getMemory().GetDescriptor();
    auto dims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < dims.ndims(); i++) {
        desc.data.dims[i] = dims[i];
        desc.data.layout_desc.blocking.padding_dims[i] = dims[i];
    }

    auto primitive_desc = concat::primitive_desc(desc, static_cast<int>(axis), srcs_pd);

    prim.reset(new concat(primitive_desc, srcs_p, getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

void MKLDNNConcatNode::initOptimalPrimitiveDescriptor() {
    if (!isOptimized()) {
        MKLDNNNode::initOptimalPrimitiveDescriptor();
        return;
    }

    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        if (config.outConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
                !isUninitTensorDesc(config.outConfs[i].desc))
            continue;

        int num = getChildEdgeAt(i)->getOutputNum();
        if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
            if (num >= 0) {
                if (isUninitTensorDesc(getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num].desc) &&
                        getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num].inPlace >= 0)
                    getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();
                if (!isUninitTensorDesc(getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num].desc) &&
                        MKLDNNExtensionUtils::initTensorsAreEqual(
                                getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num].desc,
                                config.outConfs[i].desc)) {
                    config.outConfs[i].desc = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num].desc;
                    continue;
                }
            }
        }
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder()
                                                              });
    }
    size_t offset = 0;
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                             config.inConfs[i].desc.getDims(), {
                                                                  config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                  config.inConfs[i].desc.getBlockingDesc().getOrder(),
                                                                  config.outConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                  config.outConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                  config.outConfs[0].desc.getBlockingDesc().getStrides()
                                                             });
        size_t axisSize = 1;
        for (size_t j = axis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
            axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[j];
        }
        offset += axisSize;
    }
    initDescriptor(config);
}
