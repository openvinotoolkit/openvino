// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_concat_node.h"

#include <map>
#include <utility>
#include <vector>
#include <mkldnn_extension_utils.h>

#include "details/ie_exception.hpp"
#include <legacy/ie_layers.h>
#include "mkldnn.hpp"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_dims.h"
#include "mkldnn_edge.h"
#include "mkldnn_memory.h"
#include "ie_parallel.hpp"
#include "mkldnn_conv_node.h"
#include "mkldnn_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_eltwise_node.h"
#include <limits>
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConcatNode::MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNConcatNode::getSupportedDescriptors() {
    auto * conLayer = dynamic_cast<ConcatLayer*>(getCnnLayer().get());

    if (conLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert concat layer.";

    axis = conLayer->_axis;

    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
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

    inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    bool isMixedPrecision = false;
    for (int i = 1; i < getCnnLayer()->insData.size(); i++) {
        if (getCnnLayer()->insData[0].lock()->getPrecision() != getCnnLayer()->insData[i].lock()->getPrecision()) {
            isMixedPrecision = true;
            break;
        }
    }

    // MKLDNN doesn't support different precision on inputs so fallback on FP32 in such case
    if (isMixedPrecision)
        inputPrecision = Precision::FP32;

    // Concat node supports int8 implementations only for NHWC and NDHWC layouts
    if (inputPrecision == Precision::U8 || inputPrecision == Precision::I8) {
        int ndims = getChildEdgeAt(0)->getDims().ndims();
        if (ndims != 2 && ndims != 4 && ndims != 5)
            inputPrecision = Precision::FP32;
    }

    // MKLDNN supports only equal precisions for inputs and output
    outputPrecision = inputPrecision;

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    MKLDNNDims dstDims = getChildEdgeAt(0)->getDims();
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);

        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        auto fmt = (inputPrecision == Precision::U8 || inputPrecision == Precision::I8) ? parentEdge->getDims().ndims() == 2 ? memory::format_tag::nc :
                                                                                          parentEdge->getDims().ndims() == 4 ? memory::format_tag::nhwc :
                                                                                                                               memory::format_tag::ndhwc
                                                                                        : memory::format_tag::any;

        dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(parentEdge->getDims(), inputDataType, fmt));
        config.inConfs.push_back(dataConfig);
    }

    auto dims = getChildEdgeAt(0)->getDims();

    config.outConfs.resize(1);
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    if ((!isMixedPrecision && outputPrecision != Precision::U8 && outputPrecision != Precision::I8) || axis != 1) {
        auto fmt = (inputPrecision == Precision::U8 || inputPrecision == Precision::I8) ? dims.ndims() == 2 ? memory::format_tag::nc :
                                                                                          dims.ndims() == 4 ? memory::format_tag::nhwc :
                                                                                                              memory::format_tag::ndhwc
                                                                                        : MKLDNNMemory::GetPlainFormat(dims);

        config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(dims, outputDataType, fmt));
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, fmt);

        if (inputPrecision != Precision::U8 && inputPrecision != Precision::I8) {
            if (dims.ndims() == 4) {
                if (dims[1] % 8 == 0) {
                    config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(
                            MKLDNNMemoryDesc(dims, outputDataType, memory::format_tag::nChw8c));
                    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, memory::format_tag::nChw8c);

                    if (dims[1] % 16 == 0) {
                        config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(
                                MKLDNNMemoryDesc(dims, outputDataType, mkldnn::memory::format_tag::nChw16c));
                        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, mkldnn::memory::format_tag::nChw16c);
                    }
                }
            } else if (dims.ndims() == 5) {
                if (dims[1] % 8 == 0) {
                    config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(
                            MKLDNNMemoryDesc(dims, outputDataType, mkldnn::memory::format_tag::nCdhw8c));
                    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, mkldnn::memory::format_tag::nCdhw8c);

                    if (dims[1] % 16 == 0) {
                        config.outConfs[0].desc = MKLDNNExtensionUtils::getUninitTensorDesc(
                                MKLDNNMemoryDesc(dims, outputDataType, mkldnn::memory::format_tag::nCdhw16c));
                        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, mkldnn::memory::format_tag::nCdhw16c);
                    }
                }
            }
        }
    }

    if (axis != 1)
        return;

    auto numOfDim = static_cast<size_t>(dstDims.ndims());

    SizeVector order(numOfDim);
    SizeVector offsets(numOfDim, 0lu);
    size_t offset = (std::numeric_limits<size_t>::max)();
    for (size_t i = 0; i < numOfDim; i++) {
        order[i] = i;
    }

    if (outputPrecision == Precision::I8 || outputPrecision == Precision::U8) {
        if (numOfDim == 4) {
            // Here we assume NHWC layout (channels are the last)

            order = {0, 2, 3, 1};
            offsets = {0, 0, 0, 0};

            SizeVector blkDims = dstDims.ToSizeVector();
            blkDims = { blkDims[0], blkDims[2], blkDims[3], blkDims[1] };

            SizeVector strides(numOfDim);
            strides.resize(numOfDim);
            // C is the last in NHWC, so all strides are max()
            for (size_t i = 0; i < numOfDim; i++) {
                strides[i] = (std::numeric_limits<size_t>::max)();
            }

            config.outConfs[0].desc = TensorDesc(outputPrecision,
                                                 dstDims.ToSizeVector(),
                                                 { blkDims, order, offset, offsets, strides });
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto parentEdge = getParentEdgeAt(i);

                SizeVector blkDims = parentEdge->getDims().ToSizeVector();
                blkDims = { blkDims[0], blkDims[2], blkDims[3], blkDims[1] };

                config.inConfs[i].inPlace = -1;     // Change to 0 here if inplace concat is supported for NHWC in mkldnn

                config.inConfs[i].desc = TensorDesc(inputPrecision, parentEdge->getDims().ToSizeVector(),
                                                    {blkDims, order, offset, offsets, strides});
            }

            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, mkldnn::memory::format_tag::nhwc);

            return;
        } else if (numOfDim == 5) {
            // Here we assume NDHWC layout (channels are the last)

            order = {0, 2, 3, 4, 1};
            offsets = {0, 0, 0, 0, 0};

            SizeVector blkDims = dstDims.ToSizeVector();
            blkDims = { blkDims[0], blkDims[2], blkDims[3], blkDims[4], blkDims[1] };

            SizeVector strides(numOfDim);
            strides.resize(numOfDim);
            // C is the last in NDHWC, so all strides are max()
            for (size_t i = 0; i < numOfDim; i++) {
                strides[i] = (std::numeric_limits<size_t>::max)();
            }

            config.outConfs[0].desc = TensorDesc(outputPrecision,
                                                 dstDims.ToSizeVector(),
                                                 { blkDims, order, offset, offsets, strides });
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto parentEdge = getParentEdgeAt(i);

                SizeVector blkDims = parentEdge->getDims().ToSizeVector();
                blkDims = { blkDims[0], blkDims[2], blkDims[3], blkDims[4], blkDims[1] };

                config.inConfs[i].inPlace = -1;     // Change to 0 here if inplace concat is supported for NDHWC in mkldnn

                config.inConfs[i].desc = TensorDesc(inputPrecision, parentEdge->getDims().ToSizeVector(),
                                                    {blkDims, order, offset, offsets, strides});
            }

            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, mkldnn::memory::format_tag::ndhwc);

            return;
        }
    }

    SizeVector strides(numOfDim);
    strides[numOfDim - 1] = 1;
    for (size_t i = 2; i <= numOfDim; i++) {
        if (numOfDim - i < axis) {
            strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
        } else {
            strides[numOfDim - i] = strides[numOfDim - i + 1] * dstDims[numOfDim - i + 1];
        }
    }

    config.outConfs[0].desc = TensorDesc(
            MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType),
            dstDims.ToSizeVector(),
            {dstDims.ToSizeVector(), order, offset, offsets, strides});
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        config.inConfs[i].inPlace = 0;
        config.inConfs[i].desc = TensorDesc(MKLDNNExtensionUtils::DataTypeToIEPrecision(inputDataType), parentEdge->getDims().ToSizeVector(),
                                            {parentEdge->getDims().ToSizeVector(), order, offset, offsets, strides});
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, MKLDNNMemory::Convert(config.outConfs[0].desc.getLayout()));

    if (numOfDim == 4lu || numOfDim == 5lu) {
        size_t blkDimsLen = numOfDim + 1;
        order.resize(blkDimsLen);
        for (size_t i = 0; i < numOfDim; i++) {
            order[i] = i;
        }
        order[numOfDim] = 1lu;
        offsets = SizeVector(blkDimsLen, 0lu);

        // nChw8c, nChw16c, nCdhw8c, nCdhw16c
        for (size_t sizeS : {8lu, 16lu}) {
            SizeVector blkDims = dstDims.ToSizeVector();
            if (blkDims[1] % sizeS)
                continue;
            blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1lu : 0lu);
            blkDims.push_back(sizeS);

            strides.resize(blkDimsLen);
            strides[blkDimsLen - 1] = 1;
            for (size_t i = 2lu; i <= blkDimsLen; i++) {
                if (blkDimsLen - i < axis) {
                    strides[blkDimsLen - i] = (std::numeric_limits<size_t>::max)();
                } else {
                    strides[blkDimsLen - i] = strides[blkDimsLen - i + 1] * blkDims[blkDimsLen - i + 1];
                }
            }
            config.outConfs[0].desc = TensorDesc(
                    MKLDNNExtensionUtils::DataTypeToIEPrecision(outputDataType),
                    dstDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

            bool canInplace = true;
            for (size_t i = 0lu; canInplace && i < getParentEdges().size(); i++) {
                auto parentEdge = getParentEdgeAt(i);
                blkDims = parentEdge->getDims().ToSizeVector();
                if (blkDims[1] % sizeS)
                    canInplace = false;

                blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1lu : 0lu);
                blkDims.push_back(sizeS);
                config.inConfs[i].desc =  TensorDesc(MKLDNNExtensionUtils::DataTypeToIEPrecision(inputDataType), parentEdge->getDims().ToSizeVector(),
                                                     {blkDims, order, offset, offsets, strides});
            }
            if (canInplace) {
                auto dstFormat = numOfDim == 4lu ? sizeS == 8lu ? mkldnn::memory::format_tag::nChw8c : mkldnn::memory::format_tag::nChw16c
                                                 : sizeS == 8lu ? mkldnn::memory::format_tag::nCdhw8c : mkldnn::memory::format_tag::nCdhw16c;
                supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, dstFormat);
            }
        }
    }
}

void MKLDNNConcatNode::selectOptimalPrimitiveDescriptor() {
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

    bool hasDoubleConnection = false;
    for (int i = 0; i < getParentEdges().size(); i++) {
        for (int j = i + 1; j < getParentEdges().size(); j++) {
            if (getParentEdgeAt(i) == getParentEdgeAt(j)) hasDoubleConnection = true;
        }
    }

    if (hasDoubleConnection) {
        // The double connection marks that some tensor should
        // be replicated. Inplace approach is not applicable
        // for that case. Descriptor with index 0 is pure copy
        // implementation
        selectPrimitiveDescriptorByIndex(0);
        return;
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

    std::map<PartialBlkDesc, size_t> formatFrequency;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        auto parent_pdesc = parent->getSelectedPrimitiveDescriptor();
        if (parent_pdesc == nullptr)
            continue;

        const auto &parent_config = parent_pdesc->getConfig();
        int outputIndex = parentEdge->getInputNum();
        if (outputIndex < 0 || outputIndex >= parent_config.outConfs.size())
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        const auto &port_desc = parent_config.outConfs[outputIndex].desc;
        if (port_desc.getLayout() == Layout::ANY)
            continue;
        auto partial_format_desc = PartialBlkDesc::extractFrom(port_desc);
        formatFrequency[partial_format_desc] += 1;
    }
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto child = childEdge->getChild();
        const auto *prim_desc = child->getSelectedPrimitiveDescriptor();
        if (prim_desc == nullptr)
            continue;

        const auto &config = prim_desc->getConfig();
        int inputIndex = childEdge->getOutputNum();
        if (inputIndex < 0 || inputIndex >= config.inConfs.size())
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        const auto &port_desc = config.inConfs[inputIndex].desc;
        if (port_desc.getLayout() == Layout::ANY)
            continue;
        auto partial_format_desc = PartialBlkDesc::extractFrom(port_desc);
        formatFrequency[partial_format_desc] += 1;
    }

    size_t maxCount = 0;
    auto convertTo = PartialBlkDesc::makePlain(getChildEdgeAt(0)->getDims().ToSizeVector());
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        }
    }

    if (canOptimize && convertTo.isAutoExtendedWith(getChildEdgeAt(0)->getDims().ToSizeVector()))
        convertTo = PartialBlkDesc::makePlain(getChildEdgeAt(0)->getDims().ToSizeVector());
    for (size_t i = 0; canOptimize && i < getParentEdges().size(); i++) {
        if (convertTo.isAutoExtendedWith(getParentEdgeAt(i)->getDims().ToSizeVector()))
            convertTo = PartialBlkDesc::makePlain(getChildEdgeAt(0)->getDims().ToSizeVector());
    }

    for (auto supportedPdIndex : canSelectPrimitive) {
        if (PartialBlkDesc::extractFrom(supportedPrimitiveDescriptors[supportedPdIndex].getConfig().inConfs[0].desc) == convertTo) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPdIndex));
            return;
        }
    }

    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        auto &primDescInfo = supportedPrimitiveDescriptors[i];
        if (primDescInfo.getImplementationType() == impl_desc_type::unknown)
            continue;
        if (convertTo == PartialBlkDesc::extractFrom(supportedPrimitiveDescriptors[i].getConfig().outConfs[0].desc)) {
            size_t num = 0;
            for (num = 0; num < getParentEdges().size(); num++) {
                if (convertTo.isAutoExtendedWith(getParentEdgeAt(num)->getDims().ToSizeVector()))
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
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    std::vector<memory::desc> srcs_d;

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

        srcs_d.emplace_back(desc);
    }

    auto desc = getChildEdgeAt(0)->getMemory().GetDescriptor();
    auto dims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < dims.ndims(); i++) {
        desc.data.dims[i] = dims[i];
        desc.data.padded_dims[i] = dims[i];
    }

    auto primitive_desc = concat::primitive_desc(desc, static_cast<int>(axis), srcs_d, getEngine());
    prim.reset(new concat(primitive_desc));
}

size_t MKLDNNConcatNode::inverseOrder(const SizeVector& order, size_t axis) {
    for (size_t i = 0; i < order.size(); i++) {
        if (axis == order[i]) {
            return i;
        }
    }
    return -1;
}

void MKLDNNConcatNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    if (!isOptimized()) {
        auto config = selected_pd->getConfig();
        if (!isInitConfig(config)) {
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                config.inConfs[i].desc = getConfiguredInputDesc(config, i);
                // MKLDNN doesn't support different precision on inputs
                config.inConfs[i].desc.setPrecision(inputPrecision);
            }

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
                config.outConfs[i].desc.setPrecision(outputPrecision);
            }

            initDescriptor(config);
        }

        return;
    }

    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        if (config.outConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
                !isUninitTensorDesc(config.outConfs[i].desc))
            continue;

        int num = getChildEdgeAt(i)->getOutputNum();
        if (num >= 0) {
            auto childConf = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
            childConf.desc.setPrecision(config.outConfs[i].desc.getPrecision());

            if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
                if (isUninitTensorDesc(childConf.desc) && childConf.inPlace >= 0)
                    getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();

                if (!isUninitTensorDesc(childConf.desc) &&
                        MKLDNNExtensionUtils::initTensorsAreEqual(childConf.desc, config.outConfs[i].desc)) {
                    config.outConfs[i].desc = childConf.desc;
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

        if (config.inConfs[0].desc.getLayout() == Layout::NHWC) {
            // This is more general and works for any "direct" Layout (such as nchw or nhwc), but it doesn't work for nchw8c
            size_t realAxis = inverseOrder(config.inConfs[0].desc.getBlockingDesc().getOrder(), axis);
            for (size_t j = realAxis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
                size_t jj = config.inConfs[0].desc.getBlockingDesc().getOrder()[j];
                axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[jj];
            }
        } else {
            // This works for nchw and nchw8c/nchw16c
            for (size_t j = axis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
                axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[j];
            }
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNConcatNode::execute(mkldnn::stream strm) {
    if (isOptimized()) {
        return;
    }

    const MKLDNNMemory& dst_memory = getChildEdgeAt(0)->getMemory();
    const mkldnn::memory::data_type data_type = dst_memory.GetDataType();
    const size_t num_src = getParentEdges().size();

    const bool isInt8 = (data_type == mkldnn_s8 || data_type == mkldnn_u8);

    if (isInt8) {
        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_memory.GetData());

        std::vector<size_t> channels;
        size_t channels_size = 0;
        std::vector<const uint8_t*> src_ptrs;
        std::vector<uint8_t*> dst_ptrs;

        for (size_t i = 0; i < num_src; i++) {
            const MKLDNNMemory& src_mem = getParentEdgeAt(i)->getMemory();
            const size_t num_channels = src_mem.GetDims()[1];

            channels.push_back(num_channels);
            src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_mem.GetData()));
            dst_ptrs.push_back(dst_ptr + channels_size);
            channels_size += num_channels;
        }

        const size_t iter_count = getParentEdgeAt(0)->getMemory().GetSize() / channels[0];

        parallel_for(iter_count, [&](int i) {
            const size_t dst_off = i * channels_size;
            for (int j = 0; j < num_src; j++) {
                cpu_memcpy(dst_ptrs[j] + dst_off, src_ptrs[j] + i * channels[j], channels[j]);
            }
        });
    } else {
        std::unordered_map<int, memory> mem_ags {{DNNL_ARG_DST, dst_memory.GetPrimitive()}};
        for (int i = 0; i < num_src; i++)
            mem_ags[DNNL_ARG_MULTIPLE_SRC + i] = getParentEdgeAt(i)->getMemory().GetPrimitive();

        (*prim).execute(strm, mem_ags);
    }
}

InferenceEngine::Precision MKLDNNConcatNode::getRuntimePrecision() const {
    return MKLDNNExtensionUtils::getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNConcatNode, Concatenation);
