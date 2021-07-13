// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_concat_node.h"

#include <map>
#include <utility>
#include <vector>
#include <mkldnn_extension_utils.h>

#include "mkldnn.hpp"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_dims.h"
#include "mkldnn_edge.h"
#include "mkldnn_memory.h"
#include "ie_parallel.hpp"
#include "mkldnn_conv_node.h"
#include "mkldnn_fake_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include "mkldnn_eltwise_node.h"
#include <limits>
#include "common/cpu_memcpy.h"
#include "common/blocked_desc_creator.h"
#include <cpu_memory_desc_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace {
    constexpr size_t channelAxis = 1lu;
}

bool MKLDNNConcatNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto concatOp = ngraph::as_type_ptr<const ngraph::op::v0::Concat>(op);
        if (!concatOp) {
            errorMessage = "Node is not an instance of the Concat operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNConcatNode::MKLDNNConcatNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto concatOp = ngraph::as_type_ptr<ngraph::op::v0::Concat>(op);
    auto axis = concatOp->get_axis();
    if (axis < 0) {
        this->axis = concatOp->get_input_shape(0).size() + axis;
    } else {
        this->axis = axis;
    }
}

void MKLDNNConcatNode::getSupportedDescriptors() {
    auto& firstParentDims = getParentEdgeAt(0)->getShape().getStaticDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto& dims = getParentEdgeAt(i)->getShape().getStaticDims();
        bool incorrectDims = false;
        for (size_t j = 0; j < firstParentDims.size(); j++) {
            if (j == axis)
                continue;
            if (dims.size() != firstParentDims.size() || firstParentDims[j] != dims[j]) {
                incorrectDims = true;
                break;
            }
        }
        if (incorrectDims || firstParentDims.size() == 0) {
            IE_THROW() << "Incorrect input dimensions for concat node " << getName();
        }
    }
}

void MKLDNNConcatNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto& originInputPrecisions = getOriginalInputPrecisions();
    inputPrecision = originInputPrecisions[0];
    bool isMixedPrecision = false;
    for (int i = 1; i < getOriginalInputsNumber(); i++) {
        if (originInputPrecisions[0] != originInputPrecisions[i]) {
            isMixedPrecision = true;
            break;
        }
    }

    // Concat doesn't support different precision on inputs so fallback on FP32 in such case
    if (isMixedPrecision)
        inputPrecision = Precision::FP32;

    // Concat supports only equal precisions for inputs and output
    outputPrecision = inputPrecision;

    auto& dstDims = getChildEdgeAt(0)->getShape().getStaticDims();
    std::vector<GeneralLayout> tdCreatorTypes = {GeneralLayout::ncsp, GeneralLayout::nspc};

    // check if blocked layouts are available the channels size should be evenly divided by the block size to avoid slow oneDNN ref implementation
    if (dstDims.size() > channelAxis) {
        for (auto item : { std::make_pair(8lu, GeneralLayout::nCsp8c), std::make_pair(16lu, GeneralLayout::nCsp16c)}) {
            SizeVector blkDims = dstDims;
            if (blkDims[channelAxis] % item.first)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto& srcDims = getParentEdgeAt(i)->getShape().getStaticDims();
                if (srcDims[channelAxis] % item.first) {
                    blocked = false;
                    break;
                }
            }
            if (blocked) {
                tdCreatorTypes.push_back(item.second);
            }
        }
    }

    std::vector<size_t> pdIndexesToReuse;

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto itrRange = BlockedDescCreator::makeFilteredRange(creatorsMap, static_cast<unsigned>(dstDims.size()), tdCreatorTypes);
    for (auto itr = itrRange.first; itr != itrRange.second; ++itr) {
        NodeConfig config;

        config.dynBatchSupport = true;
        config.outConfs.resize(1);
        config.outConfs[0].inPlace = -1;
        config.outConfs[0].constant = false;
        config.outConfs[0].desc = itr->second->createUniqueDesc(outputPrecision, dstDims);

        config.inConfs.resize(getParentEdges().size());

        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            config.inConfs[i].inPlace = -1;
            config.inConfs[i].constant = false;
            config.inConfs[i].desc = MemoryDescUtils::applyUndefinedOffset(
                    itr->second->createDesc(inputPrecision, getParentEdgeAt(i)->getShape().getStaticDims()));
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
        if (itr->first != GeneralLayout::nspc) {
            pdIndexesToReuse.push_back(supportedPrimitiveDescriptors.size() - 1);
        }
    }

    if (axis != channelAxis)
        return;

    // Optimized inplace case

    for (auto refPdIndex : pdIndexesToReuse) {
        const auto& refConfig = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        auto config = refConfig;

        const auto &order = refConfig.outConfs[0].desc->as<BlockedMemoryDesc>()->getOrder();
        const auto &blkDims = refConfig.outConfs[0].desc->as<BlockedMemoryDesc>()->getBlockDims();
        auto numOfDim = blkDims.size();

        SizeVector offsets(numOfDim, 0lu);
        SizeVector strides(numOfDim);
        strides.back() = 1lu;
        size_t offset = (std::numeric_limits<size_t>::max)();

        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }

        config.outConfs[0].desc = MKLDNNPlugin::make_unique<BlockedMemoryDesc>(outputPrecision, dstDims, blkDims, order, offset, offsets, strides);

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            const auto& srcBlkDims = refConfig.inConfs[i].desc->as<BlockedMemoryDesc>()->getBlockDims();
            const auto& dims = refConfig.inConfs[i].desc->getShape().getStaticDims();

            config.inConfs[i].inPlace = 0;
            config.inConfs[i].desc = MKLDNNPlugin::make_unique<BlockedMemoryDesc>(inputPrecision, dims, srcBlkDims, order, offset, offsets, strides);
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }
}

void MKLDNNConcatNode::selectOptimalPrimitiveDescriptor() {
    std::vector<size_t> canSelectPrimitive;

    bool canOptimize = true;

    // The double connection marks that some tensor should
    // be replicated. Inplace approach is not applicable
    // for that case.
    for (int i = 0; i < getParentEdges().size(); i++) {
        for (int j = i + 1; j < getParentEdges().size(); j++) {
            if (getParentEdgeAt(i) == getParentEdgeAt(j)) canOptimize = false;
        }
    }

    if (axis != channelAxis) {
        canOptimize = false;
    }

    std::map<GeneralLayout, size_t> formatFrequency;
    std::vector<GeneralLayout> supportedLayouts = {GeneralLayout::ncsp, GeneralLayout::nspc, GeneralLayout::nCsp8c, GeneralLayout::nCsp16c};

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        auto parent_pdesc = parent->getSelectedPrimitiveDescriptor();
        if (parent_pdesc == nullptr)
            continue;

        const auto &parent_config = parent_pdesc->getConfig();
        int outputIndex = parentEdge->getInputNum();
        if (outputIndex < 0 || outputIndex >= parent_config.outConfs.size())
            IE_THROW() << "Cannot find index of output node";
        const auto &port_desc = parent_config.outConfs[outputIndex].desc;
        for (auto& item : supportedLayouts) {
            if (port_desc->checkGeneralLayout(item)) {
                formatFrequency[item] += 1;
            }
        }
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
            IE_THROW() << "Cannot find index of output node";
        const auto &port_desc = config.inConfs[inputIndex].desc;
        for (auto& item : supportedLayouts) {
            if (port_desc->checkGeneralLayout(item)) {
                formatFrequency[item] += 1;
            }
        }
    }

    size_t maxCount = 0;
    auto outDims = getChildEdgeAt(0)->getShape().getStaticDims();
    GeneralLayout convertTo = GeneralLayout::ncsp;
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        } else if (it.second == maxCount) {
            if (isInQuantizedGraph && it.first == GeneralLayout::nspc) {
                convertTo = it.first;
            } else if (it.first == GeneralLayout::nCsp8c || it.first == GeneralLayout::nCsp16c) {
                convertTo = it.first;
            }
        }
    }

    for (auto& item : { std::make_pair(8lu, GeneralLayout::nCsp8c), std::make_pair(16lu, GeneralLayout::nCsp16c) }) {
        if (convertTo == item.second) {
            if (outDims[1] % item.first != 0) {
                convertTo = GeneralLayout::ncsp;
                break;
            }
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto& inpDims = getParentEdgeAt(i)->getShape().getStaticDims();
                if (inpDims[1] % item.first != 0) {
                    convertTo = GeneralLayout::ncsp;
                    break;
                }
            }
        }
    }

    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); ++i) {
        if (supportedPrimitiveDescriptors[i].getConfig().outConfs[0].desc->checkGeneralLayout(convertTo)) {
            if (IMPLICATION(supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown, canOptimize)) {
                canSelectPrimitive.push_back(i);
            }
        }
    }

    if (canSelectPrimitive.size() == 1) {
        selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
        return;
    }

    // if there are more than one PD with similar data layouts - select the optimized one
    for (auto indx : canSelectPrimitive) {
        if (supportedPrimitiveDescriptors[indx].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(indx));
            return;
        }
    }

    // if there are no matching data layouts, select first optimized implementation
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        if (canOptimize && supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(i));
            return;
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
        IE_THROW() << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    //check if selected Tensor descriptor has nspc layout and concat axis is C
    if (axis == channelAxis && getChildEdgeAt(0)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nspc)) {
        canOptimizeNspc = true;
        return;
    }

    std::vector<memory::desc> srcs_d;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            IE_THROW() << "Source memory from " << parent->getName() << " didn't allocate for node "
                               << getName() << ".";
        }

        auto desc = srcMemPtr->GetDescriptor();
        auto& dims = getParentEdgeAt(i)->getShape().getStaticDims();
        for (size_t j = 0; j < dims.size(); j++) {
            desc.data.dims[j] = dims[j];
        }

        srcs_d.emplace_back(desc);
    }

    auto desc = getChildEdgeAt(0)->getMemory().GetDescriptor();
    auto& dims = getChildEdgeAt(0)->getShape().getStaticDims();
    for (size_t i = 0; i < dims.size(); i++) {
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
        IE_THROW() << "Preferable primitive descriptor is not set.";

   if (!isOptimized()) {
       MKLDNNNode::initOptimalPrimitiveDescriptor();
        auto config = selected_pd->getConfig();
        if (!isConfigDefined(config)) {
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                config.inConfs[i].desc = getDefinedInputDesc(config, i);
                // Concat doesn't support different precision on inputs
                config.inConfs[i].desc->setPrecision(inputPrecision);
            }

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].desc = getDefinedOutputDesc(config, i);
                config.outConfs[i].desc->setPrecision(outputPrecision);
            }

            initDescriptor(config);
        }
    }

    auto config = selected_pd->getConfig();
    if (isConfigDefined(config))
        return;

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        if (config.outConfs[i].desc->isDefined())
            continue;

        int num = getChildEdgeAt(i)->getOutputNum();
        if (num >= 0) {
            auto childConf = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
            childConf.desc->setPrecision(config.outConfs[i].desc->getPrecision());

            if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
                if (!childConf.desc->isDefined() && childConf.inPlace >= 0)
                    getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();

                if (childConf.desc->isDefined() && childConf.desc->isCompatible(*config.outConfs[i].desc)) {
                    config.outConfs[i].desc = childConf.desc->clone();
                    continue;
                }
            }
        }

        // reset undefined offsets
        config.outConfs[i].desc = MemoryDescUtils::resetOffset(config.outConfs[i].desc.get());
    }
    auto firstOutBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.outConfs[0].desc);
    size_t offset = 0;
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        auto inpBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.inConfs[i].desc);
        config.inConfs[i].desc = MKLDNNPlugin::make_unique<BlockedMemoryDesc>(inpBlockingDesc.getPrecision(),
                                                                inpBlockingDesc.getShape().getStaticDims(),
                                                                inpBlockingDesc.getBlockDims(),
                                                                inpBlockingDesc.getOrder(),
                                                                firstOutBlockingDesc.getOffsetPadding() + offset,
                                                                firstOutBlockingDesc.getOffsetPaddingToData(),
                                                                firstOutBlockingDesc.getStrides());
        size_t axisSize = 1;

        auto firstInpBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.inConfs[0].desc);
        if (firstInpBlockingDesc.checkGeneralLayout(GeneralLayout::nspc)) {
            // This is more general and works for any "direct" Layout (such as nchw or nhwc), but it doesn't work for blocked
            size_t realAxis = inverseOrder(firstInpBlockingDesc.getOrder(), axis);
            for (size_t j = realAxis; j < inpBlockingDesc.getBlockDims().size(); j++) {
                size_t jj = firstInpBlockingDesc.getOrder()[j];
                axisSize *= inpBlockingDesc.getBlockDims()[jj];
            }
        } else {
            // This works for nchw and nchw8c/nchw16c
            for (size_t j = axis; j < inpBlockingDesc.getBlockDims().size(); j++) {
                axisSize *= inpBlockingDesc.getBlockDims()[j];
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

    if (canOptimizeNspc) {
        execNspcSpecCase();
        return;
    }

    const MKLDNNMemory& dst_memory = getChildEdgeAt(0)->getMemory();
    const size_t num_src = getParentEdges().size();
    std::unordered_map<int, memory> mem_ags {{DNNL_ARG_DST, dst_memory.GetPrimitive()}};
    for (int i = 0; i < num_src; i++)
        mem_ags[DNNL_ARG_MULTIPLE_SRC + i] = getParentEdgeAt(i)->getMemory().GetPrimitive();

    (*prim).execute(strm, mem_ags);
}

InferenceEngine::Precision MKLDNNConcatNode::getRuntimePrecision() const {
    return MKLDNNExtensionUtils::getMaxPrecision(getInputPrecisions());
}

void MKLDNNConcatNode::execNspcSpecCase() {
    const MKLDNNMemory& dst_memory = getChildEdgeAt(0)->getMemory();
    const size_t num_src = getParentEdges().size();
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_memory.GetData());
    const size_t dataSize = MKLDNNExtensionUtils::sizeOfDataType(dst_memory.GetDataType());

    std::vector<size_t> channelsDataSize;
    size_t channels_size = 0;
    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;

    for (size_t i = 0; i < num_src; i++) {
        const MKLDNNMemory& src_mem = getParentEdgeAt(i)->getMemory();
        const size_t num_channels = src_mem.GetDims()[channelAxis];

        channelsDataSize.push_back(num_channels * dataSize);
        src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_mem.GetData()));
        dst_ptrs.push_back(dst_ptr + channels_size);
        channels_size += num_channels * dataSize;
    }

    const size_t iter_count = getParentEdgeAt(0)->getMemory().GetSize() / channelsDataSize[0];

    parallel_for(iter_count, [&](int i) {
        const size_t dst_off = i * channels_size;
        for (int j = 0; j < num_src; j++) {
            cpu_memcpy(dst_ptrs[j] + dst_off, src_ptrs[j] + i * channelsDataSize[j], channelsDataSize[j]);
        }
    });
}

REG_MKLDNN_PRIM_FOR(MKLDNNConcatNode, Concatenation);
