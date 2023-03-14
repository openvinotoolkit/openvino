// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat.h"

#include <map>
#include <utility>
#include <vector>
#include <dnnl_extension_utils.h>

#include <onednn/dnnl.h>
#include <onednn/iml_type_mapper.h>
#include <edge.h>
#include <cpu_memory.h>
#include "ie_parallel.hpp"
#include "conv.h"
#include "fake_quantize.h"
#include "pooling.h"
#include "eltwise.h"
#include <limits>
#include "common/cpu_memcpy.h"
#include "common/blocked_desc_creator.h"
#include <memory_desc/cpu_memory_desc_utils.h>
using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {
    constexpr size_t channelAxis = 1lu;
}

bool Concat::isExecutable() const {
    return !hasEmptyOutputTensors() && !isOptimized();
}

bool Concat::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

Concat::Concat(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto inRank = getInputShapeAtPort(0).getRank();
    auto concatOp = ngraph::as_type_ptr<ngraph::op::v0::Concat>(op);
    auto axis = concatOp->get_axis();
    if (axis < 0) {
        axis += inRank;
    }
    if (axis >= static_cast<int64_t>(inRank) || axis < 0) {
        IE_THROW() << "Concat node with name '" << getName() << "' has invalid value of axis parameter: " << axis;
    }
    this->axis = axis;
}

void Concat::getSupportedDescriptors() {
    const auto& firstParentDims = getInputShapeAtPort(0).getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        const auto& dims = getInputShapeAtPort(i).getDims();
        bool incorrectDims = false;
        for (size_t j = 0; j < firstParentDims.size(); j++) {
            if (j == axis)
                continue;
            if (dims.size() != firstParentDims.size() || !dimsEqualWeak(firstParentDims[j], dims[j])) {
                incorrectDims = true;
                break;
            }
        }
        if (incorrectDims || firstParentDims.size() == 0) {
            IE_THROW() << "Incorrect input dimensions for concat node " << getName();
        }
    }

    // we need the first dims before axis to be 1 to avoid the reorder in the edge between the first parent and this concat
    // TODO [DS]: inplace
    if (!isDynamicNode()) {
        const auto& childDims = outputShapes[0].getStaticDims();
        if (std::all_of(childDims.begin(), childDims.begin() + axis, [](size_t dim) { return  dim == 1; }))
            canBeInPlace = true;
    }
}

void Concat::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto& originInputPrecisions = getOriginalInputPrecisions();
    inputPrecision = originInputPrecisions[0];
    bool isMixedPrecision = false;
    for (int i = 1; i < inputShapes.size(); i++) {
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

    const auto& dstShape = getOutputShapeAtPort(0);
    std::vector<LayoutType> tdCreatorTypes = {LayoutType::ncsp, LayoutType::nspc};

    // check if blocked layouts are available the channels size should be evenly divided by the block size to avoid slow oneDNN ref implementation
    if (dstShape.getRank() > channelAxis) {
        for (auto item : { std::make_pair(8lu, LayoutType::nCsp8c), std::make_pair(16lu, LayoutType::nCsp16c)}) {
            const VectorDims &blkDims = dstShape.getDims();
            if (blkDims[channelAxis] == Shape::UNDEFINED_DIM || blkDims[channelAxis] % item.first != 0)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                auto& srcDims = getInputShapeAtPort(i).getDims();
                if (srcDims[channelAxis] == Shape::UNDEFINED_DIM || srcDims[channelAxis] % item.first != 0) {
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

    auto itrRange = BlockedDescCreator::makeFilteredRange(creatorsMap, static_cast<unsigned>(dstShape.getRank()), tdCreatorTypes);
    for (auto itr = itrRange.first; itr != itrRange.second; ++itr) {
        NodeConfig config;

        config.dynBatchSupport = true;
        config.outConfs.resize(1);
        config.outConfs[0].inPlace(-1);
        config.outConfs[0].constant(false);
        config.outConfs[0].setMemDesc(itr->second->createSharedDesc(outputPrecision, dstShape));

        config.inConfs.resize(getParentEdges().size());

        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            config.inConfs[i].inPlace(-1);
            config.inConfs[i].constant(false);
            auto desc = itr->second->createSharedDesc(inputPrecision, getInputShapeAtPort(i));
            // TODO [DS]: inplace
            if (isDynamicNode()) {
                config.inConfs[i].setMemDesc(desc);
            } else {
                config.inConfs[i].setMemDesc(desc, BLOCKED_DESC_EMPTY_MASK);
            }
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
        if (itr->first != LayoutType::nspc) {
            pdIndexesToReuse.push_back(supportedPrimitiveDescriptors.size() - 1);
        }
    }

    // required to prevent incorrect memory sharing of a constant with other tensors on edges
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (getParentEdgeAt(i)->getParent()->isConstant()) {
            return;
        }
    }

    // TODO [DS]: inplace
    if (!canBeInPlace || std::any_of(inputShapes.begin(), inputShapes.end(), [](const Shape& shape) { return shape.hasZeroDims(); }))
        return;

    // Optimized inplace case
    for (auto refPdIndex : pdIndexesToReuse) {
        const auto& refConfig = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        auto config = refConfig;

        auto denseOutDesc = refConfig.outConfs[0].getMemDesc()->as<CpuBlockedMemoryDesc>();
        const auto &order = denseOutDesc->getOrder();
        const auto &blkDims = denseOutDesc->getBlockDims();
        auto numOfDim = blkDims.size();

        SizeVector offsets(numOfDim, 0lu);
        SizeVector strides(numOfDim);
        strides.back() = 1lu;
        size_t offset = Shape::UNDEFINED_DIM;
        BlockedMemoryDesc::CmpMask mask = BLOCKED_DESC_SKIP_OFFSET_MASK; // any offset

        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = Shape::UNDEFINED_DIM;
                mask.reset(numOfDim - i); // any strides on certain axis
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }

        config.outConfs[0].setMemDesc(std::dynamic_pointer_cast<CpuBlockedMemoryDesc>(refConfig.outConfs[0].getMemDesc()), mask);

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            const auto& srcBlkDims = refConfig.inConfs[i].getMemDesc()->as<CpuBlockedMemoryDesc>()->getBlockDims();
            const auto& shape = refConfig.inConfs[i].getMemDesc()->getShape();

            config.inConfs[i].inPlace(0);
            config.inConfs[i].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(inputPrecision, shape, srcBlkDims, order, offset, offsets, strides), mask);
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }
}

void Concat::selectOptimalPrimitiveDescriptor() {
    std::vector<size_t> canSelectPrimitive;

    // The double connection marks that some tensor should
    // be replicated. Inplace approach is not applicable
    // for that case.
    for (int i = 0; i < getParentEdges().size(); i++) {
        for (int j = i + 1; j < getParentEdges().size(); j++) {
            if (getParentEdgeAt(i) == getParentEdgeAt(j)) canBeInPlace = false;
        }
    }

    std::map<LayoutType, size_t> formatFrequency;
    std::vector<LayoutType> supportedLayouts = {LayoutType::ncsp, LayoutType::nspc, LayoutType::nCsp8c, LayoutType::nCsp16c};
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
        const auto &port_desc = parent_config.outConfs[outputIndex].getMemDesc();
        for (auto& item : supportedLayouts) {
            if (port_desc->hasLayoutType(item)) {
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
        const auto &port_desc = config.inConfs[inputIndex].getMemDesc();
        for (auto& item : supportedLayouts) {
            if (port_desc->hasLayoutType(item)) {
                formatFrequency[item] += 1;
            }
        }
    }

    size_t maxCount = 0;
    const auto &outDims = getOutputShapeAtPort(0).getDims();
    LayoutType convertTo = LayoutType::ncsp;
    for (auto &it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        } else if (it.second == maxCount) {
            if (context->isGraphQuantized() && it.first == LayoutType::nspc) {
                convertTo = it.first;
            } else if (it.first == LayoutType::nCsp8c || it.first == LayoutType::nCsp16c) {
                convertTo = it.first;
            }
        }
    }

    for (auto& item : { std::make_pair(8lu, LayoutType::nCsp8c), std::make_pair(16lu, LayoutType::nCsp16c) }) {
        if (convertTo == item.second) {
            if (outDims[channelAxis] == Shape::UNDEFINED_DIM || outDims[1] % item.first != 0) {
                convertTo = LayoutType::ncsp;
                break;
            }
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                const auto& inpDims = getInputShapeAtPort(i).getDims();
                if (inpDims[channelAxis] == Shape::UNDEFINED_DIM || inpDims[1] % item.first != 0) {
                    convertTo = LayoutType::ncsp;
                    break;
                }
            }
        }
    }

    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); ++i) {
        if (supportedPrimitiveDescriptors[i].getConfig().outConfs[0].getMemDesc()->hasLayoutType(convertTo)) {
            if (IMPLICATION(supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown, canBeInPlace)) {
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
        if (canBeInPlace && supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(i));
            return;
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

bool Concat::created() const {
    return getType() == Type::Concatenation;
}

bool Concat::isOptimized() const {
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].inPlace() >= 0;
}

bool Concat::needPrepareParams() const {
    if (canOptimizeNspc) {
        return false;
    }
    return inputShapesModified();
}

void Concat::prepareParams() {
    if (canOptimizeNspc || isOptimized())
        return;

    const auto& dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    auto dstMemDesc = dstMemPtr->GetDescWithType<BlockedMemoryDesc>();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    const auto& outputStrides = dstMemDesc->getStrides();
    size_t curConcatOffset = 0;
    const size_t elemSize = DnnlExtensionUtils::sizeOfDataType(dstMemPtr->GetDataType());
    const auto& src0BlkMemDesc = getParentEdgesAtPort(0)[0]->getMemoryPtr()->getDescPtr()->as<BlockedMemoryDesc>();
    const auto& outputOrder = src0BlkMemDesc->getOrder();
    for (size_t i = 0; i < outputOrder.size(); i++) {
        if (outputOrder[i] == axis) {
            reorderedAxis = i;
            break;
        }
    }
    const auto& outputShape = dstMemDesc->getBlockDims();
    for (size_t i = 0; i < reorderedAxis; i++) {
        if (outputShape[i] != 1) {
            hasOuterLoop = true;
        }
    }
    std::vector<memory::desc> srcs_d;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        const auto& srcMemPtr = getParentEdgesAtPort(i)[0]->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->isAllocated()) {
            auto parent = getParentEdgeAt(i)->getParent();
            IE_THROW() << "Source memory from " << parent->getName() << " didn't allocate for node "
                       << getName() << ".";
        }

        if (canExecRef) {
            const auto srcMemDesc = srcMemPtr->getDescPtr()->as<BlockedMemoryDesc>();
            const auto& inputShape = srcMemDesc->getBlockDims();
            const auto& strides = srcMemDesc->getStrides();
            inputStrides[i].resize(MAX_RANK_REF, 0);
            std::transform(strides.begin(), strides.end(), inputStrides[i].begin(), [&elemSize](const Dim& i) {
                return i * elemSize;
            });
            size_t nElem = 1;
            for (size_t j = reorderedAxis; j < inputShape.size(); j++) {
                nElem *= inputShape[j];
            }
            nelemToCopy[i] = nElem * elemSize;
            dstOffset[i] = outputStrides[reorderedAxis] * curConcatOffset * elemSize;
            curConcatOffset += inputShape[reorderedAxis];
        } else {
            if (srcMemPtr->GetShape().hasZeroDims()) {
                continue;
            }
            auto desc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
            const auto& dims = srcMemPtr->getStaticDims();
            for (size_t j = 0; j < dims.size(); j++) {
                desc.data.dims[j] = dims[j];
            }
            srcs_d.emplace_back(desc);
        }
    }
    if (!canExecRef) {
        auto desc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        const auto& dims = dstMemPtr->getStaticDims();
        for (size_t i = 0; i < dims.size(); i++) {
            desc.data.dims[i] = dims[i];
            desc.data.padded_dims[i] = dims[i];
        }

        auto primitive_desc = concat::primitive_desc(desc, static_cast<int>(axis), srcs_d, getEngine());
        prim = concat(primitive_desc);
    }
}

size_t Concat::inverseOrder(const SizeVector& order, size_t axis) {
    for (size_t i = 0; i < order.size(); i++) {
        if (axis == order[i]) {
            return i;
        }
    }
    return -1;
}

void Concat::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

   if (!isOptimized()) {
       Node::initOptimalPrimitiveDescriptor();
        auto config = selected_pd->getConfig();
        if (!isConfigDefined(config)) {
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                // Concat doesn't support different precision on inputs
                config.inConfs[i].setMemDesc(getConsistentInputDesc(config, i)->getMemDesc()->cloneWithNewPrecision(inputPrecision));
            }

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].setMemDesc(getConsistentOutputDesc(config, i)->getMemDesc()->cloneWithNewPrecision(outputPrecision));
            }

            initDescriptor(config);
        }
    }

    auto config = selected_pd->getConfig();
    if (!isDynamicNode() && !isConfigDefined(config)) {
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            int num = getChildEdgeAt(i)->getOutputNum();
            if (num >= 0) {
                auto childConf = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
                childConf.setMemDesc(childConf.getMemDesc()->cloneWithNewPrecision(config.outConfs[i].getMemDesc()->getPrecision()));

                if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
                    if (!childConf.getMemDesc()->isDefined() && childConf.inPlace() >= 0)
                        getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();

                    if (childConf.getMemDesc()->isDefined() && config.outConfs[i].getPortDesc()->isCompatible(*childConf.getPortDesc())) {
                        config.outConfs[i].setMemDesc(childConf.getMemDesc());
                        continue;
                    }
                }
            }

            // reset mask
            config.outConfs[i].setMemDesc(config.outConfs[i].getMemDesc());
        }
        auto firstOutBlockingDesc = config.outConfs[0].getMemDesc()->as<BlockedMemoryDesc>();
        size_t offset = 0;
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            auto oldDesc = config.inConfs[i].getMemDesc();
            auto inpBlockingDesc = oldDesc->as<BlockedMemoryDesc>();

            config.inConfs[i].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(inpBlockingDesc->getPrecision(),
                                                                            inpBlockingDesc->getShape(),
                                                                            inpBlockingDesc->getBlockDims(),
                                                                            inpBlockingDesc->getOrder(),
                                                                            firstOutBlockingDesc->getOffsetPadding() + offset,
                                                                            firstOutBlockingDesc->getOffsetPaddingToData(),
                                                                            firstOutBlockingDesc->getStrides()), BLOCKED_DESC_FULL_MASK);
            size_t axisSize = 1;

            auto firstInpBlockingDesc = config.inConfs[0].getMemDesc()->as<BlockedMemoryDesc>();
            if (firstInpBlockingDesc->hasLayoutType(LayoutType::nspc)) {
                // This is more general and works for any "direct" Layout (such as nchw or nhwc), but it doesn't work for blocked
                size_t realAxis = inverseOrder(firstInpBlockingDesc->getOrder(), axis);
                for (size_t j = realAxis; j < inpBlockingDesc->getBlockDims().size(); j++) {
                    size_t jj = firstInpBlockingDesc->getOrder()[j];
                    axisSize *= inpBlockingDesc->getBlockDims()[jj];
                }
            } else {
                // This works for nchw and nchw8c/nchw16c
                for (size_t j = axis; j < inpBlockingDesc->getBlockDims().size(); j++) {
                    axisSize *= inpBlockingDesc->getBlockDims()[j];
                }
            }
            offset += axisSize;
        }
        initDescriptor(config);
    }
    //block layout may have axis greater than rank, disable ref_concat
    auto primDesc = getSelectedPrimitiveDescriptor();
    auto memDesc = primDesc->getConfig().outConfs[0].getMemDesc()->as<BlockedMemoryDesc>();
    auto rank = memDesc->getShape().getRank();
    bool isBlocked = rank != memDesc->getBlockDims().size();
    if (!isBlocked && rank <= MAX_RANK_REF) {
        canExecRef = true;
        nelemToCopy.resize(getParentEdges().size(), 0);
        dstOffset.resize(getParentEdges().size());
        inputStrides.resize(getParentEdges().size());
        srcPtrs.resize(getParentEdges().size());
    }
    // check if selected Tensor descriptor has nspc layout and concat axis is C
    canOptimizeNspc = axis == channelAxis && getSelectedPrimitiveDescriptor()->getConfig().outConfs.front().getMemDesc()->hasLayoutType(LayoutType::nspc);
}

void Concat::execute(dnnl::stream strm) {
    if (isOptimized()) {
        return;
    }

    if (canOptimizeNspc) {
        execNspcSpecCase();
        return;
    }

    if (canExecRef) {
        execRef();
    } else {
        const Memory& dst_memory = getChildEdgeAt(0)->getMemory();
        const size_t num_src = getParentEdges().size();
        std::unordered_map<int, memory> mem_ags {{DNNL_ARG_DST, dst_memory.GetPrimitive()}};
        size_t nonZeroInShapes = 0;
        for (int i = 0; i < num_src; i++) {
            const auto& srcMem = getParentEdgesAtPort(i)[0]->getMemory();
            if (srcMem.GetShape().hasZeroDims()) {
                continue;
            }
            mem_ags[DNNL_ARG_MULTIPLE_SRC + nonZeroInShapes] = srcMem.GetPrimitive();
            nonZeroInShapes++;
        }
        prim.execute(strm, mem_ags);
    }
}

InferenceEngine::Precision Concat::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void Concat::execNspcSpecCase() {
    const Memory& dst_memory = getChildEdgeAt(0)->getMemory();
    const size_t num_src = getParentEdges().size();
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_memory.GetData());
    const size_t dataSize = DnnlExtensionUtils::sizeOfDataType(dst_memory.GetDataType());

    std::vector<size_t> channelsDataSize;
    size_t channels_size = 0;
    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;

    size_t nonZeroInShapes = 0;
    int firstNonZeroEdge = -1;
    for (size_t i = 0; i < num_src; i++) {
        const Memory& src_mem = getParentEdgesAtPort(i)[0]->getMemory();
        if (src_mem.GetShape().hasZeroDims()) {
            continue;
        }
        const size_t num_channels = src_mem.getStaticDims()[channelAxis];

        channelsDataSize.push_back(num_channels * dataSize);
        src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_mem.GetData()));
        dst_ptrs.push_back(dst_ptr + channels_size);
        channels_size += num_channels * dataSize;

        if (firstNonZeroEdge == -1) {
            firstNonZeroEdge = i;
        }

        nonZeroInShapes++;
    }

    const size_t iter_count = getParentEdgeAt(firstNonZeroEdge)->getMemory().GetSize() / channelsDataSize[0];

    parallel_for(iter_count, [&](int i) {
        const size_t dst_off = i * channels_size;
        for (int j = 0; j < nonZeroInShapes; j++) {
            cpu_memcpy(dst_ptrs[j] + dst_off, src_ptrs[j] + i * channelsDataSize[j], channelsDataSize[j]);
        }
    });
}

void Concat::execRef() {
    const size_t numSrc = getParentEdges().size();
    const Memory& dstMemory = getChildEdgeAt(0)->getMemory();
    const size_t elemSize = DnnlExtensionUtils::sizeOfDataType(dstMemory.GetDataType());
    const auto dstMemBlkDesc = dstMemory.getDescPtr()->as<BlockedMemoryDesc>();
    const auto& outputShape = dstMemBlkDesc->getBlockDims();
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dstMemory.GetData());
    for (size_t i = 0; i < numSrc; i++) {
        const Memory& srcMem = getParentEdgesAtPort(i)[0]->getMemory();
        srcPtrs[i] = reinterpret_cast<const uint8_t*>(srcMem.GetPtr());
    }
    size_t outputStrides[MAX_RANK_REF] = {0};
    const auto strides = dstMemBlkDesc->getStrides();
    std::transform(strides.begin(), strides.end(), outputStrides, [&elemSize](const Dim& i) {
        return i * elemSize;
    });
    if (!hasOuterLoop) {
        int nthr = parallel_get_max_threads();
        if (nthr == 1) {
            for (size_t a = 0; a < srcPtrs.size(); ++a) {
                const auto inData = srcPtrs[a];
                auto outputData = &dstPtr[dstOffset[a]];
                std::memcpy(outputData, inData, nelemToCopy[a]);
            }
        } else {
            parallel_nt(nthr, [&](int ithr, int nthr) {
                for (size_t a = 0; a < srcPtrs.size(); ++a) {
                    size_t start = 0, end = 0;
                    splitter(nelemToCopy[a], nthr, ithr, start, end);
                    const uint8_t* i = srcPtrs[a] + start;
                    uint8_t* o = dstPtr + dstOffset[a] + start;
                    std::memcpy(o, i, end - start);
                }
            });
        }
    } else {
        size_t physDims[5] = {1, 1, 1, 1, 1};
        for (size_t i = 0; i < reorderedAxis; i++) {
            physDims[i] = outputShape[i];
        }
        const auto L1Size = dnnl::utils::get_cache_size(1, true);
        UNUSED(L1Size); // for Windows
        parallel_for6d(physDims[0], physDims[1], physDims[2], physDims[3], physDims[4], numSrc,
                                [&](size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t a) {
            // check if zero memory
            if (srcPtrs[a] == nullptr) return;

            size_t inOff = inputStrides[a][0] * n0 + inputStrides[a][1] * n1 + inputStrides[a][2] * n2
                            + inputStrides[a][3] * n3 + inputStrides[a][4] * n4;
            size_t outOff = outputStrides[0] * n0 + outputStrides[1] * n1 + outputStrides[2] * n2
                             + outputStrides[3] * n3 + outputStrides[4] * n4;
            const uint8_t *i = &srcPtrs[a][inOff];
            uint8_t *o = &dstPtr[dstOffset[a] + outOff];

#if defined(__GNUC__)
            // Heuristic:
            // memcpy works generally faster for data sizes not
            // exceeding L1 cache.
            if (nelemToCopy[a] > L1Size) {
                // The code below performs data copying: o[e] = i[e]
                // and uses a workaround to make GNU compilers optimize it
                uint8_t *ptro = o;
                const uint8_t *ptri = i;
                // head part: bytes before 4 byte-align's address
                const size_t headPart = sizeof(uint32_t)
                                         - reinterpret_cast<uint64_t>(ptro)
                                           % sizeof(uint32_t);

                // main part: bytes in 4 byte-align
                const size_t mainPart
                        = (nelemToCopy[a] - headPart) / sizeof(uint32_t);
                // tail part: bytes after 4 byte-align
                const size_t tailPart
                        = (nelemToCopy[a]) - headPart
                          - (mainPart * sizeof(uint32_t));
                // copy head part
                for (size_t e = 0; e < headPart; ++e) {
                    *ptro = *ptri;
                    ++ptro;
                    ++ptri;
                }
                // copy main part
                std::memcpy(ptro, ptri, mainPart * sizeof(uint32_t));
                ptro += mainPart * sizeof(uint32_t);
                ptri += mainPart * sizeof(uint32_t);
                // copy tail part
                for (size_t e = 0; e < tailPart; ++e) {
                    *ptro = *ptri;
                    ++ptro;
                    ++ptri;
                }
            } else {
                std::memcpy(o, i, nelemToCopy[a]);
            }
#else
            std::memcpy(o, i, nelemToCopy[a]);
#endif
        });
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
