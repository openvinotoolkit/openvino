// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat.h"

#include <cpu_memory.h>
#include <edge.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <onednn/iml_type_mapper.h>
#include <partitioned_mem_blk.h>

#include <map>
#include <utility>
#include <vector>

#include "common/blocked_desc_creator.h"
#include "common/cpu_memcpy.h"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/concat.hpp"
using namespace dnnl;

namespace ov::intel_cpu::node {
namespace {
constexpr size_t channelAxis = 1lu;
}

bool Concat::neverExecute() const {
    return isInPlace() || getSelectedPrimitiveDescriptor()->hasZeroOutputDims();
}

bool Concat::isExecutable() const {
    return !isInPlace() && !hasEmptyOutputTensors();
}

bool Concat::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto concatOp = ov::as_type_ptr<const ov::op::v0::Concat>(op);
        if (!concatOp) {
            errorMessage = "Node is not an instance of the Concat operation.";
            return false;
        }
        if (concatOp->get_output_element_type(0) == ov::element::string) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Concat::Concat(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto inRank = getInputShapeAtPort(0).getRank();
    auto concatOp = ov::as_type_ptr<ov::op::v0::Concat>(op);
    auto axis = concatOp->get_axis();
    if (axis < 0) {
        axis += inRank;
    }
    if (axis >= static_cast<int64_t>(inRank) || axis < 0) {
        THROW_CPU_NODE_ERR("has invalid value of axis parameter: ", axis);
    }
    this->axis = axis;
}

void Concat::getSupportedDescriptors() {
    const auto& firstParentDims = getInputShapeAtPort(0).getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        const auto& dims = getInputShapeAtPort(i).getDims();
        bool incorrectDims = false;
        for (size_t j = 0; j < firstParentDims.size(); j++) {
            if (j == axis) {
                continue;
            }
            if (dims.size() != firstParentDims.size() || !dimsEqualWeak(firstParentDims[j], dims[j])) {
                incorrectDims = true;
                break;
            }
        }
        if (incorrectDims || firstParentDims.size() == 0) {
            THROW_CPU_NODE_ERR("has incorrect input dimensions");
        }
    }

    // we need the first dims before axis to be 1 to avoid the reorder in the edge between the first parent and this
    // concat

    const auto& childDims = outputShapes[0].getDims();
    if (childDims[axis] != Shape::UNDEFINED_DIM &&
        std::all_of(childDims.begin(), childDims.begin() + axis, [](size_t dim) {
            return dim == 1;
        })) {
        canBeInPlace = true;
    }
}

void Concat::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto& originInputPrecisions = getOriginalInputPrecisions();
    inputPrecision = originInputPrecisions[0];
    bool isMixedPrecision = false;
    for (size_t i = 1; i < inputShapes.size(); i++) {
        if (originInputPrecisions[0] != originInputPrecisions[i]) {
            isMixedPrecision = true;
            break;
        }
    }

    // Concat doesn't support different precision on inputs so fallback on FP32 in such case
    if (isMixedPrecision) {
        inputPrecision = ov::element::f32;
    }

    // Concat supports only equal precisions for inputs and output
    outputPrecision = inputPrecision;

    const auto& dstShape = getOutputShapeAtPort(0);
    std::vector<LayoutType> tdCreatorTypes = {LayoutType::ncsp, LayoutType::nspc};

    // check if blocked layouts are available the channels size should be evenly divided by the block size to avoid slow
    // oneDNN ref implementation and allow inPlace memory usage if possible
    if (dstShape.getRank() > channelAxis) {
        for (auto& item : {std::make_pair(8lu, LayoutType::nCsp8c), std::make_pair(16lu, LayoutType::nCsp16c)}) {
            const VectorDims& blkDims = dstShape.getDims();
            if (blkDims[channelAxis] == Shape::UNDEFINED_DIM || blkDims[channelAxis] % item.first != 0) {
                continue;
            }

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

    auto itrRange =
        BlockedDescCreator::makeFilteredRange(creatorsMap, static_cast<unsigned>(dstShape.getRank()), tdCreatorTypes);
    for (auto itr = itrRange.first; itr != itrRange.second; ++itr) {
        NodeConfig config;

        config.outConfs.resize(1);
        config.outConfs[0].inPlace(-1);
        config.outConfs[0].constant(false);
        config.outConfs[0].setMemDesc(itr->second->createSharedDesc(outputPrecision, dstShape));

        config.inConfs.resize(getParentEdges().size());

        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            config.inConfs[i].inPlace(-1);
            config.inConfs[i].constant(false);
            auto desc = itr->second->createSharedDesc(inputPrecision, getInputShapeAtPort(i));
            config.inConfs[i].setMemDesc(desc);
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
        if (itr->first != LayoutType::nspc) {
            pdIndexesToReuse.push_back(supportedPrimitiveDescriptors.size() - 1);
        } else if (canBeInPlace) {
            // canBeInPlace means all dims before axis are 1, so for nspc layout we only need check sp dimensions in
            // axis=1 cases here
            const auto& childDims = outputShapes[0].getDims();
            if (axis != 1 || std::all_of(childDims.crbegin(), childDims.crend() - 2, [](const Dim dim) {
                    return 1 == dim;
                })) {
                pdIndexesToReuse.push_back(supportedPrimitiveDescriptors.size() - 1);
            }
        }
    }

    // required to prevent incorrect memory sharing of a constant with other tensors on edges
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (getParentEdgeAt(i)->getParent()->isConstant()) {
            return;
        }
    }

    if (!canBeInPlace || std::any_of(inputShapes.begin(), inputShapes.end(), [](const Shape& shape) {
            return shape.hasZeroDims();
        })) {
        return;
    }

    // Optimized inplace case
    for (auto refPdIndex : pdIndexesToReuse) {
        auto config = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        ;
        for (auto& inConf : config.inConfs) {
            inConf.inPlace(0);
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }
}

void Concat::selectOptimalPrimitiveDescriptor() {
    std::vector<size_t> canSelectPrimitive;

    // The double connection marks that some tensor should
    // be replicated. Inplace approach is not applicable
    // for that case.
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        for (size_t j = i + 1; j < getParentEdges().size(); j++) {
            if (getParentEdgeAt(i) == getParentEdgeAt(j)) {
                canBeInPlace = false;
            }
        }
    }

    std::map<LayoutType, size_t> formatFrequency;
    std::vector<LayoutType> supportedLayouts = {LayoutType::ncsp,
                                                LayoutType::nspc,
                                                LayoutType::nCsp8c,
                                                LayoutType::nCsp16c};
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        auto parent_pdesc = parent->getSelectedPrimitiveDescriptor();
        if (parent_pdesc == nullptr) {
            continue;
        }

        const auto& parent_config = parent_pdesc->getConfig();
        int outputIndex = parentEdge->getInputNum();
        if (outputIndex < 0 || outputIndex >= static_cast<int>(parent_config.outConfs.size())) {
            THROW_CPU_NODE_ERR("Cannot find index of output node");
        }
        const auto& port_desc = parent_config.outConfs[outputIndex].getMemDesc();
        for (auto& item : supportedLayouts) {
            if (port_desc->hasLayoutType(item)) {
                formatFrequency[item] += 1;
            }
        }
    }
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto child = childEdge->getChild();
        const auto* prim_desc = child->getSelectedPrimitiveDescriptor();
        if (prim_desc == nullptr) {
            continue;
        }

        const auto& config = prim_desc->getConfig();
        int inputIndex = childEdge->getOutputNum();
        if (inputIndex < 0 || inputIndex >= static_cast<int>(config.inConfs.size())) {
            THROW_CPU_NODE_ERR("Cannot find index of output node");
        }
        const auto& port_desc = config.inConfs[inputIndex].getMemDesc();
        for (auto& item : supportedLayouts) {
            if (port_desc->hasLayoutType(item)) {
                formatFrequency[item] += 1;
            }
        }
    }

    size_t maxCount = 0;
    const auto& outDims = getOutputShapeAtPort(0).getDims();
    LayoutType convertTo = LayoutType::ncsp;
    for (auto& it : formatFrequency) {
        if (it.second > maxCount) {
            maxCount = it.second;
            convertTo = it.first;
        } else if (it.second == maxCount) {
            if ((context->isGraphQuantized() && it.first == LayoutType::nspc) || it.first == LayoutType::nCsp8c ||
                it.first == LayoutType::nCsp16c) {
                convertTo = it.first;
            }
        }
    }

    for (auto& item : {std::make_pair(8lu, LayoutType::nCsp8c), std::make_pair(16lu, LayoutType::nCsp16c)}) {
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
            if (IMPLICATION(supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown,
                            canBeInPlace)) {
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

bool Concat::needPrepareParams() const {
    if (canOptimizeNspc || isInPlace()) {
        return false;
    }
    return inputShapesModified();
}

void Concat::prepareParams() {
    if (canOptimizeNspc || isInPlace()) {
        return;
    }

    const auto& dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Destination memory is undefined.");
    }
    auto dstMemDesc = dstMemPtr->getDescWithType<BlockedMemoryDesc>();
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }

    const auto& outputStrides = dstMemDesc->getStrides();
    size_t curConcatOffset = 0;
    const size_t elemSize = DnnlExtensionUtils::sizeOfDataType(dstMemPtr->getDataType());
    const auto& src0BlkMemDesc = getSrcMemoryAtPort(0)->getDescPtr()->as<BlockedMemoryDesc>();
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

    canOptimize1DCase = false;
    if (outputShape.size() == 1 && outputStrides[0] == 1 && outputShape[0] <= 64 && elemSize == 4) {
        // output is small 1d vector (which is typical in shape inference subgraph),
        // in this case, inputs are also small 1d vector and single thread naive impl is faster
        canOptimize1DCase = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            const auto& srcMemPtr = getSrcMemoryAtPort(i);
            const auto srcMemDesc = srcMemPtr->getDescPtr()->as<BlockedMemoryDesc>();
            const auto& inputShape = srcMemDesc->getBlockDims();
            const auto& strides = srcMemDesc->getStrides();
            if (inputShape.size() != 1 || strides.size() != 1) {
                canOptimize1DCase = false;
                break;
            }
        }
        if (canOptimize1DCase) {
            return;
        }
    }

    std::vector<memory::desc> srcs_d;
    nelemTotal = 0;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        const auto& srcMemPtr = getSrcMemoryAtPort(i);
        if (!srcMemPtr || !srcMemPtr->isDefined()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_CPU_NODE_ERR("Source memory from ", parent->getName(), " is undefined.");
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
            nelemTotal += nelemToCopy[i];
        } else {
            if (srcMemPtr->getShape().hasZeroDims()) {
                continue;
            }
            auto desc = srcMemPtr->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();

            const auto& dims = srcMemPtr->getStaticDims();
            for (size_t j = 0; j < dims.size(); j++) {
                desc.get()->dims[j] = dims[j];
            }
            srcs_d.emplace_back(desc);
        }
    }

    if (!canExecRef) {
        auto desc = dstMemPtr->getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();

        const auto& dims = dstMemPtr->getStaticDims();
        for (size_t i = 0; i < dims.size(); i++) {
            desc.get()->dims[i] = dims[i];
            desc.get()->padded_dims[i] = dims[i];
        }

        auto primitive_desc = concat::primitive_desc(getEngine(), desc, static_cast<int>(axis), srcs_d);
        prim = concat(primitive_desc);
#ifdef CPU_DEBUG_CAPS
        if (prim) {
            auto pd = prim.get_primitive_desc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
    }
}

size_t Concat::inverseOrder(const VectorDims& order, size_t axis) {
    for (size_t i = 0; i < order.size(); i++) {
        if (axis == order[i]) {
            return i;
        }
    }
    return -1;
}

void Concat::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }

    if (!isInPlace()) {
        Node::initOptimalPrimitiveDescriptor();
        auto config = selected_pd->getConfig();
        if (!isConfigDefined(config)) {
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                // Concat doesn't support different precision on inputs
                config.inConfs[i].setMemDesc(
                    getConsistentInputDesc(config, i)->getMemDesc()->cloneWithNewPrecision(inputPrecision));
            }

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].setMemDesc(
                    getConsistentOutputDesc(config, i)->getMemDesc()->cloneWithNewPrecision(outputPrecision));
            }

            initDescriptor(config);
        }
    }

    // block layout may have axis greater than rank, disable ref_concat
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
    canOptimizeNspc =
        axis == channelAxis &&
        getSelectedPrimitiveDescriptor()->getConfig().outConfs.front().getMemDesc()->hasLayoutType(LayoutType::nspc);
}

void Concat::execute(const dnnl::stream& strm) {
    if (isInPlace()) {
        return;
    }

    if (canOptimize1DCase) {
        exec1DCase();
        return;
    }

    if (canOptimizeNspc) {
        execNspcSpecCase();
        return;
    }

    if (canExecRef) {
        execRef();
    } else {
        const auto& dst_memory = getChildEdgeAt(0)->getMemory();
        const size_t num_src = getParentEdges().size();
        std::unordered_map<int, memory> mem_ags{{DNNL_ARG_DST, dst_memory.getPrimitive()}};
        size_t nonZeroInShapes = 0;
        for (size_t i = 0; i < num_src; i++) {
            const auto& srcMem = getParentEdgeAt(i)->getMemory();
            if (srcMem.getShape().hasZeroDims()) {
                continue;
            }
            mem_ags[DNNL_ARG_MULTIPLE_SRC + nonZeroInShapes] = srcMem.getPrimitive();
            nonZeroInShapes++;
        }
        prim.execute(strm, mem_ags);
    }
}

ov::element::Type Concat::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void Concat::exec1DCase() {
    DEBUG_LOG(getName(), " exec1DCase");
    auto* dst = getDstDataAtPortAs<uint32_t>(0);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        const auto& srcMemPtr = getSrcMemoryAtPort(i);
        const auto& srcShape = srcMemPtr->getStaticDims();
        const auto* src = srcMemPtr->getDataAs<const uint32_t>();
        for (size_t i = 0; i < srcShape[0]; i++) {
            *dst++ = src[i];
        }
    }
}

void Concat::execNspcSpecCase() {
    const auto& dst_memory = getChildEdgeAt(0)->getMemory();
    const size_t num_src = getParentEdges().size();
    auto* dst_ptr = dst_memory.getDataAs<uint8_t>();
    const size_t dataSize = DnnlExtensionUtils::sizeOfDataType(dst_memory.getDataType());

    std::vector<size_t> channelsDataSize;
    size_t channels_size = 0;
    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;

    size_t nonZeroInShapes = 0;
    int firstNonZeroEdge = -1;
    for (size_t i = 0; i < num_src; i++) {
        const auto& src_mem = getParentEdgeAt(i)->getMemory();
        if (src_mem.getShape().hasZeroDims()) {
            continue;
        }
        const size_t num_channels = src_mem.getStaticDims()[channelAxis];

        channelsDataSize.push_back(num_channels * dataSize);
        src_ptrs.push_back(src_mem.getDataAs<const uint8_t>());
        dst_ptrs.push_back(dst_ptr + channels_size);
        channels_size += num_channels * dataSize;

        if (firstNonZeroEdge == -1) {
            firstNonZeroEdge = i;
        }

        nonZeroInShapes++;
    }
    const Shape& shape = getSrcMemoryAtPort(firstNonZeroEdge)->getShape();
    const size_t iter_count = shape.getElementsCount() / shape.getStaticDims()[channelAxis];

    parallel_for(iter_count, [&](int i) {
        const size_t dst_off = i * channels_size;
        for (size_t j = 0; j < nonZeroInShapes; j++) {
            cpu_memcpy(dst_ptrs[j] + dst_off, src_ptrs[j] + i * channelsDataSize[j], channelsDataSize[j]);
        }
    });
}

void Concat::execRef() {
    const size_t numSrc = getParentEdges().size();
    const auto& dstMemory = getChildEdgeAt(0)->getMemory();
    auto* dstPtr = dstMemory.getDataAs<uint8_t>();
    for (size_t i = 0; i < numSrc; i++) {
        const auto& srcMem = getParentEdgeAt(i)->getMemory();
        srcPtrs[i] = srcMem.getDataAs<const uint8_t>();
    }

    if (!hasOuterLoop) {
        if (nelemTotal < 64 * 1024 || parallel_get_max_threads() == 1) {
            for (size_t a = 0; a < srcPtrs.size(); ++a) {
                const auto inData = srcPtrs[a];
                auto outputData = &dstPtr[dstOffset[a]];
                std::memcpy(outputData, inData, nelemToCopy[a]);
            }
        } else {
            int nthr = parallel_get_max_threads();
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
        const size_t elemSize = DnnlExtensionUtils::sizeOfDataType(dstMemory.getDataType());
        const auto dstMemBlkDesc = dstMemory.getDescPtr()->as<BlockedMemoryDesc>();
        const auto& outputShape = dstMemBlkDesc->getBlockDims();
        size_t outputStrides[MAX_RANK_REF] = {0};
        const auto strides = dstMemBlkDesc->getStrides();
        std::transform(strides.begin(), strides.end(), outputStrides, [&elemSize](const Dim& i) {
            return i * elemSize;
        });
        size_t physDims[5] = {1, 1, 1, 1, 1};
        for (size_t i = 0; i < reorderedAxis; i++) {
            physDims[i] = outputShape[i];
        }
        const auto L1Size = dnnl::utils::get_cache_size(1, true);
        UNUSED(L1Size);  // for Windows
        parallel_for6d(physDims[0],
                       physDims[1],
                       physDims[2],
                       physDims[3],
                       physDims[4],
                       numSrc,
                       [&](size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t a) {
                           // check if zero memory
                           if (srcPtrs[a] == nullptr) {
                               return;
                           }

                           size_t inOff = inputStrides[a][0] * n0 + inputStrides[a][1] * n1 + inputStrides[a][2] * n2 +
                                          inputStrides[a][3] * n3 + inputStrides[a][4] * n4;
                           size_t outOff = outputStrides[0] * n0 + outputStrides[1] * n1 + outputStrides[2] * n2 +
                                           outputStrides[3] * n3 + outputStrides[4] * n4;
                           const uint8_t* i = &srcPtrs[a][inOff];
                           uint8_t* o = &dstPtr[dstOffset[a] + outOff];

#if defined(__GNUC__)
                           // Heuristic:
                           // memcpy works generally faster for data sizes not
                           // exceeding L1 cache.
                           if (nelemToCopy[a] > L1Size) {
                               // The code below performs data copying: o[e] = i[e]
                               // and uses a workaround to make GNU compilers optimize it
                               uint8_t* ptro = o;
                               const uint8_t* ptri = i;
                               // head part: bytes before 4 byte-align's address
                               const size_t headPart =
                                   sizeof(uint32_t) - reinterpret_cast<uint64_t>(ptro) % sizeof(uint32_t);

                               // main part: bytes in 4 byte-align
                               const size_t mainPart = (nelemToCopy[a] - headPart) / sizeof(uint32_t);
                               // tail part: bytes after 4 byte-align
                               const size_t tailPart = (nelemToCopy[a]) - headPart - (mainPart * sizeof(uint32_t));
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

void Concat::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_DOWN) || !isInPlace()) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }
    auto& config = selected_pd->getConfig();
    size_t numberOfInputs = config.inConfs.size();
    size_t inplaceOutIndx = selected_pd->getConfig().inConfs[0].inPlace();
    auto baseDim = outputShapes.front().getDims()[axis];
    CPU_NODE_ASSERT(baseDim != Shape::UNDEFINED_DIM,
                    "can't use inPlace memory with concatenation on dynamic dimension");

    auto edges = getChildEdgesAtPort(inplaceOutIndx);
    auto itr = std::find_if(edges.begin(), edges.end(), [](const EdgePtr& edge) {
        return edge->getStatus() == Edge::Status::Allocated;
    });
    CPU_NODE_ASSERT(itr != edges.end(), "Could not find allocated child edge");

    auto baseMemBlock = (*itr)->getMemory().getMemoryBlock();
    CPU_NODE_ASSERT(baseMemBlock != nullptr, "NULL base memory block");

    ptrdiff_t offset = 0;
    for (size_t i = 0; i < numberOfInputs; ++i) {
        auto partDim = inputShapes[i].getDims()[axis];
        CPU_NODE_ASSERT(partDim != Shape::UNDEFINED_DIM,
                        "can't use inPlace memory with concatenation on dynamic dimension");

        auto parentEdge = getParentEdgeAt(i);

        CPU_NODE_ASSERT(parentEdge->getStatus() == Edge::Status::NotAllocated,
                        "Unexpected inplace resolve call to an allocated edge: ",
                        *parentEdge);

        auto memDesc = selected_pd->getConfig().inConfs[i].getMemDesc();
        MemoryPtr newMem;
        if (partDim != 0) {
            auto memBlock = std::make_shared<PartitionedMemoryBlock>(baseMemBlock, baseDim, offset, partDim);
            newMem = std::make_shared<Memory>(getEngine(), memDesc, memBlock);
        } else {
            // empty tensor, no need to reference a part, default memory is enough
            newMem = std::make_shared<Memory>(getEngine(), memDesc);
        }

        parentEdge->reuse(newMem);
        offset += partDim;
    }
}

}  // namespace ov::intel_cpu::node
