// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split.h"
#include "common/cpu_memcpy.h"
#include "common/blocked_desc_creator.h"
#include <vector>
#include "dnnl_types.h"
#include "dnnl_extension_utils.h"
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "utils/ngraph_utils.hpp"
#include <partitioned_mem_blk.h>
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

#define THROW_ERROR(...) OPENVINO_THROW("Split layer with name '", getName(), "' ", __VA_ARGS__)

using namespace dnnl;

namespace ov {
namespace intel_cpu {
namespace node {

bool Split::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ov::op::v1::Split::get_type_info_static(), ov::op::v1::VariadicSplit::get_type_info_static())) {
            errorMessage = "Only opset1 Split and VariadicSplit operations are supported";
            return false;
        }
        auto axisOp = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        if (!axisOp) {
            errorMessage = "Constant expected as the axis input.";
            return false;
        }
        if (op->get_input_size() > 2 && op->get_input_partial_shape(2).is_dynamic()) {
            errorMessage = "Expected static 'split_lengths' shape because dynamic number of outputs is not supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Split::Split(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, NgraphShapeInferFactory(op, PortMask(1, 2))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (ov::as_type_ptr<const ov::op::v1::Split>(op)) {
        INPUTS_NUM = 2;
    } else if (ov::as_type_ptr<const ov::op::v1::VariadicSplit>(op)) {
        INPUTS_NUM = 3;
        if (!ov::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2))) {
            this->splitLengths.resize(op->get_input_shape(2)[0]);
            this->constSplitLengths = false;
        }
    }

    const auto inRank = getInputShapeAtPort(0).getRank();
    auto axisOp = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto axis = axisOp->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += inRank;
    }
    if (axis >= static_cast<int64_t>(inRank)) {
        THROW_ERROR("Split node with name '", op->get_friendly_name(), "' has invalid value of axis parameter: ", axis);
    }
    this->axis = axis;
}

void Split::getSupportedDescriptors() {
}

void Split::initSupportedPrimitiveDescriptors() {
    constexpr size_t channelsPos = 1lu;

    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto &srcShape = getInputShapeAtPort(0);
    const auto &dstFirstDims = getOutputShapeAtPort(0).getDims();
    for (size_t i = 0; i < outputShapes.size(); i++) {
        const auto &o_Dims = outputShapes[i].getDims();
        if (dstFirstDims.size() != o_Dims.size()) {
            THROW_ERROR("only supports output blobs with equal number of dimensions");
        }

        for (size_t j = 0; j < dstFirstDims.size(); j++) {
            if (j == axis)
                continue;
            if (!dimsEqualWeak(o_Dims[j], dstFirstDims[j]))
                THROW_ERROR("has incorrect output dimensions");
        }
    }

    ov::element::Type inpPrecision = getOriginalInputPrecisionAtPort(0);
    const auto axisPrecision = ov::element::i32;

    // Set plain and tailC formats
    std::vector<LayoutType> tdCreatorTypes{ LayoutType::ncsp, LayoutType::nspc };

    // Support channel blocked format only if we manipulate complete blocks
    if (srcShape.getRank() > 2) {
        for (auto item : { std::make_pair(8lu, LayoutType::nCsp8c), std::make_pair(16lu, LayoutType::nCsp16c) }) {
            const auto &blkDims = srcShape.getDims();
            if (blkDims[channelsPos] == Shape::UNDEFINED_DIM || blkDims[channelsPos] % item.first != 0)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < outputShapes.size(); i++) {
                const auto &outBlkDims = getOutputShapeAtPort(i).getDims();
                if (outBlkDims[channelsPos] == Shape::UNDEFINED_DIM || outBlkDims[channelsPos] % item.first != 0) {
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
    auto itrRange = BlockedDescCreator::makeFilteredRange(creatorsMap, static_cast<unsigned>(srcShape.getRank()), tdCreatorTypes);
    for (auto itr = itrRange.first; itr != itrRange.second; ++itr) {
        NodeConfig config;

        config.inConfs.resize(INPUTS_NUM);
        config.inConfs[0].inPlace(-1);
        config.inConfs[0].constant(false);
        config.inConfs[0].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(inpPrecision, srcShape)));
        config.inConfs[1].inPlace(-1);
        config.inConfs[1].constant(true);
        config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(axisPrecision, Shape(VectorDims{1})));
        if (INPUTS_NUM == 3) {
            config.inConfs[2].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(axisPrecision, Shape(VectorDims{outputShapes.size()})));
            config.inConfs[2].constant(constSplitLengths);
        }

        config.outConfs.resize(outputShapes.size());

        for (size_t i = 0; i < outputShapes.size(); i++) {
            config.outConfs[i].inPlace(-1);
            config.outConfs[i].constant(false);
            config.outConfs[i].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(inpPrecision, outputShapes[i])));
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);

        if (itr->first == LayoutType::ncsp) {
            // at least the plain layout can be optimized inplace.
            pdIndexesToReuse.emplace_back(supportedPrimitiveDescriptors.size() - 1);
        } else if (itr->first == LayoutType::nCsp8c || itr->first == LayoutType::nCsp16c) {
            if (axis < 2) {
                pdIndexesToReuse.emplace_back(supportedPrimitiveDescriptors.size() - 1);
            }
        }
    }

    // in place only makes sense when we split by dense blocks since strided tensors are not supported by most nodes.
    const auto& parentdDims = inputShapes[0].getDims();
    if (parentdDims[axis] != Shape::UNDEFINED_DIM &&
        std::all_of(parentdDims.begin(), parentdDims.begin() + axis, [](size_t dim) { return  dim == 1; }) &&
        std::all_of(outputShapes.begin(), outputShapes.end(),
            [OV_CAPTURE_CPY_AND_THIS](const Shape& shape){ return shape.getDims()[axis] != Shape::UNDEFINED_DIM; })) {
        for (auto refPdIndex : pdIndexesToReuse) {
            auto config = supportedPrimitiveDescriptors[refPdIndex].getConfig();

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].inPlace(0);
            }
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    }

    // Special nspc -> ncsp case when splitting channels
    if (axis == 1 && (dstFirstDims.size() == 4 || dstFirstDims.size() == 5)) {
        NodeConfig config;

        config.inConfs.resize(INPUTS_NUM);
        config.inConfs[0].inPlace(-1);
        config.inConfs[0].constant(false);
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(inpPrecision, srcShape));
        config.inConfs[1].inPlace(-1);
        config.inConfs[1].constant(true);
        config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(axisPrecision, Shape(VectorDims{1})));
        if (INPUTS_NUM == 3) {
            config.inConfs[2].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(axisPrecision, Shape(VectorDims{outputShapes.size()})));
            config.inConfs[2].constant(constSplitLengths);
        }
        config.outConfs.resize(outputShapes.size());

        for (size_t i = 0; i < outputShapes.size(); i++) {
            config.outConfs[i].inPlace(-1);
            config.outConfs[i].constant(false);
            config.outConfs[i].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inpPrecision, outputShapes[i]));
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
    }
}

bool Split::needShapeInfer() const {
    if (Node::needShapeInfer()) {
        return true;
    } else if (!constSplitLengths) {
        const auto& lengthsMemPtr = getSrcMemoryAtPort(2);
        const auto curLengthsSize = lengthsMemPtr->getStaticDims()[0];
        if (curLengthsSize != splitLengths.size()) {
            return true;
        }
        const int* curLengthsValues = lengthsMemPtr->getDataAs<int>();
        for (size_t i = 0; i < curLengthsSize; ++i) {
            if (curLengthsValues[i] != splitLengths[i]) {
                return true;
            }
        }
    }
    return false;
}

bool Split::needPrepareParams() const {
    if (isInPlace()) {
        return false;
    }
    return needShapeInfer();
}

void Split::createPrimitive() {
    if (outputShapesDefined()) {
        Node::createPrimitive();
    }
}

void Split::prepareParams() {
    const auto &srcMemPtr = getSrcMemoryAtPort(0);
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_ERROR("has undefined input memory");
    }

    if (!constSplitLengths) {
        const auto& splitLengthsPtr = getSrcMemoryAtPort(2);
        const int* curSplitLengths = splitLengthsPtr->getDataAs<int>();
        const auto curLengthsSize = splitLengthsPtr->getStaticDims()[0];
        splitLengths.assign(curSplitLengths, curSplitLengths + curLengthsSize);
    }

    dstMemPtrs.clear();
    std::vector<BlockedMemoryDescCPtr> outDescs;
    for (size_t port = 0; port < outputShapes.size(); ++port) {
        const auto &outMemPtr = this->getDstMemoryAtPort(port);
        if (!outMemPtr || !outMemPtr->isDefined()) {
            THROW_ERROR("has undefined destination memory");
        }

        if (outMemPtr->getShape().hasZeroDims()) {
            continue;
        }

        dstMemPtrs.emplace_back(port, outMemPtr);

        if (!canUseOptimizedNspc2Ncsp) {
            outDescs.push_back(outMemPtr->getDescWithType<BlockedMemoryDesc>());
        }
    }

    if (!canUseOptimizedNspc2Ncsp) {
        const auto inDesc = srcMemPtr->getDescWithType<BlockedMemoryDesc>();
        execPtr = std::make_shared<SplitOptimizedExecutor>(inDesc, outDescs, axis);
    }
}

bool Split::isExecutable() const {
    return !isInPlace() && !isInputTensorAtPortEmpty(0);
}

void Split::execute(dnnl::stream strm) {
    if (isInPlace()) {
        return;
    }

    if (dstMemPtrs.empty())
        THROW_ERROR("Output data pointers have not been initialized.");

    const auto &srcMem = getParentEdgeAt(0)->getMemory();

    if (canUseOptimizedNspc2Ncsp) {
        optimizedNspc2Ncsp(srcMem.getStaticDims()[0]);
        return;
    }

    uint8_t* srcData = srcMem.getDataAs<uint8_t>();
    OPENVINO_ASSERT(execPtr != nullptr);
    execPtr->exec(srcData, getRawDstMemPtrs());
}

bool Split::created() const {
    return getType() == Type::Split;
}

void Split::initOptimalPrimitiveDescriptor() {
    Node::initOptimalPrimitiveDescriptor();
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_ERROR("Preferable primitive descriptor is not set.");

    auto config = selected_pd->getConfig();
    canUseOptimizedNspc2Ncsp = false;
    OPENVINO_ASSERT(config.inConfs.size() > 0);
    const auto inConfDesc = config.inConfs[0].getMemDesc();
    if (axis == 1 && one_of(inConfDesc->getShape().getRank(), 4u, 5u) && inConfDesc->hasLayoutType(LayoutType::nspc)) {
        canUseOptimizedNspc2Ncsp = true;
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            if (!config.outConfs[i].getMemDesc()->hasLayoutType(LayoutType::ncsp))
                canUseOptimizedNspc2Ncsp = false;
        }
    }
}

void Split::selectOptimalPrimitiveDescriptor() {
    // Enforce the reference implementation for the planar layout if the implementation is in the impl priorities list.
    // This is needed mostly for the testing purposes, since for the planar layout Split works always in place, we need to enforce
    // the reference implementation when it is selected in a test to test that piece of code.
    if (!customImplPriorities.empty() && customImplPriorities[0] == impl_desc_type::ref) {
        for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); ++i) {
            auto& pd = supportedPrimitiveDescriptors[i];
            if (pd.getConfig().inConfs[0].getMemDesc()->hasLayoutType(LayoutType::ncsp) &&
                impl_desc_type::ref == pd.getImplementationType()) {
                    selectPrimitiveDescriptorByIndex(static_cast<int>(i));
                return;
            }
        }
    }

    // check the descriptors and select the ones that have the same data format as the input
    std::vector<size_t> canSelectPrimitive;
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        auto parentEdge = getParentEdgeAt(0);
        auto parentPtr = parentEdge->getParent();
        auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

        if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
            int inNum = parentEdge->getInputNum();
            if (inNum < 0 || static_cast<size_t>(inNum) >= parent_spd->getConfig().outConfs.size()) {
                inNum = 0;
            }
            if (supportedPrimitiveDescriptors[i].getConfig().inConfs[0].getMemDesc()->isCompatible(*parent_spd->getConfig().outConfs[inNum].getMemDesc())) {
                canSelectPrimitive.push_back(i);
            }
        }
    }
    if (canSelectPrimitive.size() == 1) {
        selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
        return;
    }

    // if there are more then one PD with similar data layouts - select the optimized one
    for (auto indx : canSelectPrimitive) {
        if (supportedPrimitiveDescriptors[indx].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(indx));
            return;
        }
    }

    // if there are no inPlace, but more than one suitable configurations, select the one that matches the output layout
    for (auto indx : canSelectPrimitive) {
        bool outputDescFullMatch = true;
        for (size_t i = 0; i < getChildEdges().size(); ++i) {
            auto childEdge = getChildEdgeAt(i);
            auto childPtr = childEdge->getChild();
            auto& vecChildSpd = childPtr->getSupportedPrimitiveDescriptors();
            const auto& outputDesc = supportedPrimitiveDescriptors[indx].getConfig().outConfs[childEdge->getInputNum()].getMemDesc();

            if (!vecChildSpd.empty()) {
                int inNum = childEdge->getOutputNum();
                if (inNum < 0) {
                    inNum = 0;
                }
                bool hasMatchDesc = false;
                for (auto& childSpd : vecChildSpd) {
                    if (static_cast<size_t>(inNum) >= childSpd.getConfig().inConfs.size()) {
                        inNum = 0;
                    }
                    if (outputDesc->isCompatible(*childSpd.getConfig().inConfs[inNum].getMemDesc())) {
                        hasMatchDesc = true;
                        break;
                    }
                }
                if (!hasMatchDesc) {
                    outputDescFullMatch = false;
                    break;
                }
            }
        }
        if (outputDescFullMatch) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(indx));
            return;
        }
    }
    if (!canSelectPrimitive.empty()) {
        selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive.front()));
        return;
    }

    // if there are no matching data layouts, select first optimized implementation
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        if (supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(i));
            return;
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

void Split::optimizedNspc2Ncsp(size_t MB) {
    auto parentEdge = getParentEdgeAt(0);
    const int rank = parentEdge->getMemory().getShape().getRank();
    const auto parentDims = parentEdge->getMemory().getStaticDims();
    const size_t IC = parentDims[1];
    const size_t D = rank == 5 ? parentDims[rank - 3] : 1;
    const size_t H = parentDims[rank - 2];
    const size_t W = parentDims[rank - 1];

    auto& srcMem = parentEdge->getMemory();
    auto srcData = srcMem.getDataAs<const uint8_t>();
    const auto dataSize = srcMem.getDesc().getPrecision().size();

    const size_t DHW = D*H*W;
    const size_t strideIB = DHW * IC * dataSize;
    const size_t strideIW = IC*dataSize;
    const size_t strideOC = DHW * dataSize;

    for (size_t i = 0, sIdx = 0; i < dstMemPtrs.size(); i++) {
        auto dstData = dstMemPtrs[i].second->getDataAs<uint8_t>();

        size_t innerSize = 1;
        auto dims = getChildEdgeAt(dstMemPtrs[i].first)->getMemory().getStaticDims();

        for (size_t j = axis; j < dims.size(); j++) {
            innerSize *= dims[j];
        }
        auto srcPtr = srcData + srcMem.getDesc().getElementOffset(sIdx) * dataSize;

        const size_t OC = dims[1];
        const size_t strideOB = OC * strideOC;

        parallel_for2d(MB, DHW, [&](size_t b, size_t j) {
            auto localSrcPtr = srcPtr + b*strideIB + j*strideIW;
            auto localDstPtr = dstData + b*strideOB + j*dataSize;
            for (size_t c = 0; c < OC; c++) {
                cpu_memcpy(localDstPtr, localSrcPtr, dataSize);
                localSrcPtr += dataSize;
                localDstPtr += strideOC;
            }
        });

        sIdx += innerSize;
    }
}

std::vector<uint8_t*> Split::getRawDstMemPtrs() const {
    std::vector<uint8_t*> result(dstMemPtrs.size());
    for (size_t i = 0; i < dstMemPtrs.size(); ++i) {
        result[i] = dstMemPtrs[i].second->getDataAs<uint8_t>();
        if (!result[i]) {
            THROW_ERROR("can't get child edge indx ", dstMemPtrs[i].first, " data.");
        }
    }
    return result;
}

Split::SplitOptimizedExecutor::SplitOptimizedExecutor(BlockedMemoryDescCPtr inDesc, const std::vector<BlockedMemoryDescCPtr> &outDescs,
                                                                const size_t axis) {
    // find axis order position
    const auto& order = inDesc->getOrder();
    unsigned axisOrderPos = std::numeric_limits<unsigned>::max();
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] == axis) {
            axisOrderPos = i;
            break;
        }
    }
    if (std::numeric_limits<unsigned>::max() == axisOrderPos) {
        OPENVINO_THROW("Can't create split executor, because can't find the axis in the input tensor order list");
    }

    const auto outputPortsCount = outDescs.size();

    uint8_t srcDataSize = inDesc->getPrecision().size();
    const auto& srcDims = inDesc->getBlockDims();
    const auto getRank = srcDims.size();

    countStrides = 1;
    for (unsigned int i = 0; i < axisOrderPos; i++)
        countStrides *= srcDims[i];

    srcDataStride = 0;
    dataSize.resize(outputPortsCount);

    for (size_t i = 0; i < outputPortsCount; i++) {
        dataSize[i] = srcDataSize;
        for (size_t j = axisOrderPos; j < getRank; j++)
            dataSize[i] *= outDescs[i]->getBlockDims()[j];

        srcDataStride += dataSize[i];
    }

    srcDataOffsets.resize(outputPortsCount);
    srcDataOffsets[0] = 0;
    for (size_t i = 1; i < outputPortsCount; i++) {
        srcDataOffsets[i] = srcDataOffsets[i - 1] + dataSize[i - 1];
    }
}

void Split::SplitOptimizedExecutor::exec(const uint8_t* srcData, const std::vector<uint8_t*>& dstRawMemPtrs) {
    size_t execCountStrides = countStrides;

    parallel_for2d(dstRawMemPtrs.size(), execCountStrides, [&](size_t i, size_t j) {
        uint8_t* dstData = dstRawMemPtrs[i];

        cpu_memcpy(&dstData[j * dataSize[i]],
                   &srcData[srcDataOffsets[i] + j * srcDataStride],
                   dataSize[i]);
    });
}

void Split::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_UP) || !isInPlace()) {
        Node::resolveInPlaceEdges(look);
        return;
    }
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        OPENVINO_THROW("Preferable primitive descriptor is not set.");
    auto& config = selected_pd->getConfig();
    size_t numberOfOutputs = config.outConfs.size();
    size_t inplaceInpIndx = selected_pd->getConfig().outConfs[0].inPlace();
    auto baseDim = inputShapes.front().getDims()[axis];
    OPENVINO_ASSERT(baseDim != Shape::UNDEFINED_DIM,
                    " Split node: ",
                    getName(),
                    " can not use inPlace memory with splitting on dynamic dimension");
    auto baseMemBlock = getParentEdgeAt(inplaceInpIndx)->getMemory().getMemoryBlock();
    ptrdiff_t offset = 0;
    for (size_t i = 0; i < numberOfOutputs; ++i) {
        auto partDim = outputShapes[i].getDims()[axis];
        OPENVINO_ASSERT(partDim != Shape::UNDEFINED_DIM,
                        " Split node: ",
                        getName(),
                        " can not use inPlace memory with splitting on dynamic dimension");
        const auto& childEdges = getChildEdgesAtPort(i);
        for (auto& childEdge : childEdges) {
            OPENVINO_ASSERT(childEdge->getStatus() == Edge::Status::NotAllocated,
                            " Unexpected edge status in node: ",
                            getName(),
                            " with type ",
                            getTypeStr());

            auto memDesc = selected_pd->getConfig().outConfs[i].getMemDesc();
            MemoryPtr newMem;
            if (partDim != 0) {
                auto memBlock = std::make_shared<PartitionedMemoryBlock>(baseMemBlock, baseDim, offset, partDim);
                newMem = std::make_shared<Memory>(getEngine(), memDesc, memBlock);
            } else {
                // empty tensor, no need to reference a part, default memory is enough
                newMem = std::make_shared<Memory>(getEngine(), memDesc);
            }

            childEdge->reuse(newMem);
        }
        offset += partDim;
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
