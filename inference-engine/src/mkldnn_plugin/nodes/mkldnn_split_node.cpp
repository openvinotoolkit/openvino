// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include "common/cpu_memcpy.h"
#include "common/blocked_desc_creator.h"
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_parallel.hpp>
#include "utils/general_utils.h"
#include <cpu_memory_desc_utils.h>

#define THROW_ERROR IE_THROW() << "Split layer with name '" << getName() <<"' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNSplitNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of(op->get_type_info(), ngraph::op::v1::Split::type_info, ngraph::op::v1::VariadicSplit::type_info)) {
            errorMessage = "Only opset1 Split and VariadicSplit operations are supported";
            return false;
        }
        auto axisOp = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        if (!axisOp) {
            errorMessage = "Constant expected as the axis input.";
            return false;
        }
        if (op->get_input_size() > 2) {
            auto splitLengthsOp = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (!splitLengthsOp) {
                errorMessage = "Constant expected as the split_lengths input.";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNSplitNode::MKLDNNSplitNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (ngraph::as_type_ptr<const ngraph::op::v1::Split>(op)) {
        INPUTS_NUM = 2;
    } else if (ngraph::as_type_ptr<const ngraph::op::v1::VariadicSplit>(op)) {
        INPUTS_NUM = 3;
    }

    auto axisOp = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto axis = axisOp->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += op->get_input_shape(0).size();
    }
    if (axis >= op->get_input_shape(0).size()) {
        THROW_ERROR << "Split node with name '" << op->get_friendly_name() << "' has invalid value of axis parameter: " << axis;
    }
    this->axis = axis;
}

void MKLDNNSplitNode::getSupportedDescriptors() {
}

void MKLDNNSplitNode::initSupportedPrimitiveDescriptors() {
    constexpr size_t channelsPos = 1lu;

    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto srcShape = getParentEdgeAt(0)->getShape();
    auto axis_size = 0;
    auto dstFirstDims = getChildEdgeAt(0)->getShape().getStaticDims();
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto o_Dims = outputShapes[i].getStaticDims();
        if (dstFirstDims.size() != o_Dims.size()) {
            THROW_ERROR << "only supports output blobs with equal number of dimensions";
        }

        axis_size += o_Dims[axis];
        for (size_t j = 0; j < dstFirstDims.size(); j++) {
            if (j == axis)
                continue;
            if (o_Dims[j] != dstFirstDims[j])
                THROW_ERROR << "has incorrect output dimensions";
        }
    }
    dstFirstDims[axis] = axis_size;
    if (std::accumulate(dstFirstDims.begin(), dstFirstDims.end(), 1, std::multiplies<size_t>()) != srcShape.getElementsCount())
        THROW_ERROR << "sizes of input blob and sum of output blobs are not equal.";

    InferenceEngine::Precision inpPrecision = getOriginalInputPrecisionAtPort(0);
    const auto axisPrecision = getOriginalInputPrecisionAtPort(1);
    auto outPrecision = inpPrecision; // the split layer doesn't convert precisions

    bool dynBatchSupport = true;
    if (axis < 1) {
        dynBatchSupport = false;
    }

    //Set plain and tailC formats
    std::vector<GeneralLayout> tdCreatorTypes{ GeneralLayout::ncsp, GeneralLayout::nspc };

    //Support channel blocked format
    if (srcShape.getRank() > 2) {
        for (auto item : { std::make_pair(8lu, GeneralLayout::nCsp8c), std::make_pair(16lu, GeneralLayout::nCsp16c) }) {
            SizeVector blkDims = srcShape.getStaticDims();
            if (blkDims[channelsPos] % item.first)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < outputShapes.size(); i++) {
                if (outputShapes[i].getStaticDims()[channelsPos] % item.first) {
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

        config.dynBatchSupport = dynBatchSupport;
        config.inConfs.resize(INPUTS_NUM);
        config.inConfs[0].inPlace = -1;
        config.inConfs[0].constant = false;
        config.inConfs[0].desc = make_unique<BlockedMemoryDesc>(itr->second->createDesc(inpPrecision, srcShape.getStaticDims()));
        config.inConfs[1].inPlace = -1;
        config.inConfs[1].constant = true;
        config.inConfs[1].desc = make_unique<BlockedMemoryDesc>(axisPrecision, SizeVector{1});
        if (INPUTS_NUM == 3) {
            config.inConfs[2].desc = make_unique<BlockedMemoryDesc>(axisPrecision, SizeVector{outputShapes.size()});
            config.inConfs[2].constant = true;
        }

        config.outConfs.resize(outputShapes.size());

        for (size_t i = 0; i < outputShapes.size(); i++) {
            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = make_unique<BlockedMemoryDesc>(itr->second->createDesc(inpPrecision, outputShapes[i].getStaticDims()));
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);

        if (itr->first == GeneralLayout::ncsp) {
            // at least the plain layout can be optimized inplace.
            pdIndexesToReuse.emplace_back(supportedPrimitiveDescriptors.size() - 1);
        } else if (itr->first == GeneralLayout::nCsp8c || itr->first == GeneralLayout::nCsp16c) {
            if (axis < 2) {
                pdIndexesToReuse.emplace_back(supportedPrimitiveDescriptors.size() - 1);
            }
        }
    }

    // Optimized inplace case
    for (auto refPdIndex : pdIndexesToReuse) {
        const auto& refConfig = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        auto config = refConfig;
        const auto inBlockingDesc = refConfig.inConfs[0].desc->as<BlockedMemoryDesc>();
        const auto& order = inBlockingDesc->getOrder();
        const auto& blkDims = inBlockingDesc->getBlockDims();
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

        config.inConfs[0].desc = make_unique<BlockedMemoryDesc>(inpPrecision, srcShape.getStaticDims(), blkDims, order, offset, offsets, strides);

        for (size_t i = 0; i < outputShapes.size(); i++) {
            auto outBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*refConfig.outConfs[i].desc);
            const auto& outBlkDims = outBlockingDesc.getBlockDims();
            const auto& dims = outBlockingDesc.getShape().getStaticDims();

            config.outConfs[i].inPlace = 0;
            config.outConfs[i].desc = make_unique<BlockedMemoryDesc>(outPrecision, dims, outBlkDims, order, offset, offsets, strides);
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }

    // Special nspc -> ncsp case when splitting channels
    if (axis == 1 && (dstFirstDims.size() == 4 || dstFirstDims.size() == 5)) {
        NodeConfig config;

        config.dynBatchSupport = dynBatchSupport;
        config.inConfs.resize(INPUTS_NUM);
        config.inConfs[0].inPlace = -1;
        config.inConfs[0].constant = false;
        config.inConfs[0].desc = make_unique<BlockedMemoryDesc>(
                creatorsMap.at(GeneralLayout::nspc)->createDesc(inpPrecision, srcShape.getStaticDims()));
        config.inConfs[1].inPlace = -1;
        config.inConfs[1].constant = true;
        config.inConfs[1].desc = make_unique<BlockedMemoryDesc>(axisPrecision, SizeVector{1});
        if (INPUTS_NUM == 3) {
            config.inConfs[2].desc = make_unique<BlockedMemoryDesc>(axisPrecision, SizeVector{outputShapes.size()});
            config.inConfs[2].constant = true;
        }
        config.outConfs.resize(outputShapes.size());

        for (size_t i = 0; i < outputShapes.size(); i++) {
            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = make_unique<BlockedMemoryDesc>(creatorsMap.at(GeneralLayout::ncsp)->createDesc(inpPrecision,
                                                                                               outputShapes[i].getStaticDims()));
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
    }
}

void MKLDNNSplitNode::createPrimitive() {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "Input memory has not been allocated.";
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (!getChildEdgeAt(i)->getMemoryPtr() || !getChildEdgeAt(i)->getMemory().GetPrimitivePtr())
            THROW_ERROR << "Destination memory has not been allocated.";
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "Preferable primitive descriptor is not set.";

    auto& memDesc = getParentEdgeAt(0)->getMemoryPtr()->GetDesc();

    canUseOptimizedNspc2Ncsp = false;
    if (axis == 1 && one_of(memDesc.getShape().getRank(), 4, 5) && memDesc.checkGeneralLayout(GeneralLayout::nspc)) {
        canUseOptimizedNspc2Ncsp = true;
        for (size_t i = 0; i < getChildEdges().size(); i++) {
            auto& childMemDesc = getChildEdgeAt(i)->getMemoryPtr()->GetDesc();
            if (!childMemDesc.checkGeneralLayout(GeneralLayout::ncsp))
                canUseOptimizedNspc2Ncsp = false;
        }
    }

    if (!isOptimized()) {
        initializeDstMemPtrs();
        if (!canUseOptimizedNspc2Ncsp)
            prepareOptimizedParams();
    }
}

void MKLDNNSplitNode::execute(mkldnn::stream strm) {
    if (isOptimized())
        return;

    if (dstMemPtrs.empty())
        THROW_ERROR << "Output data pointers have not been initialized.";

    int MB = batchToProcess();

    if (canUseOptimizedNspc2Ncsp) {
        optimizedNspc2Ncsp(MB);
        return;
    }

    uint8_t* srcData = reinterpret_cast<uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    size_t batch = this->getParentEdgeAt(0)->getShape().getStaticDims()[0];

    if (batch != MB)
        optimizedParams.countStrides = optimizedParams.countStrides / batch * MB;

    parallel_for2d(dstMemPtrs.size(), optimizedParams.countStrides, [&](size_t i, size_t j) {
        uint8_t* dstData = dstMemPtrs[i];

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
        THROW_ERROR << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isConfigDefined(config))
        return;
    //TODO [mkutakov]: why did we introduce this loop?
//    for (size_t i = 0; i < config.outConfs.size(); i++) {
////        if (config.outConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
////                !isUninitTensorDesc(config.outConfs[i].desc))
////            continue;
//        if (config.outConfs[i].desc->isDefined())
//            continue;
//
//        int num = getChildEdgeAt(i)->getOutputNum();
//        if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
//            if (num >= 0) {
//                auto childConf = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
//                childConf.desc->setPrecision(config.outConfs[i].desc->getPrecision());
//
//                if (!childConf.desc->isDefined() && childConf.inPlace >= 0)
//                    getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();
//
//                if (!childConf.desc->isDefined() && childConf.desc->isCompatible(*config.outConfs[i].desc)) {
//                    config.outConfs[i].desc = childConf.desc->clone();
//                    continue;
//                }
//            }
//        }
//        // TODO [DS]: Why do we need this code?
////        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
////                                                              config.outConfs[i].desc.getDims(), {
////                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
////                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder()
////                                                              });
//    }

    for (size_t i = 0; i < config.inConfs.size(); i++) {
        if (config.inConfs[i].desc->isDefined())
            continue;

        int num = getParentEdgeAt(i)->getOutputNum();
        if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()) {
            if (num >= 0) {
                const auto& parentConfig = getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
                if (!parentConfig.desc->isDefined() && parentConfig.inPlace >= 0)
                    getParentEdgeAt(i)->getParent()->initOptimalPrimitiveDescriptor();
                if (parentConfig.desc->isDefined() && parentConfig.desc->isCompatible(*config.inConfs[i].desc)) {
                    config.inConfs[i].desc = parentConfig.desc->clone();
                    continue;
                }
            }
        }

        // reset undefined offsets
        config.inConfs[i].desc = MemoryDescUtils::resetOffset(config.inConfs[i].desc.get());
    }
    if (config.outConfs.size() != outputShapes.size())
        THROW_ERROR << "has invalid config";

    auto firstInBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.inConfs[0].desc);
    size_t offset = 0;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto outBlockingDesc = MemoryDescUtils::convertToBlockedDescriptor(*config.outConfs[i].desc);
        config.outConfs[i].desc = make_unique<BlockedMemoryDesc>(outBlockingDesc.getPrecision(),
                                                                 outBlockingDesc.getShape().getStaticDims(),
                                                                 outBlockingDesc.getBlockDims(),
                                                                 outBlockingDesc.getOrder(),
                                                                 firstInBlockingDesc.getOffsetPadding() + offset,
                                                                 firstInBlockingDesc.getOffsetPaddingToData(),
                                                                 firstInBlockingDesc.getStrides());

        size_t axisSize = 1;
        for (size_t j = axis; j < outBlockingDesc.getBlockDims().size(); j++) {
            axisSize *= outBlockingDesc.getBlockDims()[j];
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {
    // Enforce the reference implementation for the planar layout if the implementation is in the impl priorities list.
    // This is needed mostly for the testing purposes, since for the planar layout Split works always in place, we need to enforce
    // the reference implementation when it is selected in a test to test that piece of code.
    if (!implPriorities.empty() && implPriorities[0] == impl_desc_type::ref) {
        for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); ++i) {
            auto& pd = supportedPrimitiveDescriptors[i];
            if (pd.getConfig().inConfs[0].desc->checkGeneralLayout(GeneralLayout::ncsp) &&
                impl_desc_type::ref == pd.getImplementationType()) {
                    selectPrimitiveDescriptorByIndex(static_cast<int>(i));
                return;
            }
        }
    }

    //check the descriptors and select the ones that have the same data format as the input

    std::vector<size_t> canSelectPrimitive;
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        auto parentEdge = getParentEdgeAt(0);
        auto parentPtr = parentEdge->getParent();
        auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

        if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
            int inNum = parentEdge->getInputNum();
            if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
                inNum = 0;
            }
            if (supportedPrimitiveDescriptors[i].getConfig().inConfs[0].desc->isCompatible(*parent_spd->getConfig().outConfs[inNum].desc)) {
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
            const auto& outputDesc = supportedPrimitiveDescriptors[indx].getConfig().outConfs[i].desc;

            if (!vecChildSpd.empty()) {
                int inNum = childEdge->getOutputNum();
                if (inNum < 0) {
                    inNum = 0;
                }
                bool hasMatchDesc = false;
                for (auto& childSpd : vecChildSpd) {
                    if (inNum >= childSpd.getConfig().inConfs.size()) {
                        inNum = 0;
                    }
                    if (outputDesc->isCompatible(*childSpd.getConfig().inConfs[inNum].desc)) {
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

void MKLDNNSplitNode::setDynamicBatchLim(int lim) {
    if (axis == 0)
        THROW_ERROR << "Dynamic batch is not supported by split layer with axis == 0 parameter";

    dynBatchLim = lim;
}

void MKLDNNSplitNode::prepareOptimizedParams() {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU Split node with name '" << getName() << "' doesn't have primitive descriptors.";
    const auto inpTensorDesc = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    const auto outputPortsCount = outputShapes.size();

    //find axis order position
    const auto& order = inpTensorDesc.getOrder();
    unsigned axisOrderPos = std::numeric_limits<unsigned>::max();
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] == axis) {
            axisOrderPos = i;
            break;
        }
    }
    if (std::numeric_limits<unsigned>::max() == axisOrderPos) {
        THROW_ERROR << "Can't find the axis in the input tensor order list";
    }

    uint8_t srcDataSize = inpTensorDesc.getPrecision().size();
    const auto& srcDims = inpTensorDesc.getBlockDims();
    const auto getRank = srcDims.size();

    optimizedParams.countStrides = 1;
    for (int i = 0; i < axisOrderPos; i++)
        optimizedParams.countStrides *= srcDims[i];

    optimizedParams.srcDataStride = 0;
    optimizedParams.dataSize.resize(outputPortsCount);

    for (size_t i = 0; i < outputPortsCount; i++) {
        auto outputEdge = this->getChildEdgesAtPort(i).front();
        optimizedParams.dataSize[i] = srcDataSize;

        auto desc = outputEdge->getMemory().GetDesc().as<BlockedMemoryDesc>();
        for (size_t j = axisOrderPos; j < getRank; j++)
            optimizedParams.dataSize[i] *= desc->getBlockDims()[j];

        optimizedParams.srcDataStride += optimizedParams.dataSize[i];
    }

    optimizedParams.srcDataOffsets.resize(outputPortsCount);
    optimizedParams.srcDataOffsets[0] = 0;
    for (size_t i = 1; i < outputPortsCount; i++) {
        optimizedParams.srcDataOffsets[i] = optimizedParams.srcDataOffsets[i - 1] + optimizedParams.dataSize[i - 1];
    }
}

void MKLDNNSplitNode::optimizedNspc2Ncsp(size_t MB) {
    auto parentEdge = getParentEdgeAt(0);
    const int rank = parentEdge->getShape().getRank();
    const auto parentDims = parentEdge->getShape().getStaticDims();
    const size_t IC = parentDims[1];
    const size_t D = rank == 5 ? parentDims[rank - 3] : 1;
    const size_t H = parentDims[rank - 2];
    const size_t W = parentDims[rank - 1];

    auto& srcMem = parentEdge->getMemory();
    auto srcData = reinterpret_cast<const uint8_t*>(srcMem.GetPtr());
    const auto dataSize = srcMem.GetDesc().getPrecision().size();

    const size_t DHW = D*H*W;
    const size_t strideIB = DHW * IC * dataSize;
    const size_t strideIW = IC*dataSize;
    const size_t strideOC = DHW * dataSize;

    for (size_t i = 0, sIdx = 0; i < outputShapes.size(); i++) {
        auto dstData = dstMemPtrs[i];

        size_t innerSize = 1;
        auto dims = outputShapes[i].getStaticDims();

        for (size_t j = axis; j < dims.size(); j++) {
            innerSize *= dims[j];
        }
        auto srcPtr = srcData + srcMem.GetDesc().getOffset(sIdx) * dataSize;

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

void MKLDNNSplitNode::initializeDstMemPtrs() {
    dstMemPtrs.clear();

    for (size_t i = 0; i < outputShapes.size(); ++i) {
        auto outputEdges = this->getChildEdgesAtPort(i);
        if (uint8_t* dstData = reinterpret_cast<uint8_t*>(outputEdges.front()->getMemoryPtr()->GetPtr())) {
            dstMemPtrs.push_back(dstData);
        } else {
            THROW_ERROR << "can't get child edge indx " << i << "data.";
        }
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNSplitNode, Split);
