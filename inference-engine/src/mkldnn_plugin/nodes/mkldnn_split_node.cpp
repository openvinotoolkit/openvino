// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include "common/cpu_memcpy.h"
#include "common/tensor_desc_creator.h"
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_parallel.hpp>
#include "utils/general_utils.h"

#define THROW_ERROR IE_THROW() << "Split layer with name '" << getName() <<"' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNSplitNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of_castable(op->get_type_info(), ngraph::op::v1::Split::type_info, ngraph::op::v1::VariadicSplit::type_info)) {
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

    auto srcDims = getParentEdgeAt(0)->getDims();
    auto axis_size = 0;
    auto dstFirstDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto o_Dims = outDims[i];
        if (dstFirstDims.ndims() != o_Dims.ndims()) {
            THROW_ERROR << "only supports output blobs with equal number of dimensions";
        }

        axis_size += o_Dims[axis];
        for (size_t j = 0; j < dstFirstDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (o_Dims[j] != dstFirstDims[j])
                THROW_ERROR << "has incorrect output dimensions";
        }
    }
    dstFirstDims[axis] = axis_size;
    if (dstFirstDims.size() != srcDims.size())
        THROW_ERROR << "sizes of input blob and sum of output blobs are not equal.";

    InferenceEngine::Precision inpPrecision = getOriginalInputPrecisionAtPort(0);
    const auto axisPrecision = getOriginalInputPrecisionAtPort(1);
    auto outPrecision = inpPrecision; // the split layer doesn't convert precisions

    bool dynBatchSupport = true;
    if (axis < 1) {
        dynBatchSupport = false;
    }

    //Set plain and tailC formats
    std::vector<TensorDescCreatorTypes> tdCreatorTypes{ TensorDescCreatorTypes::ncsp, TensorDescCreatorTypes::nspc };

    //Support channel blocked format
    if (srcDims.ndims() > 2) {
        for (auto item : { std::make_pair(8lu, TensorDescCreatorTypes::nCsp8c), std::make_pair(16lu, TensorDescCreatorTypes::nCsp16c) }) {
            SizeVector blkDims = srcDims.ToSizeVector();
            if (blkDims[channelsPos] % item.first)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < outDims.size(); i++) {
                if (outDims[i].ToSizeVector()[channelsPos] % item.first) {
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

    auto& creatorsMap = TensorDescCreator::getCommonCreators();
    auto itrRange = TensorDescCreator::makeFilteredRange(creatorsMap, static_cast<unsigned>(srcDims.ndims()), tdCreatorTypes);
    for (auto itr = itrRange.first; itr != itrRange.second; ++itr) {
        InferenceEngine::LayerConfig config;

        config.dynBatchSupport = dynBatchSupport;
        config.inConfs.resize(INPUTS_NUM);
        config.inConfs[0].inPlace = -1;
        config.inConfs[0].constant = false;
        config.inConfs[0].desc = itr->second->createDesc(inpPrecision, srcDims.ToSizeVector());
        config.inConfs[1].inPlace = -1;
        config.inConfs[1].constant = true;
        config.inConfs[1].desc.setDims({1});
        config.inConfs[1].desc.setPrecision(axisPrecision);
        if (INPUTS_NUM == 3) {
            config.inConfs[2].desc = TensorDesc(axisPrecision, SizeVector{outDims.size()}, TensorDesc::getLayoutByDims(SizeVector{outDims.size()}));
            config.inConfs[2].constant = true;
        }

        config.outConfs.resize(outDims.size());

        std::vector<memory::format_tag> outFormats;

        for (size_t i = 0; i < outDims.size(); i++) {
            auto o_Dims = outDims[i];

            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = itr->second->createDesc(inpPrecision, o_Dims.ToSizeVector());
            outFormats.push_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFormats);

        if (itr->first == TensorDescCreatorTypes::ncsp) {
            // at least the plain layout can be optimized inplace.
            pdIndexesToReuse.emplace_back(supportedPrimitiveDescriptors.size() - 1);
        } else if (itr->first == TensorDescCreatorTypes::nCsp8c || itr->first == TensorDescCreatorTypes::nCsp16c) {
            if (axis < 2) {
                pdIndexesToReuse.emplace_back(supportedPrimitiveDescriptors.size() - 1);
            }
        }
    }

    // Optimized inplace case
    for (auto refPdIndex : pdIndexesToReuse) {
        const auto& refConfig = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        auto config = refConfig;

        const auto& order = refConfig.inConfs[0].desc.getBlockingDesc().getOrder();
        const auto& blkDims = refConfig.inConfs[0].desc.getBlockingDesc().getBlockDims();
        auto numOfDim = blkDims.size();

        std::vector<memory::format_tag> outFormats;
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

        config.inConfs[0].desc = TensorDesc(inpPrecision, srcDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

        for (size_t i = 0; i < outDims.size(); i++) {
            const auto& outBlkDims = refConfig.outConfs[i].desc.getBlockingDesc().getBlockDims();
            const auto& dims = refConfig.outConfs[i].desc.getDims();

            config.outConfs[i].inPlace = 0;
            config.outConfs[i].desc = TensorDesc(outPrecision, dims, {outBlkDims, order, offset, offsets, strides});
            outFormats.emplace_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);
    }

    // Special nspc -> ncsp case when splitting channels
    if (axis == 1 && (dstFirstDims.ndims() == 4 || dstFirstDims.ndims() == 5)) {
        InferenceEngine::LayerConfig config;

        config.dynBatchSupport = dynBatchSupport;
        config.inConfs.resize(INPUTS_NUM);
        config.inConfs[0].inPlace = -1;
        config.inConfs[0].constant = false;
        config.inConfs[0].desc = creatorsMap.at(TensorDescCreatorTypes::nspc)->createDesc(inpPrecision, srcDims.ToSizeVector());
        config.inConfs[1].inPlace = -1;
        config.inConfs[1].constant = true;
        config.inConfs[1].desc.setDims({1});
        config.inConfs[1].desc.setPrecision(axisPrecision);
        if (INPUTS_NUM == 3) {
            config.inConfs[2].desc = TensorDesc(axisPrecision, SizeVector{outDims.size()}, TensorDesc::getLayoutByDims(SizeVector{outDims.size()}));
            config.inConfs[2].constant = true;
        }
        config.outConfs.resize(outDims.size());

        std::vector<memory::format_tag> outFormats;

        for (size_t i = 0; i < outDims.size(); i++) {
            auto o_Dims = outDims[i];

            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = creatorsMap.at(TensorDescCreatorTypes::ncsp)->createDesc(inpPrecision, o_Dims.ToSizeVector());
            outFormats.push_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFormats);
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

    canUseOptimizedNspc2Ncsp = true;
    if (axis != 1)
        canUseOptimizedNspc2Ncsp = false;

    if (getParentEdgeAt(0)->getBlob()->getTensorDesc().getLayout() != NHWC &&
        getParentEdgeAt(0)->getBlob()->getTensorDesc().getLayout() != NDHWC)
        canUseOptimizedNspc2Ncsp = false;

    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (getChildEdgeAt(i)->getBlob()->getTensorDesc().getLayout() != NCHW &&
            getChildEdgeAt(i)->getBlob()->getTensorDesc().getLayout() != NCDHW)
            canUseOptimizedNspc2Ncsp = false;
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
    size_t batch = this->getParentEdgeAt(0)->getDims()[0];

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
    if (config.outConfs.size() != outDims.size())
        THROW_ERROR << "has invalid config";
    size_t offset = 0;
    for (size_t i = 0; i < outDims.size(); i++) {
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

void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {
    // Enforce the reference implementation for the planar layout if the implementation is in the impl priorities list.
    // This is needed mostly for the testing purposes, since for the planar layout Split works always in place, we need to enforce
    // the reference implementation when it is selected in a test to test that piece of code.
    if (!implPriorities.empty() && implPriorities[0] == impl_desc_type::ref) {
        auto plain = PartialBlkDesc::makePlain(getParentEdgeAt(0)->getDims().ToSizeVector());
        for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); ++i) {
            auto& pd = supportedPrimitiveDescriptors[i];
            if (PartialBlkDesc::extractFrom(pd.getConfig().inConfs[0].desc) == plain &&
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
            if (MKLDNNExtensionUtils::initTensorsAreEqual(
                    supportedPrimitiveDescriptors[i].getConfig().inConfs[0].desc,
                    parent_spd->getConfig().outConfs[inNum].desc)) {
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
                    if (MKLDNNExtensionUtils::initTensorsAreEqual(outputDesc, childSpd.getConfig().inConfs[inNum].desc)) {
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
    const auto& inpTensorDesc = selectedPrimitiveDescriptor->getConfig().inConfs[0].desc;
    const auto outputPortsCount = outDims.size();

    //find axis order position
    const auto& order = inpTensorDesc.getBlockingDesc().getOrder();
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
    const auto& srcDims = inpTensorDesc.getBlockingDesc().getBlockDims();
    const auto nDims = srcDims.size();

    optimizedParams.countStrides = 1;
    for (int i = 0; i < axisOrderPos; i++)
        optimizedParams.countStrides *= srcDims[i];

    optimizedParams.srcDataStride = 0;
    optimizedParams.dataSize.resize(outputPortsCount);

    for (size_t i = 0; i < outputPortsCount; i++) {
        auto outputEdge = this->getChildEdgesAtPort(i).front();
        optimizedParams.dataSize[i] = srcDataSize;

        for (size_t j = axisOrderPos; j < nDims; j++)
            optimizedParams.dataSize[i] *= outputEdge->getDesc().getBlockingDesc().getBlockDims()[j];

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
    const int ndims = parentEdge->getDims().ndims();
    const size_t IC = parentEdge->getDims()[1];
    const size_t D = ndims == 5 ? parentEdge->getDims()[ndims - 3] : 1;
    const size_t H = parentEdge->getDims()[ndims - 2];
    const size_t W = parentEdge->getDims()[ndims - 1];

    auto srcBlob = parentEdge->getBlob();
    auto srcData = srcBlob->cbuffer().as<const uint8_t*>();
    const auto dataSize = srcBlob->getTensorDesc().getPrecision().size();

    const size_t DHW = D*H*W;
    const size_t strideIB = DHW * IC * dataSize;
    const size_t strideIW = IC*dataSize;
    const size_t strideOC = DHW * dataSize;

    for (size_t i = 0, sIdx = 0; i < outDims.size(); i++) {
        auto dstData = dstMemPtrs[i];

        size_t innerSize = 1;
        auto dims = outDims[i].ToSizeVector();

        for (size_t j = axis; j < dims.size(); j++) {
            innerSize *= dims[j];
        }
        auto srcPtr = srcData + srcBlob->getTensorDesc().offset(sIdx) * dataSize;

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

    for (size_t i = 0; i < outDims.size(); ++i) {
        auto outputEdges = this->getChildEdgesAtPort(i);
        if (uint8_t* dstData = reinterpret_cast<uint8_t*>(outputEdges.front()->getMemoryPtr()->GetPtr())) {
            dstMemPtrs.push_back(dstData);
        } else {
            THROW_ERROR << "can't get child edge indx " << i << "data.";
        }
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNSplitNode, Split);
