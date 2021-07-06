// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_strided_slice_node.h"

#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

#include "ie_parallel.hpp"
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "common/blocked_desc_creator.h"
#include "utils/general_utils.h"
#include "mkldnn_input_node.h"

#include <string>
#include <tuple>
#include <algorithm>
#include "caseless.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <cpu_memory_desc_utils.h>

#define THROW_ERROR IE_THROW() << "StridedSlice layer with name '" << getName() << "' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

static inline size_t parallel_init(size_t start, size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

bool MKLDNNStridedSliceNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto ss = std::dynamic_pointer_cast<const ngraph::opset1::StridedSlice>(op);
        if (!ss) {
            errorMessage = "Only opset1 StridedSlice operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNStridedSliceNode::MKLDNNStridedSliceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        const auto ss = std::dynamic_pointer_cast<const ngraph::opset1::StridedSlice>(op);

        const size_t nDims = std::max(inputShapes[DATA_ID].getRank(), outputShapes[0].getRank());

        auto createMask = [&](const std::vector<int64_t> &origMask, const int bit = 0, bool needReverse = false) {
            std::vector<int> mask(origMask.begin(), origMask.end());
            if (needReverse) {
                for (size_t i = 0; i < mask.size(); i++)
                    mask[i] = 1 - mask[i];
            }
            for (size_t i = mask.size(); i < nDims; ++i) mask.push_back(bit);
            return mask;
        };

        beginMask = createMask(ss->get_begin_mask(), 1, true);
        endMask = createMask(ss->get_end_mask(), 1, true);
        newAxisMask = createMask(ss->get_new_axis_mask());
        shrinkAxisMask = createMask(ss->get_shrink_axis_mask());

        auto origEllipsisMask = ss->get_ellipsis_mask();
        for (const auto &o : origEllipsisMask) {
            ellipsisMask.push_back(o);
        }
        if (ellipsisMask.size() == 0) {
            for (size_t i = ellipsisMask.size(); i < nDims; ++i) ellipsisMask.push_back(0);
        }

    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNStridedSliceNode::getSupportedDescriptors() {
    auto isConstantNode = [](const MKLDNNNodePtr &node) {
        return node->getType() == Input && node->isConstant();
    };

    params.parametersAreConstant = isConstantNode(getParentEdgesAtPort(BEGIN_ID)[0]->getParent()) &&
                                   isConstantNode(getParentEdgesAtPort(END_ID)[0]->getParent());

    const SizeVector srcDims = inputShapes[DATA_ID].getStaticDims();
    const SizeVector dstDims = outputShapes[0].getStaticDims();
    const size_t nSrcDims = srcDims.size();
    const size_t nDims = std::max(nSrcDims, dstDims.size());

    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        THROW_ERROR << "has incorrect number of input edges";
    if (!getChildEdges().size())
        THROW_ERROR << "has incorrect number of output edges";

    beginDims = inputShapes[BEGIN_ID].getStaticDims();
    if (beginDims.size() != 1)
        THROW_ERROR << " should have begin vector with 1 dimension";

    endDims = inputShapes[END_ID].getStaticDims();
    if (endDims.size() != 1)
        THROW_ERROR << "should have end vector with 1 dimension";
    if (beginDims[0] != endDims[0])
        THROW_ERROR << "should have begin vector with size equal to end vector size";

    if (inputShapes.size() > STRIDE_ID) {
        if (!isConstantNode(getParentEdgesAtPort(STRIDE_ID)[0]->getParent()))
            params.parametersAreConstant = false;

        strideDims = inputShapes[STRIDE_ID].getStaticDims();
        if (strideDims.size() > 1)
            THROW_ERROR << "should have stride vector with 1 dimension";
        if (beginDims[0] != strideDims[0])
            THROW_ERROR << "should have stride vector with size equal to begin vector size";
    }

    int ellipsisMaskCounter = 0;
    params.ellipsisPos1 = -1;
    for (size_t i = 0; i < ellipsisMask.size(); i++) {
        ellipsisMaskCounter += ellipsisMask[i];
        params.ellipsisPos1 = ellipsisMask[i] == 1 && params.ellipsisPos1 == -1 ? i : params.ellipsisPos1;
    }
    if (ellipsisMaskCounter > 1)
        THROW_ERROR << "has incorrect 'Ellipsis_mask'. Only one non-zero bit is allowed";

    int newAxis = std::accumulate(newAxisMask.begin(), newAxisMask.end(), 0);
    int shrinkAxis = std::accumulate(shrinkAxisMask.begin(), shrinkAxisMask.end(), 0);
    params.equalDims = newAxis == 0 && shrinkAxis == 0;

    if (params.parametersAreConstant) {
        auto fillingInParameters = [&](std::vector<int> &parameter, const size_t type, const size_t size, const int value) {
            const auto constNode = std::dynamic_pointer_cast<MKLDNNInputNode>(getParentEdgesAtPort(type)[0]->getParent());
            if (!constNode) {
                THROW_ERROR << "can't cast node on " << type << " port to MKLDNNInputNode";
            }
            auto blob = constNode->getMemoryPtr();
            if (blob->GetDataType() != mkldnn::memory::data_type::s32)
                THROW_ERROR << "supports only parameters input with precision I32";
            const int *ptr = static_cast<const int*>(blob->GetPtr());
            parameter.assign(ptr, ptr + size);

            if (ellipsisMaskCounter == 0 && size < nDims) {
                for (size_t i = size; i < nDims; i++) parameter.push_back(value);
            }
        };

        if (beginDims.size())
            fillingInParameters(begin, BEGIN_ID, beginDims[0], 0);
        if (endDims.size())
            fillingInParameters(end, END_ID, endDims[0], 0);
        if (strideDims.size())
            fillingInParameters(stride, STRIDE_ID, strideDims[0], 1);

        if (nSrcDims > 3 && params.equalDims && ellipsisMaskCounter == 1)
            addHiddenDims(nSrcDims);
    }
}

void MKLDNNStridedSliceNode::addHiddenDims(const size_t nSrcDims) {
    // all masks and input parameters are for planar layouts. So if we use blocked or per channel layout and
    // there is ellipsis should to add default values in hidden dimensions to know real order of mask or parameter values
    size_t afterDims = ellipsisMask.size() - params.ellipsisPos1 - 1;
    size_t ellipsisPos2 = nSrcDims - afterDims - 1;

    auto addHiddenDims = [&](std::vector<int>& data, const int bit = 0) {
        std::vector<int> temp;
        for (size_t i = 0; i < params.ellipsisPos1; i++)
            temp.push_back(data[i]);
        for (size_t i = params.ellipsisPos1; i < ellipsisPos2 + 1; i++)
            temp.push_back(bit);
        for (size_t i = 1; i < nSrcDims - ellipsisPos2; i++)
            temp.push_back(data[i + params.ellipsisPos1]);
        data = temp;
    };

    addHiddenDims(begin);
    addHiddenDims(end);
    addHiddenDims(stride, 1);
    addHiddenDims(beginMask);
    addHiddenDims(endMask);
    addHiddenDims(ellipsisMask);
    addHiddenDims(newAxisMask);
    addHiddenDims(shrinkAxisMask);
}

void MKLDNNStridedSliceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const bool hasStrides = getParentEdges().size() > 3;
    InferenceEngine::Precision dataPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    InferenceEngine::Precision beginPrecision = getOriginalInputPrecisionAtPort(BEGIN_ID);
    auto beginDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(beginPrecision);
    InferenceEngine::Precision endPrecision = getOriginalInputPrecisionAtPort(END_ID);
    auto endDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(endPrecision);
    InferenceEngine::Precision stridePrecision;
    if (hasStrides)
        stridePrecision = getOriginalInputPrecisionAtPort(STRIDE_ID);

    auto srcDims = getParentEdgeAt(DATA_ID)->getShape().getStaticDims();
    auto dstDims = getChildEdgeAt(0)->getShape().getStaticDims();
    size_t nDims = srcDims.size();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(getParentEdges().size());
    config.inConfs[DATA_ID].inPlace = -1;
    config.inConfs[BEGIN_ID].inPlace = -1;
    config.inConfs[END_ID].inPlace = -1;
    config.inConfs[DATA_ID].constant = false;
    config.inConfs[BEGIN_ID].constant = true;
    config.inConfs[END_ID].constant = true;
    if (hasStrides) {
        config.inConfs[STRIDE_ID].inPlace = -1;
        config.inConfs[STRIDE_ID].constant = true;
    }
    config.outConfs.resize(1);

    std::vector<GeneralLayout> supportedTypes;
    if (nDims > 2 && params.equalDims) {
        auto canUseBlocked = [=](const size_t blockSize) {
            return srcDims[1] % blockSize == 0 && abs(stride[1]) == 1 && (begin[1] > srcDims[1] || begin[1] % blockSize == 0);
        };

        supportedTypes.push_back(GeneralLayout::nspc);
        if (canUseBlocked(8lu))
            supportedTypes.push_back(GeneralLayout::nCsp8c);
        if (canUseBlocked(16lu))
            supportedTypes.push_back(GeneralLayout::nCsp16c);
    }
    supportedTypes.push_back(GeneralLayout::ncsp);
    auto creators = BlockedDescCreator::getCommonCreators();
    auto range = BlockedDescCreator::makeFilteredRange(creators, nDims, supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].desc = itr->second->createUniqueDesc(dataPrecision, getParentEdgeAt(DATA_ID)->getShape().getStaticDims());
        config.inConfs[BEGIN_ID].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(BEGIN_ID)->getShape().getStaticMklDims(), beginDataType,
                                                                      mkldnn::memory::format_tag::x);
        config.inConfs[END_ID].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(END_ID)->getShape().getStaticMklDims(), endDataType,
                                                                    mkldnn::memory::format_tag::x);
        if (hasStrides)
            config.inConfs[STRIDE_ID].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(STRIDE_ID)->getShape().getStaticMklDims(),
                                                              MKLDNNExtensionUtils::IEPrecisionToDataType(stridePrecision),
                                                              mkldnn::memory::format_tag::x);

        config.outConfs[0].desc = itr->second->createUniqueDesc(dataPrecision, getChildEdgeAt(DATA_ID)->getShape().getStaticDims());
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
    }
}

void MKLDNNStridedSliceNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor.";

    auto srcBlockingDesc = getParentEdgeAt(DATA_ID)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    auto dstBlockingDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    auto srcOrder = srcBlockingDesc.getOrder();
    params.srcDims = srcBlockingDesc.getBlockDims();
    params.dstDims = dstBlockingDesc.getBlockDims();
    params.dataSize = getSelectedPrimitiveDescriptor()->getConfig().inConfs[DATA_ID].desc->getPrecision().size();

    if (params.parametersAreConstant) {
        size_t realNDims = params.dstDims.size();
        if (!getParentEdgeAt(DATA_ID)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::ncsp))
            orderParametersByLayouts();

        SizeVector newSrcDims, newDstDims;
        dimsNormalization(newSrcDims, newDstDims);
        dimsGluing(realNDims, newSrcDims, newDstDims);

        if (params.dstDims.size() == 1 || params.nDimsForWork != 1)
            indicesCalculation();
    }
}

void MKLDNNStridedSliceNode::orderParametersByLayouts() {
    const bool isPerChannelLayout = getParentEdgeAt(DATA_ID)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nspc);
    const bool isBlockedLayout = getParentEdgeAt(DATA_ID)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nCsp8c) ||
                                 getParentEdgeAt(DATA_ID)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nCsp16c);
    auto srcOrder = getParentEdgeAt(DATA_ID)->getMemory().GetDescWithType<BlockedMemoryDesc>().getOrder();

    if (isBlockedLayout) {
        const size_t blk = params.srcDims.back();
        begin[1] = begin[1] / blk;
        end[1] = ceil(end[1] / static_cast<float>(blk));
        begin.push_back(0);
        end.push_back(0);
        stride.push_back(1);
        beginMask.push_back(0);
        endMask.push_back(0);
        ellipsisMask.push_back(0);
        newAxisMask.push_back(0);
        shrinkAxisMask.push_back(0);
    } else if (isPerChannelLayout) {
        auto sortByOrder = [&](std::vector<int>& data) {
            std::vector<int> temp(srcOrder.size());
            for (size_t i = 0; i < srcOrder.size(); i++)
                temp[i] = data[srcOrder[i]];
            data = temp;
        };

        sortByOrder(begin);
        sortByOrder(end);
        sortByOrder(stride);
        sortByOrder(beginMask);
        sortByOrder(endMask);
        sortByOrder(ellipsisMask);
        sortByOrder(newAxisMask);
        sortByOrder(shrinkAxisMask);
    }
}

void MKLDNNStridedSliceNode::dimsNormalization(SizeVector& newSrcDims, SizeVector& newDstDims) {
    // creating new src and dst dimensions and parameters of the same size using masks
    //
    // example 1: before srcDims = [5, 6, 8, 3, 2], begin = [1, 0], end = [4, 0], stride = [1, 1]
    //            beginMask = [0, 1], endMask = [0, 1], ellipsisMask = [1, 0], newAxisMas = [0, 0], shrinkAxisMask = [0, 0]
    //            after srcDims = [5, 6, 8, 3, 2], begin = [1, 0, 0, 0, 0], end = [4, 5, 7, 2, 1], stride = [1, 1, 1, 1, 1], dstDims = [4, 6, 8, 3, 2]
    //
    // example 2: before srcDims = [5, 6, 8, 3, 2], begin = [0, 3, 0, 0, 0], end = [0, 3, 0, 0, 0], stride = [1, 1, 1, 1, 1]
    //            beginMask = [1, 0, 1, 1, 1], endMask = [1, 0, 1, 1, 1], ellipsisMask = [0, 0, 0, 0, 0], newAxisMask = [0, 0, 0, 0, 0],
    //            shrinkAxisMask = [0, 1, 0, 0, 0]
    //            after srcDims = [5, 6, 8, 3, 2], begin = [0, 3, 0, 0, 0], end = [4, 3, 7, 2, 1], stride = [1, 1, 1, 1, 1], dstDims = [5, 1, 8, 3, 2]
    //
    // example 3: before srcDims = [5, 8, 3, 2], begin = [0, 0, 0, 0], end = [0, 0, 0, 0], stride = [1, 1, 1, 1]
    //            beginMask = [1, 0, 1, 1, 1], endMask = [1, 0, 1, 1, 1], ellipsisMask = [0, 0, 0, 0, 0], newAxisMask = [0, 1, 0, 0, 0],
    //            shrinkAxisMask = [0, 0, 0, 0, 0]
    //            after srcDims = [5, 1, 8, 3, 2], begin = [0, 0, 0, 0, 0], end = [4, 0, 7, 2, 1], stride = [1, 1, 1, 1, 1], dstDims = [5, 1, 8, 3, 2]

    auto clipping = [](int& idx, const int min, const int max) {
        idx = (idx > min) ? idx : min;
        idx = (idx < max) ? idx : (max - 1);
    };

    auto correcting = [](int& dim, const size_t shift) {
        dim = dim >= 0 ? dim : shift + dim;
    };

    std::vector<int> beginTemp;
    std::vector<int> endTemp;
    std::vector<int> strideTemp;
    size_t srcIdx = 0;
    for (size_t axis = 0; axis < begin.size(); ++axis) {
        if (ellipsisMask[axis] == 1) {
            int nNewAxisAfterEllipses = 0;
            int nSrcAxisBeforeEllipses = 0;
            for (size_t i = 0; i < axis; ++i) {
                if (newAxisMask[i] != 1)
                    nSrcAxisBeforeEllipses++;
            }
            for (size_t i = axis + 1; i < begin.size(); ++i) {
                if (newAxisMask[i] == 1)
                    nNewAxisAfterEllipses++;
            }

            size_t nSrcAxisAfterEllipses = (begin.size() - axis - nNewAxisAfterEllipses - 1);
            size_t nHiddenDims = params.srcDims.size() - nSrcAxisAfterEllipses - nSrcAxisBeforeEllipses;
            for (size_t i = 0; i < nHiddenDims; ++i) {
                newSrcDims.push_back(params.srcDims[srcIdx]);
                newDstDims.push_back(params.srcDims[srcIdx]);
                beginTemp.push_back(0);
                endTemp.push_back(params.srcDims[srcIdx] - 1);
                strideTemp.push_back(1);

                srcIdx++;
            }
        } else {
            if (newAxisMask[axis] == 1) {
                beginTemp.push_back(0);
                endTemp.push_back(0);
                strideTemp.push_back(1);
                newSrcDims.push_back(1);
                newDstDims.push_back(1);
            } else if (shrinkAxisMask[axis] == 1) {
                int b = beginMask[axis] == 1 ? begin[axis] : 0;
                correcting(b, params.srcDims[srcIdx]);
                clipping(b, 0, params.srcDims[srcIdx]);
                beginTemp.push_back(b);
                endTemp.push_back(b);
                strideTemp.push_back(1);
                newSrcDims.push_back(params.srcDims[srcIdx]);
                newDstDims.push_back(1);

                srcIdx++;
            } else {
                int b = beginMask[axis] == 1 ? begin[axis] : (stride[axis] > 0 ? 0 : -1);
                correcting(b, params.srcDims[srcIdx]);
                clipping(b, 0, params.srcDims[srcIdx]);

                int e = endMask[axis] == 1 ? (stride[axis] > 0 ? end[axis] - 1 : end[axis] + 1) :
                        (stride[axis] > 0 ? -1 : 0);
                correcting(e, params.srcDims[srcIdx]);
                clipping(e, 0, params.srcDims[srcIdx]);

                beginTemp.push_back(b);
                endTemp.push_back(e);
                strideTemp.push_back(stride[axis]);
                newSrcDims.push_back(params.srcDims[srcIdx]);
                newDstDims.push_back(ceil(static_cast<float>(abs(e - b) + 1) / static_cast<float>(abs(strideTemp.back()))));

                srcIdx++;
            }
        }
    }

    begin = beginTemp;
    end = endTemp;
    stride = strideTemp;

    params.dstDims = newDstDims;
    params.srcDims = newSrcDims;
    params.dstStrides.resize(newDstDims.size());
    params.srcStrides.resize(newSrcDims.size());
    params.dstStrides[params.dstStrides.size() - 1] = params.srcStrides[params.srcStrides.size() - 1] = 1;
    for (int i = newDstDims.size() - 2; i >= 0; --i) {
        params.dstStrides[i] = params.dstStrides[i + 1] * params.dstDims[i + 1];
        params.srcStrides[i] = params.srcStrides[i + 1] * params.srcDims[i + 1];
    }
}

void MKLDNNStridedSliceNode::dimsGluing(const size_t realNDims, const SizeVector& newSrcDims, const SizeVector& newDstDims) {
    // gluing of dimensions if there aren't begin, end and stride != 1 on this axis
    // example: before gluing srcDims = [5, 6, 8, 3, 2], begin = [1, 0, 0, 0, 0], stride = [1, 1, 2, 1, 1], dstDims = [4, 6, 4, 3, 2]
    //          after gluing  srcDims = [30, 8, 6],      begin = [6, 0, 0],       stride = [1, 2, 1],       dstDims = [24, 4, 6]

    std::pair<size_t, size_t> secondDim = { 0, begin.size() };
    SizeVector indexes(1, 0);
    for (int idx = 0; idx < begin.size(); idx++) {
        if (begin[idx] != 0 || end[idx] != params.srcDims[idx] - 1 || stride[idx] != 1) {
            indexes.push_back(std::max(idx - 1, 0));
            indexes.push_back(stride[idx] == 1 ? idx : idx + 1);

            if (idx != 0 && secondDim.first == 0)
                secondDim.first = idx;
            else if (idx != 0 && secondDim.second == begin.size())
                secondDim.second = idx;
        }
    }

    if (indexes.back() < 2) {
        indexes[indexes.size() - 1] = 1;
        secondDim.first = 1;
    }

    const size_t nGluingLastDims = params.dstStrides[std::max(static_cast<int>(indexes.back() - 1), 0)];
    const bool vLastDim = indexes.back() < begin.size();
    indexes[indexes.size() - 1] = vLastDim ? indexes.back() : begin.size() - 1;
    indexes.push_back(begin.size() - 1);

    for (int idx = indexes.size() - 1; idx >= 0; idx -= 2) {
        if (indexes[idx - 1] < indexes[idx]) {
            for (size_t jdx = indexes[idx]; jdx > indexes[idx - 1]; --jdx) {
                params.dstDims[indexes[idx - 1]] *= params.dstDims[jdx];
                params.srcDims[indexes[idx - 1]] *= params.srcDims[jdx];
                params.dstStrides[indexes[idx - 1]] /= params.dstDims[jdx];
                params.srcStrides[indexes[idx - 1]] /= params.srcDims[jdx];

                begin[indexes[idx - 1]] *= params.dstDims[jdx];
            }
            const size_t beginShift = indexes[idx - 1] + 1;
            const size_t endShift = indexes[idx] + 1;

            params.dstDims.erase(params.dstDims.begin() + beginShift, params.dstDims.begin() + endShift);
            params.srcDims.erase(params.srcDims.begin() + beginShift, params.srcDims.begin() + endShift);
            params.dstStrides.erase(params.dstStrides.begin() + beginShift, params.dstStrides.begin() + endShift);
            params.srcStrides.erase(params.srcStrides.begin() + beginShift, params.srcStrides.begin() + endShift);

            begin.erase(begin.begin() + beginShift, begin.begin() + endShift);
            stride.erase(stride.begin() + beginShift, stride.begin() + endShift);
        }
    }

    params.workAmount = params.dstDims[0] * params.dstStrides[0] / nGluingLastDims;
    params.lastDstDim = nGluingLastDims * params.dataSize;
    params.nDimsForWork = params.dstDims.size() - static_cast<size_t>(vLastDim);

    if (params.nDimsForWork == 1 && realNDims > 2) {
        const size_t realSrcDim = newSrcDims[secondDim.first];
        const size_t realDstDim = newDstDims[secondDim.first];

        params.dstStrides.insert(params.dstStrides.begin() + 1, params.dstStrides[0] / realDstDim);
        params.srcStrides.insert(params.srcStrides.begin() + 1, params.srcStrides[0] / realSrcDim);

        for (size_t idx = secondDim.first + 1; idx < secondDim.second; idx++)
            begin[1] /= newDstDims[idx];

        const size_t maxThreads = parallel_get_max_threads();
        if (params.dstDims[0] < maxThreads) {
            params.dstDims[1] /= realDstDim;
            params.srcDims[1] /= realSrcDim;
            params.dstDims.insert(params.dstDims.begin() + 1, realDstDim);
            params.srcDims.insert(params.srcDims.begin() + 1, realSrcDim);
        }

        if (params.dstDims.size() > 2)
            params.lastDstDim /= newDstDims[secondDim.first];
    }
}

void MKLDNNStridedSliceNode::indicesCalculation() {
    // indices calculation before execution for the best performance
    params.nThreads = parallel_get_max_threads();
    params.srcIndices.resize(params.workAmount, 0);
    params.dstIndices.resize(params.workAmount, 0);

    auto getSrcIdx = [this](const SizeVector& indexes){
        size_t srcIdx = 0;
        for (int i = 0; i < params.nDimsForWork; ++i)
            srcIdx += (begin[i] + indexes[i] * stride[i]) * params.srcStrides[i];
        return srcIdx * params.dataSize;
    };

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector coords(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);
        parallel_init(start, params.nDimsForWork, params.dstDims, coords);

        size_t srcIdx = getSrcIdx(coords);
        for (size_t j = start; j < end; ++j) {
            params.dstIndices[j] = j * params.lastDstDim;
            params.srcIndices[j] = srcIdx;

            bool out = false;
            for (int k = params.nDimsForWork - 1; k >= 0; k--) {
                coords[k]++;
                if (coords[k] < params.dstDims[k]) {
                    srcIdx += stride[k] * params.srcStrides[k] * params.dataSize;
                    break;
                } else {
                    coords[k] = 0;
                    out = true;
                }
            }

            if (out)
                srcIdx = getSrcIdx(coords);
        }
    });
}

void MKLDNNStridedSliceNode::execute(mkldnn::stream strm) {
    if (!params.parametersAreConstant) {
        auto srcDims = getParentEdgeAt(DATA_ID)->getShape().getStaticDims();
        auto dstDims = getChildEdgeAt(0)->getShape().getStaticDims();
        const size_t nDims = std::max(srcDims.size(), dstDims.size());
        const size_t ellipsisMaskCounter = std::accumulate(ellipsisMask.begin(), ellipsisMask.end(), 0);

        auto fillingInParameters = [&](std::vector<int> &parameter, const size_t type, const size_t size, const int value) {
            const int *ptr = reinterpret_cast<const int*>(this->getParentEdgeAt(type)->getMemoryPtr()->GetPtr());
            parameter.assign(ptr, ptr + size);

            if (ellipsisMaskCounter == 0 && size < nDims) {
                for (size_t i = size; i < nDims; i++) parameter.push_back(value);
            }
        };

        if (beginDims.size())
            fillingInParameters(begin, BEGIN_ID, beginDims[0], 0);
        if (endDims.size())
            fillingInParameters(end, END_ID, endDims[0], 0);
        if (strideDims.size())
            fillingInParameters(stride, STRIDE_ID, strideDims[0], 1);

        if (srcDims.size() > 3 && params.equalDims && ellipsisMaskCounter != 0)
            addHiddenDims(srcDims.size());

        if (!getParentEdgeAt(DATA_ID)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::ncsp))
            orderParametersByLayouts();

        SizeVector newSrcDims, newDstDims;
        dimsNormalization(newSrcDims, newDstDims);
        dimsGluing(dstDims.size(), newSrcDims, newDstDims);

        if (params.dstDims.size() == 1 || params.nDimsForWork != 1)
            indicesCalculation();
    }

    if (params.dstDims.size() > 1 && params.nDimsForWork == 1)
        stridedSliceV();
    else
        stridedSlice();
}

void MKLDNNStridedSliceNode::stridedSliceV() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(DATA_ID)->getMemoryPtr()->GetPtr()) +
                             (begin[0] * params.srcStrides[0] + begin[1] * params.srcStrides[1]) * params.dataSize;
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const size_t dstIdx = params.dstStrides[0] * params.dataSize;
    const size_t srcIdx = stride[0] * params.srcStrides[0] * params.dataSize;
    const size_t dstShift = params.dstStrides[1] * params.dataSize;
    const size_t srcShift = stride[1] * params.srcStrides[1] * params.dataSize;

    if (params.dstDims.size() > 2) {
        parallel_for2d(params.dstDims[0], params.dstDims[1], [&](const size_t i, const size_t j) {
            cpu_memcpy(&dstData[i * dstIdx + j * dstShift], &srcData[i * srcIdx + j * srcShift], params.lastDstDim);
        });
    } else {
        parallel_for(params.dstDims[0], [&](const size_t i) {
            cpu_memcpy(&dstData[i * dstIdx], &srcData[i * srcIdx], params.lastDstDim);
        });
    }
}

void MKLDNNStridedSliceNode::stridedSlice() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(DATA_ID)->getMemoryPtr()->GetPtr()) +
            (stride.back() == 1 && stride.size() > 1 ? begin[params.nDimsForWork] * params.srcStrides[params.nDimsForWork] * params.dataSize : 0);
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(params.workAmount, nthr, ithr, start, end);

        for (size_t iwork = start; iwork < end; ++iwork)
            cpu_memcpy(&dstData[params.dstIndices[iwork]], &srcData[params.srcIndices[iwork]], params.lastDstDim);
    });
}

bool MKLDNNStridedSliceNode::created() const {
    return getType() == StridedSlice;
}

REG_MKLDNN_PRIM_FOR(MKLDNNStridedSliceNode, StridedSlice);
