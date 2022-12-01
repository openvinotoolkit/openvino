// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice.h"

#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "input.h"
#include <ngraph/opsets/opset1.hpp>

#include <string>

#define THROW_ERROR IE_THROW() << NameFromType(getType()) << " node with name '" << getName() << "' "

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

static inline size_t parallel_init(size_t start, size_t nDims, const VectorDims& dims, VectorDims& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

bool StridedSlice::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v1::StridedSlice>(op) &&
                !ov::is_type<ov::op::v8::Slice>(op)) {
            errorMessage = "Only StridedSlice from opset1 and Slice from opset8 operations are supported.";
            return false;
        }

        if (!ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(BEGIN_ID)) ||
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_shared_ptr(END_ID)) ||
                (op->get_input_size() > STRIDE_ID && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(STRIDE_ID))) ||
                (op->get_input_size() > AXES_ID && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXES_ID)))) {
            // TODO: Support begin, end, stride, axis inputs for dynamic shapes.
            errorMessage = "Only Constant 'begin', 'end', 'stride' and 'axis' inputs are supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

StridedSlice::StridedSlice(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
        Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    isStridedSliceOp = ov::is_type<ov::op::v1::StridedSlice>(op);

    if ((isStridedSliceOp && (inputShapes.size() < 3 || inputShapes.size() > 4)) ||
            (!isStridedSliceOp && (inputShapes.size() < 4 || inputShapes.size() > 5))) {
        THROW_ERROR << "has incorrect number of input edges";
    }
    if (outputShapes.size() != 1) {
        THROW_ERROR << "has incorrect number of output edges";
    }
    for (size_t i = 0lu; i < op->get_input_size(); i++) {
        isConstantInput[i] = ov::is_type<ov::op::v0::Constant>(op->inputs()[i].get_node());
    }

    attrs.beginDims = getInputShapeAtPort(BEGIN_ID).getStaticDims();
    attrs.endDims = getInputShapeAtPort(END_ID).getStaticDims();
    if (attrs.beginDims.size() != 1)
        THROW_ERROR << "should have begin vector with 1 dimension";
    if (attrs.endDims.size() != 1)
        THROW_ERROR << "should have end vector with 1 dimension";
    if (attrs.beginDims[0] != attrs.endDims[0])
        THROW_ERROR << "should have begin vector with size equal to end vector size";

    if (inputShapes.size() > STRIDE_ID) {
        isStrideSpecified = true;
        attrs.strideDims = getInputShapeAtPort(STRIDE_ID).getStaticDims();
        if (attrs.strideDims.size() > 1)
            THROW_ERROR << "should have stride vector with 1 dimension";
        if (attrs.beginDims[0] != attrs.strideDims[0])
            THROW_ERROR << "should have stride vector with size equal to begin vector size";
    }

    if (inputShapes.size() > AXES_ID) {
        isAxesSpecified = true;
        attrs.axesDims = inputShapes[AXES_ID].getStaticDims();
        if (attrs.axesDims.size() != 1)
            THROW_ERROR << "should have axes vector with 1 dimension.";
        if (attrs.beginDims[0] != attrs.axesDims[0])
            THROW_ERROR << "should have axes vector with size equal to begin vector size.";
    }

    if (isStridedSliceOp) {
        auto ss = ov::as_type_ptr<const ov::op::v1::StridedSlice>(op);

        const size_t inputRank = getInputShapeAtPort(DATA_ID).getRank();
        const size_t outputRank = getOutputShapeAtPort(0).getRank();

        const size_t nDims = std::max(inputRank, outputRank);

        auto createMask = [&](const std::vector<int64_t> &origMask, const int bit = 0, bool needReverse = false) {
            std::vector<int> mask(origMask.size());
            for (size_t i = 0; i < mask.size(); i++) {
                mask[i] = static_cast<int>(origMask[i]);
                if (needReverse) {
                    mask[i] = 1 - mask[i];
                }
            }
            for (size_t i = mask.size(); i < nDims; ++i) mask.push_back(bit);
            return mask;
        };

        attrs.beginMask = createMask(ss->get_begin_mask(), 1, true);
        attrs.endMask = createMask(ss->get_end_mask(), 1, true);
        attrs.newAxisMask = createMask(ss->get_new_axis_mask());
        attrs.shrinkAxisMask = createMask(ss->get_shrink_axis_mask());

        auto origEllipsisMask = ss->get_ellipsis_mask();
        bool isEllipsis = false;
        for (const auto &o : origEllipsisMask) {
            isEllipsis = isEllipsis || o != 0;
            attrs.ellipsisMask.push_back(o);
        }
        if (attrs.ellipsisMask.size() == 0 || !isEllipsis) {
            for (size_t i = attrs.ellipsisMask.size(); i < nDims; ++i) attrs.ellipsisMask.push_back(0);
        }
    } else {
        const size_t length = outputShapes[0].getRank();
        if (inputShapes.size() > AXES_ID) {
            attrs.beginMask = std::vector<int>(length, 0);
            attrs.endMask = std::vector<int>(length, 0);
        } else {
            attrs.beginMask = std::vector<int>(length, 1);
            attrs.endMask = std::vector<int>(length, 1);
        }
        attrs.newAxisMask = std::vector<int>(length, 0);
        attrs.shrinkAxisMask = std::vector<int>(length, 0);
        attrs.ellipsisMask = std::vector<int>(length, 0);
    }
}

void StridedSlice::getSupportedDescriptors() {
    const size_t inputRank = getInputShapeAtPort(DATA_ID).getRank();
    const size_t outputRank = getOutputShapeAtPort(0).getRank();
    const size_t nDims = std::max(inputRank, outputRank);

    int ellipsisMaskCounter = 0;
    int ellipsisPos1 = -1;
    if (isStridedSliceOp) {
        for (size_t i = 0; i < attrs.ellipsisMask.size(); i++) {
            ellipsisMaskCounter += attrs.ellipsisMask[i];
            ellipsisPos1 = attrs.ellipsisMask[i] == 1 && ellipsisPos1 == -1 ? i : ellipsisPos1;
        }
        if (ellipsisMaskCounter > 1)
            THROW_ERROR << "has incorrect 'Ellipsis_mask'. Only one non-zero bit is allowed";

        int newAxis = std::accumulate(attrs.newAxisMask.begin(), attrs.newAxisMask.end(), 0);
        int shrinkAxis = std::accumulate(attrs.shrinkAxisMask.begin(), attrs.shrinkAxisMask.end(), 0);
        attrs.equalDims = newAxis == 0 && shrinkAxis == 0;
    } else {
        attrs.equalDims = true;
    }

    auto fillingInParameters = [&](std::vector<int> &parameter, const size_t type, const size_t size, const int value) {
        const auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgesAtPort(type)[0]->getParent());
        if (!constNode) {
            THROW_ERROR << "can't cast node on " << type << " port to Input";
        }
        auto blob = constNode->getMemoryPtr();
        if (blob->GetDataType() != dnnl::memory::data_type::s32)
            THROW_ERROR << "supports only parameters input with precision I32";
        const int *ptr = static_cast<const int*>(blob->GetPtr());
        parameter.assign(ptr, ptr + size);

        if (type != AXES_ID && ellipsisMaskCounter == 0 && size < nDims) {
            for (size_t i = size; i < nDims; i++) parameter.push_back(value);
        }
    };

    if (attrs.beginDims.size())
        fillingInParameters(attrs.begin, BEGIN_ID, attrs.beginDims[0], 0);
    if (attrs.endDims.size())
        fillingInParameters(attrs.end, END_ID, attrs.endDims[0], 0);
    if (attrs.strideDims.size())
        fillingInParameters(attrs.stride, STRIDE_ID, attrs.strideDims[0], 1);
    if (attrs.axesDims.size()) {
        fillingInParameters(attrs.axes, AXES_ID, attrs.axesDims[0], 0);
        std::vector<int> beginTmp(outputRank, 0);
        std::vector<int> endTmp(outputRank, -1);
        std::vector<int> strideTmp(outputRank, 1);
        size_t i = 0lu;
        for (auto& a : attrs.axes) {
            if (a < 0)
                a += outputRank;
            beginTmp[a] = attrs.begin[i];
            endTmp[a] = attrs.end[i];
            strideTmp[a] = attrs.stride[i++];
            attrs.beginMask[a] = 1;
            attrs.endMask[a] = 1;
        }
        attrs.begin = beginTmp;
        attrs.end = endTmp;
        attrs.stride = strideTmp;
    }

    if (inputRank > 3 && attrs.equalDims && ellipsisMaskCounter == 1)
        addHiddenDims(inputRank, ellipsisPos1);
}


void StridedSlice::addHiddenDims(const size_t nSrcDims, int ellipsisPos1) {
    // all masks and input parameters are for planar layouts. So if we use blocked or per channel layout and
    // there is ellipsis should to add default values in hidden dimensions to know real order of mask or parameter values
    size_t afterDims =  attrs.begin.size() - ellipsisPos1 - 1;
    size_t ellipsisPos2 = nSrcDims - afterDims - 1;

    auto addHiddenDims = [&](std::vector<int>& data, const int bit = 0) {
        std::vector<int> temp;
        for (size_t i = 0; i < ellipsisPos1; i++)
            temp.push_back(data[i]);
        for (size_t i = ellipsisPos1; i < ellipsisPos2 + 1; i++)
            temp.push_back(bit);
        for (size_t i = 1; i < nSrcDims - ellipsisPos2; i++)
            temp.push_back(data[i + ellipsisPos1]);
        data = temp;
    };

    addHiddenDims(attrs.begin);
    addHiddenDims(attrs.end);
    addHiddenDims(attrs.stride, 1);
    addHiddenDims(attrs.beginMask);
    addHiddenDims(attrs.endMask);
    addHiddenDims(attrs.ellipsisMask);
    addHiddenDims(attrs.newAxisMask);
    addHiddenDims(attrs.shrinkAxisMask);
}

void StridedSlice::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const InferenceEngine::Precision dataPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    const InferenceEngine::Precision iPrecision = Precision::I32;
    attrs.dataSize = dataPrecision.size();

    const size_t nDims = getInputShapeAtPort(DATA_ID).getRank();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(getParentEdges().size());
    config.inConfs[DATA_ID].inPlace(-1);
    config.inConfs[BEGIN_ID].inPlace(-1);
    config.inConfs[END_ID].inPlace(-1);
    config.inConfs[DATA_ID].constant(isConstantInput[DATA_ID]);
    config.inConfs[BEGIN_ID].constant(isConstantInput[BEGIN_ID]);
    config.inConfs[END_ID].constant(isConstantInput[END_ID]);
    if (isStrideSpecified) {
        config.inConfs[STRIDE_ID].inPlace(-1);
        config.inConfs[STRIDE_ID].constant(isConstantInput[STRIDE_ID]);
    }
    if (isAxesSpecified) {
        config.inConfs[AXES_ID].inPlace(-1);
        config.inConfs[AXES_ID].constant(isConstantInput[AXES_ID]);
    }
    config.outConfs.resize(1);

    std::vector<LayoutType> supportedTypes;
    if (nDims > 2 && attrs.equalDims) {
        auto canUseBlocked = [&](const size_t blockSize) {
            const auto& srcDims = getInputShapeAtPort(DATA_ID).getDims();
            if (srcDims[1] == Shape::UNDEFINED_DIM)
                return false;
            auto channelBeginNormalized = attrs.begin[1] > 0 ? attrs.begin[1] : attrs.begin[1] + static_cast<std::int64_t>(srcDims[1]);
            return srcDims[1] % blockSize == 0 && abs(attrs.stride[1]) == 1 &&
            (channelBeginNormalized > srcDims[1] || channelBeginNormalized % blockSize == 0 || channelBeginNormalized < 0 || attrs.beginMask[1] == 0);
        };

        supportedTypes.push_back(LayoutType::nspc);
        if (canUseBlocked(8lu))
            supportedTypes.push_back(LayoutType::nCsp8c);
        if (canUseBlocked(16lu))
            supportedTypes.push_back(LayoutType::nCsp16c);
    }
    supportedTypes.push_back(LayoutType::ncsp);
    auto creators = BlockedDescCreator::getCommonCreators();
    auto range = BlockedDescCreator::makeFilteredRange(creators, nDims, supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[DATA_ID].setMemDesc(itr->second->createSharedDesc(dataPrecision, getInputShapeAtPort(DATA_ID)));
        config.inConfs[BEGIN_ID].setMemDesc(creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(BEGIN_ID)));
        config.inConfs[END_ID].setMemDesc(creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(END_ID)));
        if (isStrideSpecified)
            config.inConfs[STRIDE_ID].setMemDesc(creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(STRIDE_ID)));
        if (isAxesSpecified)
            config.inConfs[AXES_ID].setMemDesc(creators.at(LayoutType::ncsp)->createSharedDesc(iPrecision, getInputShapeAtPort(AXES_ID)));

        config.outConfs[0].setMemDesc(itr->second->createSharedDesc(dataPrecision, getOutputShapeAtPort(DATA_ID)));
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref);
    }
}

bool StridedSlice::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void StridedSlice::createPrimitive() {
    if (!isExecutable()) {
        return;
    }
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << "has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_ERROR << "has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor.";

    if (!srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp))
        orderParametersByLayouts(srcMemPtr);

    if (inputShapesDefined()) {
        prepareParams();
        updateLastInputDims();
    }
}

void StridedSlice::orderParametersByLayouts(const MemoryPtr& srcMemPtr) {
    size_t blk = 1;
    bool isBlockedLayout = false;
    if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        isBlockedLayout = true;
        blk = 16;
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c)) {
        isBlockedLayout = true;
        blk = 8;
    }
    const bool isPerChannelLayout = srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc);
    auto srcOrder = srcMemPtr->GetDescWithType<BlockedMemoryDesc>()->getOrder();

    if (isBlockedLayout) {
        attrs.begin[1] = attrs.begin[1] / blk;
        attrs.end[1] = ceil(attrs.end[1] / static_cast<float>(blk));
        attrs.begin.push_back(0);
        attrs.end.push_back(0);
        attrs.stride.push_back(1);
        attrs.beginMask.push_back(0);
        attrs.endMask.push_back(0);
        attrs.ellipsisMask.push_back(0);
        attrs.newAxisMask.push_back(0);
        attrs.shrinkAxisMask.push_back(0);
    } else if (isPerChannelLayout) {
        auto sortByOrder = [&](std::vector<int>& data) {
            std::vector<int> temp(srcOrder.size());
            for (size_t i = 0; i < srcOrder.size(); i++)
                temp[i] = data[srcOrder[i]];
            data = temp;
        };

        sortByOrder(attrs.begin);
        sortByOrder(attrs.end);
        sortByOrder(attrs.stride);
        sortByOrder(attrs.beginMask);
        sortByOrder(attrs.endMask);
        if (isStridedSliceOp) {
            sortByOrder(attrs.ellipsisMask);
            sortByOrder(attrs.newAxisMask);
            sortByOrder(attrs.shrinkAxisMask);
        }
    }
}

void StridedSlice::prepareParams() {
    execPtr = std::make_shared<StridedSliceExecutor>(attrs,
                                                     getParentEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims(),
                                                     getChildEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims());
}

StridedSlice::StridedSliceExecutor::StridedSliceExecutor(const StridedSliceAttributes& attrs,
                                                                   const VectorDims& srcBlockedDims,
                                                                   const VectorDims& dstBlockedDims) {
    StridedSliceParams params;
    params.srcBlockedDims = srcBlockedDims;
    params.dstBlockedDims = dstBlockedDims;
    params.attrs = attrs;

    size_t realNDims = params.dstBlockedDims.size();
    dimsNormalization(params);
    dimsGluing(params, realNDims);
    indicesCalculation(params);
}

void StridedSlice::StridedSliceExecutor::dimsNormalization(StridedSliceParams& params) {
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

    VectorDims newSrcDims, newDstDims;
    std::vector<int> beginTemp;
    std::vector<int> endTemp;
    std::vector<int> strideTemp;
    size_t srcIdx = 0;
    for (size_t axis = 0; axis < params.attrs.begin.size(); ++axis) {
        if (params.attrs.ellipsisMask[axis] == 1) {
            int nNewAxisAfterEllipses = 0;
            int nSrcAxisBeforeEllipses = 0;
            for (size_t i = 0; i < axis; ++i) {
                if (params.attrs.newAxisMask[i] != 1)
                    nSrcAxisBeforeEllipses++;
            }
            for (size_t i = axis + 1; i < params.attrs.begin.size(); ++i) {
                if (params.attrs.newAxisMask[i] == 1)
                    nNewAxisAfterEllipses++;
            }

            size_t nSrcAxisAfterEllipses = (params.attrs.begin.size() - axis - nNewAxisAfterEllipses - 1);
            size_t nHiddenDims = params.srcBlockedDims.size() - nSrcAxisAfterEllipses - nSrcAxisBeforeEllipses;
            for (size_t i = 0; i < nHiddenDims; ++i) {
                newSrcDims.push_back(params.srcBlockedDims[srcIdx]);
                newDstDims.push_back(params.srcBlockedDims[srcIdx]);
                beginTemp.push_back(0);
                endTemp.push_back(params.srcBlockedDims[srcIdx] - 1);
                strideTemp.push_back(1);

                srcIdx++;
            }
        } else {
            if (params.attrs.newAxisMask[axis] == 1) {
                beginTemp.push_back(0);
                endTemp.push_back(0);
                strideTemp.push_back(1);
                newSrcDims.push_back(1);
                newDstDims.push_back(1);
            } else if (params.attrs.shrinkAxisMask[axis] == 1) {
                int b = params.attrs.beginMask[axis] == 1 ? params.attrs.begin[axis] : 0;
                correcting(b, params.srcBlockedDims[srcIdx]);
                clipping(b, 0, params.srcBlockedDims[srcIdx]);
                beginTemp.push_back(b);
                endTemp.push_back(b);
                strideTemp.push_back(1);
                newSrcDims.push_back(params.srcBlockedDims[srcIdx]);
                newDstDims.push_back(1);

                srcIdx++;
            } else {
                int b = params.attrs.beginMask[axis] == 1 ? params.attrs.begin[axis] : (params.attrs.stride[axis] > 0 ? 0 : -1);
                correcting(b, params.srcBlockedDims[srcIdx]);
                clipping(b, 0, params.srcBlockedDims[srcIdx]);

                int e = params.attrs.endMask[axis] == 1 ? (params.attrs.stride[axis] > 0 ? params.attrs.end[axis] - 1 : params.attrs.end[axis] + 1) :
                        (params.attrs.stride[axis] > 0 ? -1 : 0);
                correcting(e, params.srcBlockedDims[srcIdx]);
                clipping(e, 0, params.srcBlockedDims[srcIdx]);

                beginTemp.push_back(b);
                endTemp.push_back(e);
                strideTemp.push_back(params.attrs.stride[axis]);
                newSrcDims.push_back(params.srcBlockedDims[srcIdx]);
                newDstDims.push_back(ceil(static_cast<float>(abs(e - b) + 1) / static_cast<float>(abs(strideTemp.back()))));

                srcIdx++;
            }
        }
    }

    params.attrs.begin = beginTemp;
    params.attrs.end = endTemp;
    params.attrs.stride = strideTemp;

    params.dstBlockedDims = newDstDims;
    params.srcBlockedDims = newSrcDims;
    params.dstStrides.resize(newDstDims.size());
    params.srcStrides.resize(newSrcDims.size());
    params.dstStrides[params.dstStrides.size() - 1] = params.srcStrides[params.srcStrides.size() - 1] = 1;
    for (int i = newDstDims.size() - 2; i >= 0; --i) {
        params.dstStrides[i] = params.dstStrides[i + 1] * params.dstBlockedDims[i + 1];
        params.srcStrides[i] = params.srcStrides[i + 1] * params.srcBlockedDims[i + 1];
    }
}

void StridedSlice::StridedSliceExecutor::dimsGluing(StridedSliceParams& params, const size_t realNDims) {
    // gluing of dimensions if there aren't begin, end and stride != 1 on this axis
    // example: before gluing srcDims = [5, 6, 8, 3, 2], begin = [1, 0, 0, 0, 0], stride = [1, 1, 2, 1, 1], dstDims = [4, 6, 4, 3, 2]
    //          after gluing  srcDims = [30, 8, 6],      begin = [6, 0, 0],       stride = [1, 2, 1],       dstDims = [24, 4, 6]

    std::pair<size_t, size_t> secondDim = { 0, params.attrs.begin.size() };
    VectorDims indexes(1, 0);
    for (int idx = 0; idx < params.attrs.begin.size(); idx++) {
        if (params.attrs.begin[idx] != 0 || params.attrs.end[idx] != params.srcBlockedDims[idx] - 1 || params.attrs.stride[idx] != 1) {
            indexes.push_back(std::max(idx - 1, 0));
            indexes.push_back(params.attrs.stride[idx] == 1 ? idx : idx + 1);

            if (idx != 0 && secondDim.first == 0)
                secondDim.first = idx;
            else if (idx != 0 && secondDim.second == params.attrs.begin.size())
                secondDim.second = idx;
        }
    }

    if (indexes.back() < 2) {
        indexes[indexes.size() - 1] = 1;
        secondDim.first = 1;
    }

    const VectorDims srcBlockedDimsBefore = params.srcBlockedDims;
    const VectorDims dstBlockedDimsBefore = params.dstBlockedDims;
    const size_t nGluingLastDims = params.dstStrides[std::max(static_cast<int>(indexes.back() - 1), 0)];
    const bool vLastDim = indexes.back() < params.attrs.begin.size();
    indexes[indexes.size() - 1] = vLastDim ? indexes.back() : params.attrs.begin.size() - 1;
    indexes.push_back(params.attrs.begin.size() - 1);

    for (int idx = indexes.size() - 1; idx >= 0; idx -= 2) {
        if (indexes[idx - 1] < indexes[idx]) {
            for (size_t jdx = indexes[idx]; jdx > indexes[idx - 1]; --jdx) {
                params.dstBlockedDims[indexes[idx - 1]] *= params.dstBlockedDims[jdx];
                params.srcBlockedDims[indexes[idx - 1]] *= params.srcBlockedDims[jdx];
                params.dstStrides[indexes[idx - 1]] /= params.dstBlockedDims[jdx];
                params.srcStrides[indexes[idx - 1]] /= params.srcBlockedDims[jdx];

                params.attrs.begin[indexes[idx - 1]] *= params.dstBlockedDims[jdx];
            }
            const size_t beginShift = indexes[idx - 1] + 1;
            const size_t endShift = indexes[idx] + 1;

            params.dstBlockedDims.erase(params.dstBlockedDims.begin() + beginShift, params.dstBlockedDims.begin() + endShift);
            params.srcBlockedDims.erase(params.srcBlockedDims.begin() + beginShift, params.srcBlockedDims.begin() + endShift);
            params.dstStrides.erase(params.dstStrides.begin() + beginShift, params.dstStrides.begin() + endShift);
            params.srcStrides.erase(params.srcStrides.begin() + beginShift, params.srcStrides.begin() + endShift);

            params.attrs.begin.erase(params.attrs.begin.begin() + beginShift, params.attrs.begin.begin() + endShift);
            params.attrs.stride.erase(params.attrs.stride.begin() + beginShift, params.attrs.stride.begin() + endShift);
        }
    }

    workAmount = params.dstBlockedDims[0] * params.dstStrides[0] / nGluingLastDims;
    lastDstDim = nGluingLastDims * params.attrs.dataSize;
    params.nDimsForWork = params.dstBlockedDims.size() - static_cast<size_t>(vLastDim);

    if (params.nDimsForWork == 1 && realNDims > 2) {
        const size_t realSrcDim = srcBlockedDimsBefore[secondDim.first];
        const size_t realDstDim = dstBlockedDimsBefore[secondDim.first];

        params.dstStrides.insert(params.dstStrides.begin() + 1, params.dstStrides[0] / realDstDim);
        params.srcStrides.insert(params.srcStrides.begin() + 1, params.srcStrides[0] / realSrcDim);

        for (size_t idx = secondDim.first + 1; idx < secondDim.second; idx++)
            params.attrs.begin[1] /= dstBlockedDimsBefore[idx];

        const size_t maxThreads = parallel_get_max_threads();
        if (params.dstBlockedDims[0] < maxThreads) {
            params.dstBlockedDims[1] /= realDstDim;
            params.srcBlockedDims[1] /= realSrcDim;
            params.dstBlockedDims.insert(params.dstBlockedDims.begin() + 1, realDstDim);
            params.srcBlockedDims.insert(params.srcBlockedDims.begin() + 1, realSrcDim);
        }

        if (params.dstBlockedDims.size() > 2)
            lastDstDim /= dstBlockedDimsBefore[secondDim.first];
    }

    // some parameter calculations for common execution
    params.isOptimized = params.nDimsForWork == 1 && params.dstBlockedDims.size() > 1;
    if (params.isOptimized) {
        if (params.dstBlockedDims.size() == 2)
            params.dstBlockedDims[1] = 1;

        workAmount = params.dstBlockedDims[0] * params.dstBlockedDims[1];
        srcShift = (params.attrs.begin[0] * params.srcStrides[0] + params.attrs.begin[1] * params.srcStrides[1]) * params.attrs.dataSize;
    } else {
        srcShift = params.attrs.stride.back() == 1 && params.attrs.stride.size() > 1 ?
                          params.attrs.begin[params.nDimsForWork] * params.srcStrides[params.nDimsForWork] * params.attrs.dataSize : 0;
    }
}

void StridedSlice::StridedSliceExecutor::indicesCalculation(const StridedSliceParams& params) {
    // indices calculation before execution for the best performance
    srcIndices.resize(workAmount, 0);
    dstIndices.resize(workAmount, 0);

    // should choose more optimal thread count
    const size_t nthr = parallel_get_max_threads();
    nThreads = nthr > workAmount ? workAmount : nthr;

    if (params.isOptimized) {
        indicesCalculationForOptimized(params);
        return;
    }

    auto getSrcIdx = [&](const VectorDims& indexes){
        size_t srcIdx = 0;
        for (int i = 0; i < params.nDimsForWork; ++i)
            srcIdx += (params.attrs.begin[i] + indexes[i] * params.attrs.stride[i]) * params.srcStrides[i];
        return srcIdx * params.attrs.dataSize;
    };

    parallel_nt(nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims coords(params.nDimsForWork, 0);
        splitter(workAmount, nthr, ithr, start, end);
        parallel_init(start, params.nDimsForWork, params.dstBlockedDims, coords);

        size_t srcIdx = getSrcIdx(coords);
        for (size_t j = start; j < end; ++j) {
            dstIndices[j] = j * lastDstDim;
            srcIndices[j] = srcIdx;

            bool out = false;
            for (int k = params.nDimsForWork - 1; k >= 0; k--) {
                coords[k]++;
                if (coords[k] < params.dstBlockedDims[k]) {
                    srcIdx += params.attrs.stride[k] * params.srcStrides[k] * params.attrs.dataSize;
                    break;
                }

                coords[k] = 0;
                out = true;
            }

            if (out)
                srcIdx = getSrcIdx(coords);
        }
    });
}

void StridedSlice::StridedSliceExecutor::indicesCalculationForOptimized(const StridedSliceParams& params) {
    const size_t dstIdx0 = params.dstStrides[0] * params.attrs.dataSize;
    const size_t dstIdx1 = params.dstStrides[1] * params.attrs.dataSize;
    const size_t srcIdx0 = params.attrs.stride[0] * params.srcStrides[0] * params.attrs.dataSize;
    const size_t srcIdx1 = params.attrs.stride[1] * params.srcStrides[1] * params.attrs.dataSize;

    for (size_t i0 = 0; i0 < params.dstBlockedDims[0]; i0++) {
        const size_t idx = i0 * params.dstBlockedDims[1];

        dstIndices[idx] = i0 * dstIdx0;
        srcIndices[idx] = i0 * srcIdx0;

        for (size_t i1 = 1; i1 < params.dstBlockedDims[1]; i1++) {
            dstIndices[idx + i1] = dstIndices[idx] + i1 * dstIdx1;
            srcIndices[idx + i1] = srcIndices[idx] + i1 * srcIdx1;
        }
    }
}

void StridedSlice::StridedSliceExecutor::exec(const uint8_t* srcData, uint8_t* dstData) {
    const uint8_t* srcShiftedData = srcData + srcShift;
    parallel_nt(nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(workAmount, nthr, ithr, start, end);

        for (size_t iwork = start; iwork < end; ++iwork)
            cpu_memcpy(&dstData[dstIndices[iwork]], &srcShiftedData[srcIndices[iwork]], lastDstDim);
    });
}

void StridedSlice::execute(dnnl::stream strm) {
    if (!execPtr)
        THROW_ERROR << "doesn't have compiled executor!";
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(0)->getMemory().GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetPtr());
    execPtr->exec(srcData, dstData);
}

void StridedSlice::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool StridedSlice::created() const {
    return getType() == Type::StridedSlice;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
