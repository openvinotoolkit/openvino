// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_strided_slice_node.h"

#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

#include "ie_parallel.hpp"
#include <legacy/ie_layers.h>
#include "common/dnnl_thread.hpp"
#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"
#include "common/dnnl_thread.hpp"

#include <string>
#include <algorithm>

#define THROW_ERROR THROW_IE_EXCEPTION << "StridedSlice layer with name '" << getName() << "' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

static inline size_t parallel_init(size_t start, size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

MKLDNNStridedSliceNode::MKLDNNStridedSliceNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNStridedSliceNode::getSupportedDescriptors() {
    auto stridedSliceLayer = getCnnLayer();

    if (stridedSliceLayer == nullptr)
        THROW_ERROR << "cannot convert from CNN layer";

    auto inData = stridedSliceLayer->insData[DATA_ID].lock();
    auto beginData = stridedSliceLayer->insData[BEGIN_ID].lock();
    auto endData = stridedSliceLayer->insData[END_ID].lock();
    if (!inData || !beginData || !endData)
        THROW_ERROR << "has nullable input data";

    const SizeVector srcDims = inData->getTensorDesc().getDims();
    const SizeVector dstDims = stridedSliceLayer->outData[0]->getTensorDesc().getDims();

    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        THROW_ERROR << "has incorrect number of input edges";
    if (!getChildEdges().size())
        THROW_ERROR << "has incorrect number of output edges";

    SizeVector beginDims = beginData->getTensorDesc().getDims();
    if (beginDims.size() != 1)
        THROW_ERROR << " should have begin vector with 1 dimension";

    SizeVector endDims = endData->getTensorDesc().getDims();
    if (endDims.size() > 1)
        THROW_ERROR << "should have end vector with 1 dimension";
    if (beginDims[0] != endDims[0])
        THROW_ERROR << "should have begin vector with size equal to end vector size";

    SizeVector strideDims;
    if (stridedSliceLayer->insData.size() > STRIDE_ID) {
        auto strideData = stridedSliceLayer->insData[STRIDE_ID].lock();
        if (!strideData)
            THROW_ERROR << "has nullable input data";

        strideDims = strideData->getTensorDesc().getDims();
        if (strideDims.size() > 1)
            THROW_ERROR << "should have stride vector with 1 dimension";
        if (beginDims[0] != strideDims[0])
            THROW_ERROR << "should have stride vector with size equal to begin vector size";
    }

    auto createMask = [&](const char* maskName, std::vector<int>& mask, const int bit) {
        std::string::size_type i;
        std::string maskStr = stridedSliceLayer->GetParamAsString(maskName, "");
        for (i = 0; i < maskStr.size(); ++i) {
            if (maskStr[i] == '1')
                mask.push_back(1);
            else if (maskStr[i] == '0')
                mask.push_back(0);
            else if (maskStr[i] != ',')
                THROW_ERROR << "has incorrect '" << maskName << "'. Mask values should be 0 or 1";
        }
        if (strcmp(maskName, "ellipsis_mask") != 0 || i == 0) {
            for (; i < srcDims.size(); ++i) mask.push_back(bit);
        }
    };

    createMask("begin_mask", beginMask, 1);
    createMask("end_mask", endMask, 1);
    createMask("new_axis_mask", newAxisMask, 0);
    createMask("shrink_axis_mask", shrinkAxisMask, 0);
    createMask("ellipsis_mask", ellipsisMask, 0);

    int ellipsisMaskCounter = 0;
    int ellipsisPos1 = -1;
    for (size_t i = 0; i < ellipsisMask.size(); i++) {
        ellipsisMaskCounter += ellipsisMask[i];
        ellipsisPos1 = ellipsisMask[i] == 1 && ellipsisPos1 == -1 ? i : ellipsisPos1;
    }
    if (ellipsisMaskCounter > 1)
        THROW_ERROR << "has incorrect 'Ellipsis_mask'. Only one non-zero bit is allowed";

    int newAxis = 0;
    for (auto na : newAxisMask)
        newAxis += na;
    size_t maxDims = srcDims.size() + newAxis;

    int shrinkAxis = 0;
    for (auto sa : shrinkAxisMask)
        shrinkAxis += sa;
    params.equalDims = srcDims.size() == maxDims && shrinkAxis == 0;

    auto fillingInParameters = [&](std::vector<int>& parameter, const size_t type, const size_t size, const int bit) {
        auto blob = getParentEdgesAtPort(type)[0]->getParent()->getCnnLayer()->blobs["custom"];
        const int* ptr = blob->cbuffer().as<const int*>() + blob->getTensorDesc().getBlockingDesc().getOffsetPadding();
        parameter.assign(ptr, ptr + size);

        if (ellipsisMaskCounter == 0 && size < dstDims.size()) {
            for (size_t i = size; i < dstDims.size(); i++)
                parameter.push_back(bit);
        }
    };

    if (beginDims.size())
        fillingInParameters(begin, BEGIN_ID, beginDims[0], 0);
    if (endDims.size())
        fillingInParameters(end, END_ID, endDims[0], -1);
    if (strideDims.size())
        fillingInParameters(stride, STRIDE_ID, strideDims[0], 1);

    if (srcDims.size() > 3 && params.equalDims && ellipsisMaskCounter == 1) {
        size_t afterDims = ellipsisMask.size() - ellipsisPos1 - 1;
        size_t ellipsisPos2 = srcDims.size() - afterDims - 1;

        auto addHiddenDims = [&](std::vector<int>& data, const int bit) {
            std::vector<int> temp;
            for (size_t i = 0; i < ellipsisPos1; i++)
                temp.push_back(data[i]);
            for (size_t i = ellipsisPos1; i < ellipsisPos2 + 1; i++)
                temp.push_back(bit);
            for (size_t i = 1; i < srcDims.size() - ellipsisPos2; i++)
                temp.push_back(data[i + ellipsisPos1]);
            data = temp;
        };

        addHiddenDims(begin, 0);
        addHiddenDims(end, 0);
        addHiddenDims(stride, 1);
        addHiddenDims(beginMask, 1);
        addHiddenDims(endMask, 1);
        addHiddenDims(ellipsisMask, 0);
        addHiddenDims(newAxisMask, 0);
        addHiddenDims(shrinkAxisMask, 0);
    }
}

void MKLDNNStridedSliceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const bool hasStrides = getParentEdges().size() > 3;
    InferenceEngine::Precision dataPrecision = getCnnLayer()->insData[DATA_ID].lock()->getPrecision();
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(dataPrecision);
    InferenceEngine::Precision beginPrecision = getCnnLayer()->insData[BEGIN_ID].lock()->getPrecision();
    auto beginDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(beginPrecision);
    InferenceEngine::Precision endPrecision = getCnnLayer()->insData[END_ID].lock()->getPrecision();
    auto endDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(endPrecision);
    InferenceEngine::Precision stridePrecision;
    if (hasStrides)
        stridePrecision = getCnnLayer()->insData[STRIDE_ID].lock()->getPrecision();

    auto srcDims = getParentEdgeAt(DATA_ID)->getDims();
    auto dstDims = getChildEdgeAt(0)->getDims();
    size_t nDims = srcDims.ndims();

    InferenceEngine::LayerConfig config;
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

    auto pushSupportedPrimitiveDescriptor = [&](const mkldnn::memory::format_tag inMemoryFormat, const mkldnn::memory::format_tag outMemoryFormat) {
        config.inConfs[DATA_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(DATA_ID)->getDims(), dataType, inMemoryFormat);
        config.inConfs[BEGIN_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(BEGIN_ID)->getDims(), beginDataType, mkldnn::memory::format_tag::x);
        config.inConfs[END_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(END_ID)->getDims(), endDataType, mkldnn::memory::format_tag::x);
        if (hasStrides)
            config.inConfs[STRIDE_ID].desc = MKLDNNMemoryDesc(getParentEdgeAt(STRIDE_ID)->getDims(),
                                                              MKLDNNExtensionUtils::IEPrecisionToDataType(stridePrecision),
                                                              mkldnn::memory::format_tag::x);

        config.outConfs[0].desc = MKLDNNMemoryDesc(dstDims, dataType, outMemoryFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, outMemoryFormat});
    };

    auto canUseBlocked = [=](const size_t blockSize) {
        return srcDims[1] % blockSize == 0 && abs(stride[1]) == 1 && (begin[1] > srcDims[1] || begin[1] % blockSize == 0);
    };

    if (nDims == 4 && params.equalDims) {
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nhwc, mkldnn::memory::format_tag::nhwc);
        if (canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw8c, mkldnn::memory::format_tag::nChw8c);
        if (canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw16c, mkldnn::memory::format_tag::nChw16c);
    } else if (nDims == 5 && params.equalDims) {
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::ndhwc, mkldnn::memory::format_tag::ndhwc);
        if (canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw8c, mkldnn::memory::format_tag::nCdhw8c);
        if (canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw16c, mkldnn::memory::format_tag::nCdhw16c);
    }
    pushSupportedPrimitiveDescriptor(MKLDNNMemory::GetPlainFormat(srcDims), MKLDNNMemory::GetPlainFormat(dstDims));
}

void MKLDNNStridedSliceNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto srcBlockingDesc = getParentEdgeAt(DATA_ID)->getDesc().getBlockingDesc();
    auto dstBlockingDesc = getChildEdgeAt(0)->getDesc().getBlockingDesc();
    auto srcOrder = srcBlockingDesc.getOrder();
    params.srcDims = srcBlockingDesc.getBlockDims();
    params.dstDims = dstBlockingDesc.getBlockDims();
    params.srcStrides = srcBlockingDesc.getStrides();
    params.dstStrides = dstBlockingDesc.getStrides();
    params.dataSize = getSelectedPrimitiveDescriptor()->getConfig().inConfs[DATA_ID].desc.getPrecision().size();

    const bool isPerChannelLayout = getParentEdgeAt(DATA_ID)->getMemory().GetDesc().isTailCFormat();
    const bool isBlockedLayout = getParentEdgeAt(DATA_ID)->getMemory().GetDesc().isBlockedCFormat();
    size_t realNDims = params.dstDims.size();

    if (isBlockedLayout) {
        const size_t blk = params.srcDims.back();
        begin[1] = begin[1] / blk;
        end[1] = ceil(end[1] / static_cast<float>(blk));
        begin.push_back(0);
        end.push_back(0);
        stride.push_back(1);
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

    auto clipping = [](int& idx, const int min, const int max) {
        idx = (idx > min) ? idx : min;
        idx = (idx < max) ? idx : (max - 1);
    };

    auto correcting = [](int& dim, const size_t shift) {
        dim = dim >= 0 ? dim : shift + dim;
    };

    // creating new src and dst dimensions of the same size using masks
    std::vector<int> beginTemp;
    std::vector<int> endTemp;
    std::vector<int> strideTemp;
    SizeVector newDstDims, newSrcDims;
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

    // gluing dimensions (reshaping)
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

        const size_t maxThreads = dnnl_get_max_threads();
        if (params.dstDims[0] < maxThreads) {
            params.dstDims[1] /= realDstDim;
            params.srcDims[1] /= realSrcDim;
            params.dstDims.insert(params.dstDims.begin() + 1, realDstDim);
            params.srcDims.insert(params.srcDims.begin() + 1, realSrcDim);
        }

        if (params.dstDims.size() > 2)
            params.lastDstDim /= newDstDims[secondDim.first];
    }

    // calculate src and dst indices
    if (params.dstDims.size() == 1 || params.nDimsForWork != 1) {
        params.nThreads = dnnl_get_max_threads();
        params.srcIndices.resize(params.workAmount, 0);
        params.dstIndices.resize(params.workAmount, 0);
        indexes.clear();

        auto getSrcIdx = [this](const SizeVector& indexes){
            size_t srcIdx = 0;
            for (int i = 0; i < params.nDimsForWork; ++i)
                srcIdx += (begin[i] + indexes[i] * stride[i]) * params.srcStrides[i];
            return srcIdx * params.dataSize;
        };

        parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            indexes.resize(params.nDimsForWork, 0);
            splitter(params.workAmount, nthr, ithr, start, end);
            parallel_init(start, params.nDimsForWork, params.dstDims, indexes);

            srcIdx = getSrcIdx(indexes);
            for (size_t j = start; j < end; j++) {
                params.dstIndices[j] = j * params.lastDstDim;
                params.srcIndices[j] = srcIdx;

                bool out = false;
                for (int k = params.nDimsForWork - 1; k >= 0; k--) {
                    indexes[k]++;
                    if (indexes[k] < params.dstDims[k]) {
                        srcIdx += stride[k] * params.srcStrides[k] * params.dataSize;
                        break;
                    } else {
                        indexes[k] = 0;
                        out = true;
                    }
                }

                if (out) {
                    srcIdx = getSrcIdx(indexes);
                    out = false;
                }
            }
        });
    }
}

void MKLDNNStridedSliceNode::execute(mkldnn::stream strm) {
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
