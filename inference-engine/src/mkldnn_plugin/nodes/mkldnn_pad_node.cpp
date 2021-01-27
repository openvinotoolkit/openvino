// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pad_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPadNode::MKLDNNPadNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNPadNode::getSupportedDescriptors() {
    auto* padLayer = dynamic_cast<PadLayer*>(getCnnLayer().get());
    if (padLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Pad layer.";

    padsBegin = padLayer->GetParamAsUInts("pads_begin");
    padsEnd = padLayer->GetParamAsUInts("pads_end");

    SizeVector srcDims = padLayer->insData[0].lock()->getTensorDesc().getDims();
    SizeVector dstDims = padLayer->outData[0]->getTensorDesc().getDims();
    if (srcDims.size() != dstDims.size() || padsBegin.size() != srcDims.size() || padsEnd.size() != srcDims.size())
        THROW_IE_EXCEPTION << padLayer->name << " Incorrect number of input/output dimensions!";

    std::string pad_mode = padLayer->GetParamAsString("pad_mode");
    if (pad_mode == "constant") {
        padMode = CONSTANT;
        padValue = padLayer->GetParamAsFloat("pad_value", 0.f);
    } else if (pad_mode == "edge") {
        padMode = EDGE;
    } else if (pad_mode == "reflect") {
        padMode = REFLECT;
        for (size_t i = 0; i < srcDims.size(); i++) {
            if ((srcDims[i] - 1) < padsBegin[i] || (srcDims[i] - 1) < padsEnd[i])
                THROW_IE_EXCEPTION << padLayer->name << " Incorrect padsBegin or padsEnd for 'reflect' pad mode";
        }
    } else if (pad_mode == "symmetric") {
        padMode = SYMMETRIC;
        for (size_t i = 0; i < srcDims.size(); i++) {
            if (srcDims[i] < padsBegin[i] || srcDims[i] < padsEnd[i])
                THROW_IE_EXCEPTION << padLayer->name << " Incorrect padsBegin or padsEnd for 'symmetric' pad mode";
        }
    } else {
        THROW_IE_EXCEPTION << padLayer->name
                           << " Incorrect pad_mode. Only constants|edge|reflect|symmetric modes are supported!";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNPadNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<InferenceEngine::Precision> supportedPrecisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32,
                                                                   InferenceEngine::Precision::BF16, InferenceEngine::Precision::I8,
                                                                   InferenceEngine::Precision::U8};
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), precision) == supportedPrecisions.end())
        precision = precision.is_float() ? InferenceEngine::Precision::FP32 : InferenceEngine::Precision::I32;
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();
    int numOfDims = srcDims.ToSizeVector().size();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    auto pushSupportedPrimitiveDescriptor = [&](memory::format_tag memoryFormat) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), dataType, memoryFormat);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), dataType, memoryFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref, memoryFormat});
    };

    if (numOfDims == 4)
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nhwc);
    else if (numOfDims == 5)
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::ndhwc);

    pushSupportedPrimitiveDescriptor(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));

    auto canUseBlocked = [=](const size_t blockSize) {
        return (padMode == CONSTANT && padsBegin[1] % blockSize == 0 && padsEnd[1] % blockSize == 0) ||
               (padMode != CONSTANT && padsBegin[1] == 0 && padsEnd[1] == 0);
    };

    if (numOfDims == 4) {
        if (srcDims[1] % 8 == 0 && canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw8c);
        if (srcDims[1] % 16 == 0 && canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw16c);
    } else if (numOfDims == 5) {
        if (srcDims[1] % 8 == 0 && canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw8c);
        if (srcDims[1] % 16 == 0 && canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw16c);
    }
}

void MKLDNNPadNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory for Pad " << getName() << " didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory for Pad " << getName() << " didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor for Pad " << getName() << " is not set.";

    params.sizeData = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();

    params.srcDims = getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getBlockDims();
    params.dstDims = getChildEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getBlockDims();

    params.srcStrides = getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getStrides();
    params.dstStrides = getChildEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getStrides();

    if (getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat()) {
        padsBegin[1] /= params.srcDims[params.srcDims.size() - 1];
        padsEnd[1] /= params.srcDims[params.srcDims.size() - 1];
        padsBegin.push_back(0);
        padsEnd.push_back(0);
    } else {
        auto order = getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getOrder();
        std::vector<unsigned int> newPadsBegin(padsBegin.size(), 0), newPadsEnd(padsEnd.size(), 0);
        for (size_t i = 0; i < padsBegin.size(); ++i) {
            newPadsBegin[i] = padsBegin[order[i]];
            newPadsEnd[i] = padsEnd[order[i]];
        }
        padsBegin = newPadsBegin;
        padsEnd = newPadsEnd;
    }

    int beginIdx = 0;
    int endIdx = padsBegin.size() - 1;

    for (int i = 0; i < padsBegin.size(); ++i) {
        if (padsBegin[i] != 0 || padsEnd[i] != 0) {
            beginIdx = i - 1;
            break;
        }
    }

    for (int i = padsBegin.size() - 1; i >= 0; --i) {
        if (padsBegin[i] != 0 || padsEnd[i] != 0) {
            endIdx = i;
            break;
        }
    }

    size_t nGluingLastDims = params.dstStrides[std::max(endIdx - 1, 0)];
    params.nDimsForWork = std::max(endIdx - std::max(beginIdx, 0), 1);
    params.workAmount = params.dstDims[0];
    for (int i = 1; i <= beginIdx; ++i) {
        params.workAmount *= params.dstDims[i];
        params.dstDims[0] *= params.dstDims[i];
        params.srcDims[0] *= params.srcDims[i];
        params.dstStrides[0] /= params.dstDims[i];
        params.srcStrides[0] /= params.srcDims[i];
    }

    if (beginIdx > 0) {
        beginIdx++;
        params.dstDims.erase(params.dstDims.begin() + 1, params.dstDims.begin() + beginIdx);
        params.srcDims.erase(params.srcDims.begin() + 1, params.srcDims.begin() + beginIdx);
        params.dstStrides.erase(params.dstStrides.begin() + 1, params.dstStrides.begin() + beginIdx);
        params.srcStrides.erase(params.srcStrides.begin() + 1, params.srcStrides.begin() + beginIdx);
        padsBegin.erase(padsBegin.begin() + 1, padsBegin.begin() + beginIdx);
        padsEnd.erase(padsEnd.begin() + 1, padsEnd.begin() + beginIdx);
    }
    params.workAmount = params.workAmount * params.dstStrides[0] / nGluingLastDims;

    params.lastDstDim = nGluingLastDims;
    params.shift = params.dstStrides[params.nDimsForWork];
    if (padMode != CONSTANT || (padMode == CONSTANT && padValue == 0)) {
        params.lastDstDim *= params.sizeData;
        params.shift *= params.sizeData;
    }

    for (size_t i = 0; i < params.srcDims.size(); ++i)
        params.srcODims.push_back(padsBegin[i] + params.srcDims[i]);

    if (padMode == REFLECT || padMode == SYMMETRIC) {
        int shift = padMode == SYMMETRIC ? 1 : 0;
        for (size_t i = 0; i < params.srcDims.size(); ++i)
            params.srcDimsForReflectOrSymmetric.push_back(params.srcDims[i] + params.srcODims[i] - 2 + shift);
    }
}

void MKLDNNPadNode::execute(mkldnn::stream strm) {
    switch (padMode) {
        case CONSTANT:
            padConstant();
            break;
        case EDGE:
            padEdge();
            break;
        case REFLECT:
            padReflectOrSymmetric();
            break;
        case SYMMETRIC:
            padReflectOrSymmetric(true);
            break;
    }
}

static inline size_t parallel_init(size_t start, size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

static inline void parallel_step(size_t nDims, const SizeVector& dims, SizeVector& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = (indexes[j] + 1) % dims[j];
        if (indexes[j] != 0)
            return;
    }
}

void MKLDNNPadNode::padConstant() {
    if (padValue == 0) {
        padConstantZero();
        return;
    }

    InferenceEngine::Precision precision = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision();
    OV_SWITCH(MKLDNNPlugin, PadConstantEmitter, this, precision,
              OV_CASE(InferenceEngine::Precision::FP32, float),
              OV_CASE(InferenceEngine::Precision::I32, int32_t),
              OV_CASE(InferenceEngine::Precision::BF16, bfloat16_t),
              OV_CASE(InferenceEngine::Precision::I8, int8_t),
              OV_CASE(InferenceEngine::Precision::U8, uint8_t));
}

template<typename T>
void MKLDNNPadNode::padConstantCommon() {
    T* srcData = reinterpret_cast<T*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    T* dstData = reinterpret_cast<T*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    T value = static_cast<T>(padValue);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t j = 0;
            for (; j < params.nDimsForWork; ++j) {
                if (indexes[j] < padsBegin[j] || indexes[j] >= params.srcODims[j])
                    break;
            }

            if (j != params.nDimsForWork) {
                std::fill_n(&dstData[dstIdx], params.lastDstDim, value);
                parallel_step(params.nDimsForWork, params.dstDims, indexes);
                continue;
            }

            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx)
                srcIdx += (indexes[idx] - padsBegin[idx]) * params.srcStrides[idx];

            std::fill_n(&dstData[dstIdx], padsBegin[params.nDimsForWork] * params.shift, value);
            cpu_memcpy(&dstData[dstIdx + padsBegin[params.nDimsForWork] * params.shift], &srcData[srcIdx],
                       params.srcDims[params.nDimsForWork] * params.shift * params.sizeData);
            std::fill_n(&dstData[dstIdx + params.srcODims[params.nDimsForWork] * params.shift],
                        padsEnd[params.nDimsForWork] * params.shift, value);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::padConstantZero() {
    uint8_t* srcData = reinterpret_cast<uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t j = 0;
            for (; j < params.nDimsForWork; ++j) {
                if (indexes[j] < padsBegin[j] || indexes[j] >= params.srcODims[j])
                    break;
            }

            if (j != params.nDimsForWork) {
                memset(&dstData[dstIdx], 0, params.lastDstDim);
                parallel_step(params.nDimsForWork, params.dstDims, indexes);
                continue;
            }

            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx)
                srcIdx += (indexes[idx] - padsBegin[idx]) * params.srcStrides[idx];
            srcIdx *= params.sizeData;

            memset(&dstData[dstIdx], 0, padsBegin[params.nDimsForWork] * params.shift);
            cpu_memcpy(&dstData[dstIdx + padsBegin[params.nDimsForWork] * params.shift], &srcData[srcIdx],
                       params.srcDims[params.nDimsForWork] * params.shift);
            memset(&dstData[dstIdx + params.srcODims[params.nDimsForWork] * params.shift], 0, padsEnd[params.nDimsForWork] * params.shift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::padEdge() {
    uint8_t* srcData = reinterpret_cast<uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx) {
                size_t shift = (indexes[idx] < padsBegin[idx]) ? 0 :
                               ((indexes[idx] >= params.srcODims[idx]) ? (params.srcDims[idx] - 1) : (indexes[idx] - padsBegin[idx]));
                srcIdx += shift * params.srcStrides[idx];
            }
            srcIdx *= params.sizeData;

            for (size_t i = 0; i < padsBegin[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + i * params.shift], &srcData[srcIdx], params.shift);

            cpu_memcpy(&dstData[dstIdx + padsBegin[params.nDimsForWork] * params.shift], &srcData[srcIdx],
                       params.srcDims[params.nDimsForWork] * params.shift);

            for (size_t i = 0; i < padsEnd[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + params.srcODims[params.nDimsForWork] * params.shift + i * params.shift],
                           &srcData[srcIdx + (params.srcDims[params.nDimsForWork] - 1) * params.shift], params.shift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::padReflectOrSymmetric(const bool isSymmetric) {
    uint8_t* srcData = reinterpret_cast<uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    size_t shift = isSymmetric ? 1 : 0;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t srcIdx = 0;
            for (size_t i = 0; i < params.nDimsForWork; ++i) {
                size_t idx = (indexes[i] < padsBegin[i]) ? (padsBegin[i] - indexes[i] - shift) :
                             ((indexes[i] >= params.srcODims[i]) ? (params.srcDimsForReflectOrSymmetric[i] - indexes[i]) : (indexes[i] - padsBegin[i]));
                srcIdx += idx * params.srcStrides[i];
            }
            srcIdx *= params.sizeData;

            for (size_t i = 0; i < padsBegin[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + i * params.shift],
                           &srcData[srcIdx + (padsBegin[params.nDimsForWork] - shift - i) * params.shift], params.shift);

            cpu_memcpy(&dstData[dstIdx + padsBegin[params.nDimsForWork] * params.shift], &srcData[srcIdx],
                       params.srcDims[params.nDimsForWork] * params.shift);

            size_t srcShift = (params.srcDimsForReflectOrSymmetric[params.nDimsForWork] - params.srcODims[params.nDimsForWork]) * params.shift;
            for (size_t i = 0; i < padsEnd[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + (params.srcODims[params.nDimsForWork] + i) * params.shift],
                           &srcData[srcIdx + srcShift - i * params.shift], params.shift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

inline void MKLDNNPadNode::getDstIdx(const InferenceEngine::SizeVector& indexes, size_t& dstIdx) const {
    for (size_t i = 0; i < params.nDimsForWork; ++i)
        dstIdx += indexes[i] * params.dstStrides[i];
    dstIdx *= (padMode == CONSTANT && padValue != 0) ? 1 : params.sizeData;
}

bool MKLDNNPadNode::created() const {
    return getType() == Pad;
}
REG_MKLDNN_PRIM_FOR(MKLDNNPadNode, Pad);
