// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pad_node.h"
#include <string>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include <ngraph/opsets/opset1.hpp>
#include <cpu_memory_desc_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNPadNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto pad = std::dynamic_pointer_cast<const ngraph::opset1::Pad>(op);
        if (!pad) {
            errorMessage = "Only opset1 Pad operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(pad->get_input_node_shared_ptr(PADS_BEGIN_ID)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(pad->get_input_node_shared_ptr(PADS_END_ID)) == nullptr ||
            (pad->get_pad_mode() == ngraph::op::PadMode::CONSTANT && pad->get_input_size() == 4 &&
                std::dynamic_pointer_cast<const ngraph::opset1::Constant>(pad->get_input_node_shared_ptr(PAD_VALUE_ID)) == nullptr)) {
            errorMessage = "Only Constant operation on 'pads_begin', 'pads_end', 'pad_value' inpus is supported";
            return false;
        }
        const auto pad_mode = pad->get_pad_mode();
        if (pad_mode != ngraph::op::PadMode::CONSTANT && pad_mode != ngraph::op::PadMode::EDGE && pad_mode != ngraph::op::PadMode::REFLECT &&
                pad_mode != ngraph::op::PadMode::SYMMETRIC) {
            errorMessage = "Has unsupported pad_mode: " + ngraph::as_string(pad_mode);
            return false;
        }
        const auto pb = pad->get_pads_begin();
        const auto pe = pad->get_pads_end();
        if (std::count_if(pb.begin(), pb.end(), [](ptrdiff_t x) { return x < 0; }) != 0 ||
                std::count_if(pe.begin(), pe.end(), [](ptrdiff_t x) { return x < 0; }) != 0) {
            errorMessage = "Doesn't support 'pads_begin' or 'pads_end' negative value";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNPadNode::MKLDNNPadNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Pad node with name '" + op->get_friendly_name() + "'";
        const auto pad = std::dynamic_pointer_cast<const ngraph::opset1::Pad>(op);

        const auto pb = pad->get_pads_begin();
        const auto pe = pad->get_pads_end();
        for (size_t i = 0; i < pb.size(); i++)
            padsBegin.push_back(static_cast<unsigned int>(pb[i]));
         for (size_t i = 0; i < pe.size(); i++)
            padsEnd.push_back(static_cast<unsigned int>(pe[i]));

        const auto pad_mode = pad->get_pad_mode();
        isPadValueSpecified = pad->get_input_size() == 4;
        if (pad_mode == ngraph::op::PadMode::CONSTANT) {
            padMode = CONSTANT;
            if (isPadValueSpecified) {
                if (!ngraph::is_scalar(pad->get_input_shape(PAD_VALUE_ID)))
                    IE_THROW() << errorPrefix << " has non scalar 'pad_value' input";
                padValue = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(pad->get_input_node_shared_ptr(PAD_VALUE_ID))->cast_vector<float>()[0];
            }
        } else if (pad_mode == ngraph::op::PadMode::EDGE) {
            padMode = EDGE;
        } else if (pad_mode == ngraph::op::PadMode::REFLECT) {
            padMode = REFLECT;
        } else if (pad_mode == ngraph::op::PadMode::SYMMETRIC) {
            padMode = SYMMETRIC;
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNPadNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "Incorrect number of output edges";

    const SizeVector srcDims = getParentEdgeAt(DATA_ID)->getShape().getStaticDims();
    const SizeVector dstDims = getChildEdgeAt(DATA_ID)->getShape().getStaticDims();
    if (srcDims.size() != dstDims.size() || padsBegin.size() != srcDims.size() || padsEnd.size() != srcDims.size())
        IE_THROW() << errorPrefix << " has incorrect number of input/output dimensions!";

    if (padMode == REFLECT) {
        for (size_t i = 0; i < srcDims.size(); i++) {
            if ((srcDims[i] - 1) < padsBegin[i] || (srcDims[i] - 1) < padsEnd[i])
                IE_THROW() << errorPrefix << " has incorrect padsBegin or padsEnd for 'reflect' pad mode";
        }
    } else if (padMode == SYMMETRIC) {
        for (size_t i = 0; i < srcDims.size(); i++) {
            if (srcDims[i] < padsBegin[i] || srcDims[i] < padsEnd[i])
                IE_THROW() << errorPrefix << " has incorrect padsBegin or padsEnd for 'symmetric' pad mode";
        }
    }
}

void MKLDNNPadNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<InferenceEngine::Precision> supportedPrecisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32,
                                                                   InferenceEngine::Precision::BF16, InferenceEngine::Precision::I8,
                                                                   InferenceEngine::Precision::U8};
    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(DATA_ID);
    if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), precision) == supportedPrecisions.end())
        precision = precision.is_float() ? InferenceEngine::Precision::FP32 : InferenceEngine::Precision::I32;
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(DATA_ID)->getShape().getStaticDims();
    int numOfDims = srcDims.size();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(isPadValueSpecified ? 4 : 3);
    config.outConfs.resize(1);

    auto pushSupportedPrimitiveDescriptor = [&](memory::format_tag memoryFormat) {
        config.inConfs[0].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(DATA_ID)->getShape().getStaticMklDims(), dataType, memoryFormat);
        config.inConfs[1].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(PADS_BEGIN_ID)->getShape().getStaticMklDims(), memory::data_type::s32,
                                                               memory::format_tag::x);
        config.inConfs[2].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(PADS_END_ID)->getShape().getStaticMklDims(), memory::data_type::s32,
                                                               memory::format_tag::x);
        if (isPadValueSpecified)
            config.inConfs[3].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(PAD_VALUE_ID)->getShape().getStaticMklDims(), memory::data_type::f32,
                                                                   memory::format_tag::x);
        config.outConfs[0].desc = make_unique<MKLDNNMemoryDesc>(getChildEdgeAt(DATA_ID)->getShape().getStaticMklDims(), dataType, memoryFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    };

    if (numOfDims == 4)
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nhwc);
    else if (numOfDims == 5)
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::ndhwc);

    pushSupportedPrimitiveDescriptor(MKLDNNMemory::GetPlainFormatByRank(getParentEdgeAt(0)->getShape().getRank()));

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
        IE_THROW() << "Destination memory for Pad " << getName() << " didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory for Pad " << getName() << " didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor for Pad " << getName() << " is not set.";

    params.sizeData = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision().size();

    const auto inBlkDesc = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.srcDims = inBlkDesc.getBlockDims();
    params.dstDims = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>().getBlockDims();

    size_t nDims = params.srcDims.size();
    params.srcStrides.resize(nDims, 1);
    params.dstStrides.resize(nDims, 1);
    for (int i = nDims - 2; i >= 0; i--) {
        params.srcStrides[i] = params.srcStrides[i + 1] * params.srcDims[i + 1];
        params.dstStrides[i] = params.dstStrides[i + 1] * params.dstDims[i + 1];
    }

    if (getParentEdgeAt(0)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nCsp16c) ||
            getParentEdgeAt(0)->getMemory().GetDesc().checkGeneralLayout(GeneralLayout::nCsp8c)) {
        padsBegin[1] /= params.srcDims[params.srcDims.size() - 1];
        padsEnd[1] /= params.srcDims[params.srcDims.size() - 1];
        padsBegin.push_back(0);
        padsEnd.push_back(0);
    } else {
        auto order = inBlkDesc.getOrder();
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

    params.lastDstDim = params.dstStrides[std::max(endIdx - 1, 0)];
    params.nDimsForWork = endIdx - std::max(beginIdx, 0);
    params.nThreads = params.nDimsForWork > 0 ? 0 : 1;
    params.workAmount = params.nDimsForWork > 0 ? params.dstDims[0] : 1lu;
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

    params.workAmount = params.workAmount * params.dstStrides[0] / params.lastDstDim;
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
    for (int j = nDims - 1; j >= 0; --j) {
        ++indexes[j];
        if (indexes[j] < dims[j])
            break;
        else
            indexes[j] = 0;
    }
}

void MKLDNNPadNode::padConstant() {
    if (padValue == 0) {
        padConstantZero();
        return;
    }

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU Pad node with name '" << getName() << "' doesn't have primitive descriptors.";
    InferenceEngine::Precision precision = selectedPrimitiveDescriptor->getConfig().inConfs[0].desc->getPrecision();
    OV_SWITCH(MKLDNNPlugin, PadConstantEmitter, this, precision,
              OV_CASE(InferenceEngine::Precision::FP32, float),
              OV_CASE(InferenceEngine::Precision::I32, int32_t),
              OV_CASE(InferenceEngine::Precision::BF16, bfloat16_t),
              OV_CASE(InferenceEngine::Precision::I8, int8_t),
              OV_CASE(InferenceEngine::Precision::U8, uint8_t));
}

template<typename T>
void MKLDNNPadNode::padConstantCommon() {
    const T* srcData = reinterpret_cast<const T*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    T* dstData = reinterpret_cast<T*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const T value = static_cast<T>(padValue);

    const size_t beginShift = padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;
    const size_t endShift = padsEnd[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
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

            std::fill_n(&dstData[dstIdx], beginShift, value);
            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize * params.sizeData);
            std::fill_n(&dstData[dstIdx + beginShift + copySize], endShift, value);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::padConstantZero() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const size_t beginShift = padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;
    const size_t endShift = padsEnd[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
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

            memset(&dstData[dstIdx], 0, beginShift);
            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize);
            memset(&dstData[dstIdx + beginShift + copySize], 0, endShift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::padEdge() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const size_t beginShift = padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
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

            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize);

            for (size_t i = 0; i < padsEnd[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + beginShift + copySize + i * params.shift],
                           &srcData[srcIdx + (params.srcDims[params.nDimsForWork] - 1) * params.shift], params.shift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::padReflectOrSymmetric(const bool isSymmetric) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    size_t shift = isSymmetric ? 1 : 0;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
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
