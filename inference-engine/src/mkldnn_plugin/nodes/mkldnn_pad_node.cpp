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

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

#define THROW_ERROR IE_THROW() << "Pad layer with name '" << getName() << "' "

bool MKLDNNPadNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto pad = ov::as_type_ptr<const ngraph::opset1::Pad>(op);
        if (!pad) {
            errorMessage = "Only opset1 Pad operation is supported";
            return false;
        }

        const auto pad_mode = pad->get_pad_mode();
        if (!MKLDNNPlugin::one_of(pad_mode, ngraph::op::PadMode::CONSTANT, ngraph::op::PadMode::EDGE, ngraph::op::PadMode::REFLECT,
            ngraph::op::PadMode::SYMMETRIC)) {
            errorMessage = "Has unsupported pad_mode: " + ngraph::as_string(pad_mode);
            return false;
        }

        if (op->get_input_node_shared_ptr(PADS_BEGIN_ID)->get_type_info() != ov::op::v0::Constant::get_type_info_static() ||
            op->get_input_node_shared_ptr(PADS_END_ID)->get_type_info() != ov::op::v0::Constant::get_type_info_static() ||
            pad->get_input_size() == 4 && pad->get_pad_mode() == ngraph::op::PadMode::CONSTANT &&
            op->get_input_node_shared_ptr(PAD_VALUE_ID)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support pads_begin, pads_end, pad_value inputs for dynamic shapes.
            errorMessage = "Only Constant 'pads_begin', 'pads_end' and 'pad_value' inputs are supported.";
            return false;
        }

        const auto pb = pad->get_pads_begin();
        const auto pe = pad->get_pads_end();
        if (std::count_if(pb.begin(), pb.end(), [](ptrdiff_t x) { return x < 0; }) != 0 ||
            std::count_if(pe.begin(), pe.end(), [](ptrdiff_t x) { return x < 0; }) != 0) {
            errorMessage =  "Doesn't support 'pads_begin' or 'pads_end' with negative values";
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
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (inputShapes.size() != 3 && inputShapes.size() != 4)
        THROW_ERROR << " has incorrect number of input edges";
    if (outputShapes.size() != 1)
        THROW_ERROR << "Incorrect number of output edges";

    const int srcDimsRank = inputShapes[DATA_ID].getRank();
    const int dstDimsRank = outputShapes[DATA_ID].getRank();
    if (srcDimsRank != dstDimsRank)
        THROW_ERROR << "has incorrect number of input/output dimensions!";

    const auto pad = ov::as_type_ptr<const ngraph::opset1::Pad>(op);

    if (op->get_input_node_shared_ptr(PADS_BEGIN_ID)->get_type_info() == ov::op::v0::Constant::get_type_info_static() &&
        op->get_input_node_shared_ptr(PADS_END_ID)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
        const auto pb = pad->get_pads_begin();
        const auto pe = pad->get_pads_end();

        for (size_t i = 0; i < pb.size(); i++)
            attrs.padsBegin.push_back(static_cast<unsigned int>(pb[i]));
        for (size_t i = 0; i < pe.size(); i++)
            attrs.padsEnd.push_back(static_cast<unsigned int>(pe[i]));

        if (attrs.padsBegin.size() != srcDimsRank || attrs.padsEnd.size() != srcDimsRank)
            THROW_ERROR << "has incorrect number of input/output dimensions!";
    }

    const auto pad_mode = pad->get_pad_mode();
    isPadValueSpecified = pad->get_input_size() == 4;
    if (pad_mode == ngraph::op::PadMode::CONSTANT) {
        attrs.padMode = CONSTANT;
        if (isPadValueSpecified && op->get_input_node_shared_ptr(PAD_VALUE_ID)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
            if (!ngraph::is_scalar(pad->get_input_shape(PAD_VALUE_ID)))
                THROW_ERROR << "has non scalar 'pad_value' input";
            attrs.padValue = ov::as_type_ptr<const ngraph::opset1::Constant>(pad->get_input_node_shared_ptr(PAD_VALUE_ID))->cast_vector<float>()[0];
        }
    } else if (pad_mode == ngraph::op::PadMode::EDGE) {
        attrs.padMode = EDGE;
    } else if (pad_mode == ngraph::op::PadMode::REFLECT) {
        attrs.padMode = REFLECT;
    } else if (pad_mode == ngraph::op::PadMode::SYMMETRIC) {
        attrs.padMode = SYMMETRIC;
    } else {
        THROW_ERROR << "has unsupported pad_mode: " + ngraph::as_string(pad_mode);
    }
}

void MKLDNNPadNode::getSupportedDescriptors() {}

void MKLDNNPadNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<InferenceEngine::Precision> supportedPrecisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32,
                                                                   InferenceEngine::Precision::BF16, InferenceEngine::Precision::I8,
                                                                   InferenceEngine::Precision::U8};
    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(DATA_ID);
    if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), precision) == supportedPrecisions.end())
        precision = precision.is_float() ? InferenceEngine::Precision::FP32 : InferenceEngine::Precision::I32;

    const auto& inputDataShape = getInputShapeAtPort(DATA_ID);
    int numOfDims = inputDataShape.getRank();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(isPadValueSpecified ? 4 : 3);
    config.outConfs.resize(1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushSupportedPrimitiveDescriptor = [&](LayoutType memoryFormat) {
        config.inConfs[0].desc = creatorsMap.at(memoryFormat)->createSharedDesc(precision, getInputShapeAtPort(DATA_ID));
        config.inConfs[1].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(InferenceEngine::Precision::I32, getInputShapeAtPort(PADS_BEGIN_ID));
        config.inConfs[2].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(InferenceEngine::Precision::I32, getInputShapeAtPort(PADS_END_ID));
        if (isPadValueSpecified)
            config.inConfs[3].desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(InferenceEngine::Precision::FP32, getInputShapeAtPort(PAD_VALUE_ID));

        config.outConfs[0].desc = creatorsMap.at(memoryFormat)->createSharedDesc(precision, getOutputShapeAtPort(DATA_ID));
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    };

    if (numOfDims == 4 || numOfDims == 5)
        pushSupportedPrimitiveDescriptor(LayoutType::nspc);

    pushSupportedPrimitiveDescriptor(LayoutType::ncsp);

    auto canUseBlocked = [&](const size_t blockSize) {
        const auto& srcDims = inputDataShape.getDims();
        return srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % blockSize == 0 &&
               ((attrs.padMode == CONSTANT && attrs.padsBegin[1] % blockSize == 0 && attrs.padsEnd[1] % blockSize == 0) ||
               (attrs.padMode != CONSTANT && attrs.padsBegin[1] == 0 && attrs.padsEnd[1] == 0));
    };

    if (numOfDims == 4 || numOfDims == 5) {
        if (canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(LayoutType::nCsp8c);
        if (canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(LayoutType::nCsp16c);
    }
}

void MKLDNNPadNode::createPrimitive() {
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory for Pad " << getName() << " didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory for Pad " << getName() << " didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor for Pad " << getName() << " is not set.";

    attrs.prc = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision();
    attrs.dataSize = attrs.prc.size();

    // pads are constant, so we can calculate new collapsing pads for first target dimensions and use it for the next dimensions
    // to avoid permanent identical pad calculations
    if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        attrs.padsBegin[1] /= 16;
        attrs.padsEnd[1] /= 16;
        attrs.padsBegin.push_back(0);
        attrs.padsEnd.push_back(0);
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c)) {
        attrs.padsBegin[1] /= 8;
        attrs.padsEnd[1] /= 8;
        attrs.padsBegin.push_back(0);
        attrs.padsEnd.push_back(0);
    } else {
        auto order = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getOrder();
        std::vector<unsigned int> newPadsBegin(attrs.padsBegin.size(), 0), newPadsEnd(attrs.padsEnd.size(), 0);
        for (size_t i = 0; i < attrs.padsBegin.size(); ++i) {
            newPadsBegin[i] = attrs.padsBegin[order[i]];
            newPadsEnd[i] = attrs.padsEnd[order[i]];
        }
        attrs.padsBegin = newPadsBegin;
        attrs.padsEnd = newPadsEnd;
    }

    // collapse dimensions
    attrs.beginPadIdx = 0;
    attrs.endPadIdx = attrs.padsBegin.size() - 1;

    for (int i = 0; i < attrs.padsBegin.size(); ++i) {
        if (attrs.padsBegin[i] != 0 || attrs.padsEnd[i] != 0) {
            attrs.beginPadIdx = i - 1;
            break;
        }
    }

    for (int i = attrs.padsBegin.size() - 1; i >= 0; --i) {
        if (attrs.padsBegin[i] != 0 || attrs.padsEnd[i] != 0) {
            attrs.endPadIdx = i;
            break;
        }
    }

    if (attrs.beginPadIdx > 0) {
        attrs.padsBegin.erase(attrs.padsBegin.begin() + 1, attrs.padsBegin.begin() + attrs.beginPadIdx + 1);
        attrs.padsEnd.erase(attrs.padsEnd.begin() + 1, attrs.padsEnd.begin() + attrs.beginPadIdx + 1);
    }

    if (inputShapesDefined()) {
        prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNPadNode::prepareParams() {
    execPtr = std::make_shared<PadExecutor>(attrs, getParentEdgeAt(0)->getMemoryPtr(), getChildEdgeAt(0)->getMemoryPtr());
}

MKLDNNPadNode::PadExecutor::PadExecutor(const PadAttrs& attrs,
                                        const MKLDNNMemoryPtr& srcMemPtr,
                                        const MKLDNNMemoryPtr& dstMemPtr) {
    params.attrs = attrs;
    params.srcMemPtr = srcMemPtr;
    params.dstMemPtr = dstMemPtr;
    params.srcDims = srcMemPtr->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    params.dstDims = dstMemPtr->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();

    size_t nDims = params.srcDims.size();
    params.srcStrides.resize(nDims, 1);
    params.dstStrides.resize(nDims, 1);
    for (int i = nDims - 2; i >= 0; i--) {
        params.srcStrides[i] = params.srcStrides[i + 1] * params.srcDims[i + 1];
        params.dstStrides[i] = params.dstStrides[i + 1] * params.dstDims[i + 1];
    }

    params.lastDstDim = params.dstStrides[std::max(params.attrs.endPadIdx - 1, 0)];
    params.nDimsForWork = params.attrs.endPadIdx - std::max(params.attrs.beginPadIdx, 0);
    params.nThreads = params.nDimsForWork > 0 ? 0 : 1;
    params.workAmount = params.nDimsForWork > 0 ? params.dstDims[0] : 1lu;
    for (int i = 1; i <= params.attrs.beginPadIdx; ++i) {
        params.workAmount *= params.dstDims[i];
        params.dstDims[0] *= params.dstDims[i];
        params.srcDims[0] *= params.srcDims[i];
        params.dstStrides[0] /= params.dstDims[i];
        params.srcStrides[0] /= params.srcDims[i];
    }

    if (params.attrs.beginPadIdx > 0) {
        params.attrs.beginPadIdx++;
        params.dstDims.erase(params.dstDims.begin() + 1, params.dstDims.begin() + params.attrs.beginPadIdx);
        params.srcDims.erase(params.srcDims.begin() + 1, params.srcDims.begin() + params.attrs.beginPadIdx);
        params.dstStrides.erase(params.dstStrides.begin() + 1, params.dstStrides.begin() + params.attrs.beginPadIdx);
        params.srcStrides.erase(params.srcStrides.begin() + 1, params.srcStrides.begin() + params.attrs.beginPadIdx);
    }

    params.workAmount = params.workAmount * params.dstStrides[0] / params.lastDstDim;
    params.shift = params.dstStrides[params.nDimsForWork];
    if (params.attrs.padMode != CONSTANT || (params.attrs.padMode == CONSTANT && params.attrs.padValue == 0)) {
        params.lastDstDim *= attrs.dataSize;
        params.shift *= attrs.dataSize;
    }

    params.srcODims.clear();
    for (size_t i = 0; i < params.srcDims.size(); ++i)
        params.srcODims.push_back(params.attrs.padsBegin[i] + params.srcDims[i]);

    params.srcDimsForReflectOrSymmetric.clear();
    if (params.attrs.padMode == REFLECT || params.attrs.padMode == SYMMETRIC) {
        int shift = params.attrs.padMode == SYMMETRIC ? 1 : 0;
        for (size_t i = 0; i < params.srcDims.size(); ++i)
            params.srcDimsForReflectOrSymmetric.push_back(params.srcDims[i] + params.srcODims[i] - 2 + shift);
    }
}

void MKLDNNPadNode::PadExecutor::exec() {
    switch (params.attrs.padMode) {
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

void MKLDNNPadNode::execute(mkldnn::stream strm) {
    execPtr->exec();
}

void MKLDNNPadNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
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

void MKLDNNPadNode::PadExecutor::padConstant() {
    if (params.attrs.padValue == 0) {
        padConstantZero();
        return;
    }

    OV_SWITCH(MKLDNNPlugin, PadConstantEmitter, this, params.attrs.prc,
              OV_CASE(InferenceEngine::Precision::FP32, float),
              OV_CASE(InferenceEngine::Precision::I32, int32_t),
              OV_CASE(InferenceEngine::Precision::BF16, bfloat16_t),
              OV_CASE(InferenceEngine::Precision::I8, int8_t),
              OV_CASE(InferenceEngine::Precision::U8, uint8_t));
}

template<typename T>
void MKLDNNPadNode::PadExecutor::padConstantCommon() {
    const T* srcData = reinterpret_cast<const T*>(params.srcMemPtr->GetPtr());
    T* dstData = reinterpret_cast<T*>(params.dstMemPtr->GetPtr());
    const T value = static_cast<T>(params.attrs.padValue);

    const size_t beginShift = params.attrs.padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;
    const size_t endShift = params.attrs.padsEnd[params.nDimsForWork] * params.shift;

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
                if (indexes[j] < params.attrs.padsBegin[j] || indexes[j] >= params.srcODims[j])
                    break;
            }

            if (j != params.nDimsForWork) {
                std::fill_n(&dstData[dstIdx], params.lastDstDim, value);
                parallel_step(params.nDimsForWork, params.dstDims, indexes);
                continue;
            }

            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx)
                srcIdx += (indexes[idx] - params.attrs.padsBegin[idx]) * params.srcStrides[idx];

            std::fill_n(&dstData[dstIdx], beginShift, value);
            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize * params.attrs.dataSize);
            std::fill_n(&dstData[dstIdx + beginShift + copySize], endShift, value);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::PadExecutor::padConstantZero() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(params.srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(params.dstMemPtr->GetPtr());

    const size_t beginShift = params.attrs.padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;
    const size_t endShift = params.attrs.padsEnd[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);
        dstIdx *= params.attrs.dataSize;

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t j = 0;
            for (; j < params.nDimsForWork; ++j) {
                if (indexes[j] < params.attrs.padsBegin[j] || indexes[j] >= params.srcODims[j])
                    break;
            }

            if (j != params.nDimsForWork) {
                memset(&dstData[dstIdx], 0, params.lastDstDim);
                parallel_step(params.nDimsForWork, params.dstDims, indexes);
                continue;
            }

            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx)
                srcIdx += (indexes[idx] - params.attrs.padsBegin[idx]) * params.srcStrides[idx];
            srcIdx *= params.attrs.dataSize;

            memset(&dstData[dstIdx], 0, beginShift);
            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize);
            memset(&dstData[dstIdx + beginShift + copySize], 0, endShift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::PadExecutor::padEdge() {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(params.srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(params.dstMemPtr->GetPtr());

    const size_t beginShift = params.attrs.padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);
        dstIdx *= params.attrs.dataSize;

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx) {
                size_t shift = (indexes[idx] < params.attrs.padsBegin[idx]) ? 0 :
                               ((indexes[idx] >= params.srcODims[idx]) ? (params.srcDims[idx] - 1) : (indexes[idx] - params.attrs.padsBegin[idx]));
                srcIdx += shift * params.srcStrides[idx];
            }
            srcIdx *= params.attrs.dataSize;

            for (size_t i = 0; i < params.attrs.padsBegin[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + i * params.shift], &srcData[srcIdx], params.shift);

            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize);

            for (size_t i = 0; i < params.attrs.padsEnd[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + beginShift + copySize + i * params.shift],
                           &srcData[srcIdx + (params.srcDims[params.nDimsForWork] - 1) * params.shift], params.shift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void MKLDNNPadNode::PadExecutor::padReflectOrSymmetric(const bool isSymmetric) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(params.srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(params.dstMemPtr->GetPtr());
    size_t shift = isSymmetric ? 1 : 0;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);
        dstIdx *= params.attrs.dataSize;

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t srcIdx = 0;
            for (size_t i = 0; i < params.nDimsForWork; ++i) {
                size_t idx = (indexes[i] < params.attrs.padsBegin[i]) ? (params.attrs.padsBegin[i] - indexes[i] - shift) :
                             ((indexes[i] >= params.srcODims[i]) ? (params.srcDimsForReflectOrSymmetric[i] - indexes[i]) :
                             (indexes[i] - params.attrs.padsBegin[i]));
                srcIdx += idx * params.srcStrides[i];
            }
            srcIdx *= params.attrs.dataSize;

            for (size_t i = 0; i < params.attrs.padsBegin[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + i * params.shift],
                           &srcData[srcIdx + (params.attrs.padsBegin[params.nDimsForWork] - shift - i) * params.shift], params.shift);

            cpu_memcpy(&dstData[dstIdx + params.attrs.padsBegin[params.nDimsForWork] * params.shift], &srcData[srcIdx],
                       params.srcDims[params.nDimsForWork] * params.shift);

            size_t srcShift = (params.srcDimsForReflectOrSymmetric[params.nDimsForWork] - params.srcODims[params.nDimsForWork]) * params.shift;
            for (size_t i = 0; i < params.attrs.padsEnd[params.nDimsForWork]; ++i)
                cpu_memcpy(&dstData[dstIdx + (params.srcODims[params.nDimsForWork] + i) * params.shift],
                           &srcData[srcIdx + srcShift - i * params.shift], params.shift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

inline void MKLDNNPadNode::PadExecutor::getDstIdx(const SizeVector& indexes, size_t& dstIdx) const {
    for (size_t i = 0; i < params.nDimsForWork; ++i)
        dstIdx += indexes[i] * params.dstStrides[i];
}

bool MKLDNNPadNode::created() const {
    return getType() == Pad;
}
REG_MKLDNN_PRIM_FOR(MKLDNNPadNode, Pad);
