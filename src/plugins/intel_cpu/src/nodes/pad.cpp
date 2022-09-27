// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad.h"
#include <string>
#include <cmath>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include <selective_build.h>
#include <ngraph/opsets/opset1.hpp>

using namespace dnnl;
using namespace InferenceEngine;

#define THROW_ERROR IE_THROW() << "Pad layer with name '" << getName() << "' "

namespace ov {
namespace intel_cpu {
namespace node {

bool Pad::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto pad = ov::as_type_ptr<const ngraph::opset1::Pad>(op);
        if (!pad) {
            errorMessage = "Only opset1 Pad operation is supported";
            return false;
        }

        const auto pad_mode = pad->get_pad_mode();
        if (!one_of(pad_mode, ngraph::op::PadMode::CONSTANT, ngraph::op::PadMode::EDGE, ngraph::op::PadMode::REFLECT,
            ngraph::op::PadMode::SYMMETRIC)) {
            errorMessage = "Has unsupported pad_mode: " + ngraph::as_string(pad_mode);
            return false;
        }

        if (op->get_input_node_shared_ptr(PADS_BEGIN_ID)->get_type_info() != ov::op::v0::Constant::get_type_info_static() ||
            op->get_input_node_shared_ptr(PADS_END_ID)->get_type_info() != ov::op::v0::Constant::get_type_info_static() ||
            (pad->get_input_size() == 4 && pad->get_pad_mode() == ngraph::op::PadMode::CONSTANT &&
            op->get_input_node_shared_ptr(PAD_VALUE_ID)->get_type_info() != ov::op::v0::Constant::get_type_info_static())) {
            // TODO: Support pads_begin, pads_end, pad_value inputs for dynamic shapes.
            errorMessage = "Only Constant 'pads_begin', 'pads_end' and 'pad_value' inputs are supported.";
            return false;
        }

        const auto pb = pad->get_pads_begin();
        const auto pe = pad->get_pads_end();
        if (std::any_of(pb.begin(), pb.end(), [](ptrdiff_t x) { return x < 0; }) ||
            std::any_of(pe.begin(), pe.end(), [](ptrdiff_t x) { return x < 0; })) {
            errorMessage =  "Doesn't support 'pads_begin' or 'pads_end' with negative values";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Pad::Pad(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (inputShapes.size() != 3 && inputShapes.size() != 4)
        THROW_ERROR << " has incorrect number of input edges";
    if (outputShapes.size() != 1)
        THROW_ERROR << "Incorrect number of output edges";

    const size_t srcDimsRank = inputShapes[DATA_ID].getRank();
    const size_t dstDimsRank = outputShapes[DATA_ID].getRank();
    if (srcDimsRank != dstDimsRank)
        THROW_ERROR << "has incorrect number of input/output dimensions!";

    auto pad = ov::as_type_ptr<const ngraph::opset1::Pad>(op);
    if (!pad) {
        THROW_ERROR << "couldn't be casted to op of opset1";
    }

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

void Pad::getSupportedDescriptors() {}

void Pad::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<InferenceEngine::Precision> supportedPrecisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32,
                                                                   InferenceEngine::Precision::BF16, InferenceEngine::Precision::I8,
                                                                   InferenceEngine::Precision::U8};
    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(DATA_ID);
    if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), precision) == supportedPrecisions.end())
        precision = precision.is_float() ? InferenceEngine::Precision::FP32 : InferenceEngine::Precision::I32;

    const auto& inputDataShape = getInputShapeAtPort(DATA_ID);
    const size_t numOfDims = inputDataShape.getRank();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(isPadValueSpecified ? 4 : 3);
    config.outConfs.resize(1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushSupportedPrimitiveDescriptor = [&](LayoutType memoryFormat) {
        config.inConfs[0].setMemDesc(creatorsMap.at(memoryFormat)->createSharedDesc(precision, getInputShapeAtPort(DATA_ID)));
        config.inConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(Precision::I32, getInputShapeAtPort(PADS_BEGIN_ID)));
        config.inConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(Precision::I32, getInputShapeAtPort(PADS_END_ID)));
        if (isPadValueSpecified)
            config.inConfs[3].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(Precision::FP32, getInputShapeAtPort(PAD_VALUE_ID)));

        config.outConfs[0].setMemDesc(creatorsMap.at(memoryFormat)->createSharedDesc(precision, getOutputShapeAtPort(DATA_ID)));
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

void Pad::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << "has not allocated source memory.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_ERROR << "has not allocated destination memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";

    // pads are constant, so we can calculate new collapsing pads for first target dimensions and use it for the next dimensions
    // to avoid permanent identical pad calculations
    const size_t blockSize = srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c) ? 16 :
                             (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c) ? 8 : 1);
    if (blockSize > 1) {
        attrs.padsBegin[1] /= blockSize;
        attrs.padsEnd[1] /= blockSize;
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

    attrs.prc = srcMemPtr->getDesc().getPrecision();

    if (inputShapesDefined()) {
        prepareParams();
        updateLastInputDims();
    }
}

bool Pad::isExecutable() const {
    return !isOutputTensorAtPortEmpty(0);
}

void Pad::prepareParams() {
    execPtr = std::make_shared<PadExecutor>(attrs,
                                            getParentEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims(),
                                            getChildEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims());
}

Pad::PadExecutor::PadExecutor(const PadAttrs& attrs,
                                        const VectorDims& srcDims,
                                        const VectorDims& dstDims) {
    params.attrs = attrs;
    params.dstDims = dstDims;

    zeroInputDimsCase = std::any_of(srcDims.begin(), srcDims.end(), [](size_t dim) { return dim == 0; } ) &&
                        std::none_of(dstDims.begin(), dstDims.end(), [](size_t dim) { return dim == 0; } );
    if (zeroInputDimsCase) {
        return;
    }

    params.srcDims = srcDims;
    params.dataSize = attrs.prc.size();

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
        params.lastDstDim *= params.dataSize;
        params.shift *= params.dataSize;
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

void Pad::PadExecutor::exec(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    if (zeroInputDimsCase) {
        padConstant(srcMemPtr, dstMemPtr);
    } else {
        switch (params.attrs.padMode) {
            case CONSTANT:
                padConstant(srcMemPtr, dstMemPtr);
                break;
            case EDGE:
                padEdge(srcMemPtr, dstMemPtr);
                break;
            case REFLECT:
                padReflectOrSymmetric(srcMemPtr, dstMemPtr);
                break;
            case SYMMETRIC:
                padReflectOrSymmetric(srcMemPtr, dstMemPtr, true);
                break;
        }
    }
}

void Pad::execute(dnnl::stream strm) {
    if (!execPtr)
        THROW_ERROR << "has not compiled executor.";

    execPtr->exec(getParentEdgeAt(0)->getMemoryPtr(), getChildEdgeAt(0)->getMemoryPtr());
}

void Pad::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

std::vector<VectorDims> Pad::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(PADS_BEGIN_ID, PADS_END_ID));
}

static inline size_t parallel_init(size_t start, size_t nDims, const VectorDims& dims, VectorDims& indexes) {
    for (int j = nDims - 1; j >= 0; j--) {
        indexes[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

static inline void parallel_step(size_t nDims, const VectorDims& dims, VectorDims& indexes) {
    for (int j = nDims - 1; j >= 0; --j) {
        ++indexes[j];
        if (indexes[j] < dims[j])
            break;
        else
            indexes[j] = 0;
    }
}

void Pad::PadExecutor::padConstant(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    if (params.attrs.padValue == 0 && !zeroInputDimsCase) {
        padConstantZero(srcMemPtr, dstMemPtr);
        return;
    }

    PadContext ctx { this, srcMemPtr, dstMemPtr };
    OV_SWITCH(intel_cpu, PadConstantEmitter, ctx, params.attrs.prc,
              OV_CASE(InferenceEngine::Precision::FP32, float),
              OV_CASE(InferenceEngine::Precision::I32, int32_t),
              OV_CASE(InferenceEngine::Precision::BF16, bfloat16_t),
              OV_CASE(InferenceEngine::Precision::I8, int8_t),
              OV_CASE(InferenceEngine::Precision::U8, uint8_t));
}

template<typename T>
void Pad::PadExecutor::padConstantCommon(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    T* dstData = reinterpret_cast<T*>(dstMemPtr->GetPtr());
    const T value = static_cast<T>(params.attrs.padValue);
    if (zeroInputDimsCase) {
        const auto workAmount = dstMemPtr->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
        parallel_for(workAmount, [&](size_t i) {
            dstData[i] = value;
        });

        return;
    }

    const T* srcData = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    const size_t beginShift = params.attrs.padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;
    const size_t endShift = params.attrs.padsEnd[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims indexes(params.nDimsForWork, 0);
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
            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize * params.dataSize);
            std::fill_n(&dstData[dstIdx + beginShift + copySize], endShift, value);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void Pad::PadExecutor::padConstantZero(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());

    const size_t beginShift = params.attrs.padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;
    const size_t endShift = params.attrs.padsEnd[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);
        dstIdx *= params.dataSize;

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
            srcIdx *= params.dataSize;

            memset(&dstData[dstIdx], 0, beginShift);
            cpu_memcpy(&dstData[dstIdx + beginShift], &srcData[srcIdx], copySize);
            memset(&dstData[dstIdx + beginShift + copySize], 0, endShift);

            parallel_step(params.nDimsForWork, params.dstDims, indexes);
        }
    });
}

void Pad::PadExecutor::padEdge(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());

    const size_t beginShift = params.attrs.padsBegin[params.nDimsForWork] * params.shift;
    const size_t copySize = params.srcDims[params.nDimsForWork] * params.shift;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);
        dstIdx *= params.dataSize;

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t srcIdx = 0;
            for (size_t idx = 0; idx < params.nDimsForWork; ++idx) {
                size_t shift = (indexes[idx] < params.attrs.padsBegin[idx]) ? 0 :
                               ((indexes[idx] >= params.srcODims[idx]) ? (params.srcDims[idx] - 1) : (indexes[idx] - params.attrs.padsBegin[idx]));
                srcIdx += shift * params.srcStrides[idx];
            }
            srcIdx *= params.dataSize;

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

void Pad::PadExecutor::padReflectOrSymmetric(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr, const bool isSymmetric) {
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());
    size_t shift = isSymmetric ? 1 : 0;

    parallel_nt(params.nThreads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        VectorDims indexes(params.nDimsForWork, 0);
        splitter(params.workAmount, nthr, ithr, start, end);

        parallel_init(start, params.nDimsForWork, params.dstDims, indexes);
        size_t dstIdx = 0;
        getDstIdx(indexes, dstIdx);
        dstIdx *= params.dataSize;

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += params.lastDstDim) {
            size_t srcIdx = 0;
            for (size_t i = 0; i < params.nDimsForWork; ++i) {
                size_t idx = (indexes[i] < params.attrs.padsBegin[i]) ? (params.attrs.padsBegin[i] - indexes[i] - shift) :
                             ((indexes[i] >= params.srcODims[i]) ? (params.srcDimsForReflectOrSymmetric[i] - indexes[i]) :
                             (indexes[i] - params.attrs.padsBegin[i]));
                srcIdx += idx * params.srcStrides[i];
            }
            srcIdx *= params.dataSize;

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

inline void Pad::PadExecutor::getDstIdx(const VectorDims& indexes, size_t& dstIdx) const {
    for (size_t i = 0; i < params.nDimsForWork; ++i)
        dstIdx += indexes[i] * params.dstStrides[i];
}

bool Pad::created() const {
    return getType() == Type::Pad;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
