// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <dnnl_types.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <selective_build.h>
#include "space_to_batch.h"
#include <nodes/common/blocked_desc_creator.h>
#include <ngraph/opsets/opset2.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool SpaceToBatch::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto spaceToBatch = std::dynamic_pointer_cast<const ngraph::opset2::SpaceToBatch>(op);
        if (!spaceToBatch) {
            errorMessage = "Only opset2 SpaceToBatch operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(3)) == nullptr) {
            errorMessage = "Only constant 'block_shape', 'pads_begin', 'pads_end' are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

SpaceToBatch::SpaceToBatch(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "BatchToSpace layer with name '" + op->get_friendly_name() + "'";

    if (inputShapes.size() != 4 || outputShapes.size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input or output edges!";

    const size_t srcRank = getInputShapeAtPort(0).getRank();
    const size_t dstRank = getOutputShapeAtPort(0).getRank();
    if (srcRank < 4 || srcRank > 5)
        IE_THROW() << errorPrefix << " has unsupported 'data' input rank: " << srcRank;
    if (srcRank != dstRank)
        IE_THROW() << errorPrefix << " has incorrect number of input/output dimensions";
    blockShapeIn = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<size_t>();
    padsBeginIn = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<size_t>();
}

void SpaceToBatch::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto &inDims = getInputShapeAtPort(0).getDims();
    const auto precision = getOriginalInputPrecisionAtPort(0);
    const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
    if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
        IE_THROW() << errorPrefix << " has unsupported precision: " << precision.name();

    addSupportedPrimDesc({{LayoutType::nspc, precision},
                          {LayoutType::ncsp},
                          {LayoutType::ncsp},
                          {LayoutType::ncsp}},
                         {{LayoutType::nspc, precision}},
                         impl_desc_type::ref_any);
    addSupportedPrimDesc({{LayoutType::ncsp, precision},
                          {LayoutType::ncsp},
                          {LayoutType::ncsp},
                          {LayoutType::ncsp}},
                         {{LayoutType::ncsp, precision}},
                         impl_desc_type::ref_any);
    if (inDims[1] != Shape::UNDEFINED_DIM && inDims[1] % 8 == 0) {
        addSupportedPrimDesc({{LayoutType::nCsp8c, precision},
                              {LayoutType::ncsp},
                              {LayoutType::ncsp},
                              {LayoutType::ncsp}},
                             {{LayoutType::nCsp8c, precision}},
                             impl_desc_type::ref_any);
    }
    if (inDims[1] != Shape::UNDEFINED_DIM && inDims[1] % 16 == 0) {
        addSupportedPrimDesc({{LayoutType::nCsp16c, precision},
                              {LayoutType::ncsp},
                              {LayoutType::ncsp},
                              {LayoutType::ncsp}},
                             {{LayoutType::nCsp16c, precision}},
                             impl_desc_type::ref_any);
    }
}

std::vector<VectorDims> SpaceToBatch::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1, 2, 3));
}

static std::vector<size_t> getShape5D(const SizeVector &shape) {
    std::vector<size_t> shape5D(5, 1);
    for (int i = 0; i < 2; i++) {
        shape5D[i] = shape[i];
        shape5D[4 - i] = shape[shape.size() - 1 - i];
    }
    shape5D[2] = shape.size() == 5 ? shape[2] : shape5D[2];
    return shape5D;
}

template<typename T>
void SpaceToBatch::SpaceToBatchKernel() {
    const auto *srcData = reinterpret_cast<const T *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<T *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const auto &inDims = getParentEdgesAtPort(0)[0]->getMemoryPtr()->getStaticDims();
    const auto &outDims = getChildEdgesAtPort(0)[0]->getMemoryPtr()->getStaticDims();

    const bool blocked = getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nCsp16c) ||
                         getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nCsp8c);
    const auto dimsSize = inDims.size();

    auto inShape5D  = getShape5D(outDims);
    auto outShape5D = getShape5D(inDims);
    auto blockShape = getShape5D(blockShapeIn);

    if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nspc)) {
        inShape5D.push_back(inShape5D[1]);
        inShape5D.erase(inShape5D.begin() + 1);
        outShape5D.push_back(outShape5D[1]);
        outShape5D.erase(outShape5D.begin() + 1);
        blockShape.push_back(blockShape[1]);
        blockShape.erase(blockShape.begin() + 1);
    }

    const auto outBlkDims = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    const size_t blockSize = blocked ? outBlkDims.back() : 1lu;
    const size_t blockCountInput = outBlkDims[1];
    const size_t blockCountOutput = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims()[1];
    const auto blockRemainder = inShape5D[1] % blockSize;
    const auto lastBlock = blockRemainder == 0 ? blockSize : blockRemainder;

    const size_t inSpatialStep = inShape5D[2] * inShape5D[3] * inShape5D[4];
    const size_t inBatchStep = (blocked ? blockSize * blockCountInput : inShape5D[1]) * inSpatialStep;

    const size_t outSpatialStep = outShape5D[2] * outShape5D[3] * outShape5D[4];
    const size_t outBatchStep = (blocked ? blockSize * blockCountOutput : outShape5D[1]) * outSpatialStep;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(inShape5D[0] * inBatchStep, nthr, ithr, start, end);
        std::fill(dstData + start, dstData + end, T(0));
    });

    size_t channels = (inShape5D[1] / blockSize);
    channels = channels == 0 ? 1 : channels;
    const size_t workAmount = inShape5D[0] * channels;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        std::vector<size_t> indxStart(2, 0);
        std::vector<size_t> indxEnd(2, 0);
        parallel_it_init(start, indxStart[0], inShape5D[0], indxStart[1], channels);
        parallel_it_init((end - 1), indxEnd[0], inShape5D[0], indxEnd[1], channels);
        std::vector<int64_t> oAdd(5, 1);
        std::vector<size_t> begin(5, 0);
        std::vector<size_t> finish(5, 1);
        for (size_t i0 = indxStart[0]; i0 < indxEnd[0] + 1; ++i0) {
            int64_t bIdx = i0 / outShape5D[0];
            const size_t srcIdx0 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
            const size_t dstIdx0 = i0 * inBatchStep;
            oAdd[4] = bIdx % blockShapeIn[dimsSize - 1] - padsBeginIn[dimsSize - 1];
            bIdx /= blockShapeIn[dimsSize - 1];
            oAdd[3] = bIdx % blockShapeIn[dimsSize - 2] - padsBeginIn[dimsSize - 2];
            bIdx /= blockShapeIn[dimsSize - 2];
            oAdd[2] = dimsSize == 5 ? bIdx % blockShapeIn[2] - padsBeginIn[2] : 0lu;
            bIdx = dimsSize == 5 ? bIdx / blockShapeIn[2] : bIdx;
            oAdd[1] = bIdx % blockShapeIn[1] - padsBeginIn[1];
            if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nspc)) {
                oAdd.push_back(oAdd[1]);
                oAdd.erase(oAdd.begin() + 1);
            }
            begin[1] = (blockShape[1] - 1 - oAdd[1]) / blockShape[1] / blockSize;
            finish[1] = (outShape5D[1] - 1 - oAdd[1]) / blockShape[1] / blockSize;
            begin[2] = (blockShape[2] - 1 - oAdd[2]) / blockShape[2];
            finish[2] = (outShape5D[2] - 1 - oAdd[2]) / blockShape[2];
            begin[3] = (blockShape[3] - 1 - oAdd[3]) / blockShape[3];
            finish[3] = (outShape5D[3] - 1 - oAdd[3]) / blockShape[3];
            begin[4] = (blockShape[4] - 1 - oAdd[4]) / blockShape[4];
            finish[4] = (outShape5D[4] - 1 - oAdd[4]) / blockShape[4];
            const int64_t addTmpOC = blocked ? 0lu : oAdd[1];
            const int64_t addTmpOc = blocked ? oAdd[1] : 0lu;
            indxStart[1] = begin[1] > indxStart[1] ? begin[1] : indxStart[1];
            const size_t lastI1 = i0 == indxEnd[0] ? (indxEnd[1] > finish[1] ? finish[1] : indxEnd[1]) : finish[1];
            for (; indxStart[1] < lastI1 + 1; ++indxStart[1]) {
                const size_t block = indxStart[1] == finish[1] ? lastBlock : blockSize;
                const int64_t tmpOC = indxStart[1] * blockShape[1] + addTmpOC;
                const size_t srcIdx1 = srcIdx0 + tmpOC * outSpatialStep * blockSize;
                const size_t dstIdx1 = dstIdx0 + indxStart[1] * inSpatialStep * blockSize;
                const size_t itEnd = blocked ? ((block - 1) * blockShape[1] + oAdd[1]) / blockSize : 0lu;
                for (size_t i2 = begin[2]; i2 < finish[2] + 1; ++i2) {
                    const int64_t tmpOd = i2 * blockShape[2] + oAdd[2];
                    const size_t srcIdx2 = srcIdx1 + tmpOd * outShape5D[3] * outShape5D[4] * blockSize;
                    const size_t dstIdx2 = dstIdx1 + i2 * inShape5D[3] * inShape5D[4] * blockSize;
                    for (size_t i3 = begin[3]; i3 < finish[3] + 1; ++i3) {
                        const int64_t tmpOh = i3 * blockShape[3] + oAdd[3];
                        const size_t srcIdx3 = srcIdx2 + tmpOh * outShape5D[4] * blockSize;
                        const size_t dstIdx3 = dstIdx2 + i3 * inShape5D[4] * blockSize;
                        for (size_t i4 = begin[4]; i4 < finish[4] + 1; ++i4) {
                            const int64_t tmpOw = i4 * blockShape[4] + oAdd[4];
                            const size_t srcIdx4 = srcIdx3 + tmpOw * blockSize;
                            const size_t dstIdx4 = dstIdx3 + i4 * blockSize;
                            for (size_t it = 0; it < itEnd + 1; ++it) {
                                const size_t i5Begin = it == 0 ? 0 : (it * blockSize - 1 - oAdd[1]) / blockShape[1] + 1;
                                const size_t i5End = it == itEnd ? (block - 1) : ((it + 1) * blockSize - 1 - oAdd[1]) / blockShape[1];
                                for (size_t i5 = i5Begin; i5 < i5End + 1; ++i5) {
                                    const int64_t tmpOc = i5 * blockShape[1] + addTmpOc;
                                    const size_t srcIdx5 = srcIdx4 + it * outSpatialStep * blockSize + (tmpOc - it * blockSize);
                                    const size_t dstIdx5 = dstIdx4 + i5;
                                    dstData[dstIdx5] = srcData[srcIdx5];
                                }
                            }
                        }
                    }
                }
            }
            indxStart[1] = 0lu;
        }
    });
}

void SpaceToBatch::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void SpaceToBatch::execute(dnnl::stream strm) {
    switch (getParentEdgeAt(0)->getMemory().getDesc().getPrecision().size()) {
        case 1: SpaceToBatchKernel<PrecisionTrait<Precision::U8>::value_type>();  break;
        case 2: SpaceToBatchKernel<PrecisionTrait<Precision::U16>::value_type>(); break;
        case 4: SpaceToBatchKernel<PrecisionTrait<Precision::I32>::value_type>(); break;
        default:
            IE_THROW() << "SpaceToBatch layer does not support precision '" + std::string(getParentEdgeAt(0)->getMemory().getDesc().getPrecision().name())
                          + "'";
    }
}

bool SpaceToBatch::created() const {
    return getType() == Type::SpaceToBatch;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
