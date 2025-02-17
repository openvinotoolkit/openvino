// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space.h"

#include <openvino/opsets/opset2.hpp>
#include <string>
#include <vector>

#include "dnnl_types.h"
#include "nodes/common/blocked_desc_creator.h"
#include "openvino/core/parallel.hpp"
#include "selective_build.h"

namespace ov::intel_cpu::node {

bool BatchToSpace::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto batchToSpace = ov::as_type_ptr<const ov::opset2::BatchToSpace>(op);
        if (!batchToSpace) {
            errorMessage = "Only opset2 BatchToSpace operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

BatchToSpace::BatchToSpace(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (inputShapes.size() != 4 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input or output edges!");
    }

    const auto& inDims = getInputShapeAtPort(0).getDims();
    const auto& outDims = getOutputShapeAtPort(0).getDims();
    if (inDims.size() < 4 || inDims.size() > 5) {
        THROW_CPU_NODE_ERR("has unsupported 'data' input rank: ", inDims.size());
    }
    if (inDims.size() != outDims.size()) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output dimensions");
    }
}

void BatchToSpace::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const auto& inDims = getInputShapeAtPort(0).getDims();
    const auto precision = getOriginalInputPrecisionAtPort(0);
    const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
    if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end()) {
        THROW_CPU_NODE_ERR("has unsupported precision: ", precision.get_type_name());
    }

    addSupportedPrimDesc({{LayoutType::nspc, precision},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::nspc, precision}},
                         impl_desc_type::ref_any);
    addSupportedPrimDesc({{LayoutType::ncsp, precision},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, precision}},
                         impl_desc_type::ref_any);
    if (inDims[1] != Shape::UNDEFINED_DIM && inDims[1] % 8 == 0) {
        addSupportedPrimDesc({{LayoutType::nCsp8c, precision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32}},
                             {{LayoutType::nCsp8c, precision}},
                             impl_desc_type::ref_any);
    }
    if (inDims[1] != Shape::UNDEFINED_DIM && inDims[1] % 16 == 0) {
        addSupportedPrimDesc({{LayoutType::nCsp16c, precision},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32},
                              {LayoutType::ncsp, ov::element::i32}},
                             {{LayoutType::nCsp16c, precision}},
                             impl_desc_type::ref_any);
    }
}

static std::vector<size_t> getShape5D(const VectorDims& shape) {
    std::vector<size_t> shape5D(5, 1);
    for (int i = 0; i < 2; i++) {
        shape5D[i] = shape[i];
        shape5D[4 - i] = shape[shape.size() - 1 - i];
    }
    shape5D[2] = shape.size() == 5 ? shape[2] : shape5D[2];
    return shape5D;
}

template <typename T>
void BatchToSpace::batchToSpaceKernel() {
    const auto* srcData = getSrcDataAtPortAs<const T>(0);
    const auto* blockShapesPtr = getSrcDataAtPortAs<int>(1);
    size_t dataRank = getSrcMemoryAtPort(0)->getShape().getRank();
    blockShapeIn.clear();
    for (size_t i = 0; i < dataRank; i++) {
        blockShapeIn.push_back(*(blockShapesPtr + i));
    }

    const auto* padsBeginPtr = getSrcDataAtPortAs<int>(2);
    cropsBeginIn.clear();
    for (size_t i = 0; i < dataRank; i++) {
        cropsBeginIn.push_back(*(padsBeginPtr + i));
    }

    auto* dstData = getDstDataAtPortAs<T>(0);

    const auto& inDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    const auto& outDims = getChildEdgeAt(0)->getMemory().getStaticDims();

    auto srcDesc = getParentEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();

    const bool blocked = srcDesc->hasLayoutType(LayoutType::nCsp8c) || srcDesc->hasLayoutType(LayoutType::nCsp16c);
    const auto dimsSize = inDims.size();

    auto inShape5D = getShape5D(inDims);
    auto outShape5D = getShape5D(outDims);
    auto blockShape = getShape5D(blockShapeIn);

    if (srcDesc->hasLayoutType(LayoutType::nspc) && one_of(srcDesc->getShape().getRank(), 4u, 5u)) {
        inShape5D.push_back(inShape5D[1]);
        inShape5D.erase(inShape5D.begin() + 1);
        outShape5D.push_back(outShape5D[1]);
        outShape5D.erase(outShape5D.begin() + 1);
        blockShape.push_back(blockShape[1]);
        blockShape.erase(blockShape.begin() + 1);
    }

    auto dstDesc = getChildEdgeAt(0)->getMemory().getDescWithType<BlockedMemoryDesc>();

    const size_t blockSize = blocked ? dstDesc->getBlockDims().back() : 1lu;
    const size_t blockCountInput = srcDesc->getBlockDims()[1];
    const size_t blockCountOutput = dstDesc->getBlockDims()[1];
    const auto blockRemainder = inShape5D[1] % blockSize;
    const auto lastBlock = blockRemainder == 0 ? blockSize : blockRemainder;

    const size_t inSpatialStep = inShape5D[2] * inShape5D[3] * inShape5D[4];
    const size_t inBatchStep = (blocked ? blockSize * blockCountInput : inShape5D[1]) * inSpatialStep;

    const size_t outSpatialStep = outShape5D[2] * outShape5D[3] * outShape5D[4];
    const size_t outBatchStep = (blocked ? blockSize * blockCountOutput : outShape5D[1]) * outSpatialStep;

    size_t channels = (inShape5D[1] / blockSize);
    channels = channels == 0 ? 1 : channels;
    const size_t workAmount = inShape5D[0] * channels;
    if (workAmount == 0) {
        THROW_CPU_NODE_ERR("has unsupported work amount == 0");
    }

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
            const size_t srcIdx0 = i0 * inBatchStep;
            const size_t dstIdx0 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
            oAdd[4] = bIdx % blockShapeIn[dimsSize - 1] - cropsBeginIn[dimsSize - 1];
            bIdx /= blockShapeIn[dimsSize - 1];
            oAdd[3] = bIdx % blockShapeIn[dimsSize - 2] - cropsBeginIn[dimsSize - 2];
            bIdx /= blockShapeIn[dimsSize - 2];
            oAdd[2] = dimsSize == 5 ? bIdx % blockShapeIn[2] - cropsBeginIn[2] : 0lu;
            bIdx = dimsSize == 5 ? bIdx / blockShapeIn[2] : bIdx;
            oAdd[1] = bIdx % blockShapeIn[1] - cropsBeginIn[1];
            if (srcDesc->hasLayoutType(LayoutType::nspc) && one_of(srcDesc->getShape().getRank(), 4u, 5u)) {
                oAdd.push_back(oAdd[1]);
                oAdd.erase(oAdd.begin() + 1);
            }

            const auto numBlockElements = blockShape[1] * blockSize;
            begin[1] = (blockShape[1] - 1 - oAdd[1]) / numBlockElements;
            finish[1] = (outShape5D[1] + numBlockElements - 1 - oAdd[1]) / numBlockElements;
            begin[2] = (blockShape[2] - 1 - oAdd[2]) / blockShape[2];
            finish[2] = (outShape5D[2] + blockShape[2] - 1 - oAdd[2]) / blockShape[2];
            begin[3] = (blockShape[3] - 1 - oAdd[3]) / blockShape[3];
            finish[3] = (outShape5D[3] + blockShape[3] - 1 - oAdd[3]) / blockShape[3];
            begin[4] = (blockShape[4] - 1 - oAdd[4]) / blockShape[4];
            finish[4] = (outShape5D[4] + blockShape[4] - 1 - oAdd[4]) / blockShape[4];
            const int64_t addTmpOC = blocked ? 0lu : oAdd[1];
            const int64_t addTmpOc = blocked ? oAdd[1] : 0lu;

            const size_t firstI1 = i0 == 0 ? std::max(begin[1], indxStart[1]) : begin[1];
            const size_t lastI1 = i0 == indxEnd[0] ? std::min(indxEnd[1] + 1, finish[1]) : finish[1];

            for (size_t i1 = firstI1; i1 < lastI1; ++i1) {
                const size_t block = i1 == finish[1] ? lastBlock : blockSize;
                const int64_t tmpOC = i1 * blockShape[1] + addTmpOC;
                const size_t srcIdx1 = srcIdx0 + i1 * inSpatialStep * blockSize;
                const size_t dstIdx1 = dstIdx0 + tmpOC * outSpatialStep * blockSize;
                const size_t itEnd = blocked ? ((block - 1) * blockShape[1] + oAdd[1]) / blockSize : 0lu;
                for (size_t i2 = begin[2]; i2 < finish[2]; ++i2) {
                    const int64_t tmpOd = i2 * blockShape[2] + oAdd[2];
                    const size_t srcIdx2 = srcIdx1 + i2 * inShape5D[3] * inShape5D[4] * blockSize;
                    const size_t dstIdx2 = dstIdx1 + tmpOd * outShape5D[3] * outShape5D[4] * blockSize;
                    for (size_t i3 = begin[3]; i3 < finish[3]; ++i3) {
                        const int64_t tmpOh = i3 * blockShape[3] + oAdd[3];
                        const size_t srcIdx3 = srcIdx2 + i3 * inShape5D[4] * blockSize;
                        const size_t dstIdx3 = dstIdx2 + tmpOh * outShape5D[4] * blockSize;
                        for (size_t i4 = begin[4]; i4 < finish[4]; ++i4) {
                            const int64_t tmpOw = i4 * blockShape[4] + oAdd[4];
                            const size_t srcIdx4 = srcIdx3 + i4 * blockSize;
                            const size_t dstIdx4 = dstIdx3 + tmpOw * blockSize;
                            for (size_t it = 0; it < itEnd + 1; ++it) {
                                const size_t i5Begin = it == 0 ? 0 : (it * blockSize - 1 - oAdd[1]) / blockShape[1] + 1;
                                const size_t i5End =
                                    it == itEnd ? (block - 1) : ((it + 1) * blockSize - 1 - oAdd[1]) / blockShape[1];
                                for (size_t i5 = i5Begin; i5 < i5End + 1; ++i5) {
                                    const int64_t tmpOc = i5 * blockShape[1] + addTmpOc;
                                    const size_t srcIdx5 = srcIdx4 + i5;
                                    const size_t dstIdx5 =
                                        dstIdx4 + it * outSpatialStep * blockSize + (tmpOc - it * blockSize);
                                    dstData[dstIdx5] = srcData[srcIdx5];
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}

void BatchToSpace::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void BatchToSpace::execute(const dnnl::stream& strm) {
    switch (getParentEdgeAt(0)->getMemory().getDesc().getPrecision().size()) {
    case 1:
        batchToSpaceKernel<element_type_traits<ov::element::u8>::value_type>();
        break;
    case 2:
        batchToSpaceKernel<element_type_traits<ov::element::u16>::value_type>();
        break;
    case 4:
        batchToSpaceKernel<element_type_traits<ov::element::i32>::value_type>();
        break;
    default:
        THROW_CPU_NODE_ERR("does not support precision '",
                           std::string(getParentEdgeAt(0)->getMemory().getDesc().getPrecision().get_type_name()),
                           "'");
    }
}

bool BatchToSpace::created() const {
    return getType() == Type::BatchToSpace;
}

}  // namespace ov::intel_cpu::node
