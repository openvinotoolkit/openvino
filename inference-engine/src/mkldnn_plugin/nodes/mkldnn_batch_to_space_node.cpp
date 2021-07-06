// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include "mkldnn_batch_to_space_node.h"
#include <nodes/common/blocked_desc_creator.h>
#include <cpu_memory_desc_utils.h>
#include <ngraph/opsets/opset2.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNBatchToSpaceNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto batchToSpace = std::dynamic_pointer_cast<const ngraph::opset2::BatchToSpace>(op);
        if (!batchToSpace) {
            errorMessage = "Only opset2 BatchToSpace operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(3)) == nullptr) {
            errorMessage = "Only constant 'block_shape', 'crops_begin', 'crops_end' are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNBatchToSpaceNode::MKLDNNBatchToSpaceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "BatchToSpace layer with name '" + op->get_friendly_name() + "'";

    if (op->get_input_size() != 4 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input or output edges!";

    inDims = op->get_input_shape(0);
    outDims = op->get_output_shape(0);
    if (inDims.size() < 4 || inDims.size() > 5)
        IE_THROW() << errorPrefix << " has unsupported 'data' input rank: " << inDims.size();
    if (inDims.size() != outDims.size())
        IE_THROW() << errorPrefix << " has incorrect number of input/output dimensions";

    blockShapeIn = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<size_t>();
    cropsBeginIn  = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<size_t>();
}

void MKLDNNBatchToSpaceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto precision = getOriginalInputPrecisionAtPort(0);
    const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8};
    if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
        IE_THROW() << errorPrefix << " has unsupported precision: " << precision.name();

    addSupportedPrimDesc({{GeneralLayout::nspc, precision},
                          {GeneralLayout::ncsp},
                          {GeneralLayout::ncsp},
                          {GeneralLayout::ncsp}},
                         {{GeneralLayout::nspc, precision}},
                         impl_desc_type::ref_any);
    addSupportedPrimDesc({{GeneralLayout::ncsp, precision},
                          {GeneralLayout::ncsp},
                          {GeneralLayout::ncsp},
                          {GeneralLayout::ncsp}},
                         {{GeneralLayout::ncsp, precision}},
                         impl_desc_type::ref_any);
    if (inDims[1] % 8 == 0) {
        addSupportedPrimDesc({{GeneralLayout::nCsp8c, precision},
                              {GeneralLayout::ncsp},
                              {GeneralLayout::ncsp},
                              {GeneralLayout::ncsp}},
                             {{GeneralLayout::nCsp8c, precision}},
                             impl_desc_type::ref_any);
    }
    if (inDims[1] % 16 == 0) {
        addSupportedPrimDesc({{GeneralLayout::nCsp16c, precision},
                              {GeneralLayout::ncsp},
                              {GeneralLayout::ncsp},
                              {GeneralLayout::ncsp}},
                             {{GeneralLayout::nCsp16c, precision}},
                             impl_desc_type::ref_any);
    }
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
void MKLDNNBatchToSpaceNode::batchToSpaceKernel() {
    const auto *srcData = reinterpret_cast<const T *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<T *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto srcDesc = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();

    const bool blocked = srcDesc.checkGeneralLayout(GeneralLayout::nCsp8c) || srcDesc.checkGeneralLayout(GeneralLayout::nCsp16c);
    const auto dimsSize = inDims.size();

    auto inShape5D = getShape5D(inDims);
    auto outShape5D = getShape5D(outDims);
    auto blockShape = getShape5D(blockShapeIn);

    if (srcDesc.checkGeneralLayout(GeneralLayout::nspc) && one_of(srcDesc.getShape().getRank(), 4, 5)) {
        inShape5D.push_back(inShape5D[1]);
        inShape5D.erase(inShape5D.begin() + 1);
        outShape5D.push_back(outShape5D[1]);
        outShape5D.erase(outShape5D.begin() + 1);
        blockShape.push_back(blockShape[1]);
        blockShape.erase(blockShape.begin() + 1);
    }

    auto dstDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();

    const size_t blockSize = blocked ? dstDesc.getBlockDims().back() : 1lu;
    const size_t blockCountInput = srcDesc.getBlockDims()[1];
    const size_t blockCountOutput = dstDesc.getBlockDims()[1];
    const auto blockRemainder = inShape5D[1] % blockSize;
    const auto lastBlock = blockRemainder == 0 ? blockSize : blockRemainder;

    const size_t inSpatialStep = inShape5D[2] * inShape5D[3] * inShape5D[4];
    const size_t inBatchStep = (blocked ? blockSize * blockCountInput : inShape5D[1]) * inSpatialStep;

    const size_t outSpatialStep = outShape5D[2] * outShape5D[3] * outShape5D[4];
    const size_t outBatchStep = (blocked ? blockSize * blockCountOutput : outShape5D[1]) * outSpatialStep;

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
            const size_t srcIdx0 = i0 * inBatchStep;
            const size_t dstIdx0 = (i0 - (bIdx * outShape5D[0])) * outBatchStep;
            oAdd[4] = bIdx % blockShapeIn[dimsSize - 1] - cropsBeginIn[dimsSize - 1];
            bIdx /= blockShapeIn[dimsSize - 1];
            oAdd[3] = bIdx % blockShapeIn[dimsSize - 2] - cropsBeginIn[dimsSize - 2];
            bIdx /= blockShapeIn[dimsSize - 2];
            oAdd[2] = dimsSize == 5 ? bIdx % blockShapeIn[2] - cropsBeginIn[2] : 0lu;
            bIdx = dimsSize == 5 ? bIdx / blockShapeIn[2] : bIdx;
            oAdd[1] = bIdx % blockShapeIn[1] - cropsBeginIn[1];
            if (srcDesc.checkGeneralLayout(GeneralLayout::nspc) && one_of(srcDesc.getShape().getRank(), 4, 5)) {
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
                const size_t srcIdx1 = srcIdx0 + indxStart[1] * inSpatialStep * blockSize;
                const size_t dstIdx1 = dstIdx0 + tmpOC * outSpatialStep * blockSize;
                const size_t itEnd = blocked ? ((block - 1) * blockShape[1] + oAdd[1]) / blockSize : 0lu;
                for (size_t i2 = begin[2]; i2 < finish[2] + 1; ++i2) {
                    const int64_t tmpOd = i2 * blockShape[2] + oAdd[2];
                    const size_t srcIdx2 = srcIdx1 + i2 * inShape5D[3] * inShape5D[4] * blockSize;
                    const size_t dstIdx2 = dstIdx1 + tmpOd * outShape5D[3] * outShape5D[4] * blockSize;
                    for (size_t i3 = begin[3]; i3 < finish[3] + 1; ++i3) {
                        const int64_t tmpOh = i3 * blockShape[3] + oAdd[3];
                        const size_t srcIdx3 = srcIdx2 + i3 * inShape5D[4] * blockSize;
                        const size_t dstIdx3 = dstIdx2 + tmpOh * outShape5D[4] * blockSize;
                        for (size_t i4 = begin[4]; i4 < finish[4] + 1; ++i4) {
                            const int64_t tmpOw = i4 * blockShape[4] + oAdd[4];
                            const size_t srcIdx4 = srcIdx3 + i4 * blockSize;
                            const size_t dstIdx4 = dstIdx3 + tmpOw * blockSize;
                            for (size_t it = 0; it < itEnd + 1; ++it) {
                                const size_t i5Begin = it == 0 ? 0 : (it * blockSize - 1 - oAdd[1]) / blockShape[1] + 1;
                                const size_t i5End = it == itEnd ? (block - 1) : ((it + 1) * blockSize - 1 - oAdd[1]) / blockShape[1];
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
            indxStart[1] = 0lu;
        }
    });
}

void MKLDNNBatchToSpaceNode::execute(mkldnn::stream strm) {
    switch (getParentEdgeAt(0)->getMemory().GetDesc().getPrecision().size()) {
        case 1: batchToSpaceKernel<PrecisionTrait<Precision::U8>::value_type>();  break;
        case 2: batchToSpaceKernel<PrecisionTrait<Precision::U16>::value_type>(); break;
        case 4: batchToSpaceKernel<PrecisionTrait<Precision::I32>::value_type>(); break;
        default:
            IE_THROW() << "BatchToSpace layer does not support precision '" <<
                std::string(getParentEdgeAt(0)->getMemory().GetDesc().getPrecision().name()) << "'";
    }
}

bool MKLDNNBatchToSpaceNode::created() const {
    return getType() == BatchToSpace;
}

REG_MKLDNN_PRIM_FOR(MKLDNNBatchToSpaceNode, BatchToSpace)
