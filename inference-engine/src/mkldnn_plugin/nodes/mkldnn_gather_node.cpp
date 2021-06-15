// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto gatherOp = ngraph::as_type_ptr<const ngraph::op::v7::Gather>(op);
        if (!gatherOp) {
            errorMessage = "Only opset7 Gather operation is supported";
            return false;
        }

        auto axesOp = gatherOp->get_input_node_shared_ptr(GATHER_AXIS);
        if (!ngraph::as_type_ptr<const ngraph::op::Constant>(axesOp)) {
            errorMessage = "Only Constant operation on 'axis' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix_ = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    auto gatherOp = ngraph::as_type_ptr<ngraph::op::v7::Gather>(op);
    if (gatherOp->get_input_size() != 3 || gatherOp->get_output_size() != 1)
        IE_THROW() << errorPrefix_ << "has incorrect number of input/output edges!";

    int dataRank = static_cast<int>(gatherOp->get_input_shape(GATHER_DATA).size());
    int indicesRank = static_cast<int>(gatherOp->get_input_shape(GATHER_INDEXES).size());

    axis = static_cast<int>(gatherOp->get_axis());
    if (axis < 0)
        axis += dataRank;
    if (axis < 0 || axis >= dataRank)
        IE_THROW() << errorPrefix_ << "has incorrect 'axis' parameter value: " << axis;

    batchDims = static_cast<int>(gatherOp->get_batch_dims());
    if (batchDims < 0)
        batchDims += indicesRank;
    if (batchDims < -std::min(dataRank, indicesRank) || batchDims > std::min(dataRank, indicesRank) || batchDims > axis)
        IE_THROW() << errorPrefix_ << "has incorrect 'batch_dims' parameter value: " << batchDims;
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, dataPrecision},
                          {TensorDescCreatorTypes::ncsp, Precision::I32},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         {{TensorDescCreatorTypes::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix_ << " has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix_ << " has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix_ << "has unidentified preferable primitive descriptor.";

    dataSize = getParentEdgeAt(GATHER_DATA)->getDesc().getPrecision().size();

    // Gather instruction is applicable just for 32 and 64 bit data and is not supported by SSE.
/*    if ((x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2))) {// &&
//                afterAxisSize_ == 1) {
        jGatherConfParams jcp;
        jcp.beforeAxisSize = beforeAxisSize_;
        jcp.indicesSize = indicesSize_ * idxPrecision.size();
        jcp.dictTypeSize = dictTypeSize_;
        jcp.axisDim = axisDim_;
        jcp.afterAxisSize = afterAxisSize_;
//            const auto vlen512 = x6   4::cpu_isa_traits<x64::avx512_common>::vlen;
        const auto vlen256 = x64::cpu_isa_traits<x64::avx2>::vlen;
//            const auto vlen128 = x64::cpu_isa_traits<x64::sse41>::vlen;
        if (x64::mayiuse(x64::avx512_common)) {
//                if (threadsNum > 2 && indicesSize_ >= 4) {
//                    if (indicesSize_ >= vlen512)  {
//                        if (indicesSize_ % vlen512 == 0) {
//                            jcp.blockedIndices512 = true;
//                        }
//                    } else if (indicesSize_ >= 32 && indicesSize_ % 32 == 0) {
//                        jcp.blockedIndices256 = true;
//                    } else if (indicesSize_ >= 16 && indicesSize_ % 16 == 0) {
//                        jcp.blockedIndices128 = true;
//                    }
//                    idxPrecision.size();
//                }
            jKernel_.reset(new jitUniGatherKernel<x64::avx512_common>(jcp));
        } else if (x64::mayiuse(x64::avx2)) {
            size_t workAmount = beforeAxisSize_ * indicesSize_;
            const size_t elPerVec = vlen256 / dictTypeSize_;
//                if (workAmount >= elPerVec) {
//                    const size_t wRest = (workAmount / vlen256) % (threadsNum - 1);
                const size_t wpt  = ((workAmount / elPerVec) / threadsNum_ + 1) * elPerVec;

//printf("LAYER: %s\n", layer->name.c_str());
                auto threadBody = [&](const int ithr, const int nthr) {
                    int64_t start = std::min(wpt * ithr, workAmount);
                    int64_t end = std::min(wpt * (ithr + 1), workAmount);
                    size_t basStart = (start / indicesSize_) % beforeAxisSize_;
                    size_t idxStart = start % indicesSize_;

                    auto& params = parallelParams[ithr];
                    params.workAmount = end - start;
                    params.idxStartInBytes = idxStart * idxPrecision.size();
                    params.dictTypeSize = dictTypeSize_;
                    params.axisDimInBytes = axisDim_ * dictTypeSize_;
                    params.axDimSumInBytes = params.axisDimInBytes * basStart;
                    params.dstShift = basStart * indicesSize_ + idxStart;
//printf("[%d] WA: %lu start: %ld; end: %ld; wa: %ld; ii: %ld\n", ithr, workAmount, jcpThr.start, jcpThr.end, jcpThr.workAmount, jcpThr.idxIterator);
                };
                parallel_nt(threadsNum_, threadBody);
//                } else {
//                }
            jKernel_.reset(new jitUniGatherKernel<x64::avx2>(jcp));
        }
        if (jKernel_)
            jKernel_->create_ker();
    }*/
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    const SizeVector srcDims = getParentEdgeAt(GATHER_DATA)->getDims().ToSizeVector();
    const SizeVector idxDims = getParentEdgeAt(GATHER_INDEXES)->getDims().ToSizeVector();
    const SizeVector dstDims = getChildEdgeAt(0)->getDims().ToSizeVector();

    indexRange = srcDims[axis];
    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    outerSize = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    dataLength = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1, std::multiplies<size_t>());
    idxBatchStride = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
    dstBatchStride = std::accumulate(dstDims.begin() + batchDims, dstDims.end(), 1, std::multiplies<size_t>());
    len = dataLength * dataSize;

    if (dataLength == 0)
        IE_THROW() << errorPrefix_ << " has incorrect input parameters dimension!";

    const int32_t* srcIndexes = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDEXES)->getMemoryPtr()->GetPtr());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    size_t beforeBatchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    size_t betweenBatchAndAxis = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    size_t afterAxisSize = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    size_t specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
    afterAxisSize *= dataSize;
    size_t srcIdxAndAfterAxisSize = srcDims[axis] * afterAxisSize;
    size_t srcAfterBatchSize = betweenBatchAndAxis * srcIdxAndAfterAxisSize;
    size_t dstIdxAndAfterAxisSize = afterAxisSize * specIndicesSize;
    size_t dstAfterBatchSize = betweenBatchAndAxis * dstIdxAndAfterAxisSize;

    if (jitKernel) {
    } else {
        for (size_t b = 0; b < beforeBatchSize; b++) {
            for (size_t i = 0; i < betweenBatchAndAxis; i++) {
                for (size_t j = 0; j < specIndicesSize; j++) {
                    int idx = srcIndexes[b * specIndicesSize + j];
                    if (idx >= 0) {
                        size_t srcIdx = srcAfterBatchSize * b + srcIdxAndAfterAxisSize * i + afterAxisSize * idx;
                        size_t dstIdx = dstAfterBatchSize * b + dstIdxAndAfterAxisSize * i + afterAxisSize * j;

                        cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], len);
                    } else {
                        memset(&dstData[dstAfterBatchSize * b + dstIdxAndAfterAxisSize * i + afterAxisSize * j], 0, len);
                    }
                }
            }
        }
//        parallel_for2d(batchSize, idxBatchStride, [&](const size_t i, const size_t j) {
//            const unsigned int idx = static_cast<uint32_t>(srcIndexes[i * idxBatchStride + j]);
//
//            // while negative indices are not supported, should set zero
//            if (idx < indexRange) {
//                for (size_t k = 0; k < outerSize; ++k) {
//                    const size_t srcStride = (i * srcBatchStride + k * dataLength * indexRange) * dataSize;
//                    const size_t dstStride = (i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize;
//
//                    cpu_memcpy(&dstData[dstStride + j * len], &srcData[srcStride + idx * len], len);
//                }
//            } else {
//                for (size_t k = 0; k < outerSize; ++k) {
//                    memset(&dstData[(i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize + j * len], 0, len);
//                }
//            }
//        });
//        parallel_for2d(batchSize, idxBatchStride, [&](const size_t i, const size_t j) {
//            const unsigned int idx = static_cast<uint32_t>(srcIndexes[i * idxBatchStride + j]);
//
//            // while negative indices are not supported, should set zero
//            if (idx < indexRange) {
//                for (size_t k = 0; k < outerSize; ++k) {
//                    const size_t srcStride = (i * srcBatchStride + k * dataLength * indexRange) * dataSize;
//                    const size_t dstStride = (i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize;
//
//                    cpu_memcpy(&dstData[dstStride + j * len], &srcData[srcStride + idx * len], len);
//                }
//            } else {
//                for (size_t k = 0; k < outerSize; ++k) {
//                    memset(&dstData[(i * dstBatchStride + k * dataLength * idxBatchStride) * dataSize + j * len], 0, len);
//                }
//            }
//        });
    }
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
