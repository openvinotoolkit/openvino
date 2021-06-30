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
#include <utils/general_utils.h>
#include "kernels/gather_uni_kernel.cpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::cpu;

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
printf("--- Axis: %d; batchDims: %d\n", axis, batchDims);
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

    dataTypeSize = getParentEdgeAt(GATHER_DATA)->getDesc().getPrecision().size();

    // Gather instruction is applicable just for 32 and 64 bit data and is not supported by SSE.
    if ((x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2))) {// &&
//                afterAxisSize_ == 1) {
        jGatherConfParams jcp;
        jcp.dataTypeSize = dataTypeSize;
//        jcp.axis = axis;
//            const auto vlen512 = x64::cpu_isa_traits<x64::avx512_common>::vlen;
//        const auto vlen256 = x64::cpu_isa_traits<x64::avx2>::vlen;
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
            jitKernel.reset(new jitUniGatherKernel<x64::avx512_common>(jcp));
        } else if (x64::mayiuse(x64::avx2)) {
//            size_t workAmount = beforeAxisSize_ * indicesSize_;
//            const size_t elPerVec = vlen256 / dictTypeSize_;
//                if (workAmount >= elPerVec) {
//                    const size_t wRest = (workAmount / vlen256) % (threadsNum - 1);
//                const size_t wpt  = ((workAmount / elPerVec) / threadsNum_ + 1) * elPerVec;

//printf("LAYER: %s\n", layer->name.c_str());
//                auto threadBody = [&](const int ithr, const int nthr) {
//                    int64_t start = std::min(wpt * ithr, workAmount);
//                    int64_t end = std::min(wpt * (ithr + 1), workAmount);
//                    size_t basStart = (start / indicesSize_) % beforeAxisSize_;
//                    size_t idxStart = start % indicesSize_;
//
//                    auto& params = parallelParams[ithr];
//                    params.workAmount = end - start;
//                    params.idxStartInBytes = idxStart * idxPrecision.size();
//                    params.dictTypeSize = dictTypeSize_;
//                    params.axisDimInBytes = axisDim_ * dictTypeSize_;
//                    params.axDimSumInBytes = params.axisDimInBytes * basStart;
//                    params.dstShift = basStart * indicesSize_ + idxStart;
////printf("[%d] WA: %lu start: %ld; end: %ld; wa: %ld; ii: %ld\n", ithr, workAmount, jcpThr.start, jcpThr.end, jcpThr.workAmount, jcpThr.idxIterator);
//                };
//                parallel_nt(threadsNum_, threadBody);
//                } else {
//                }
            jitKernel.reset(new jitUniGatherKernel<x64::avx2>(jcp));
        }
        if (jitKernel)
            jitKernel->create_ker();
    }
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    const SizeVector srcDims = getParentEdgeAt(GATHER_DATA)->getDims().ToSizeVector();
    const SizeVector idxDims = getParentEdgeAt(GATHER_INDEXES)->getDims().ToSizeVector();
//    const SizeVector dstDims = getChildEdgeAt(0)->getDims().ToSizeVector();

    size_t axisDim = srcDims[axis];
//    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
//    outerSize = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
//    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1, std::multiplies<size_t>());
//    idxBatchStride = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
//    dstBatchStride = std::accumulate(dstDims.begin() + batchDims, dstDims.end(), 1, std::multiplies<size_t>());

    const int32_t* srcIndices = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDEXES)->getMemoryPtr()->GetPtr());
    for (int i = 0; i < getParentEdgeAt(GATHER_INDEXES)->getDims().size(); i++) {
        if (srcIndices[i] < 0 || srcIndices[i] >= 100)
            std::cout << "INVALID INDEX: " << srcIndices[i] << std::endl;
    }
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    size_t beforeBatchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    size_t betweenBatchAndAxis = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    size_t afterAxisSize = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    size_t specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
//    size_t srcAfterBatchSize = betweenBatchAndAxis * specIndicesSize * afterAxisSize;
    size_t afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
    int axisAndAfterAxisSize = srcDims[axis] * afterAxisSizeInBytes;
    int srcAfterBatchSize = betweenBatchAndAxis * axisAndAfterAxisSize;
    size_t dstIdxAndAfterAxisSize = afterAxisSizeInBytes * specIndicesSize;
    size_t dstAfterBatchSize = betweenBatchAndAxis * dstIdxAndAfterAxisSize;

    if (jitKernel) {
        auto threadBody = [&](const int ithr, const int nthr) {
//int tmp[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//int retVal = 0;
            size_t start = 0, end = 0;
            size_t workAmount = 0;
            size_t totalWork = beforeBatchSize * betweenBatchAndAxis * specIndicesSize;
            int vecLen = jitKernel->geElPerVec();
            if (afterAxisSize == 4) {
//                ;
            } else if (x64::mayiuse(x64::avx2)) {
//                if (one_of(afterAxisSize, 1, 2, 3, 5, 6, 7)) {
                    size_t wpt = ((totalWork / vecLen) / nthr + 1) * vecLen;

                    start = std::min(wpt * ithr, totalWork);
                    end = std::min(wpt * (ithr + 1), totalWork);
                    workAmount = end - start;
//                }
            } else if (x64::mayiuse(x64::avx512_common)) {
//                if (one_of(afterAxisSize, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15)) {
////                    ;
//                }
            } else {
//                ;
            }

//printf("%s [%d] axisDimInBytes: %d; axDimSumInBytes: %d; idxStartInBytes: %d; dstShift: %lu\n",
//        name_.c_str(), ithr, params.axisDimInBytes, params.axDimSumInBytes, params.idxStartInBytes, params.dstShift);

//            int axisDimInBytes = axisDim * dataTypeSize;
            int dts = dataTypeSize;
            int minusOne = -1;
//            int indicesSize = beforeBatchSize * specIndicesSize;
//            int beforeAxisSize = std::accumulate(srcDims.begin(), srcDims.begin() + axis, 1, std::multiplies<size_t>());
//            int basStart = (start / indicesSize) % beforeAxisSize;
//            int idxStart = start % indicesSize;
//            int axDimSumInBytes = axisDimInBytes * basStart;
//            int idxStartInBytes = idxStart * sizeof(int);
//            int specIndicesSizeMinOne = specIndicesSize - 1;

//printf("[%d] start: %lu; end: %lu\n", ithr, start, end);
    std::string seqStr = std::string("[") + std::to_string(ithr) + "] start: " + std::to_string(start) + "; end: " + std::to_string(end);
    std::string bIdx = "\nbatchchIndices {", btw = "}\nbetweenBatchAndAxisIdx {", spIdx = "}\nspecIndices {";
            std::vector<int> batchIndices(vecLen);
            std::vector<int> betweenBatchAndAxisIdx(vecLen);
            std::vector<int> specIndices(vecLen);
            int afterBatchSize = betweenBatchAndAxis * specIndicesSize;
            for (int i = 0; i < vecLen; i++) {
                batchIndices[i] = (start + i) / afterBatchSize;
                betweenBatchAndAxisIdx[i] = ((start + i) / specIndicesSize) % betweenBatchAndAxis;
                specIndices[i] = (start + i) % specIndicesSize;

bIdx += std::to_string(batchIndices[i]) + ";";
btw += std::to_string(betweenBatchAndAxisIdx[i]) + ";";
spIdx += std::to_string(specIndices[i]) + ";";
            }
seqStr += bIdx + btw + spIdx + "}\n";
            int beforeAxisCounter = betweenBatchAndAxisIdx[0];//start / betweenBatchAndAxis;
printf("%sbeforeAxisCounter: %d\n", seqStr.c_str(), beforeAxisCounter);

            uint32_t vlen = jitKernel->getVecLen();
            int idxTypeSize = sizeof(int32_t);
            int specIndicesSizeInt = specIndicesSize * idxTypeSize;
            size_t idxIter = specIndices[0] * idxTypeSize;

            auto arg = gatherJitExecArgs();
            arg.src = srcData;
            arg.dst = dstData + afterAxisSizeInBytes * start;
            arg.indices = srcIndices;
            arg.dataTypeSize = &dts;
            arg.idxTypeSize = &idxTypeSize;
            arg.idxIter = idxIter;
//            arg.axisDim = &axisDimInBytes;
//            arg.axDimSum = &axDimSumInBytes;
//            arg.idxStartB = idxStartInBytes;
            arg.specIndices = specIndices.data();
            arg.batchIndices = batchIndices.data();
            arg.betweenBatchAndAxisIdx = betweenBatchAndAxisIdx.data();
            arg.afterAxisBlockSize = afterAxisSize;
            arg.axisAndAfterAxisSize = &axisAndAfterAxisSize;
            arg.srcAfterBatchSize = &srcAfterBatchSize;
            arg.betweenBatchAndAxisSize = betweenBatchAndAxis;
            arg.beforeAxisCounter = beforeAxisCounter;
            arg.shufMask8bitUni  = shufMask8bitUni_;
            arg.permMask8bitA2   = permMask8bitA2_;
            arg.permMask8bitA5   = permMask8bitA5_;
            arg.shufMask16bitUni = shufMask16bitUni_;
            arg.permMask16bitA2  = permMask16bitA2_;
            arg.permMask16bitA5  = permMask16bitA5_;
            arg.vecLen = &vlen;
            arg.specIndicesSize = specIndicesSize * idxTypeSize;
            arg.specIndicesSizePtr = &specIndicesSizeInt;
            arg.minusOne = &minusOne;
            arg.workAmount = workAmount;
//                    arg.tmp = tmp;
//                    arg.retVal = &retVal;
            (*jitKernel)(&arg);
    std::string tmpStr = "dst: ";
for (int s = 0; s < vecLen; s++) {
    tmpStr += std::to_string((reinterpret_cast<int*>(dstData + afterAxisSizeInBytes * start))[s]) + ";";
}
printf("[%d] %s\n", ithr, tmpStr.c_str());
//printf("retVal: %d\n", retVal);
//printf("retVal\n");
        };

        parallel_nt(0, threadBody);
    } else {
        parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
            int ii = srcIndices[b * specIndicesSize + j];
            if (ii < 0)
                ii += axisDim;
            size_t idx = ii;
            size_t c2 = dstAfterBatchSize * b + afterAxisSizeInBytes * j;
            if (idx < axisDim) {
                size_t c1 = srcAfterBatchSize * b + afterAxisSizeInBytes * idx;
                for (size_t i = 0; i < betweenBatchAndAxis; i++) {
                    size_t srcIdx = c1 + axisAndAfterAxisSize * i;
                    size_t dstIdx = c2 + dstIdxAndAfterAxisSize * i;

                    cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], afterAxisSizeInBytes);
                }
            } else {
                for (size_t i = 0; i < betweenBatchAndAxis; i++) {
                    memset(&dstData[c2 + dstIdxAndAfterAxisSize * i], 0, afterAxisSizeInBytes);
                }
            }
        });
//        parallel_for2d(batchSize, idxBatchStride, [&](const size_t i, const size_t j) {
//            const unsigned int idx = static_cast<uint32_t>(srcIndices[i * idxBatchStride + j]);
//
//            // while negative indices are not supported, should set zero
//            if (idx < axisDim) {
//                for (size_t k = 0; k < outerSize; ++k) {
//                    const size_t srcStride = (i * srcBatchStride + k * dataLength * axisDim) * dataTypeSize;
//                    const size_t dstStride = (i * dstBatchStride + k * dataLength * idxBatchStride) * dataTypeSize;
//
//                    cpu_memcpy(&dstData[dstStride + j * afterAxisSizeInBytes], &srcData[srcStride + idx * afterAxisSizeInBytes], afterAxisSizeInBytes);
//                }
//            } else {
//                for (size_t k = 0; k < outerSize; ++k) {
//                    memset(&dstData[(i * dstBatchStride + k * dataLength * idxBatchStride) * dataTypeSize + j * afterAxisSizeInBytes],
//                            0, afterAxisSizeInBytes);
//                }
//            }
//        });
    }
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
