// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include <utils/general_utils.h>
#include "kernels/gather_uni_kernel.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::cpu;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ngraph::op::v1::Gather::get_type_info_static(),
                ngraph::op::v7::Gather::get_type_info_static(),
                ngraph::op::v8::Gather::get_type_info_static())) {
            errorMessage = "Not supported Gather operation version. CPU plug-in supports only 1, 7 and 8 versions.";
            return false;
        }

        if (op->get_input_node_shared_ptr(GATHER_AXIS)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Axis input for dynamic shapes.
            errorMessage = "Only Constant operation on 'axis' input is supported.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    if (op->get_input_size() != 3 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input/output edges!";

    dataSrcRank = static_cast<int>(op->get_input_shape(GATHER_DATA).size());
    int indicesRank = static_cast<int>(op->get_input_shape(GATHER_INDEXES).size());

    auto gatherBase = ngraph::as_type_ptr<ngraph::op::util::GatherBase>(op);
    axis = static_cast<int>(gatherBase->get_axis());
    if (axis < 0)
        axis += dataSrcRank;
    if (axis < 0 || axis >= dataRank)
        IE_THROW() << errorPrefix << "has incorrect 'axis' parameter value: " << axis;

    if (one_of(op->get_type_info(),
                ngraph::op::v7::Gather::type_info,
                ngraph::op::v8::Gather::type_info)) {
        if (op->get_type_info() == ngraph::op::v7::Gather::type_info) {
            batchDims = static_cast<int>(ngraph::as_type_ptr<ngraph::op::v7::Gather>(op)->get_batch_dims());
            reverseIndexing = false;
        } else if (op->get_type_info() == ngraph::op::v8::Gather::type_info) {
            batchDims = static_cast<int>(ngraph::as_type_ptr<ngraph::op::v8::Gather>(op)->get_batch_dims());
            reverseIndexing = true;
            // TODO: remove this WA when NMS & Gather will support dynamic shape.
            if (!op->get_input_element_type(1).is_signed())
                reverseIndexing = false;
        }

        if (batchDims < 0)
            batchDims += indicesRank;
        if (batchDims < 0 || batchDims > std::min(dataRank, indicesRank) || batchDims > axis)
    }
    dataSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32, isAxisInputConst}},
                         {{LayoutType::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNode::prepareParams() {
    auto& srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has unidentified preferable primitive descriptor.";

    const auto& srcDims = srcMemPtr->getStaticDims();
    const auto& idxDims = getParentEdgeAt(GATHER_INDEXES)->getMemory().getStaticDims();
    const auto& dstDims = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    if (!isAxisInputConst) {
        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            IE_THROW() << errorPrefix << "has incorrect input parameter axis value: " << axis;
    }

    indexRange = srcDims[axis];
    batchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<size_t>());
    outerSize = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<size_t>());
    dataLength = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<size_t>());
    srcBatchStride = std::accumulate(srcDims.begin() + batchDims, srcDims.end(), 1, std::multiplies<size_t>());
    idxBatchStride = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<size_t>());
    dstBatchStride = std::accumulate(dstDims.begin() + batchDims, dstDims.end(), 1, std::multiplies<size_t>());
    len = dataLength * dataSize;
    if (dataLength == 0)
        IE_THROW() << errorPrefix << "had incorrect input parameters dimension!";
}

bool MKLDNNGatherNode::needPrepareParams() const {
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
    return result;
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    const int32_t* srcIndexes = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDEXES)->getMemoryPtr()->GetPtr());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const int axisDim = srcDims[axis];

    const uint64_t beforeBatchSize = std::accumulate(srcDims.begin(), srcDims.begin() + batchDims, 1, std::multiplies<uint64_t>());
    const uint64_t betweenBatchAndAxis = std::accumulate(srcDims.begin() + batchDims, srcDims.begin() + axis, 1, std::multiplies<uint64_t>());
    const uint64_t afterAxisSize = std::accumulate(srcDims.begin() + axis + 1, srcDims.end(), 1, std::multiplies<uint64_t>());
    const uint64_t specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1, std::multiplies<uint64_t>());
    const uint64_t afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
    const uint64_t axisAndAfterAxisSizeInBytes = srcDims[axis] * afterAxisSizeInBytes;
    const uint64_t srcAfterBatchSizeInBytes = betweenBatchAndAxis * axisAndAfterAxisSizeInBytes;

    if (jitKernel && afterAxisSize == 1) {
        const uint64_t totalWork = beforeBatchSize * betweenBatchAndAxis * specIndicesSize;
        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

        auto threadBody = [&](const int ithr, const int nthr) {
            const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
            const uint64_t start = std::min(wpt * ithr, totalWork);
            const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
            const uint64_t workAmount = end - start;

            auto arg = gatherJitExecArgs();

            arg.src = srcData;
            arg.dst = dstData + afterAxisSizeInBytes * start;
            arg.indices = srcIndices;
            arg.start = &start;
            arg.axisDim = &axisDim;
            arg.afterAxisBlockSize = afterAxisSize;
            arg.axisAndAfterAxisSizeInBytes = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeInBytes = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxis;
            arg.specIndicesSize = &specIndicesSize;
            arg.workAmount = workAmount;

            const size_t idxElPerVec = jitKernel->getIdxElPerVec();
            if (specIndicesSize < idxElPerVec) {
                int permIdx[16];
                int beforeAxisDiff[16];
                permIdx[0] = idxElPerVec - specIndicesSize;
                int div = idxElPerVec / specIndicesSize;
                int remainder = idxElPerVec % specIndicesSize;
                for (int i = 1; i < idxElPerVec; i++) {
                    permIdx[i] = permIdx[i - 1] + 1;
                    if (permIdx[i] == idxElPerVec)
                        permIdx[i] = idxElPerVec - specIndicesSize;
                }
                int specIndices[16] = {0};
                for (int i = 0; i < idxElPerVec; i++) {
                    specIndices[i] = (start + i) % specIndicesSize;
                }
                for (int i = 0; i < idxElPerVec; i++) {
                    if (specIndices[i] < specIndicesSize - remainder)
                        beforeAxisDiff[i] = axisDim * div;
                    else
                        beforeAxisDiff[i] = axisDim * (div + 1);
                }
                arg.permIdx = permIdx;
                arg.beforeAxisDiff = beforeAxisDiff;
            }

            (*jitKernel)(&arg);
        };

        parallel_nt(0, threadBody);
    } else {
        const size_t dstIdxAndAfterAxisSize = afterAxisSizeInBytes * specIndicesSize;
        const size_t dstAfterBatchSize = betweenBatchAndAxis * dstIdxAndAfterAxisSize;
        parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
            int ii = srcIndices[b * specIndicesSize + j];
            if (ii < 0)
                ii += axisDim;
            size_t idx = ii;
            size_t c2 = dstAfterBatchSize * b + afterAxisSizeInBytes * j;
            if (idx < axisDim) {
                size_t c1 = srcAfterBatchSizeInBytes * b + afterAxisSizeInBytes * idx;
                for (size_t i = 0; i < betweenBatchAndAxis; i++) {
                    size_t srcIdx = c1 + axisAndAfterAxisSizeInBytes * i;
                    size_t dstIdx = c2 + dstIdxAndAfterAxisSize * i;

                    cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], afterAxisSizeInBytes);
                }
            } else {
                for (size_t i = 0; i < betweenBatchAndAxis; i++) {
                    memset(&dstData[c2 + dstIdxAndAfterAxisSize * i], 0, afterAxisSizeInBytes);
                }
            }
        });
    }
}

void MKLDNNGatherNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
