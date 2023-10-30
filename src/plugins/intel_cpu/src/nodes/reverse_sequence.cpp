// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "ie_parallel.hpp"
#include "reverse_sequence.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool ReverseSequence::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto revSeq = std::dynamic_pointer_cast<const ngraph::opset1::ReverseSequence>(op);
        if (!revSeq) {
            errorMessage = "Only opset1 ReverseSequence operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ReverseSequence::ReverseSequence(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    errorPrefix = "ReverseSequence layer with name '" + op->get_friendly_name() + "'";
    const auto revSeq = std::dynamic_pointer_cast<const ngraph::opset1::ReverseSequence>(op);
    if (revSeq == nullptr)
        OPENVINO_THROW("Operation with name '",
                       op->get_friendly_name(),
                       "' is not an instance of ReverseSequence from opset1.");

    if (inputShapes.size() != 2  || outputShapes.size() != 1)
        OPENVINO_THROW(errorPrefix, " has incorrect number of input/output edges!");

    const auto dataRank = getInputShapeAtPort(REVERSESEQUENCE_DATA).getRank();

    if (dataRank < 2)
        OPENVINO_THROW(errorPrefix, " 'data' rank should be greater than or equal to 2");

    if (getInputShapeAtPort(REVERSESEQUENCE_LENGTHS).getRank() != 1)
        OPENVINO_THROW(errorPrefix, " 'seq_lengths' should be 1D tensor");

    if (dataRank != getOutputShapeAtPort(0).getRank())
        OPENVINO_THROW(errorPrefix, " has input/output rank mismatch");

    seq_axis = revSeq->get_sequence_axis();

    if (seq_axis < 0 || seq_axis >= static_cast<int>(dataRank))
        OPENVINO_THROW(errorPrefix, " has incorrect 'seq_axis' parameters dimensions and axis number!");

    batch_axis = revSeq->get_batch_axis();

    if (batch_axis < 0 || batch_axis >= static_cast<int>(dataRank))
        OPENVINO_THROW(errorPrefix, " has incorrect 'batch_axis' parameters dimensions and axis number!");
}

void ReverseSequence::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    lengthsPrecision = getOriginalInputPrecisionAtPort(REVERSESEQUENCE_LENGTHS);
    if (lengthsPrecision != Precision::I32 && lengthsPrecision != Precision::FP32)
        lengthsPrecision = Precision::I32;

    addSupportedPrimDesc({{LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, lengthsPrecision}},
                         {{LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void ReverseSequence::prepareParams() {
    const auto& dataMemPtr = getParentEdgeAt(REVERSESEQUENCE_DATA)->getMemoryPtr();
    const auto& seqLengthsMemPtr = getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemoryPtr();
    const auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();

    if (!dataMemPtr || !dataMemPtr->isAllocated())
        OPENVINO_THROW(errorPrefix, " has not allocated input memory of 'data'");
    if (!seqLengthsMemPtr || !seqLengthsMemPtr->isAllocated())
        OPENVINO_THROW(errorPrefix, " has not allocated input memory of 'seq_lengths'");
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        OPENVINO_THROW(errorPrefix, " has not allocated output memory");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        OPENVINO_THROW(errorPrefix, " has unidentified preferable primitive descriptor");

    const VectorDims& dataDims = dataMemPtr->getStaticDims();
    const VectorDims& seqLengthsDims = seqLengthsMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    execPtr = std::make_shared<ReverseSequenceExecutor>(dataDims, seqLengthsDims, dstDims, batch_axis, seq_axis);
}

void ReverseSequence::executeDynamicImpl(dnnl::stream strm) {
    execute(std::move(strm));
}

ReverseSequence::ReverseSequenceExecutor::ReverseSequenceExecutor(const VectorDims& dataDims,
    const VectorDims& seqLengthsDims, const VectorDims& dstDims, int batchAxis, int seqAxis)
        : batchAxis{batchAxis}
        , seqAxis{seqAxis} {
    for (size_t i = 0; i < dataDims.size(); ++i) {
        if (dataDims[i] != dstDims[i])
            OPENVINO_THROW("Input/output tensors dimensions mismatch");
    }

    if (seqLengthsDims[0] != dataDims[batchAxis])
        OPENVINO_THROW("'seq_lengths' dimension mismatch");

    srcStrides.resize(dataDims.size());
    srcStrides[srcStrides.size() - 1] = 1;
    for (int i = srcStrides.size() - 2; i >= 0; --i) {
        srcStrides[i] = srcStrides[i + 1] * dataDims[i + 1];
    }

    workAmountDst = srcStrides[0] * dataDims[0];
}

template<typename T>
void ReverseSequence::ReverseSequenceExecutor::exec(const MemoryPtr& dataMemPtr, const MemoryPtr& seqLengthsMemPtr, const MemoryPtr& dstMemPtr) {
    const VectorDims& srcDims = dataMemPtr->getStaticDims();
    const auto *srcData = reinterpret_cast<const float *>(dataMemPtr->getData());
    auto *dstData = reinterpret_cast<float *>(dstMemPtr->getData());
    auto *seqLengthsData = reinterpret_cast<T *>(seqLengthsMemPtr->getData());

    for (size_t i = 0; i < srcDims[batchAxis]; ++i) {
        if (static_cast<int32_t>(seqLengthsData[i]) > static_cast<int>(srcDims[seqAxis])) {
            OPENVINO_THROW("Incorrect input 'seq_lengths' values!");
        }
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t i, start = 0, end = 0, srcIdx = 0;
        SizeVector counters(srcDims.size(), 0);
        splitter(workAmountDst, nthr, ithr, start, end);
        for (int j = srcDims.size() - 1, i = start; j >= 0; --j) {
            counters[j] = i % srcDims[j];
            i /= srcDims[j];
        }

        for (size_t iwork = start; iwork < end; ++iwork) {
            for (i = 0, srcIdx = 0; i < srcDims.size(); ++i) {
                size_t idx = counters[i];
                if (static_cast<int>(i) == seqAxis &&
                    static_cast<int>(idx) < static_cast<int32_t>(seqLengthsData[counters[batchAxis]])) {
                    idx = static_cast<int32_t>(seqLengthsData[counters[batchAxis]]) - idx - 1;
                }
                srcIdx += idx * srcStrides[i];
            }
            dstData[iwork] = srcData[srcIdx];
            for (int j = srcDims.size() - 1; j >= 0; --j) {
                counters[j] = (counters[j] + 1) % srcDims[j];
                if (counters[j] != 0) break;
            }
        }
    });
}

void ReverseSequence::execute(dnnl::stream strm) {
    if (!execPtr)
        OPENVINO_THROW(errorPrefix, " has no compiled executor");

    const auto precision = getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemory().getDesc().getPrecision();
    if (!one_of(precision, Precision::FP32, Precision::I32))
        OPENVINO_THROW("ReverseSequence layer does not support ", precision , " precision");

    if (precision == Precision::FP32)
        execPtr->exec<float>(getParentEdgeAt(REVERSESEQUENCE_DATA)->getMemoryPtr(),
                             getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemoryPtr(),
                             getChildEdgeAt(0)->getMemoryPtr());
    else
        execPtr->exec<int32_t>(getParentEdgeAt(REVERSESEQUENCE_DATA)->getMemoryPtr(),
                               getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemoryPtr(),
                               getChildEdgeAt(0)->getMemoryPtr());
}

bool ReverseSequence::created() const {
    return getType() == Type::ReverseSequence;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
