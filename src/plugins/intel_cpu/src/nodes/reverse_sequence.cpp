// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence.h"

#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov::intel_cpu::node {

bool ReverseSequence::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        const auto revSeq = ov::as_type_ptr<const ov::opset1::ReverseSequence>(op);
        if (!revSeq) {
            errorMessage = "Only opset1 ReverseSequence operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ReverseSequence::ReverseSequence(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto revSeq = ov::as_type_ptr<const ov::opset1::ReverseSequence>(op);
    if (revSeq == nullptr) {
        THROW_CPU_NODE_ERR("is not an instance of ReverseSequence from opset1.");
    }

    if (inputShapes.size() != 2 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    const auto dataRank = getInputShapeAtPort(REVERSESEQUENCE_DATA).getRank();

    if (dataRank < 2) {
        THROW_CPU_NODE_ERR("'data' rank should be greater than or equal to 2");
    }

    if (getInputShapeAtPort(REVERSESEQUENCE_LENGTHS).getRank() != 1) {
        THROW_CPU_NODE_ERR("'seq_lengths' should be 1D tensor");
    }

    if (dataRank != getOutputShapeAtPort(0).getRank()) {
        THROW_CPU_NODE_ERR("has input/output rank mismatch");
    }

    seq_axis = revSeq->get_sequence_axis();

    if (seq_axis < 0 || seq_axis >= static_cast<int>(dataRank)) {
        THROW_CPU_NODE_ERR("has incorrect 'seq_axis' parameters dimensions and axis number!");
    }

    batch_axis = revSeq->get_batch_axis();

    if (batch_axis < 0 || batch_axis >= static_cast<int>(dataRank)) {
        THROW_CPU_NODE_ERR("has incorrect 'batch_axis' parameters dimensions and axis number!");
    }
}

void ReverseSequence::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    lengthsPrecision = getOriginalInputPrecisionAtPort(REVERSESEQUENCE_LENGTHS);
    if (lengthsPrecision != ov::element::i32 && lengthsPrecision != ov::element::f32) {
        lengthsPrecision = ov::element::i32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, lengthsPrecision}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ReverseSequence::prepareParams() {
    const auto& dataMemPtr = getSrcMemoryAtPort(REVERSESEQUENCE_DATA);
    const auto& seqLengthsMemPtr = getSrcMemoryAtPort(REVERSESEQUENCE_LENGTHS);
    const auto& dstMemPtr = getDstMemoryAtPort(0);

    if (!dataMemPtr || !dataMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined input memory of 'data'");
    }
    if (!seqLengthsMemPtr || !seqLengthsMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined input memory of 'seq_lengths'");
    }
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined output memory");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("has unidentified preferable primitive descriptor");
    }

    const VectorDims& dataDims = dataMemPtr->getStaticDims();
    const VectorDims& seqLengthsDims = seqLengthsMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    execPtr = std::make_shared<ReverseSequenceExecutor>(dataDims, seqLengthsDims, dstDims, batch_axis, seq_axis);
}

void ReverseSequence::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

ReverseSequence::ReverseSequenceExecutor::ReverseSequenceExecutor(const VectorDims& dataDims,
                                                                  const VectorDims& seqLengthsDims,
                                                                  const VectorDims& dstDims,
                                                                  int batchAxis,
                                                                  int seqAxis)
    : batchAxis{batchAxis},
      seqAxis{seqAxis} {
    for (size_t i = 0; i < dataDims.size(); ++i) {
        if (dataDims[i] != dstDims[i]) {
            OPENVINO_THROW("Input/output tensors dimensions mismatch");
        }
    }

    if (seqLengthsDims[0] != dataDims[batchAxis]) {
        OPENVINO_THROW("'seq_lengths' dimension mismatch");
    }

    srcStrides.resize(dataDims.size());
    srcStrides[srcStrides.size() - 1] = 1;
    for (int i = srcStrides.size() - 2; i >= 0; --i) {
        srcStrides[i] = srcStrides[i + 1] * dataDims[i + 1];
    }

    workAmountDst = srcStrides[0] * dataDims[0];
}

template <typename T>
void ReverseSequence::ReverseSequenceExecutor::exec(const MemoryPtr& dataMemPtr,
                                                    const MemoryPtr& seqLengthsMemPtr,
                                                    const MemoryPtr& dstMemPtr) {
    const VectorDims& srcDims = dataMemPtr->getStaticDims();
    const auto* srcData = dataMemPtr->getDataAs<const float>();
    auto* dstData = dstMemPtr->getDataAs<float>();
    auto* seqLengthsData = seqLengthsMemPtr->getDataAs<T>();

    for (size_t i = 0; i < srcDims[batchAxis]; ++i) {
        if (static_cast<int32_t>(seqLengthsData[i]) > static_cast<int>(srcDims[seqAxis])) {
            OPENVINO_THROW("Incorrect input 'seq_lengths' values!");
        }
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t i, start = 0, end = 0, srcIdx = 0;
        VectorDims counters(srcDims.size(), 0);
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
                if (counters[j] != 0) {
                    break;
                }
            }
        }
    });
}

void ReverseSequence::execute(const dnnl::stream& strm) {
    if (!execPtr) {
        THROW_CPU_NODE_ERR("has no compiled executor");
    }

    const auto precision = getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemory().getDesc().getPrecision();
    if (!one_of(precision, ov::element::f32, ov::element::i32)) {
        THROW_CPU_NODE_ERR("does not support ", precision, " precision");
    }

    if (precision == ov::element::f32) {
        execPtr->exec<float>(getSrcMemoryAtPort(REVERSESEQUENCE_DATA),
                             getSrcMemoryAtPort(REVERSESEQUENCE_LENGTHS),
                             getDstMemoryAtPort(0));
    } else {
        execPtr->exec<int32_t>(getSrcMemoryAtPort(REVERSESEQUENCE_DATA),
                               getSrcMemoryAtPort(REVERSESEQUENCE_LENGTHS),
                               getDstMemoryAtPort(0));
    }
}

bool ReverseSequence::created() const {
    return getType() == Type::ReverseSequence;
}

}  // namespace ov::intel_cpu::node
