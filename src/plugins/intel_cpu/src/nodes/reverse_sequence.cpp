// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool ReverseSequence::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        const auto revSeq = ov::as_type_ptr<const ov::op::v0::ReverseSequence>(op);
        if (!revSeq) {
            errorMessage = "Only v0 ReverseSequence operation is supported";
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

    const auto revSeq = ov::as_type_ptr<const ov::op::v0::ReverseSequence>(op);
    CPU_NODE_ASSERT(revSeq, "is not an instance of v0 ReverseSequence.");

    CPU_NODE_ASSERT(inputShapes.size() == 2 && outputShapes.size() == 1, "has incorrect number of input/output edges!");

    const auto dataRank = getInputShapeAtPort(REVERSESEQUENCE_DATA).getRank();

    CPU_NODE_ASSERT(dataRank >= 2, "'data' rank should be greater than or equal to 2");

    CPU_NODE_ASSERT(getInputShapeAtPort(REVERSESEQUENCE_LENGTHS).getRank() == 1, "'seq_lengths' should be 1D tensor");

    CPU_NODE_ASSERT(dataRank == getOutputShapeAtPort(0).getRank(), "has input/output rank mismatch");

    seq_axis = revSeq->get_sequence_axis();

    CPU_NODE_ASSERT(seq_axis >= 0 && seq_axis < static_cast<int>(dataRank),
                    "has incorrect 'seq_axis' parameters dimensions and axis number!");

    batch_axis = revSeq->get_batch_axis();

    CPU_NODE_ASSERT(batch_axis >= 0 && batch_axis < static_cast<int>(dataRank),
                    "has incorrect 'batch_axis' parameters dimensions and axis number!");
}

void ReverseSequence::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    lengthsPrecision = getOriginalInputPrecisionAtPort(REVERSESEQUENCE_LENGTHS);
    if (none_of(lengthsPrecision, ov::element::i32, ov::element::f32)) {
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

    CPU_NODE_ASSERT(dataMemPtr && dataMemPtr->isDefined(), "has undefined input memory of 'data'");
    CPU_NODE_ASSERT(seqLengthsMemPtr && seqLengthsMemPtr->isDefined(), "has undefined input memory of 'seq_lengths'");
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined output memory");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor() != nullptr, "has unidentified preferable primitive descriptor");

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
        OPENVINO_ASSERT(dataDims[i] == dstDims[i], "Input/output tensors dimensions mismatch");
    }

    OPENVINO_ASSERT(seqLengthsDims[0] == dataDims[batchAxis], "'seq_lengths' dimension mismatch");
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
        OPENVINO_ASSERT(static_cast<int32_t>(seqLengthsData[i]) <= static_cast<int>(srcDims[seqAxis]),
                        "Incorrect input 'seq_lengths' values!");
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t i = 0;
        size_t start = 0;
        size_t end = 0;
        size_t srcIdx = 0;
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

void ReverseSequence::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(execPtr, "has no compiled executor");

    const auto precision = getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemory().getDesc().getPrecision();
    CPU_NODE_ASSERT(any_of(precision, ov::element::f32, ov::element::i32),
                    "does not support ",
                    precision,
                    " precision");

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
