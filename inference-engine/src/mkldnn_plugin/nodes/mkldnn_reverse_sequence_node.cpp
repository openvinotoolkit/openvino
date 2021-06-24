// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_reverse_sequence_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNReverseSequenceNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
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

MKLDNNReverseSequenceNode::MKLDNNReverseSequenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                         MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "ReverseSequence layer with name '" + op->get_friendly_name() + "'";
    const auto revSeq = std::dynamic_pointer_cast<const ngraph::opset1::ReverseSequence>(op);

    if (getOriginalInputsNumber() != 2 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    src_dims = op->get_input_shape(REVERSESEQUENCE_DATA);

    SizeVector seq_lengths_dims = op->get_input_shape(REVERSESEQUENCE_LENGTHS);
    if (seq_lengths_dims.size() != 1)
        IE_THROW() << errorPrefix << " has incorrect 2nd input rank: " << seq_lengths_dims.size();

    SizeVector dst_dims = op->get_output_shape(0);
    if (src_dims.size() != dst_dims.size())
        IE_THROW() << errorPrefix << " has incorrect number of input/output sizes!";

    for (size_t i = 0; i < dst_dims.size(); i++) {
        if (src_dims[i] != dst_dims[i])
            IE_THROW() << errorPrefix << " has incorrect number of input/output dimension!";
    }

    seq_axis = revSeq->get_sequence_axis();

    if (seq_axis < 0 || seq_axis >= static_cast<int>(src_dims.size()))
        IE_THROW() << errorPrefix << " has incorrect 'seq_axis' parameters dimensions and axis number!";

    batch_axis = revSeq->get_batch_axis();

    if (batch_axis < 0 || batch_axis >= static_cast<int>(src_dims.size()))
        IE_THROW() << errorPrefix << " has incorrect 'batch_axis' parameters dimensions and axis number!";

    if (seq_lengths_dims[0] != dst_dims[batch_axis])
        IE_THROW() << errorPrefix << " has incorrect 'seq_lengths_dims' parameters dimension!";

    srcStrides.resize(src_dims.size());
    srcStrides[srcStrides.size() - 1] = 1;
    for (int i = srcStrides.size() - 2; i >= 0; i--) {
        srcStrides[i] = srcStrides[i + 1] * src_dims[i + 1];
    }

    work_amount_dst = srcStrides[0] * src_dims[0];
}

void MKLDNNReverseSequenceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    lengthsPrecision = getOriginalInputPrecisionAtPort(REVERSESEQUENCE_LENGTHS);
    if (lengthsPrecision != Precision::I32 && lengthsPrecision != Precision::FP32)
        lengthsPrecision = Precision::I32;

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, Precision::FP32},
                          {TensorDescCreatorTypes::ncsp, lengthsPrecision}},
                         {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNReverseSequenceNode::execute(mkldnn::stream strm) {
    size_t i;
    const float *src_data = reinterpret_cast<const float *>(getParentEdgeAt(REVERSESEQUENCE_DATA)->getMemoryPtr()->GetPtr());
    float* dst_data = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    switch (getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getDesc().getPrecision()) {
        case Precision::FP32: {
            float *seq_lengths_data = reinterpret_cast<float *>(getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemoryPtr()->GetPtr());
            for (i = 0; i < src_dims[batch_axis]; i++) {
                if (static_cast<int32_t>(seq_lengths_data[i]) > static_cast<int>(src_dims[seq_axis])) {
                    std::string errorMsg = "Incorrect input 'seq_lengths' values!";
                    IE_THROW() << errorMsg;
                }
            }

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t i, start = 0, end = 0, src_idx = 0;
                SizeVector counters(src_dims.size(), 0);
                splitter(work_amount_dst, nthr, ithr, start, end);
                for (int j = src_dims.size() - 1, i = start; j >= 0; j--) {
                    counters[j] = i % src_dims[j];
                    i /= src_dims[j];
                }

                for (size_t iwork = start; iwork < end; ++iwork) {
                    for (i = 0, src_idx = 0; i < src_dims.size(); ++i) {
                        size_t idx = counters[i];
                        if (static_cast<int>(i) == seq_axis &&
                            static_cast<int>(idx) < static_cast<int32_t>(seq_lengths_data[counters[batch_axis]])) {
                            idx = static_cast<int32_t>(seq_lengths_data[counters[batch_axis]]) - idx - 1;
                        }
                        src_idx += idx * srcStrides[i];
                    }
                    dst_data[iwork] = src_data[src_idx];
                    for (int j = src_dims.size() - 1; j >= 0; j--) {
                        counters[j] = (counters[j] + 1) % src_dims[j];
                        if (counters[j] != 0) break;
                    }
                }
            });
        }
        break;
        case Precision::I32: {
            int32_t *seq_lengths_data = reinterpret_cast<int32_t *>(getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getMemoryPtr()->GetPtr());
            for (i = 0; i < src_dims[batch_axis]; i++) {
                if (seq_lengths_data[i] > static_cast<int>(src_dims[seq_axis])) {
                    std::string errorMsg = "Incorrect input 'seq_lengths' values!";
                    IE_THROW() << errorMsg;
                }
            }

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t i, start = 0, end = 0, src_idx = 0;
                SizeVector counters(src_dims.size(), 0);
                splitter(work_amount_dst, nthr, ithr, start, end);
                for (int j = src_dims.size() - 1, i = start; j >= 0; j--) {
                    counters[j] = i % src_dims[j];
                    i /= src_dims[j];
                }

                for (size_t iwork = start; iwork < end; ++iwork) {
                    for (i = 0, src_idx = 0; i < src_dims.size(); ++i) {
                        size_t idx = counters[i];
                        if (static_cast<int>(i) == seq_axis &&
                            static_cast<int>(idx) < seq_lengths_data[counters[batch_axis]]) {
                            idx = seq_lengths_data[counters[batch_axis]] - idx - 1;
                        }
                        src_idx += idx * srcStrides[i];
                    }
                    dst_data[iwork] = src_data[src_idx];
                    for (int j = src_dims.size() - 1; j >= 0; j--) {
                        counters[j] = (counters[j] + 1) % src_dims[j];
                        if (counters[j] != 0) break;
                    }
                }
            });
        }
        break;
        default:
            IE_THROW() << "ReverseSequence layer does not support "
                        << getParentEdgeAt(REVERSESEQUENCE_LENGTHS)->getDesc().getPrecision()  << " precision";
    }
}

bool MKLDNNReverseSequenceNode::created() const {
    return getType() == ReverseSequence;
}

REG_MKLDNN_PRIM_FOR(MKLDNNReverseSequenceNode, ReverseSequence)
