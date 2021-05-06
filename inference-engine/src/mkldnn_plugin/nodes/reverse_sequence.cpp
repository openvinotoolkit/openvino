// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>

using namespace MKLDNNPlugin;

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ReverseSequenceImpl: public ExtLayerBase {
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
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

    std::string errorPrefix;

public:
    explicit ReverseSequenceImpl(const std::shared_ptr<ngraph::Node>& op) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            errorPrefix = "ReverseSequence layer with name '" + op->get_friendly_name() + "'";
            const auto revSeq = std::dynamic_pointer_cast<const ngraph::opset1::ReverseSequence>(op);

            if (op->get_input_size() != 2 || op->get_output_size() != 1)
                IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

            src_dims = op->get_input_shape(REVERSESEQUENCE_DATA);

            Precision lengthsPrecision = details::convertPrecision(op->get_input_element_type(REVERSESEQUENCE_LENGTHS));
            if (lengthsPrecision != Precision::I32 && lengthsPrecision != Precision::FP32)
                lengthsPrecision = Precision::I32;

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

            addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                           {TensorDescCreatorTypes::ncsp, lengthsPrecision}},
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        size_t i;
        const float *src_data = inputs[REVERSESEQUENCE_DATA]->cbuffer().as<const float *>() +
                                inputs[REVERSESEQUENCE_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
                          outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        switch (inputs[REVERSESEQUENCE_LENGTHS]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                float *seq_lengths_data = inputs[REVERSESEQUENCE_LENGTHS]->cbuffer().as<float *>() +
                                          inputs[REVERSESEQUENCE_LENGTHS]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                for (i = 0; i < src_dims[batch_axis]; i++) {
                    if (static_cast<int32_t>(seq_lengths_data[i]) > static_cast<int>(src_dims[seq_axis])) {
                        if (resp) {
                            std::string errorMsg = "Incorrect input 'seq_lengths' values!";
                            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                        }
                        return PARAMETER_MISMATCH;
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
                int32_t *seq_lengths_data = inputs[REVERSESEQUENCE_LENGTHS]->cbuffer().as<int32_t *>() +
                                            inputs[REVERSESEQUENCE_LENGTHS]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                for (i = 0; i < src_dims[batch_axis]; i++) {
                    if (seq_lengths_data[i] > static_cast<int>(src_dims[seq_axis])) {
                        if (resp) {
                            std::string errorMsg = "Incorrect input 'seq_lengths' values!";
                            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                        }
                        return PARAMETER_MISMATCH;
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
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    const size_t REVERSESEQUENCE_DATA = 0;
    const size_t REVERSESEQUENCE_LENGTHS = 1;

    int seq_axis;
    int batch_axis;
    SizeVector src_dims;
    SizeVector srcStrides;
    size_t work_amount_dst;
};

REG_FACTORY_FOR(ReverseSequenceImpl, ReverseSequence);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
