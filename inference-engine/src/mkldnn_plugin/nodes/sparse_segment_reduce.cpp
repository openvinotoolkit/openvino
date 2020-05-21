// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <limits>
#include <functional>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SparseSegmentReduceImpl : public ExtLayerBase {
private:
    // supported operations for the reduction
    enum ReducedOp { sum, mean, sqrtn};

public:
    explicit SparseSegmentReduceImpl(const CNNLayer* layer) {
        try {
            // check a number of input/output edges
            if (layer->insData.size() != 3 || layer->outData.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }

            // check operation by which it reduces
            std::string reduce_mode = layer->type;
            if (reduce_mode == "SparseSegmentSum") reduction_op = ReducedOp::sum;
            else if (reduce_mode == "SparseSegmentMean") reduction_op = ReducedOp::mean;
            else if (reduce_mode == "SparseSegmentSqrtN") reduction_op = ReducedOp::sqrtn;
            else
                THROW_IE_EXCEPTION << layer->name << " Incorrect SparseSegmentReduce layer type!";

            // check a precision of input tensors
            Precision input_data_precision = layer->insData[INPUT_DATA_PORT].lock()->getTensorDesc().getPrecision();
            if (input_data_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision of the input data. Only FP32 is supported!";
            }
            Precision input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
            if (input_indices_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision of the input indices. Only FP32 is supported!";
            }
            Precision input_segment_ids_precision = layer->insData[INPUT_SEGMENT_IDS_PORT].lock()->getTensorDesc().getPrecision();
            if (input_segment_ids_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision of segment IDs. Only FP32 is supported!";
            }

            // check shapes of the second and third input tensors
            input_indices_dims = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getDims();
            if (input_indices_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input indices. It must be a one-dimensional tensor.";
            }
            input_segment_ids_dims = layer->insData[INPUT_SEGMENT_IDS_PORT].lock()->getTensorDesc().getDims();
            if (input_segment_ids_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input segment IDs. It must be a one-dimensional tensor.";
            }
            if (input_indices_dims[0] != input_segment_ids_dims[0]) {
                THROW_IE_EXCEPTION << layer->name << " Shapes for input indices and segment IDs must match.";
            }

            // check a precision of output tensor
            Precision output_precision = layer->insData[OUTPUT_PORT].lock()->getTensorDesc().getPrecision();
            if (output_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision of output data. Only FP32 is supported!";
            }

            // check shapes of output tensor
            input_data_dims = layer->insData[INPUT_DATA_PORT].lock()->getTensorDesc().getDims();
            output_dims = layer->outData[OUTPUT_PORT]->getTensorDesc().getDims();
            if (output_dims.size() != input_data_dims.size()) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output.";
            }
            if (output_dims[0] != input_segment_ids_dims[0]) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output.";
            }
            for (size_t i = 1; i < output_dims.size(); i++) {
                if (output_dims[i] != input_data_dims[i]) {
                    THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output.";
                }
            }

            // confugure layouts of input and output ports
            addConfig(layer,
                { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                { DataConfigurator(ConfLayout::PLN) });
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *input_data_ptr = inputs[INPUT_DATA_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_DATA_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *input_indices_ptr = inputs[INPUT_INDICES_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_INDICES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *input_segment_ids_ptr = inputs[INPUT_SEGMENT_IDS_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_SEGMENT_IDS_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float *output_ptr = outputs[OUTPUT_PORT]->cbuffer().as<float *>() +
            inputs[OUTPUT_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        // compute a number of elements in data slice
        size_t num_indices = input_indices_dims[0];
        size_t num_slices = input_data_dims[0];
        size_t num_elements_in_slice = std::accumulate(input_data_dims.begin(), input_data_dims.end(), 1, std::multiplies<size_t>()) / num_slices;

        // check that indices in a range [0; num_slices)
        if (std::any_of(input_indices_ptr, input_indices_ptr + num_indices,
            [num_slices](float idx) {return idx < 0.f || idx >= static_cast<float>(num_slices);})) {
            return GENERAL_ERROR;
        }

        // check that segment IDs are sorted
        for (size_t i = 1; i < num_indices; i++) {
            if (input_segment_ids_ptr[i] < input_segment_ids_ptr[i - 1]) {
                return GENERAL_ERROR;
            }
        }

        // compute start indices for segments in indices tensor
        size_t num_segments = static_cast<size_t>(input_segment_ids_ptr[num_indices - 1]) + 1;
        std::vector<size_t> segment_starts(num_segments);
        int prev_segment_id = -1;
        for (size_t i = 0; i < num_indices; i++) {
            if (i > 0 && input_segment_ids_ptr[i] == input_segment_ids_ptr[i - 1]) {
                continue;
            }
            int cur_segment_id = static_cast<int>(input_segment_ids_ptr[i]);
            for (int tmp_segment_ids = prev_segment_id + 1; tmp_segment_ids <= cur_segment_id; tmp_segment_ids++) {
                segment_starts[tmp_segment_ids] = i;
            }
            prev_segment_id = cur_segment_id;
        }

        // zero output buffer
        std::memset(output_ptr, 0, output_dims[0] * num_elements_in_slice * sizeof(float));

        // compute the result for each segment in parallel
        parallel_for(num_segments, [&](size_t segment_id) {
            float *segment_ptr = output_ptr + segment_id * num_elements_in_slice;
            size_t start = segment_starts[segment_id];
            size_t end = (segment_id == (num_segments - 1)) ? num_indices : segment_starts[segment_id + 1];

            // scatter data and reduce for one segment
            for (size_t idx = start; idx < end; idx++) {
                size_t indice = input_indices_ptr[idx];
                std::transform(segment_ptr, segment_ptr + num_elements_in_slice,
                    input_data_ptr + indice * num_elements_in_slice,
                    segment_ptr, std::plus<float>());
            }
        });

        if (reduction_op == ReducedOp::mean) {
            parallel_for(num_segments, [&](size_t segment_id) {
                float *segment_ptr = output_ptr + segment_id * num_elements_in_slice;
                size_t start = segment_starts[segment_id];
                size_t end = (segment_id == (num_segments - 1)) ? num_indices : segment_starts[segment_id + 1];
                float num_adds = static_cast<float>(end - start);
                if (num_adds > 0) {
                    std::transform(segment_ptr, segment_ptr + num_elements_in_slice, segment_ptr,
                        [num_adds](float elem) { return elem / num_adds; });
                }
            });
        }

        if (reduction_op == ReducedOp::sqrtn) {
            parallel_for(num_segments, [&](size_t segment_id) {
                float *segment_ptr = output_ptr + segment_id * num_elements_in_slice;
                size_t start = segment_starts[segment_id];
                size_t end = (segment_id == (num_segments - 1)) ? num_indices : segment_starts[segment_id + 1];
                float sqrtn = sqrtf(static_cast<float>(end - start));
                if (sqrtn > 0) {
                    std::transform(segment_ptr, segment_ptr + num_elements_in_slice, segment_ptr,
                        [sqrtn](float elem) { return elem / sqrtn; });
                }
            });
        }

        return OK;
    }

private:
    const size_t INPUT_DATA_PORT = 0;
    const size_t INPUT_INDICES_PORT = 1;
    const size_t INPUT_SEGMENT_IDS_PORT = 2;
    const size_t OUTPUT_PORT = 0;

    SizeVector input_data_dims;
    SizeVector input_indices_dims;
    SizeVector input_segment_ids_dims;
    SizeVector output_dims;

    ReducedOp reduction_op;
};

REG_FACTORY_FOR(SparseSegmentReduceImpl, SparseSegmentMean);
REG_FACTORY_FOR(SparseSegmentReduceImpl, SparseSegmentSqrtN);
REG_FACTORY_FOR(SparseSegmentReduceImpl, SparseSegmentSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
