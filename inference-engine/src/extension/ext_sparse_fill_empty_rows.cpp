// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <cassert>
#include <algorithm>
#include <limits>
#include "ie_parallel.hpp"
#include "simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SparseFillEmptyRowsImpl : public ExtLayerBase {
public:
    explicit SparseFillEmptyRowsImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 4 || layer->outData.size() != 3) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }

            Precision input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
            if (input_indices_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only FP32 is supported!";
            }

            // check dimensions of input tensors
            SizeVector input_indices_dims = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getDims();
            if (input_indices_dims.size() != 2 || input_indices_dims[1] != 2) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input indices. It must be Nx2 dimension tensor.";
            }
            SizeVector input_values_dims = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getDims();
            if (input_values_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input values. It must be N dimension tensor.";
            }
            if (input_indices_dims[0] != input_values_dims[0]) {
                THROW_IE_EXCEPTION << layer->name << " Mismatch of the first dimensions of input indices and values.";
            }
            SizeVector input_dense_shape_dims = layer->insData[INPUT_DENSE_SHAPE_PORT].lock()->getTensorDesc().getDims();
            if (input_dense_shape_dims.size() != 1 || input_dense_shape_dims[0] != 2) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input dense shape.";
            }
            SizeVector input_default_value_dims = layer->insData[INPUT_DEFAULT_VALUE_PORT].lock()->getTensorDesc().getDims();
            if (input_default_value_dims[0] != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input dense shape.";
            }
            inMaxNumValues = input_indices_dims[0];

            // check dimensions of output tensors
            SizeVector output_indices_dims = layer->outData[OUTPUT_INDICES_PORT]->getTensorDesc().getDims();
            if (output_indices_dims.size() != 2 || output_indices_dims[1] != 2) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output indices. It must be Nx2 dimension tensor.";
            }
            SizeVector output_values_dims = layer->outData[OUTPUT_VALUES_PORT]->getTensorDesc().getDims();
            if (output_values_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output values. It must be N dimension tensor.";
            }
            if (output_indices_dims[0] != output_values_dims[0]) {
                THROW_IE_EXCEPTION << layer->name << " Mismatch of the first dimensions of output indices and values.";
            }
            SizeVector output_empty_rows_indicator_dims = layer->outData[OUTPUT_EMPTY_ROWS_INDICATOR_PORT]->getTensorDesc().getDims();
            if (output_empty_rows_indicator_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output empty rows indicator. It must be 1-D tensor.";
            }
            outMaxNumValues = output_indices_dims[0];
            if (outMaxNumValues < inMaxNumValues) {
                THROW_IE_EXCEPTION << layer->name << " The first dimension size of input indices can not be greater the first dimension of output indices.";
            }

            // TODO: check that dense shape value is set
            addConfig(layer,
                {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)});
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *input_indices_ptr = inputs[INPUT_INDICES_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_INDICES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *input_values_ptr = inputs[INPUT_VALUES_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_VALUES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *dense_shape_ptr = inputs[INPUT_DENSE_SHAPE_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_DENSE_SHAPE_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *default_value_ptr = inputs[INPUT_DEFAULT_VALUE_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_DEFAULT_VALUE_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        float default_value = default_value_ptr[0];
        float num_rows = dense_shape_ptr[0];
        float num_cols = dense_shape_ptr[1];

        // compute actual number of values by searching out of range indice that serves as a marker
        size_t in_actual_num_values = 0;
        for (in_actual_num_values = 0; in_actual_num_values < inMaxNumValues; in_actual_num_values++) {
            float indice_x = input_indices_ptr[2 * in_actual_num_values];
            float indice_y = input_indices_ptr[2 * in_actual_num_values + 1];
            if (indice_x < 0 || indice_y < 0 || indice_x >= num_rows || indice_y >= num_cols) break;
        }

        // create auxiliary container for sorting
        std::vector<std::array<float, 3>> indices_values(in_actual_num_values);
        parallel_for(in_actual_num_values, [&](size_t i) {
            float row = input_indices_ptr[2 * i];
            float col = input_indices_ptr[2 * i + 1];
            float value = input_values_ptr[i];
            std::array<float, 3> elem = { row, col, value };
            indices_values[i] = elem;
        });

        // sort values by row
        parallel_sort(indices_values.begin(), indices_values.end(),
            [](const std::array<float, 3>& first, const std::array<float, 3>& second) {
            return first[0] < second[0];
        });

        // unsplit indices and values
        std::vector<float> indices_with_sorted_rows(in_actual_num_values * 2);
        std::vector<float> values_for_sorted_rows(in_actual_num_values);
        parallel_for(in_actual_num_values, [&](size_t i) {
            auto elem = indices_values[i];
            indices_with_sorted_rows[i * 2] = elem[0];
            indices_with_sorted_rows[i * 2 + 1] = elem[1];
            values_for_sorted_rows[i] = elem[2];
        });

        // compute start indice for each row and a number of values at each row
        std::vector<int> values_at_row(static_cast<unsigned int>(num_rows));
        std::fill(values_at_row.begin(), values_at_row.end(), 0);
        float prev_row_with_value = -1.0f;
        unsigned int total_num_values = 0;
        std::vector<std::array<float, 3>>::iterator curr_it, prev_it;
        for (float row_ind = 0.0; row_ind < num_rows; row_ind = row_ind + 1.0f) {
            curr_it = std::find_if(indices_values.begin(), indices_values.end(),
                [row_ind](std::array<float, 3> elem) { return elem[0] == row_ind; });
            if (curr_it != indices_values.end()) {
                if (prev_row_with_value != -1.0f) {
                    unsigned int num_values_at_prev_row = static_cast<unsigned int>(std::distance(prev_it, curr_it));
                    values_at_row[static_cast<int>(prev_row_with_value)] = num_values_at_prev_row;
                    total_num_values += num_values_at_prev_row;
                }
                prev_row_with_value = row_ind;
                prev_it = curr_it;
            } else {
                total_num_values++;
            }
        }
        if (prev_row_with_value != -1.0) {
            unsigned int num_values_at_prev_row = static_cast<unsigned int>(std::distance(prev_it, indices_values.end()));
            values_at_row[static_cast<int>(prev_row_with_value)] = num_values_at_prev_row;
            total_num_values += num_values_at_prev_row;
        }

        // check that output buffer size is sufficient
        if (outMaxNumValues < total_num_values) return GENERAL_ERROR;

        // create output indices
        float *output_indices_ptr = outputs[OUTPUT_INDICES_PORT]->cbuffer().as<float *>() +
            inputs[OUTPUT_INDICES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float *output_values_ptr = outputs[OUTPUT_VALUES_PORT]->cbuffer().as<float *>() +
            inputs[OUTPUT_VALUES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float *output_empty_rows_indicator_ptr = outputs[OUTPUT_EMPTY_ROWS_INDICATOR_PORT]->cbuffer().as<float *>() +
            inputs[OUTPUT_EMPTY_ROWS_INDICATOR_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        auto output_indices_size = outputs[OUTPUT_INDICES_PORT]->byteSize();
        memset(output_indices_ptr, 0, output_indices_size);

        auto output_values_size = outputs[OUTPUT_VALUES_PORT]->byteSize();
        memset(output_values_ptr, 0, output_values_size);

        auto output_empty_rows_indicator_size = outputs[OUTPUT_EMPTY_ROWS_INDICATOR_PORT]->byteSize();
        memset(output_empty_rows_indicator_ptr, 0, output_empty_rows_indicator_size);


        unsigned int curr_pos_from_copy = 0;
        unsigned int curr_pos_to_copy = 0;
        for (int row_ind = 0; row_ind < static_cast<int>(num_rows); row_ind++) {
            unsigned int num_values_at_row = values_at_row[row_ind];
            if (num_values_at_row == 0) {
                output_empty_rows_indicator_ptr[row_ind] = 1.0;
                output_values_ptr[curr_pos_to_copy] = default_value;
                output_indices_ptr[curr_pos_to_copy * 2] = static_cast<float>(row_ind);
                output_indices_ptr[curr_pos_to_copy * 2 + 1] = 0.0;
                curr_pos_to_copy++;
            } else {
                output_empty_rows_indicator_ptr[row_ind] = 0.0;
                std::copy(values_for_sorted_rows.begin() + curr_pos_from_copy,
                    values_for_sorted_rows.begin() + curr_pos_from_copy + num_values_at_row,
                    output_values_ptr + curr_pos_to_copy);
                std::copy(indices_with_sorted_rows.begin() + 2 * curr_pos_from_copy,
                    indices_with_sorted_rows.begin() + 2 * curr_pos_from_copy + 2 * num_values_at_row, output_indices_ptr + curr_pos_to_copy * 2);
                curr_pos_to_copy += num_values_at_row;
                curr_pos_from_copy += num_values_at_row;
            }
        }

        // mark the end of output using (-1, -1) indice
        if (total_num_values < outMaxNumValues) {
            output_indices_ptr[total_num_values * 2] = -1.0;
            output_indices_ptr[total_num_values * 2 + 1] = -1.0;
        }

        return OK;
    }

private:
    const size_t INPUT_INDICES_PORT = 0;
    const size_t INPUT_VALUES_PORT = 1;
    const size_t INPUT_DENSE_SHAPE_PORT = 2;
    const size_t INPUT_DEFAULT_VALUE_PORT = 3;
    const size_t OUTPUT_INDICES_PORT = 0;
    const size_t OUTPUT_VALUES_PORT = 1;
    const size_t OUTPUT_EMPTY_ROWS_INDICATOR_PORT = 2;

    size_t inMaxNumValues = 0;
    size_t outMaxNumValues = 0;
};

REG_FACTORY_FOR(ImplFactory<SparseFillEmptyRowsImpl>, SparseFillEmptyRows);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
