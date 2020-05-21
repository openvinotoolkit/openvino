// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <cassert>
#include <algorithm>
#include <limits>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExperimentalSparseWeightedReduceImpl : public ExtLayerBase {
private:
    // supported operations for the reduction
    enum ReducedOp {sum};

public:
    explicit ExperimentalSparseWeightedReduceImpl(const CNNLayer* layer) {
        try {
            if ((layer->insData.size() != 5 && layer->insData.size() != 6) || layer->outData.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }
            if (layer->insData.size() == 6) {
                with_weights = true;
            }

            // check operation by which it reduces
            std::string reduce_mode = layer->type;
            if (reduce_mode == "ExperimentalSparseWeightedSum") reduction_op = ReducedOp::sum;
            else
                THROW_IE_EXCEPTION << layer->name << " Incorrect ExperimentalSparseWeightedReduce layer type!";

            // check a precision of input tensors
            input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
            input_values_precision = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getPrecision();
            input_dense_shape_precision = layer->insData[INPUT_DENSE_SHAPE_PORT].lock()->getTensorDesc().getPrecision();
            input_parameters_table_precision = layer->insData[INPUT_PARAMETERS_TABLE_PORT].lock()->getTensorDesc().getPrecision();
            input_default_value_precision = layer->insData[INPUT_DEFAULT_VALUE_PORT].lock()->getTensorDesc().getPrecision();

            bool are_other_precisions_valid = (input_indices_precision == Precision::I32 &&
                input_values_precision == Precision::I32 &&
                input_dense_shape_precision == Precision::I32);
            if (are_other_precisions_valid == false) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision of the input tensors.";
            }

            if (input_parameters_table_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect precision of the input parameters table values. Only FP32 is supported!";
            }

            if (input_default_value_precision != Precision::I32) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect precision of the input default value. Only I32 is supported!";
            }

            if (with_weights) {
                Precision input_weights_precision = layer->insData[INPUT_WEIGHTS_PORT].lock()->getTensorDesc().getPrecision();
                if (input_weights_precision != Precision::FP32) {
                    THROW_IE_EXCEPTION << layer->name
                                       << " Incorrect precision of the input weights values. Only FP32 is supported!";
                }
            }

            // check dimensions of input tensors
            SizeVector input_indices_dims = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getDims();
            if (input_indices_dims.size() != 2 || input_indices_dims[1] != 2) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect dimensions for input indices. It must be Nx2 dimension tensor.";
            }
            SizeVector input_values_dims = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getDims();
            if (input_values_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect dimensions for input values. It must be N dimension tensor.";
            }
            if (input_indices_dims[0] != input_values_dims[0]) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Mismatch of the first dimensions of input indices and values.";
            }
            SizeVector input_dense_shape_dims = layer->insData[INPUT_DENSE_SHAPE_PORT].lock()->getTensorDesc().getDims();
            if (input_dense_shape_dims.size() != 1 || input_dense_shape_dims[0] != 2) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect dimensions for input dense shape.";
            }
            SizeVector input_parameters_table_dims = layer->insData[INPUT_PARAMETERS_TABLE_PORT].lock()->getTensorDesc().getDims();
            if (input_parameters_table_dims.size() < 2) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect dimensions for input parameters table.";
            }
            SizeVector input_default_value_dims = layer->insData[INPUT_DEFAULT_VALUE_PORT].lock()->getTensorDesc().getDims();
            if (input_default_value_dims.size() != 0) {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect dimensions for input default value.";
            }
            if (with_weights) {
                SizeVector input_weights_dims = layer->insData[INPUT_WEIGHTS_PORT].lock()->getTensorDesc().getDims();
                if (input_weights_dims.size() != 1) {
                    THROW_IE_EXCEPTION << layer->name
                                       << " Incorrect dimensions for input weights. It must be N dimension tensor.";
                }
                if (input_weights_dims[0] != input_values_dims[0]) {
                    THROW_IE_EXCEPTION << layer->name
                                       << " Mismatch of the first dimensions of input weights and values.";
                }
            }
            input_num_values = input_indices_dims[0];

            // check dimensions of output tensors
            SizeVector output_dims = layer->outData[OUTPUT_PORT]->getTensorDesc().getDims();
            if (output_dims.size() != input_parameters_table_dims.size()) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for the output tensor.";
            }
            output_batch_size = output_dims[0];
            output_elem_size = 1;
            for (size_t ind = 1; ind < input_parameters_table_dims.size(); ind++) {
                output_elem_size *= input_parameters_table_dims[ind];
            }

            // TODO: check that dense shape value is set
            if (with_weights) {
                addConfig(layer,
                { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                    DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                { DataConfigurator(ConfLayout::PLN) });
            } else {
                addConfig(layer,
                { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                    DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN) });
            }
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const int *input_indices_i32_ptr = inputs[INPUT_INDICES_PORT]->cbuffer().as<const int *>() +
            inputs[INPUT_INDICES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int *input_values_i32_ptr = inputs[INPUT_VALUES_PORT]->cbuffer().as<const int *>() +
            inputs[INPUT_VALUES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *input_parameters_table_ptr = inputs[INPUT_PARAMETERS_TABLE_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_PARAMETERS_TABLE_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int *input_default_value_ptr = inputs[INPUT_DEFAULT_VALUE_PORT]->cbuffer().as<const int *>() +
            inputs[INPUT_DEFAULT_VALUE_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t input_default_value = static_cast<size_t>(*input_default_value_ptr);
        const float *input_weights_ptr = nullptr;
        if (with_weights) {
            input_weights_ptr = inputs[INPUT_WEIGHTS_PORT]->cbuffer().as<const float *>() +
                inputs[INPUT_WEIGHTS_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        }
        float *output_ptr = outputs[OUTPUT_PORT]->cbuffer().as<float *>() +
            inputs[OUTPUT_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        // fill the output tensor with default values
        for (size_t batch_ind = 0; batch_ind < output_batch_size; batch_ind++) {
            const float *param_elem_ptr = input_parameters_table_ptr + input_default_value * output_elem_size;
            float *output_elem_ptr = output_ptr + batch_ind * output_elem_size;
            for (size_t ind = 0; ind < output_elem_size; ind++) {
                output_elem_ptr[ind] = param_elem_ptr[ind];
            }
        }

        // initialize a vector with segment number values
        std::vector<float> segment_nums(output_batch_size, 0.0f);

        // compute the output tensor
        int prev_indice_x = -1;
        for (size_t curr_value_ind = 0; curr_value_ind < input_num_values; curr_value_ind++) {
            int indice_x = 0;
            size_t value = 0;
            indice_x = input_indices_i32_ptr[2 * curr_value_ind];
            value = static_cast<size_t>(input_values_i32_ptr[curr_value_ind]);
            const float *param_elem_ptr = input_parameters_table_ptr + value * output_elem_size;
            float *output_elem_ptr = output_ptr + indice_x * output_elem_size;
            if (prev_indice_x != indice_x) {
                // zero a slice
                prev_indice_x = indice_x;
                for (size_t ind = 0; ind < output_elem_size; ind++) {
                    output_elem_ptr[ind] = 0.0f;
                }
            }
            float weight = 1.0f;
            if (with_weights) {
                weight = input_weights_ptr[curr_value_ind];
            }
            segment_nums[indice_x] += weight;
            for (size_t ind = 0; ind < output_elem_size; ind++) {
                output_elem_ptr[ind] += param_elem_ptr[ind] * weight;
            }
        }

        return OK;
    }

private:
    const size_t INPUT_INDICES_PORT = 0;
    const size_t INPUT_VALUES_PORT = 1;
    const size_t INPUT_DENSE_SHAPE_PORT = 2;
    const size_t INPUT_PARAMETERS_TABLE_PORT = 3;
    const size_t INPUT_DEFAULT_VALUE_PORT = 4;
    const size_t INPUT_WEIGHTS_PORT = 5;
    const size_t OUTPUT_PORT = 0;

    size_t input_num_values = 0;
    size_t output_batch_size = 0;
    size_t output_elem_size = 0;

    ReducedOp reduction_op;
    bool with_weights = false;

    Precision input_indices_precision;
    Precision input_values_precision;
    Precision input_dense_shape_precision;
    Precision input_parameters_table_precision;
    Precision input_default_value_precision;
};

REG_FACTORY_FOR(ExperimentalSparseWeightedReduceImpl, ExperimentalSparseWeightedSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
