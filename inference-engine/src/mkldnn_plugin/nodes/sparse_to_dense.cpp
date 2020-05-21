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

class SparseToDenseImpl : public ExtLayerBase {
public:
    explicit SparseToDenseImpl(const CNNLayer* layer) {
        try {
            if ((layer->insData.size() != 3 && layer->insData.size() != 4) || layer->outData.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }
            if (layer->insData.size() == 4) {
                with_default_value = true;
            }

            // check precisions for input tensors
            Precision input_indices_precision = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getPrecision();
            if (input_indices_precision != Precision::I32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for input indices. Only I32 is supported!";
            }
            Precision input_dense_shape_precision = layer->insData[INPUT_DENSE_SHAPE_PORT].lock()->getTensorDesc().getPrecision();
            if (input_dense_shape_precision != Precision::I32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for input dense shape. Only I32 is supported!";
            }
            Precision input_values_precision = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getPrecision();
            if (input_values_precision != Precision::I32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for input values. Only I32 is supported!";
            }
            if (with_default_value) {
                Precision input_default_value_precision = layer->insData[INPUT_DEFAULT_VALUE_PORT].lock()->getTensorDesc().getPrecision();
                if (input_default_value_precision != Precision::I32) {
                    THROW_IE_EXCEPTION << layer->name << " Incorrect input precision for input default value. Only I32 is supported!";
                }
            }

            // check dimensions of input tensors
            SizeVector input_dense_shape_dims = layer->insData[INPUT_DENSE_SHAPE_PORT].lock()->getTensorDesc().getDims();
            if (input_dense_shape_dims.size() != 1 || input_dense_shape_dims[0] < 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input dense shape. It must be 1D dimension tensor.";
            }
            dense_tensor_rank = input_dense_shape_dims[0];
            SizeVector input_indices_dims = layer->insData[INPUT_INDICES_PORT].lock()->getTensorDesc().getDims();
            if (input_indices_dims.size() != 2 || input_indices_dims[1] != dense_tensor_rank) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input indices.";
            }
            SizeVector input_values_dims = layer->insData[INPUT_VALUES_PORT].lock()->getTensorDesc().getDims();
            if (input_values_dims.size() != 1 || input_values_dims[0] != input_indices_dims[0]) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input values.";
            }
            if (with_default_value) {
                SizeVector input_default_value_dims = layer->insData[INPUT_DEFAULT_VALUE_PORT].lock()->getTensorDesc().getDims();
                if (input_default_value_dims.size() != 0) {
                    THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for input default value.";
                }
            }
            input_num_values = input_values_dims[0];

            // TODO: check that dense shape value is set
            if (with_default_value) {
                addConfig(layer,
                { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                    DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                { DataConfigurator(ConfLayout::PLN) });
            } else {
                addConfig(layer,
                { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN),
                    DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN) });
            }
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const int *input_indices_ptr = inputs[INPUT_INDICES_PORT]->cbuffer().as<const int *>() +
            inputs[INPUT_INDICES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int *input_dense_shape_ptr = inputs[INPUT_DENSE_SHAPE_PORT]->cbuffer().as<const int *>() +
            inputs[INPUT_DENSE_SHAPE_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int *input_values_ptr = inputs[INPUT_VALUES_PORT]->cbuffer().as<const int *>() +
            inputs[INPUT_VALUES_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        int default_value = 0;
        if (with_default_value) {
            const int *input_default_value_ptr = inputs[INPUT_DEFAULT_VALUE_PORT]->cbuffer().as<const int *>() +
                inputs[INPUT_DEFAULT_VALUE_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            default_value = *input_default_value_ptr;
        }
        int *output_ptr = outputs[OUTPUT_PORT]->cbuffer().as<int *>() +
            inputs[OUTPUT_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        size_t output_num_values = 1;
        for (size_t ind = 0; ind < dense_tensor_rank; ind++) {
            output_num_values *= input_dense_shape_ptr[ind];
        }

        // fill the output tensor with the default value
        for (size_t ind = 0; ind < output_num_values; ind++) {
            output_ptr[ind] = default_value;
        }

        // walkthrough all indices and fill the output tensor with corresponding values
        for (size_t ind = 0; ind < input_num_values; ind++) {
            int value = input_values_ptr[ind];
            size_t placement = 0;
            const int *tmp_indice_ptr = input_indices_ptr + ind * dense_tensor_rank;
            size_t num_values_in_slice = output_num_values;
            for (size_t subindice_ind = 0; subindice_ind < dense_tensor_rank; subindice_ind++) {
                num_values_in_slice /= input_dense_shape_ptr[subindice_ind];
                size_t subindice = static_cast<size_t>(tmp_indice_ptr[subindice_ind]);
                if (subindice >= input_dense_shape_ptr[subindice_ind]) {
                    if (resp) {
                        std::string errorMsg = "Value of index is out of bound!";
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return GENERAL_ERROR;
                }
                placement += subindice * num_values_in_slice;
            }
            output_ptr[placement] = value;
        }

        return OK;
    }

private:
    const size_t INPUT_INDICES_PORT = 0;
    const size_t INPUT_DENSE_SHAPE_PORT = 1;
    const size_t INPUT_VALUES_PORT = 2;
    const size_t INPUT_DEFAULT_VALUE_PORT = 3;
    const size_t OUTPUT_PORT = 0;

    size_t dense_tensor_rank = 0;
    size_t input_num_values = 0;
    bool with_default_value = false;
};

REG_FACTORY_FOR(SparseToDenseImpl, SparseToDense);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
