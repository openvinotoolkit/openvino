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

class BucketizeImpl : public ExtLayerBase {
public:
    explicit BucketizeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }

            // check one attribute
            with_right = layer->GetParamAsBool("with_right_bound");

            // check precisions for input tensors
            Precision input_tensor_precision = layer->insData[INPUT_TENSOR_PORT].lock()->getTensorDesc().getPrecision();
            if (input_tensor_precision != Precision::FP32) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision of the input. Only FP32 is supported!";
            }
            if (with_bins) {
                Precision input_bins_precision = layer->insData[INPUT_BINS_PORT].lock()->getTensorDesc().getPrecision();
                if (input_bins_precision != Precision::FP32) {
                    THROW_IE_EXCEPTION << layer->name
                                       << " Incorrect input precision of the boundaries tensor. Only FP32 is supported!";
                }
            }

            // check dimensions of input tensors
            SizeVector input_tensor_dims = layer->insData[INPUT_TENSOR_PORT].lock()->getTensorDesc().getDims();
            if (input_tensor_dims.size() < 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions of the input.";
            }
            SizeVector input_bin_dims = layer->insData[INPUT_BINS_PORT].lock()->getTensorDesc().getDims();
            if (input_bin_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions of the boundaries tensor.";
            }
            if (input_bin_dims[0] != 0) {
                with_bins = true;
            }
            num_bin_values = input_bin_dims[0];

            num_values = 1;
            for (size_t ind = 0; ind < input_tensor_dims.size(); ind++) {
                num_values *= input_tensor_dims[ind];
            }

            // TODO: check that dense shape value is set
            addConfig(layer,
            { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
            { DataConfigurator(ConfLayout::PLN) });
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *input_tensor_ptr = inputs[INPUT_TENSOR_PORT]->cbuffer().as<const float *>() +
            inputs[INPUT_TENSOR_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float *input_bins_ptr = nullptr;
        if (with_bins) {
            input_bins_ptr = inputs[INPUT_BINS_PORT]->cbuffer().as<const float *>() +
                inputs[INPUT_BINS_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        }
        int *output_tensor_ptr = outputs[OUTPUT_TENSOR_PORT]->cbuffer().as<int *>() +
            inputs[OUTPUT_TENSOR_PORT]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (with_bins == false) {
            for (size_t ind = 0; ind < num_values; ind++) {
                output_tensor_ptr[ind] = 0;
            }
            return OK;
        }

        for (size_t ind = 0; ind < num_values; ind++) {
            float value = input_tensor_ptr[ind];

            // find a bin to which value belongs
            output_tensor_ptr[ind] = -1;
            for (size_t bin_ind = 0; bin_ind < num_bin_values; bin_ind++) {
                if (with_right && value <= input_bins_ptr[bin_ind]) {
                    output_tensor_ptr[ind] = static_cast<int>(bin_ind);
                    break;
                } else if (!with_right && value < input_bins_ptr[bin_ind]) {
                    output_tensor_ptr[ind] = static_cast<int>(bin_ind);
                    break;
                }
            }
            if (output_tensor_ptr[ind] == -1) {
                output_tensor_ptr[ind] = static_cast<int>(num_bin_values);
            }
        }

        return OK;
    }

private:
    const size_t INPUT_TENSOR_PORT = 0;
    const size_t INPUT_BINS_PORT = 1;
    const size_t OUTPUT_TENSOR_PORT = 0;

    size_t num_values = 0;
    size_t num_bin_values = 0;
    bool with_right = false;
    bool with_bins = false;
};

REG_FACTORY_FOR(BucketizeImpl, Bucketize);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
