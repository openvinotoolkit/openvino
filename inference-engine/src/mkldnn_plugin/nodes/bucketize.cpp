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

            auto input = layer->insData[INPUT_TENSOR_PORT].lock();
            if (!input) {
                THROW_IE_EXCEPTION << "Missing input for " << layer->name << " layer";
            }
            auto boundaries = layer->insData[INPUT_BINS_PORT].lock();
            if (!boundaries) {
                THROW_IE_EXCEPTION << "Missing boundaries input for " << layer->name << " layer";
            }

            // check precisions for input and output tensors
            input_precision = input->getTensorDesc().getPrecision();
            if (input_precision != Precision::FP32 && input_precision != Precision::I32 &&
                input_precision != Precision::I64) {
                input_precision = Precision::FP32;
            }
            boundaries_precision = boundaries->getTensorDesc().getPrecision();
            if (boundaries_precision != Precision::FP32 && boundaries_precision != Precision::I32 &&
                boundaries_precision != Precision::I64) {
                boundaries_precision = Precision::FP32;
            }
            output_precision = layer->outData[OUTPUT_TENSOR_PORT]->getTensorDesc().getPrecision();
            if (output_precision != Precision::I32 && output_precision != Precision::I64) {
                output_precision = Precision::I32;
            }

            // check dimensions of input tensors
            SizeVector input_tensor_dims = input->getTensorDesc().getDims();
            if (input_tensor_dims.size() < 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions of the input.";
            }
            SizeVector input_bin_dims = boundaries->getTensorDesc().getDims();
            if (input_bin_dims.size() != 1) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions of the boundaries tensor.";
            }
            if (input_bin_dims[0] != 0) {
                with_bins = true;
            }
            num_bin_values = input_bin_dims[0];

            num_values = std::accumulate(input_tensor_dims.begin(), input_tensor_dims.end(), 1, std::multiplies<size_t>());

            addConfig(layer,
            { DataConfigurator(ConfLayout::PLN, input_precision), DataConfigurator(ConfLayout::PLN, boundaries_precision) },
            { DataConfigurator(ConfLayout::PLN, output_precision) });
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        auto precision_mask = getPrecisionMask(input_precision, boundaries_precision, output_precision);

        switch (precision_mask) {
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::FP32>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I32>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I32>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I64>::value_type,
                PrecisionTrait<Precision::I64>::value_type>(inputs[0], inputs[1], outputs[0]);
            break;
        default:
            return GENERAL_ERROR;
        }

        return OK;
    }

private:
    template <typename T, typename T_BOUNDARIES, typename T_IND>
    void bucketize(Blob::Ptr input, Blob::Ptr boundaries, Blob::Ptr output) {
        const auto *input_data = input->cbuffer().as<const T *>();
        const auto *boundaries_data = boundaries->cbuffer().as<const T_BOUNDARIES *>();
        auto *output_data = output->buffer().as<T_IND *>();

        if (with_bins == false) {
            memset(output_data, 0, num_values * sizeof(T_IND));
            return;
        }

        // boundaries are assumed to be sorted and to have unique elements
        parallel_for(num_values, [&](size_t ind) {
            T value = input_data[ind];
            if (with_right) {
                auto low = std::lower_bound(boundaries_data, boundaries_data + num_bin_values, value);
                output_data[ind] = static_cast<T_IND>(low - boundaries_data);
            } else {
                auto up = std::upper_bound(boundaries_data, boundaries_data + num_bin_values, value);
                output_data[ind] = static_cast<T_IND>(up - boundaries_data);
            }
        });
    }

    const size_t INPUT_TENSOR_PORT = 0;
    const size_t INPUT_BINS_PORT = 1;
    const size_t OUTPUT_TENSOR_PORT = 0;

    size_t num_values = 0;
    size_t num_bin_values = 0;
    bool with_right = false;
    bool with_bins = false;

    Precision input_precision;
    Precision boundaries_precision;
    Precision output_precision;
};

REG_FACTORY_FOR(BucketizeImpl, Bucketize);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
