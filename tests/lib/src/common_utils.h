// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/openvino.hpp>
#include <limits>
#include <random>


template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;


/**
 * @brief Fill OpenVINO tensor with random values (OV API 2.0)
 */
template<typename T, typename T2>
void fillTensorRandom(ov::Tensor& tensor,
                      T rand_min = std::numeric_limits<uint8_t>::min(),
                      T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    size_t tensor_size = tensor.get_size();
    if (0 == tensor_size) {
        throw std::runtime_error(
            "Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    }
    T* data = tensor.data<T>();
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++)
        data[i] = static_cast<T>(distribution(gen));
}


/**
 * @brief Fill infer_request tensors with random values (OV API 2)
 */
template<typename T>
void fillTensors(ov::InferRequest &infer_request, std::vector<T> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input_tensor = infer_request.get_tensor(inputs[i]);
        auto type = inputs[i].get_element_type();
        if (type == ov::element::f32) {
            fillTensorRandom<float, float>(input_tensor);
        } else if (type == ov::element::f64) {
            fillTensorRandom<double, double>(input_tensor);
        } else if (type == ov::element::f16) {
            fillTensorRandom<ov::float16, float>(input_tensor);
        } else if (type == ov::element::i32) {
            fillTensorRandom<int32_t, int32_t>(input_tensor);
        } else if (type == ov::element::i64) {
            fillTensorRandom<int64_t, int64_t>(input_tensor);
        } else if ((type == ov::element::u8) || (type == ov::element::boolean)) {
            // uniform_int_distribution<uint8_t> is not allowed in the C++17
            // standard and vs2017/19
            fillTensorRandom<uint8_t, uint32_t>(input_tensor);
        } else if (type == ov::element::i8) {
            // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
            // and vs2017/19
            fillTensorRandom<int8_t, int32_t>(input_tensor,
                                              std::numeric_limits<int8_t>::min(),
                                              std::numeric_limits<int8_t>::max());
        } else if (type == ov::element::u16) {
            fillTensorRandom<uint16_t, uint16_t>(input_tensor);
        } else if (type == ov::element::i16) {
            fillTensorRandom<int16_t, int16_t>(input_tensor);
        } else if (type == ov::element::boolean) {
            fillTensorRandom<uint8_t, uint32_t>(input_tensor, 0, 1);
        } else {
            throw std::logic_error(
                    "Input precision is not supported for " + inputs[i].get_element_type().get_type_name());
        }
        infer_request.set_input_tensor(i, input_tensor);
    }
}
