// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {
namespace utils {
struct InputGenerateData {
    double start_from = 0;
    uint32_t range = 10;
    int32_t resolution = 1;
    int32_t seed = 1;

    InputGenerateData(double _start_from = 0, uint32_t _range = 10, int32_t _resolution = 1, int32_t _seed = 1)
        : start_from(_start_from),
          range(_range),
          resolution(_resolution),
          seed(_seed){};
};

ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const InputGenerateData& inGenData = InputGenerateData(0, 10, 1, 1));

// Legacy impl for contrig repo
// todo: remove this after dependent repos clean up
ov::Tensor create_and_fill_tensor(const ov::element::Type element_type,
                                  const ov::Shape& shape,
                                  const uint32_t range,
                                  const double_t start_from = 0,
                                  const int32_t resolution = 1,
                                  const int seed = 1);

template <class T>
static ov::runtime::Tensor create_tensor(const ov::element::Type& element_type,
                                         const ov::Shape& shape,
                                         const std::vector<T>& values,
                                         const size_t size = 0) {
    const size_t real_size = size ? size : values.size() * sizeof(T) / element_type.size();
    ov::runtime::Tensor tensor{element_type, shape};
    std::memcpy(tensor.data(), values.data(), std::min(real_size * element_type.size(), sizeof(T) * values.size()));
    return tensor;
}

ov::runtime::Tensor create_and_fill_tensor_act_dft(const ov::element::Type element_type,
                                                   const ov::Shape& shape,
                                                   const uint32_t range = 10,
                                                   const double_t start_from = 0,
                                                   const int32_t resolution = 1,
                                                   const int seed = 1);

ov::runtime::Tensor create_and_fill_tensor_unique_sequence(const ov::element::Type element_type,
                                                           const ov::Shape& shape,
                                                           const int32_t start_from = 0,
                                                           const int32_t resolution = 1,
                                                           const int seed = 1);

ov::runtime::Tensor create_and_fill_tensor_normal_distribution(const ov::element::Type element_type,
                                                               const ov::Shape& shape,
                                                               const float mean,
                                                               const float stddev,
                                                               const int seed = 1);

ov::runtime::Tensor create_and_fill_tensor_consistently(const ov::element::Type element_type,
                                                        const ov::Shape& shape,
                                                        const uint32_t range,
                                                        const int32_t start_from,
                                                        const int32_t resolution);

void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             const double abs_threshold = std::numeric_limits<double>::max(),
             const double rel_threshold = std::numeric_limits<double>::max());
}  // namespace utils
}  // namespace test
}  // namespace ov
