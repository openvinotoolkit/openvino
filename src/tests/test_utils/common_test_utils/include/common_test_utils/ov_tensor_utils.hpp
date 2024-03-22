// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {
namespace utils {

// todo: remove w/a to generate correct constant data (replace parameter to const) in conformance with defined range
struct ConstRanges {
    static double max, min;
    static bool is_defined;

    static void set(double _min, double _max) {
        min = _min;
        max = _max;
        is_defined = true;
    }

    static void reset() {
        min = std::numeric_limits<double>::max();
        max = std::numeric_limits<double>::min();
        is_defined = false;
    }
};

struct InputGenerateData {
    double start_from = 0;
    uint32_t range = 10;
    int32_t resolution = 1;
    int32_t seed = 1;

    InputGenerateData(double _start_from = 0, uint32_t _range = 10, int32_t _resolution = 1, int32_t _seed = 1)
        : start_from(_start_from),
          range(_range),
          resolution(_resolution),
          seed(_seed) {
        if (ConstRanges::is_defined) {
            auto min_orig = start_from;
            auto max_orig = start_from + range * resolution;
            auto min_ref = ConstRanges::min;
            auto max_ref = ConstRanges::max;
            if (min_orig < min_ref || min_orig == 0)
                start_from = min_ref;
            range =
                (uint32_t)round((max_orig > max_ref || max_orig == 10 ? max_ref : max_orig - start_from) - start_from);
        }
    };
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
static ov::Tensor create_tensor(const ov::element::Type& element_type,
                                const ov::Shape& shape,
                                const std::vector<T>& values,
                                const size_t size = 0) {
    const size_t real_size = size ? size : values.size() * sizeof(T) / element_type.size();
    ov::Tensor tensor{element_type, shape};
    std::memcpy(tensor.data(), values.data(), std::min(real_size * element_type.size(), sizeof(T) * values.size()));
    return tensor;
}

ov::Tensor create_and_fill_tensor_act_dft(const ov::element::Type element_type,
                                          const ov::Shape& shape,
                                          const uint32_t range = 10,
                                          const double_t start_from = 0,
                                          const int32_t resolution = 1,
                                          const int seed = 1);

ov::Tensor create_and_fill_tensor_unique_sequence(const ov::element::Type element_type,
                                                  const ov::Shape& shape,
                                                  const int32_t start_from = 0,
                                                  const int32_t resolution = 1,
                                                  const int seed = 1);

ov::Tensor create_and_fill_tensor_normal_distribution(const ov::element::Type element_type,
                                                      const ov::Shape& shape,
                                                      const float mean,
                                                      const float stddev,
                                                      const int seed = 1);

ov::Tensor create_and_fill_tensor_consistently(const ov::element::Type element_type,
                                               const ov::Shape& shape,
                                               const uint32_t range,
                                               const int32_t start_from,
                                               const int32_t resolution);

ov::Tensor create_and_fill_tensor_real_distribution(const ov::element::Type element_type,
                                                    const ov::Shape& shape,
                                                    const float min,
                                                    const float max,
                                                    const int seed);

void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             const double abs_threshold = std::numeric_limits<double>::max(),
             const double rel_threshold = std::numeric_limits<double>::max());

void compare_str(const ov::Tensor& expected, const ov::Tensor& actual);
}  // namespace utils
}  // namespace test
}  // namespace ov
