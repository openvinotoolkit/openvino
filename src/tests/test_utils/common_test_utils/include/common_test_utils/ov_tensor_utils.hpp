// Copyright (C) 2018-2025 Intel Corporation
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
    bool input_attribute = false;

    InputGenerateData(double _start_from = 0,
                      uint32_t _range = 10,
                      int32_t _resolution = 1,
                      int32_t _seed = 1,
                      bool _input_attribute = false)
        : start_from(_start_from),
          range(_range),
          resolution(_resolution),
          seed(_seed),
          input_attribute(_input_attribute) {
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

    bool correct_range(const InputGenerateData new_range) {
        bool success = true;

        double new_max = new_range.start_from + new_range.range;
        double current_max = start_from + range;

        if (start_from == new_range.start_from) {
            // nothing to do - -----start_curr/new+++++++++++++++range*res curr/new-----------------------
            // nothing to do - -----start_curr/new+++++++++++++++range*res curr----------range*res new----
            // reduce range  - -----start_curr/new+++++++++++++++range*res new-----------range*res curr---
            if (current_max > new_max) {
                range = new_range.range;
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
            }
        } else if (start_from > new_range.start_from) {
            // nothing to do        - -----start_new-----start_curr++++++++++range*res curr/new-------------------
            // nothing to do        - -----start_new-----start_curr++++++++++range*res curr------range*res new----
            // reduce range         - -----start_new-----start_curr++++++++++range*res new-------range*res curr---
            // could not find range - -----start_new---range*res new-----start_curr-----range*res curr---
            if (start_from > new_max) {
                success = false;
                // std::cout << " FAIL TO FIND RANGE: current->start_from > new_range->start_from + new_range->range "
                //           << " current->start_from: " << std::to_string(start_from)
                //           << " new_range->start_from: " << std::to_string(new_range.start_from)
                //           << " new_range max: " << std::to_string(new_max) << std::endl;
            } else if (current_max > new_max) {
                range = (uint32_t)round(new_max - start_from);
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
            }
        } else if (start_from < new_range.start_from) {
            // reset to new         - -----start_curr-----start_new++++++++++range*res curr/new-------------------
            // reset to new         - -----start_curr-----start_new++++++++++range*res new-------range*res curr---
            // recalculate range    - -----start_curr-----start_new++++++++++range*res curr------range*res new----
            // could not find range - -----start_curr---range*res curr-----start_new-----range*res new---
            if (current_max < new_range.start_from) {
                success = false;
                // std::cout << " FAIL TO FIND RANGE: current->start_from + current->range < new_range->start_from "
                //           << " new_range start_from: " << std::to_string(new_range.start_from)
                //           << " current->start_from: " << std::to_string(start_from)
                //           << " current max: " << std::to_string(current_max) << std::endl;
            } else if (current_max >= new_max) {
                start_from = new_range.start_from;
                range = new_range.range;
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
            } else {
                range = (uint32_t)round(current_max - new_range.start_from);
                resolution = new_range.resolution > resolution ? new_range.resolution : resolution;
                start_from = new_range.start_from;
            }
        }

        return success;
    };
};

// Pre-defaned eps based on mantissa bit depth
inline double get_eps_by_ov_type(const ov::element::Type& elem_type) {
    switch (elem_type) {
    case ov::element::f64:
        return 1e-8;
    case ov::element::f32:
        return 1e-4;
    case ov::element::f16:
        return 1e-3;
    case ov::element::bf16:
        return 1e-2;
    case ov::element::nf4:
        return 1e-1;
    default:
        return 0.f;
    }
}

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
namespace tensor_comparation {
double calculate_threshold(const double abs_threshold, const double rel_threshold, const double ref_value);

double calculate_default_abs_threshold(const ov::element::Type& expected_type,
                                       const ov::element::Type& actual_type = ov::element::dynamic,
                                       const ov::element::Type& inference_precision = ov::element::dynamic);

double calculate_default_rel_threshold(const ov::element::Type& expected_type,
                                       const ov::element::Type& actual_type = ov::element::dynamic,
                                       const ov::element::Type& inference_precision = ov::element::dynamic);
}  // namespace tensor_comparation

// function to compare tensors using different metrics:
// `expected` : reference tensor.
// `actual` : plugin/calculated tensor.
// `inference_precision` : real plugin calculation precision. Default abs and rel thresholds will be calculated using
// this value.
// `abs_threshold` : abs difference between reference and calculated value.
// `rel_threshold` : define first incorrect element rank in mantissa.
// `mvn_threshold` : avg value of `std::diff(ref_value - calculated_value) / threshold`.
//  Shows tensor jitter relative to treshold. The value is [0.f, 1.f].
// `topk_threshold` : percentage of incorrect values in tensor. The value is [0.f, 1.f].
void compare(const ov::Tensor& expected,
             const ov::Tensor& actual,
             const ov::element::Type& inference_precision,
             const double abs_threshold = -1,
             const double rel_threshold = -1,
             const double topk_threshold = 1.f,
             const double mvn_threshold = 1.f);

inline void compare(const ov::Tensor& expected,
                    const ov::Tensor& actual,
                    const double abs_threshold = -1,
                    const double rel_threshold = -1,
                    const double topk_threshold = 1.f,
                    const double mvn_threshold = 1.f) {
    compare(expected, actual, ov::element::dynamic, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);
}

// todo: replace this function by `compare(expected, actual)`
void compare_str(const ov::Tensor& expected, const ov::Tensor& actual);
}  // namespace utils
}  // namespace test
}  // namespace ov
