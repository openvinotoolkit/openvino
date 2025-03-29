// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "openvino/core/shape.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace conformance {

constexpr double DEFAULT_MIN_VALUE = std::numeric_limits<double>::min();
constexpr double DEFAULT_MAX_VALUE = std::numeric_limits<double>::max();
constexpr double DEFAULT_EPSILON = std::numeric_limits<float>::epsilon();

struct InputInfo {
    struct Range {
        double min, max;

        Range(double in_min, double in_max) : min(in_min), max(in_max) {}

        Range& operator=(const Range& ranges) {
            if (ranges.max != DEFAULT_MAX_VALUE) {
                this->max = this->max != DEFAULT_MAX_VALUE ? std::max(this->max, ranges.max) : ranges.max;
            }
            if (ranges.min != DEFAULT_MIN_VALUE) {
                this->min = this->min != DEFAULT_MIN_VALUE ? std::min(this->min, ranges.min) : ranges.min;
            }
            return *this;
        }

        bool operator==(const Range& ranges) const {
            double max_delta = (this->max - ranges.max) > 0 ? this->max - ranges.max : ranges.max - this->max;
            double min_delta = (this->min - ranges.min) > 0 ? this->min - ranges.min : ranges.min - this->min;
            return max_delta <= DEFAULT_EPSILON && min_delta <= DEFAULT_EPSILON;
        }
    };

    Range ranges;
    bool is_const;
    ov::PartialShape max_shape, min_shape;

    InputInfo(const ov::PartialShape& shape = {},
              double in_min = DEFAULT_MIN_VALUE,
              double in_max = DEFAULT_MAX_VALUE,
              bool in_is_const = false) :
              is_const(in_is_const),
              ranges(Range(in_min, in_max)),
              max_shape(shape),
              min_shape(shape) {}

    bool operator==(const InputInfo& input_info_ref) const {
        return this->is_const == input_info_ref.is_const &&
               this->ranges == input_info_ref.ranges &&
               this->max_shape == input_info_ref.max_shape &&
               this->min_shape == input_info_ref.min_shape;
    }

    InputInfo operator=(const InputInfo& input_info) {
        auto default_in_info = InputInfo();
        if (input_info == default_in_info) {
            this->is_const = input_info.is_const;
        } else if (this->is_const != input_info.is_const) {
            throw std::runtime_error("Cast Const to Parameter! Impossible to update Input Info!");
        }
        this->ranges = input_info.ranges;
        if (ov::shape_size(this->max_shape.get_max_shape()) < ov::shape_size(input_info.max_shape.get_max_shape())) {
            this->max_shape = input_info.max_shape;
        }
        if (ov::shape_size(this->min_shape.get_min_shape()) > ov::shape_size(input_info.min_shape.get_min_shape())) {
            this->min_shape = input_info.min_shape;
        }
        return *this;
    }
};

}  // namespace conformance
}  // namespace ov
