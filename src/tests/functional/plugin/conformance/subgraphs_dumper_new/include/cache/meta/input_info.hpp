// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

namespace ov {
namespace tools {
namespace subgraph_dumper {

constexpr double DEFAULT_MIN_VALUE = std::numeric_limits<double>::min();
constexpr double DEFAULT_MAX_VALUE = std::numeric_limits<double>::max();

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
    };

    Range ranges;
    bool is_const;

    InputInfo(double in_min = DEFAULT_MIN_VALUE,
              double in_max = DEFAULT_MAX_VALUE,
              bool in_is_const = false) :
              is_const(in_is_const),
              ranges(Range(in_min, in_max)) {}

    bool operator==(const InputInfo& input_info_ref) const {
        return this->is_const == input_info_ref.is_const && this->ranges.max == input_info_ref.ranges.max && this->ranges.min == input_info_ref.ranges.min;
    }
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
