// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include "openvino/openvino.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

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

    InputInfo(double in_min = DEFAULT_MIN_VALUE,
              double in_max = DEFAULT_MAX_VALUE,
              bool in_is_const = false) :
              is_const(in_is_const),
              ranges(Range(in_min, in_max)) {}

    bool operator==(const InputInfo& input_info_ref) const {
        return this->is_const == input_info_ref.is_const && this->ranges == input_info_ref.ranges;
    }
};

using ExtractedPattern = std::pair<std::shared_ptr<ov::Model>, std::map<std::string, InputInfo>>;

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
