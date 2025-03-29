// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace reference {
namespace nms_common {
struct Rectangle {
    Rectangle(float x_left, float y_left, float x_right, float y_right)
        : x1{x_left},
          y1{y_left},
          x2{x_right},
          y2{y_right} {}

    Rectangle() = default;

    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct BoxInfo {
    BoxInfo(const Rectangle& r, int64_t idx, float sc, int64_t suppress_idx, int64_t batch_idx, int64_t class_idx)
        : box{r},
          index{idx},
          suppress_begin_index{suppress_idx},
          batch_index{batch_idx},
          class_index{class_idx},
          score{sc} {}

    BoxInfo() = default;

    inline bool operator<(const BoxInfo& rhs) const {
        return score < rhs.score || (score == rhs.score && index > rhs.index);
    }

    inline bool operator>(const BoxInfo& rhs) const {
        return !(score < rhs.score || (score == rhs.score && index > rhs.index));
    }

    Rectangle box;
    int64_t index = 0;
    int64_t suppress_begin_index = 0;
    int64_t batch_index = 0;
    int64_t class_index = 0;
    float score = 0.0f;
};

void nms_common_postprocessing(void* prois,
                               void* pscores,
                               void* pselected_num,
                               const element::Type& output_type,
                               const std::vector<float>& selected_outputs,
                               const std::vector<int64_t>& selected_indices,
                               const std::vector<int64_t>& valid_outputs,
                               const element::Type& selected_outputs_type);

}  // namespace nms_common
}  // namespace reference
}  // namespace ov
