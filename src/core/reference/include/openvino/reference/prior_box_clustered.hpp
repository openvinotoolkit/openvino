// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/core/except.hpp"
#include "openvino/op/prior_box_clustered.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T>
void prior_box_clustered(const T* data,
                         const T* img,
                         float* dst_data,
                         const Shape& out_shape,
                         const op::v0::PriorBoxClustered::Attributes& attrs) {
    size_t num_priors_ = attrs.widths.size();

    auto variances = attrs.variances;
    OPENVINO_ASSERT(variances.size() == 1 || variances.size() == 4 || variances.empty());
    if (variances.empty())
        variances.push_back(0.1f);

    // Execute
    const int64_t layer_width = data[1];
    const int64_t layer_height = data[0];

    int64_t img_width = img[1];
    int64_t img_height = img[0];

    float step_w = attrs.step_widths == 0 ? attrs.step : attrs.step_widths;
    float step_h = attrs.step_heights == 0 ? attrs.step : attrs.step_heights;

    if (step_w == 0 && step_h == 0) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }

    size_t var_size = variances.size();
    for (int64_t h = 0; h < layer_height; ++h) {
        for (int64_t w = 0; w < layer_width; ++w) {
            float center_x = (w + attrs.offset) * step_w;
            float center_y = (h + attrs.offset) * step_h;

            for (size_t s = 0; s < num_priors_; ++s) {
                float box_width = attrs.widths[s];
                float box_height = attrs.heights[s];

                float xmin = (center_x - box_width / 2.0f) / img_width;
                float ymin = (center_y - box_height / 2.0f) / img_height;
                float xmax = (center_x + box_width / 2.0f) / img_width;
                float ymax = (center_y + box_height / 2.0f) / img_height;

                if (attrs.clip) {
                    xmin = (std::min)((std::max)(xmin, 0.0f), 1.0f);
                    ymin = (std::min)((std::max)(ymin, 0.0f), 1.0f);
                    xmax = (std::min)((std::max)(xmax, 0.0f), 1.0f);
                    ymax = (std::min)((std::max)(ymax, 0.0f), 1.0f);
                }

                auto get_idx = [&](uint64_t cnt) -> uint64_t {
                    return h * layer_width * num_priors_ * cnt + w * num_priors_ * cnt + s * cnt;
                };

                uint64_t idx = get_idx(4);
                dst_data[idx + 0] = xmin;
                dst_data[idx + 1] = ymin;
                dst_data[idx + 2] = xmax;
                dst_data[idx + 3] = ymax;

                idx = get_idx(4);

                // At this point we have either:
                // 1. A single variance value (to be repeated 4 times for each prior)
                // 2. 4 variance values
                if (var_size == 1) {
                    for (size_t j = 0; j < 4; j++)
                        dst_data[idx + j + out_shape[1]] = variances[0];
                } else {
                    for (size_t j = 0; j < var_size; j++)
                        dst_data[idx + j + out_shape[1]] = variances[j];
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
