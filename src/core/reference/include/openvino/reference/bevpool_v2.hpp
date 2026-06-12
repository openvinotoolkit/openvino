// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "openvino/op/bevpool_v2.hpp"

namespace ov {
namespace reference {

template <typename T>
bool bevpool_v2(const T* cf_data,
                size_t cf_len,
                const T* dw_data,
                size_t dw_len,
                const std::vector<int64_t>& idx_values,
                const std::vector<int64_t>& itv_values,
                T* out_data,
                size_t out_len,
                uint32_t input_channels,
                uint32_t output_channels,
                uint32_t image_width,
                uint32_t image_height,
                uint32_t feature_width,
                uint32_t feature_height,
                const ov::op::v15::Bound& d_bound) {
    if (itv_values.size() % 3 != 0) {
        return false;
    }

    const auto feature_area = static_cast<int64_t>(image_width) * static_cast<int64_t>(image_height);
    const auto depth_bins = static_cast<int64_t>((d_bound.max - d_bound.min) / d_bound.step);
    const auto depth_span = depth_bins * feature_area;
    const auto out_plane = static_cast<int64_t>(feature_width) * static_cast<int64_t>(feature_height);
    if (feature_area <= 0 || depth_bins <= 0 || depth_span <= 0 || out_plane <= 0) {
        return false;
    }

    std::fill(out_data, out_data + out_len, T{0});

    const auto interval_count = itv_values.size() / 3;
    for (size_t interval = 0; interval < interval_count; ++interval) {
        const auto start = itv_values[interval * 3 + 0];
        const auto end = itv_values[interval * 3 + 1];
        const auto bev_base = itv_values[interval * 3 + 2];
        if (start < 0 || end < start || static_cast<size_t>(end) > idx_values.size()) {
            continue;
        }

        for (uint32_t c = 0; c < output_channels; ++c) {
            float acc = 0.f;
            for (int64_t i = start; i < end; ++i) {
                const auto dw_index = idx_values[static_cast<size_t>(i)];
                if (dw_index < 0 || dw_index >= static_cast<int64_t>(dw_len)) {
                    continue;
                }

                const auto camera_idx = dw_index / depth_span;
                const auto feature_idx = dw_index % feature_area;
                const auto cf_offset = (camera_idx * feature_area + feature_idx) * static_cast<int64_t>(input_channels) +
                                       static_cast<int64_t>(c);
                if (cf_offset < 0 || cf_offset >= static_cast<int64_t>(cf_len)) {
                    continue;
                }

                acc += static_cast<float>(cf_data[cf_offset]) * static_cast<float>(dw_data[dw_index]);
            }

            const auto out_index = bev_base + static_cast<int64_t>(c) * out_plane;
            if (out_index >= 0 && out_index < static_cast<int64_t>(out_len)) {
                out_data[out_index] = static_cast<T>(acc);
            }
        }
    }

    return true;
}

}  // namespace reference
}  // namespace ov
