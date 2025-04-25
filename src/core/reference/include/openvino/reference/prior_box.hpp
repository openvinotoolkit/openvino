// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/core/except.hpp"
#include "openvino/op/prior_box.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
static inline float clip_great(float x, float threshold) {
    return x < threshold ? x : threshold;
}

static inline float clip_less(float x, float threshold) {
    return x > threshold ? x : threshold;
}

template <typename T>
void prior_box(const T* data,
               const T* img,
               float* dst_data,
               const Shape& out_shape,
               const op::v8::PriorBox::Attributes& attrs) {
    const int64_t W = data[1];
    const int64_t H = data[0];
    const int64_t IW = img[1];
    const int64_t IH = img[0];

    const int64_t OH = out_shape[1];
    const int64_t OW = 1;

    std::vector<float> aspect_ratios = {1.0f};
    for (const auto& aspect_ratio : attrs.aspect_ratio) {
        bool exist = false;
        for (const auto existed_value : aspect_ratios)
            exist |= std::fabs(aspect_ratio - existed_value) < 1e-6;

        if (!exist) {
            aspect_ratios.push_back(aspect_ratio);
            if (attrs.flip) {
                aspect_ratios.push_back(1.0f / aspect_ratio);
            }
        }
    }

    std::vector<float> variance = attrs.variance;
    OPENVINO_ASSERT(variance.size() == 1 || variance.size() == 4 || variance.empty());
    if (variance.empty())
        variance.push_back(0.1f);

    int64_t num_priors = op::v8::PriorBox::number_of_priors(attrs);

    float step = attrs.step;
    auto min_size = attrs.min_size;
    if (!attrs.scale_all_sizes) {
        // mxnet-like PriorBox
        if (step == -1)
            step = 1.f * IH / H;
        else
            step *= IH;
        for (auto& size : min_size)
            size *= IH;
    }

    int64_t idx = 0;
    float center_x, center_y, box_width, box_height, step_x, step_y;
    float IWI = 1.0f / static_cast<float>(IW);
    float IHI = 1.0f / static_cast<float>(IH);

    if (step == 0) {
        step_x = static_cast<float>(IW) / W;
        step_y = static_cast<float>(IH) / H;
    } else {
        step_x = step;
        step_y = step;
    }

    auto calculate_data =
        [&dst_data, &IWI, &IHI, &idx](float center_x, float center_y, float box_width, float box_height, bool clip) {
            if (clip) {
                // order: xmin, ymin, xmax, ymax
                dst_data[idx++] = clip_less((center_x - box_width) * IWI, 0);
                dst_data[idx++] = clip_less((center_y - box_height) * IHI, 0);
                dst_data[idx++] = clip_great((center_x + box_width) * IWI, 1);
                dst_data[idx++] = clip_great((center_y + box_height) * IHI, 1);
            } else {
                dst_data[idx++] = (center_x - box_width) * IWI;
                dst_data[idx++] = (center_y - box_height) * IHI;
                dst_data[idx++] = (center_x + box_width) * IWI;
                dst_data[idx++] = (center_y + box_height) * IHI;
            }
        };

    for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
            if (step == 0) {
                center_x = (w + 0.5f) * step_x;
                center_y = (h + 0.5f) * step_y;
            } else {
                center_x = (attrs.offset + w) * step;
                center_y = (attrs.offset + h) * step;
            }

            for (size_t s = 0; s < attrs.fixed_size.size(); ++s) {
                auto fixed_size_ = static_cast<size_t>(attrs.fixed_size[s]);
                box_width = box_height = fixed_size_ * 0.5f;

                if (!attrs.fixed_ratio.empty()) {
                    for (float ar : attrs.fixed_ratio) {
                        auto density_ = static_cast<int64_t>(attrs.density[s]);
                        auto shift = static_cast<int64_t>(attrs.fixed_size[s] / density_);
                        ar = std::sqrt(ar);
                        float box_width_ratio = attrs.fixed_size[s] * 0.5f * ar;
                        float box_height_ratio = attrs.fixed_size[s] * 0.5f / ar;
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                                calculate_data(center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true);
                            }
                        }
                    }
                } else {
                    if (!attrs.density.empty()) {
                        auto density_ = static_cast<int64_t>(attrs.density[s]);
                        auto shift = static_cast<int64_t>(attrs.fixed_size[s] / density_);
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                                calculate_data(center_x_temp, center_y_temp, box_width, box_height, true);
                            }
                        }
                    }
                    //  Rest of priors
                    for (float ar : aspect_ratios) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        auto density_ = static_cast<int64_t>(attrs.density[s]);
                        auto shift = static_cast<int64_t>(attrs.fixed_size[s] / density_);
                        ar = std::sqrt(ar);
                        float box_width_ratio = attrs.fixed_size[s] * 0.5f * ar;
                        float box_height_ratio = attrs.fixed_size[s] * 0.5f / ar;
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                                calculate_data(center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true);
                            }
                        }
                    }
                }
            }

            for (size_t ms_idx = 0; ms_idx < min_size.size(); ms_idx++) {
                box_width = min_size[ms_idx] * 0.5f;
                box_height = min_size[ms_idx] * 0.5f;
                calculate_data(center_x, center_y, box_width, box_height, false);

                if (attrs.min_max_aspect_ratios_order) {
                    if (attrs.max_size.size() > ms_idx) {
                        box_width = box_height = std::sqrt(min_size[ms_idx] * attrs.max_size[ms_idx]) * 0.5f;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }

                    if (attrs.scale_all_sizes || (!attrs.scale_all_sizes && (ms_idx == min_size.size() - 1))) {
                        size_t s_idx = attrs.scale_all_sizes ? ms_idx : 0;
                        for (float ar : aspect_ratios) {
                            if (std::fabs(ar - 1.0f) < 1e-6) {
                                continue;
                            }

                            ar = std::sqrt(ar);
                            box_width = min_size[s_idx] * 0.5f * ar;
                            box_height = min_size[s_idx] * 0.5f / ar;
                            calculate_data(center_x, center_y, box_width, box_height, false);
                        }
                    }
                } else {
                    if (attrs.scale_all_sizes || (!attrs.scale_all_sizes && (ms_idx == min_size.size() - 1))) {
                        size_t s_idx = attrs.scale_all_sizes ? ms_idx : 0;
                        for (float ar : aspect_ratios) {
                            if (std::fabs(ar - 1.0f) < 1e-6) {
                                continue;
                            }

                            ar = std::sqrt(ar);
                            box_width = min_size[s_idx] * 0.5f * ar;
                            box_height = min_size[s_idx] * 0.5f / ar;
                            calculate_data(center_x, center_y, box_width, box_height, false);
                        }
                    }

                    if (attrs.max_size.size() > ms_idx) {
                        box_width = box_height = std::sqrt(min_size[ms_idx] * attrs.max_size[ms_idx]) * 0.5f;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }
                }
            }
        }
    }

    if (attrs.clip) {
        for (int64_t i = 0; i < H * W * num_priors * 4; ++i) {
            dst_data[i] = (std::min)((std::max)(dst_data[i], 0.0f), 1.0f);
        }
    }

    uint64_t channel_size = OH * OW;
    if (variance.size() == 1) {
        for (uint64_t i = 0; i < channel_size; ++i) {
            dst_data[i + channel_size] = variance[0];
        }
    } else {
        for (int64_t i = 0; i < H * W * num_priors; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                dst_data[i * 4 + j + channel_size] = variance[j];
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
