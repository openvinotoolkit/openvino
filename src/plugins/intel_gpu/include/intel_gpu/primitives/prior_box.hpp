// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "primitive.hpp"

#include <cmath>
#include <vector>
#include <limits>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Generates a set of default bounding boxes with different sizes and aspect ratios.
/// @details The prior-boxes are shared across all the images in a batch (since they have the same width and height).
/// First feature stores the mean of each prior coordinate.
/// Second feature stores the variance of each prior coordinate.
struct prior_box : public primitive_base<prior_box> {
    CLDNN_DECLARE_PRIMITIVE(prior_box)

    /// @brief Constructs prior-box primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param img_size Image width and height.
    /// @param min_sizes Minimum box sizes in pixels.
    /// @param max_sizes Maximum box sizes in pixels.
    /// @param aspect_ratios Various of aspect ratios. Duplicate ratios will be ignored.
    /// @param flip If true, will flip each aspect ratio. For example, if there is aspect ratio "r", aspect ratio "1.0/r" we will generated as well.
    /// @param clip If true, will clip the prior so that it is within [0, 1].
    /// @param variance Variance for adjusting the prior boxes.
    /// @param step_width Step width.
    /// @param step_height Step height.
    /// @param offset Offset to the top left corner of each cell.
    prior_box(const primitive_id& id,
              const primitive_id& input,
              const tensor& img_size,
              const std::vector<float>& min_sizes,
              const std::vector<float>& max_sizes = {},
              const std::vector<float>& aspect_ratios = {},
              const bool flip = true,
              const bool clip = false,
              const std::vector<float>& variance = {},
              const float step_width = 0.f,
              const float step_height = 0.f,
              const float offset = 0.5f,
              const bool scale_all_sizes = true,
              const std::vector<float>& fixed_ratio = {},
              const std::vector<float>& fixed_size = {},
              const std::vector<float>& density = {},
              const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          img_size(img_size),
          min_sizes(min_sizes),
          max_sizes(max_sizes),
          flip(flip),
          clip(clip),
          step_width(step_width),
          step_height(step_height),
          offset(offset),
          scale_all_sizes(scale_all_sizes),
          fixed_ratio(fixed_ratio),
          fixed_size(fixed_size),
          density(density),
          clustered(false) {
        init(aspect_ratios, variance);
    }

    /// @brief Constructs prior-box primitive, which supports v8 features.
    prior_box(const primitive_id& id,
              const std::vector<primitive_id>& inputs,
              const tensor& output_size,
              const tensor& img_size,
              const std::vector<float>& min_sizes,
              const std::vector<float>& max_sizes = {},
              const std::vector<float>& aspect_ratios = {},
              const bool flip = true,
              const bool clip = false,
              const std::vector<float>& variance = {},
              const float offset = 0.5f,
              const bool scale_all_sizes = true,
              const std::vector<float>& fixed_ratio = {},
              const std::vector<float>& fixed_size = {},
              const std::vector<float>& density = {},
              const float step = 0.0f,
              const bool min_max_aspect_ratios_order = true
              )
        : primitive_base{id, inputs},
          output_size(output_size),
          img_size(img_size),
          min_sizes(min_sizes),
          max_sizes(max_sizes),
          flip(flip),
          clip(clip),
          offset(offset),
          scale_all_sizes(scale_all_sizes),
          fixed_ratio(fixed_ratio),
          fixed_size(fixed_size),
          density(density),
          support_opset8{true},
          step{step},
          min_max_aspect_ratios_order{min_max_aspect_ratios_order},
          clustered(false) {
        init(aspect_ratios, variance);
    }

    /// @brief Constructs prior-box primitive, which executes clustered version.
    prior_box(const primitive_id& id,
              const primitive_id& input,
              const tensor& img_size,
              const bool clip,
              const std::vector<float>& variance,
              const float step_width,
              const float step_height,
              const float offset,
              const std::vector<float>& widths,
              const std::vector<float>& heights,
              data_types output_dt,
              const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding, optional_data_type{output_dt}),
          img_size(img_size),
          flip(false),
          clip(clip),
          variance(variance),
          step_width(step_width),
          step_height(step_height),
          offset(offset),
          scale_all_sizes(false),
          widths(widths),
          heights(heights),
          clustered(true) {
    }

    /// @brief Spatial size of generated grid with boxes.
    tensor output_size{};
    /// @brief Image width and height.
    tensor img_size{};
    /// @brief  Minimum box sizes in pixels.
    std::vector<float> min_sizes{};
    /// @brief Maximum box sizes in pixels.
    std::vector<float> max_sizes{};
    /// @brief Various of aspect ratios. Duplicate ratios will be ignored.
    std::vector<float> aspect_ratios{};
    /// @brief If true, will flip each aspect ratio. For example, if there is aspect ratio "r", aspect ratio "1.0/r" we will generated as well.
    bool flip{false};
    /// @brief If true, will clip the prior so that it is within [0, 1].
    bool clip{false};
    /// @brief Variance for adjusting the prior boxes.
    std::vector<float> variance{};
    /// @brief Step width.
    float step_width{0.0f};
    /// @brief Step height.
    float step_height{0.0f};
    /// @brief Offset to the top left corner of each cell.
    float offset{0.0f};
    /// @brief If false, only first min_size is scaled by aspect_ratios
    bool scale_all_sizes{true};

    std::vector<float> fixed_ratio{};
    std::vector<float> fixed_size{};
    std::vector<float> density{};

    // required for v8
    bool support_opset8{false};
    float step{0.0f};
    bool min_max_aspect_ratios_order{true};

    /// @brief Required for clustered version.
    std::vector<float> widths{};
    /// @brief Required for clustered version.
    std::vector<float> heights{};

    bool is_clustered() const { return clustered; }

private:
    bool clustered;

    void init(const std::vector<float>& ratios, const std::vector<float>& variances) {
        constexpr auto default_aspect_ratio = 1.0f;
        aspect_ratios.push_back(default_aspect_ratio);
        constexpr auto aspect_ratio_threshold = 1e-6;
        for (auto new_aspect_ratio : ratios) {
            bool already_exist = false;
            for (auto aspect_ratio : aspect_ratios) {
                if (std::fabs(new_aspect_ratio - aspect_ratio) < aspect_ratio_threshold) {
                    already_exist = true;
                    break;
                }
            }
            if (!already_exist) {
                if (std::fabs(new_aspect_ratio) < std::numeric_limits<float>::epsilon()) {
                    throw std::runtime_error("prior_box aspect ratio can't be zero!");
                }
                aspect_ratios.push_back(new_aspect_ratio);
                if (flip) {
                    aspect_ratios.push_back(1.0f / new_aspect_ratio);
                }
            }
        }

        const auto variances_size = variances.size();
        if (variances_size == 0) {
            constexpr auto default_variance = 0.1f;
            variance.push_back(default_variance);
        } else if (variances_size == 1 || variances_size == 4) {
            variance.resize(variances_size);
            std::copy(variances.cbegin(), variances.cend(), variance.begin());
        } else {
            throw std::runtime_error("Variances size must be 0, 1, or 4");
        }
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
