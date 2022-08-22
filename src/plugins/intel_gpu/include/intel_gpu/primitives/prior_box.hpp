// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "primitive.hpp"

namespace cldnn {

struct prior_box_attributes {
    std::vector<float> min_size;       // Desired min_size of prior boxes
    std::vector<float> max_size;       // Desired max_size of prior boxes
    std::vector<float> aspect_ratio;   // Aspect ratios of prior boxes
    std::vector<float> density;        // This is the square root of the number of boxes of each type
    std::vector<float> fixed_ratio;    // This is an aspect ratio of a box
    std::vector<float> fixed_size;     // This is an initial box size (in pixels)
    bool clip;                         // Clip output to [0,1]
    bool flip;                         // Flip aspect ratios
    float step;                        // Distance between prior box centers
    float offset;                      // Box offset relative to top center of image
    std::vector<float> variance;       // Values to adjust prior boxes with
    bool scale_all_sizes;              // Scale all sizes
    bool min_max_aspect_ratios_order;  // Order of output prior box
    std::vector<float> widths;         // Widths
    std::vector<float> heights;        // Heights
    float step_widths;                 // Distance between box centers in width
    float step_heights;                // Distance between box centers in heigth
};

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
    prior_box(const primitive_id& id,
              const std::vector<primitive_id>& inputs,
              int32_t height,
              int32_t width,
              int32_t image_height,
              int32_t image_width,
              prior_box_attributes attributes,
              const cldnn::data_types output_type)
        : primitive_base{id, inputs, ext_prim_id, padding(), optional_data_type(output_type)},
          attributes{attributes},
          width{width},
          height{height},
          image_width{image_width},
          image_height{image_height} {
        aspect_ratios = {1.0f};
        for (const auto& aspect_ratio : attributes.aspect_ratio) {
            bool exist = false;
            for (const auto existed_value : aspect_ratios) {
                exist |= std::fabs(aspect_ratio - existed_value) < 1e-6;
            }

            if (!exist) {
                aspect_ratios.push_back(aspect_ratio);
                if (attributes.flip) {
                    aspect_ratios.push_back(1.0f / aspect_ratio);
                }
            }
        }

        variance = attributes.variance;
        if (variance.empty()) {
            variance.push_back(0.1f);
        }

        float step = attributes.step;
        auto min_size = attributes.min_size;
        if (!attributes.scale_all_sizes) {
            // mxnet-like PriorBox
            if (step == -1) {
                step = 1.f * image_height / height;
            } else {
                step *= image_height;
            }
            for (auto& size : min_size) {
                size *= image_height;
            }
        }

        reverse_image_width = 1.0f / image_width;
        reverse_image_height = 1.0f / image_height;

        if (!is_clustered() && step == 0) {
            step_x = image_width / width;
            step_y = image_height / height;
        } else {
            step_x = step;
            step_y = step;
        }
    }
    bool is_clustered() const {
        return !(attributes.widths.empty() && attributes.heights.empty());
    }

public:
    prior_box_attributes attributes;
    // calculated attributes
    std::vector<float> aspect_ratios;
    std::vector<float> variance;
    float reverse_image_width, reverse_image_height;
    float step_x, step_y;
    int64_t width, height;
    int64_t image_width, image_height;
};

}  // namespace cldnn
