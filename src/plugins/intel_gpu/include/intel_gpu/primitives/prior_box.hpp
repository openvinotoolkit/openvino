// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include "openvino/op/prior_box.hpp"
#include "openvino/op/prior_box_clustered.hpp"

#include <cmath>
#include <vector>
#include <limits>

namespace cldnn {

/// @brief Generates a set of default bounding boxes with different sizes and aspect ratios.
/// @details The prior-boxes are shared across all the images in a batch (since they have the same width and height).
/// First feature stores the mean of each prior coordinate.
/// Second feature stores the variance of each prior coordinate.
struct prior_box : public primitive_base<prior_box> {
    CLDNN_DECLARE_PRIMITIVE(prior_box)

    prior_box() : primitive_base("", {}) {}

    using PriorBoxV0Op = ov::op::v0::PriorBox;
    using PriorBoxV8Op = ov::op::v8::PriorBox;
    using PriorBoxClusteredOp = ov::op::v0::PriorBoxClustered;

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
    /// @param step_width Step.
    /// @param offset Offset to the top left corner of each cell.
    prior_box(const primitive_id& id,
              const std::vector<input_info>& inputs,
              const tensor& output_size,
              const tensor& img_size,
              const std::vector<float>& min_sizes,
              const std::vector<float>& max_sizes = {},
              const std::vector<float>& aspect_ratios = {},
              const bool flip = true,
              const bool clip = false,
              const std::vector<float>& variance = {},
              const float step = 0.0f,
              const float offset = 0.5f,
              const bool scale_all_sizes = true,
              const std::vector<float>& fixed_ratio = {},
              const std::vector<float>& fixed_size = {},
              const std::vector<float>& density = {},
              const bool support_opset8 = false,
              const bool min_max_aspect_ratios_order = true)
        : primitive_base{id, inputs},
          output_size(output_size),
          img_size(img_size),
          min_sizes(min_sizes),
          max_sizes(max_sizes),
          flip(flip),
          clip(clip),
          step{step},
          offset(offset),
          scale_all_sizes(scale_all_sizes),
          fixed_ratio(fixed_ratio),
          fixed_size(fixed_size),
          density(density),
          support_opset8{support_opset8},
          min_max_aspect_ratios_order{min_max_aspect_ratios_order},
          clustered(false) {
        init(aspect_ratios, variance);
    }

    /// @brief Constructs prior-box primitive, which executes clustered version.
    prior_box(const primitive_id& id,
              const std::vector<input_info>& inputs,
              const tensor& img_size,
              const bool clip,
              const std::vector<float>& variance,
              const float step_width,
              const float step_height,
              const float offset,
              const std::vector<float>& widths,
              const std::vector<float>& heights,
              data_types output_dt)
        : primitive_base(id, inputs, 1, {optional_data_type{output_dt}}),
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

    PriorBoxV0Op::Attributes get_attrs_v0() const {
        PriorBoxV0Op::Attributes attrs;
        attrs.min_size = min_sizes;
        attrs.max_size = max_sizes;
        attrs.aspect_ratio = aspect_ratios;
        attrs.density = density;
        attrs.fixed_ratio = fixed_ratio;
        attrs.fixed_size = fixed_size;
        attrs.clip = clip;
        attrs.flip = flip;
        attrs.step = step;
        attrs.offset = offset;
        attrs.variance = variance;
        attrs.scale_all_sizes = scale_all_sizes;
        return attrs;
    }

    PriorBoxV8Op::Attributes get_attrs_v8() const {
        PriorBoxV8Op::Attributes attrs;
        attrs.min_size = min_sizes;
        attrs.max_size = max_sizes;
        attrs.aspect_ratio = aspect_ratios;
        attrs.density = density;
        attrs.fixed_ratio = fixed_ratio;
        attrs.fixed_size = fixed_size;
        attrs.clip = clip;
        attrs.flip = flip;
        attrs.step = step;
        attrs.offset = offset;
        attrs.variance = variance;
        attrs.scale_all_sizes = scale_all_sizes;
        attrs.min_max_aspect_ratios_order = min_max_aspect_ratios_order;
        return attrs;
    }

    PriorBoxClusteredOp::Attributes get_attrs_clustered() const {
        PriorBoxClusteredOp::Attributes attrs;
        attrs.widths = widths;
        attrs.heights = heights;
        attrs.clip = clip;
        attrs.step_widths = step_width;
        attrs.step_heights = step_height;
        attrs.step = step;
        attrs.offset = offset;
        attrs.variances = variance;
        return attrs;
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
    /// @brief Step width for clustered version.
    float step_width{0.0f};
    /// @brief Step height for clustered version.
    float step_height{0.0f};
    /// @brief Step.
    float step{0.0f};
    /// @brief Offset to the top left corner of each cell.
    float offset{0.0f};
    /// @brief If false, only first min_size is scaled by aspect_ratios
    bool scale_all_sizes{true};

    std::vector<float> fixed_ratio{};
    std::vector<float> fixed_size{};
    std::vector<float> density{};

    // required for v8
    bool support_opset8{false};
    bool min_max_aspect_ratios_order{true};

    /// @brief Required for clustered version.
    std::vector<float> widths{};
    /// @brief Required for clustered version.
    std::vector<float> heights{};

    bool is_clustered() const { return clustered; }
    bool is_v8_support() const { return support_opset8;}

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, img_size.spatial[0]);
        seed = hash_combine(seed, img_size.spatial[1]);

        seed = hash_range(seed, min_sizes.begin(), min_sizes.end());
        seed = hash_range(seed, max_sizes.begin(), max_sizes.end());
        seed = hash_range(seed, aspect_ratios.begin(), aspect_ratios.end());

        seed = hash_combine(seed, flip);
        seed = hash_combine(seed, clip);

        seed = hash_range(seed, variance.begin(), variance.end());
        seed = hash_combine(seed, step_width);
        seed = hash_combine(seed, step_height);
        seed = hash_combine(seed, offset);
        seed = hash_combine(seed, scale_all_sizes);

        seed = hash_range(seed, fixed_ratio.begin(), fixed_ratio.end());
        seed = hash_range(seed, fixed_size.begin(), fixed_size.end());
        seed = hash_range(seed, density.begin(), density.end());

        seed = hash_combine(seed, support_opset8);
        seed = hash_combine(seed, step);
        seed = hash_combine(seed, min_max_aspect_ratios_order);

        seed = hash_range(seed, widths.begin(), widths.end());
        seed = hash_range(seed, heights.begin(), heights.end());

        seed = hash_combine(seed, clustered);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const prior_box>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(img_size) &&
               cmp_fields(min_sizes) &&
               cmp_fields(max_sizes) &&
               cmp_fields(aspect_ratios) &&
               cmp_fields(flip) &&
               cmp_fields(clip) &&
               cmp_fields(variance) &&
               cmp_fields(step_width) &&
               cmp_fields(step_height) &&
               cmp_fields(offset) &&
               cmp_fields(scale_all_sizes) &&
               cmp_fields(fixed_ratio) &&
               cmp_fields(fixed_size) &&
               cmp_fields(density) &&
               cmp_fields(support_opset8) &&
               cmp_fields(step) &&
               cmp_fields(min_max_aspect_ratios_order) &&
               cmp_fields(widths) &&
               cmp_fields(heights) &&
               cmp_fields(clustered);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<prior_box>::save(ob);
        ob << output_size;
        ob << img_size;
        ob << min_sizes;
        ob << max_sizes;
        ob << aspect_ratios;
        ob << flip;
        ob << clip;
        ob << variance;
        ob << step_width;
        ob << step_height;
        ob << offset;
        ob << scale_all_sizes;
        ob << fixed_ratio;
        ob << fixed_size;
        ob << density;
        ob << support_opset8;
        ob << step;
        ob << min_max_aspect_ratios_order;
        ob << widths;
        ob << heights;
        ob << clustered;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<prior_box>::load(ib);
        ib >> output_size;
        ib >> img_size;
        ib >> min_sizes;
        ib >> max_sizes;
        ib >> aspect_ratios;
        ib >> flip;
        ib >> clip;
        ib >> variance;
        ib >> step_width;
        ib >> step_height;
        ib >> offset;
        ib >> scale_all_sizes;
        ib >> fixed_ratio;
        ib >> fixed_size;
        ib >> density;
        ib >> support_opset8;
        ib >> step;
        ib >> min_max_aspect_ratios_order;
        ib >> widths;
        ib >> heights;
        ib >> clustered;
    }

private:
    bool clustered = false;

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
}  // namespace cldnn
