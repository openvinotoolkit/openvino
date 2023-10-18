// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief ROIAlign is a pooling layer used over feature maps of
/// non-uniform input sizes and outputs a feature map of a fixed size.
struct roi_align : public primitive_base<roi_align> {
    CLDNN_DECLARE_PRIMITIVE(roi_align)

    roi_align() : primitive_base("", {}) {}

    /// @brief Pooling mode for the @ref roi_align
    enum PoolingMode { max, avg };

    /// @brief Aligned mode for the @ref roi_align
    enum AlignedMode { asymmetric, half_pixel_for_nn, half_pixel };

    /// @brief Constructs roi_align primitive.
    /// @param id This primitive id.
    /// @param inputs Inputs data primitive ids.
    /// @param pooled_h Height of the ROI output feature map.
    /// @param pooled_w Width of the ROI output feature map.
    /// @param sampling_ratio Number of bins over height and width to use to calculate each output feature map element.
    /// @param spatial_scale multiplicative spatial scale factor to translate ROI coordinates
    /// from their input spatial scale to the scale used when pooling.
    /// @param pooling_mode Method to perform pooling to produce output feature map elements.
    /// @param aligned_mode Method to coordinates alignment.
    roi_align(const primitive_id& id,
              const std::vector<input_info>& inputs,
              int pooled_h,
              int pooled_w,
              int sampling_ratio,
              float spatial_scale,
              PoolingMode pooling_mode,
              AlignedMode aligned_mode,
              const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}),
          pooled_h{pooled_h},
          pooled_w{pooled_w},
          sampling_ratio{sampling_ratio},
          spatial_scale{spatial_scale},
          pooling_mode{pooling_mode},
          aligned_mode{aligned_mode} {}

    /// @brief Height of the ROI output feature map.
    int pooled_h = 0;
    /// @brief Width of the ROI output feature map.
    int pooled_w = 0;
    /// @brief Number of bins over height and width to use to calculate each output feature map element.
    int sampling_ratio = 0;
    /// @brief multiplicative spatial scale factor to translate ROI coordinates
    /// from their input spatial scale to the scale used when pooling.
    float spatial_scale = false;
    /// @brief Method to perform pooling to produce output feature map elements.
    PoolingMode pooling_mode = PoolingMode::max;
    /// @brief Method to coordinate alignment.
    AlignedMode aligned_mode = AlignedMode::asymmetric;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, sampling_ratio);
        seed = hash_combine(seed, spatial_scale);
        seed = hash_combine(seed, pooling_mode);
        seed = hash_combine(seed, aligned_mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const roi_align>(rhs);

        return pooled_h == rhs_casted.pooled_h &&
               pooled_w == rhs_casted.pooled_w &&
               sampling_ratio == rhs_casted.sampling_ratio &&
               spatial_scale == rhs_casted.spatial_scale &&
               pooling_mode == rhs_casted.pooling_mode &&
               aligned_mode == rhs_casted.aligned_mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<roi_align>::save(ob);
        ob << pooled_h;
        ob << pooled_w;
        ob << sampling_ratio;
        ob << spatial_scale;
        ob << make_data(&pooling_mode, sizeof(PoolingMode));
        ob << make_data(&aligned_mode, sizeof(AlignedMode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<roi_align>::load(ib);
        ib >> pooled_h;
        ib >> pooled_w;
        ib >> sampling_ratio;
        ib >> spatial_scale;
        ib >> make_data(&pooling_mode, sizeof(PoolingMode));
        ib >> make_data(&aligned_mode, sizeof(AlignedMode));
    }
};
}  // namespace cldnn
