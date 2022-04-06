// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief ROIAlign is a pooling layer used over feature maps of
/// non-uniform input sizes and outputs a feature map of a fixed size.
struct roi_align : public primitive_base<roi_align> {
    CLDNN_DECLARE_PRIMITIVE(roi_align)

    /// @brief Pooling mode for the @ref roi_align
    enum PoolingMode {
        Max,
        Avg
    };

    /// @brief Constructs roi_align primitive.
    /// @param id This primitive id.
    /// @param inputs Inputs data primitive ids.
    /// @param pooled_h Height of the ROI output feature map.
    /// @param pooled_w Width of the ROI output feature map.
    /// @param sampling_ratio Number of bins over height and width to use to calculate each output feature map element.
    /// @param spatial_scale multiplicative spatial scale factor to translate ROI coordinates
    /// from their input spatial scale to the scale used when pooling.
    /// @param mode Method to perform pooling to produce output feature map elements.
    /// @param shrink_axis_mask Array of bits, that provide shrinks the dimensionality by 1, taking on the value at index begin[i].
    roi_align(const primitive_id& id,
              const std::vector<primitive_id>& inputs,
              int pooled_h,
              int pooled_w,
              int sampling_ratio,
              float spatial_scale,
              PoolingMode mode,
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding())
        : primitive_base(id, inputs, ext_prim_id, output_padding),
          pooled_h {pooled_h},
          pooled_w {pooled_w},
          sampling_ratio {sampling_ratio},
          spatial_scale {spatial_scale},
          mode {mode}
    {}

    /// @brief Height of the ROI output feature map.
    int pooled_h;
    /// @brief Width of the ROI output feature map.
    int pooled_w;
    /// @brief Number of bins over height and width to use to calculate each output feature map element.
    int sampling_ratio;
    /// @brief multiplicative spatial scale factor to translate ROI coordinates
    /// from their input spatial scale to the scale used when pooling.
    float spatial_scale;
    /// @brief Method to perform pooling to produce output feature map elements.
    PoolingMode mode;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
