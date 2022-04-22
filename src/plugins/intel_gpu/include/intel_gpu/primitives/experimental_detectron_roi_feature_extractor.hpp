// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
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

/// @brief experimental detectron ROI feature extractor
struct experimental_detectron_roi_feature_extractor : public primitive_base<experimental_detectron_roi_feature_extractor> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_roi_feature_extractor)

    /// @brief Constructs experimental_detectron_roi_feature_extractor primitive
    /// @param id This primitive id
    /// @param inputs Inputs for primitive id (ROIs, {pyramid levels, ...}, second_output)
    /// @param output_dim Attribute specifies the width and height of the output tensor
    /// @param pyramid_scales Scales of pyramid levels
    /// @param sampling_ratio Attribute specifies the number of sampling points per the output value
    /// @param aligned Attribute specifies add offset (-0.5) to ROIs sizes or not
    experimental_detectron_roi_feature_extractor(const primitive_id& id,
                                                 const std::vector<input_info>& inputs,
                                                 int output_dim,
                                                 const std::vector<int64_t>& pyramid_scales,
                                                 int sampling_ratio,
                                                 bool aligned,
                                                 const primitive_id& ext_prim_id = "",
                                                 const padding& output_padding = padding()) :
            primitive_base(id, inputs, ext_prim_id, {output_padding}),
            output_dim(output_dim),
            pooled_height(output_dim),
            pooled_width(output_dim),
            pyramid_scales(pyramid_scales),
            sampling_ratio(sampling_ratio),
            aligned(aligned) {}

    int output_dim = 0;
    int pooled_height = 0;
    int pooled_width = 0;
    std::vector<int64_t> pyramid_scales;
    int sampling_ratio = 0;
    bool aligned = false;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
