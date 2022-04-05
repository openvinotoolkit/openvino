// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include <string>
#include <vector>
#include <utility>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs RoI Align using image pyramid.
/// @details Applies RoI Align to layer from the image pyramid.
/// @par Level of the pyramid is selected by equation:
///    floor(START_LEVEL + log2(sqrt(w * h) / IMAGE_SIZE)
/// @par Where:
///   @li w, h - width and heigt of the region
///   @li START_LEVEL - scale of first level of the pyramid
///   @li IMAGE_SIZE - original image size
/// @par RoI Align algorithm performs max-pooling on region of interest
///   using billinear interpolation of surrounding values to avoid quantization.
struct pyramid_roi_align : public primitive_base<pyramid_roi_align> {
    CLDNN_DECLARE_PRIMITIVE(pyramid_roi_align)

    /// @param id This primitive id.
    /// @param rois Input RoI boxes as tuple [x1, y1, x2, y2] describing two opposite corners of the region.
    /// @param P2 First level of the image pyramid.
    /// @param P3 Second level of the image pyramid.
    /// @param P4 Third level of the image pyramid.
    /// @param P5 Fourth level of the image pyramid.
    /// @param output_size Output pooling size from the region pooling.
    /// @param sampling_ratio Number of sampling points per output value.
    /// @param pyramid_scales Scales of each level of pyramid in relation to original image.
    /// @param pyramid_starting_level Starting level of the pyramid that should be used for region of whole image.
    pyramid_roi_align(const primitive_id& id,
                      const primitive_id& rois,
                      const primitive_id& P2,
                      const primitive_id& P3,
                      const primitive_id& P4,
                      const primitive_id& P5,
                      int output_size,
                      int sampling_ratio,
                      std::vector<int> pyramid_scales,
                      int pyramid_starting_level,
                      const primitive_id& ext_prim_id = "",
                      const padding &output_padding = padding())
        : primitive_base(id,
                         { rois, P2, P3, P4, P5 },
                         ext_prim_id,
                         output_padding)
        , output_size(output_size)
        , sampling_ratio(sampling_ratio)
        , pyramid_scales(std::move(pyramid_scales))
        , pyramid_starting_level(pyramid_starting_level)
    {}

    int output_size;
    int sampling_ratio;
    std::vector<int> pyramid_scales;
    int pyramid_starting_level;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
