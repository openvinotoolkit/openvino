// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Constructs experimental_detectron_prior_grid_generator primitive.
struct experimental_detectron_prior_grid_generator
    : public primitive_base<experimental_detectron_prior_grid_generator> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_prior_grid_generator)

    experimental_detectron_prior_grid_generator(const primitive_id& id,
                                                const std::vector<primitive_id>& input,
                                                bool flatten,
                                                uint64_t h,
                                                uint64_t w,
                                                float stride_x,
                                                float stride_y,
                                                uint64_t featmap_height,
                                                uint64_t featmap_width,
                                                uint64_t image_height,
                                                uint64_t image_width,
                                                const primitive_id& ext_prim_id = {})
        : primitive_base{id, input, ext_prim_id},
          flatten{flatten},
          h{h},
          w{w},
          stride_x{stride_x},
          stride_y{stride_y},
          featmap_height{featmap_height},
          featmap_width{featmap_width},
          image_height{image_height},
          image_width{image_width} {}

    bool flatten;
    uint64_t h;
    uint64_t w;
    float stride_x;
    float stride_y;
    uint64_t featmap_height;
    uint64_t featmap_width;
    uint64_t image_height;
    uint64_t image_width;
};

}  // namespace cldnn
