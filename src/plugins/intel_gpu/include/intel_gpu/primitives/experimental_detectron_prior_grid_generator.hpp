// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {
struct experimental_detectron_prior_grid_generator: public primitive_base<experimental_detectron_prior_grid_generator> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_prior_grid_generator)

    experimental_detectron_prior_grid_generator(
        const primitive_id &id,
        const std::vector<primitive_id> &input,
        const layout &output_layout,
        bool flatten,
        int64_t h,
        int64_t w,
        float stride_x,
        float stride_y,
        int64_t featmap_height,
        int64_t featmap_width,
        int64_t number_of_priors,
        int64_t image_height,
        int64_t image_width,
        const primitive_id &ext_prim_id = { }) :
        primitive_base { id, input, ext_prim_id, output_layout.data_padding },
        output_layout { output_layout },
        flatten { flatten },
        h { h },
        w { w },
        stride_x { stride_x },
        stride_y { stride_y },
        featmap_height { featmap_height },
        featmap_width { featmap_width },
        number_of_priors { number_of_priors },
        image_height { image_height },
        image_width { image_width }
    { }
    layout output_layout;
    bool flatten;
    int64_t h;
    int64_t w;
    float stride_x;
    float stride_y;
    int64_t featmap_height;
    int64_t featmap_width;
    int64_t number_of_priors;
    int64_t image_height;
    int64_t image_width;
};
}  // namespace cldnn
