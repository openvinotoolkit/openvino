// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {



/// @brief Constructs experimental_detectron_prior_grid_generator primitive.
struct experimental_detectron_prior_grid_generator
    : public primitive_base<experimental_detectron_prior_grid_generator> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_prior_grid_generator)

    experimental_detectron_prior_grid_generator(const primitive_id& id,
                                                const std::vector<input_info>& inputs,
                                                bool flatten,
                                                uint64_t h,
                                                uint64_t w,
                                                float stride_x,
                                                float stride_y,
                                                uint64_t featmap_height,
                                                uint64_t featmap_width,
                                                uint64_t image_height,
                                                uint64_t image_width)
        : primitive_base{id, inputs},
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

    size_t hash() const override {
        if (!seed) {
            seed = hash_combine(seed, flatten);
            seed = hash_combine(seed, h);
            seed = hash_combine(seed, w);
            seed = hash_combine(seed, stride_x);
            seed = hash_combine(seed, stride_y);
            seed = hash_combine(seed, featmap_height);
            seed = hash_combine(seed, featmap_width);
            seed = hash_combine(seed, image_height);
            seed = hash_combine(seed, image_width);
        }
        return seed;
    }
};

}  // namespace cldnn
