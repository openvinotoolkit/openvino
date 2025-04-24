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

    experimental_detectron_prior_grid_generator() : primitive_base("", {}) {}

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

    bool flatten = false;
    uint64_t h = 0;
    uint64_t w = 0;
    float stride_x = 0.0f;
    float stride_y = 0.0f;
    uint64_t featmap_height = 0;
    uint64_t featmap_width = 0;
    uint64_t image_height = 0;
    uint64_t image_width = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, flatten);
        seed = hash_combine(seed, h);
        seed = hash_combine(seed, w);
        seed = hash_combine(seed, stride_x);
        seed = hash_combine(seed, stride_y);
        seed = hash_combine(seed, featmap_height);
        seed = hash_combine(seed, featmap_width);
        seed = hash_combine(seed, image_height);
        seed = hash_combine(seed, image_width);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const experimental_detectron_prior_grid_generator>(rhs);

        return flatten == rhs_casted.flatten &&
               h == rhs_casted.h &&
               w == rhs_casted.w &&
               stride_x == rhs_casted.stride_x &&
               stride_y == rhs_casted.stride_y &&
               featmap_height == rhs_casted.featmap_height &&
               featmap_width == rhs_casted.featmap_width &&
               image_height == rhs_casted.image_height &&
               image_width == rhs_casted.image_width;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<experimental_detectron_prior_grid_generator>::save(ob);
        ob << flatten;
        ob << h;
        ob << w;
        ob << stride_x;
        ob << stride_y;
        ob << featmap_height;
        ob << featmap_width;
        ob << image_height;
        ob << image_width;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<experimental_detectron_prior_grid_generator>::load(ib);
        ib >> flatten;
        ib >> h;
        ib >> w;
        ib >> stride_x;
        ib >> stride_y;
        ib >> featmap_height;
        ib >> featmap_width;
        ib >> image_height;
        ib >> image_width;
    }
};

}  // namespace cldnn
