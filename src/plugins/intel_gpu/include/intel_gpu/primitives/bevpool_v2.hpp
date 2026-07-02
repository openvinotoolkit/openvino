// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

struct bound3d {
    float min = 0.f;
    float max = 0.f;
    float step = 1.f;
};

struct bevpool_v2 : public primitive_base<bevpool_v2> {
    CLDNN_DECLARE_PRIMITIVE(bevpool_v2)

    bevpool_v2() : primitive_base("", {}) {}

    bevpool_v2(const primitive_id& id,
               const std::vector<input_info>& inputs,
               uint32_t input_channels,
               uint32_t output_channels,
               uint32_t image_width,
               uint32_t image_height,
               uint32_t feature_width,
               uint32_t feature_height,
               const bound3d& x_bound,
               const bound3d& y_bound,
               const bound3d& z_bound,
               const bound3d& d_bound)
        : primitive_base(id, inputs),
          input_channels(input_channels),
          output_channels(output_channels),
          image_width(image_width),
          image_height(image_height),
          feature_width(feature_width),
          feature_height(feature_height),
          x_bound(x_bound),
          y_bound(y_bound),
          z_bound(z_bound),
          d_bound(d_bound) {}

    uint32_t input_channels = 0;
    uint32_t output_channels = 0;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t feature_width = 0;
    uint32_t feature_height = 0;
    bound3d x_bound{};
    bound3d y_bound{};
    bound3d z_bound{};
    bound3d d_bound{};

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, input_channels);
        seed = hash_combine(seed, output_channels);
        seed = hash_combine(seed, image_width);
        seed = hash_combine(seed, image_height);
        seed = hash_combine(seed, feature_width);
        seed = hash_combine(seed, feature_height);
        seed = hash_combine(seed, x_bound.min);
        seed = hash_combine(seed, x_bound.max);
        seed = hash_combine(seed, x_bound.step);
        seed = hash_combine(seed, y_bound.min);
        seed = hash_combine(seed, y_bound.max);
        seed = hash_combine(seed, y_bound.step);
        seed = hash_combine(seed, z_bound.min);
        seed = hash_combine(seed, z_bound.max);
        seed = hash_combine(seed, z_bound.step);
        seed = hash_combine(seed, d_bound.min);
        seed = hash_combine(seed, d_bound.max);
        seed = hash_combine(seed, d_bound.step);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const bevpool_v2>(rhs);
        return input_channels == rhs_casted.input_channels &&
               output_channels == rhs_casted.output_channels &&
               image_width == rhs_casted.image_width &&
               image_height == rhs_casted.image_height &&
               feature_width == rhs_casted.feature_width &&
               feature_height == rhs_casted.feature_height &&
               x_bound.min == rhs_casted.x_bound.min &&
               x_bound.max == rhs_casted.x_bound.max &&
               x_bound.step == rhs_casted.x_bound.step &&
               y_bound.min == rhs_casted.y_bound.min &&
               y_bound.max == rhs_casted.y_bound.max &&
               y_bound.step == rhs_casted.y_bound.step &&
               z_bound.min == rhs_casted.z_bound.min &&
               z_bound.max == rhs_casted.z_bound.max &&
               z_bound.step == rhs_casted.z_bound.step &&
               d_bound.min == rhs_casted.d_bound.min &&
               d_bound.max == rhs_casted.d_bound.max &&
               d_bound.step == rhs_casted.d_bound.step;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<bevpool_v2>::save(ob);
        ob << input_channels;
        ob << output_channels;
        ob << image_width;
        ob << image_height;
        ob << feature_width;
        ob << feature_height;
        ob << x_bound.min << x_bound.max << x_bound.step;
        ob << y_bound.min << y_bound.max << y_bound.step;
        ob << z_bound.min << z_bound.max << z_bound.step;
        ob << d_bound.min << d_bound.max << d_bound.step;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<bevpool_v2>::load(ib);
        ib >> input_channels;
        ib >> output_channels;
        ib >> image_width;
        ib >> image_height;
        ib >> feature_width;
        ib >> feature_height;
        ib >> x_bound.min >> x_bound.max >> x_bound.step;
        ib >> y_bound.min >> y_bound.max >> y_bound.step;
        ib >> z_bound.min >> z_bound.max >> z_bound.step;
        ib >> d_bound.min >> d_bound.max >> d_bound.step;
    }
};

}  // namespace cldnn
