// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "pooling.hpp"
#include "primitive.hpp"
#include <vector>

namespace cldnn {
struct roi_pooling : public primitive_base<roi_pooling> {
    CLDNN_DECLARE_PRIMITIVE(roi_pooling)

    roi_pooling() : primitive_base("", {}),
                    mode(pooling_mode::max),
                    position_sensitive(false),
                    pooled_width(0),
                    pooled_height(0),
                    spatial_scale(0.0f),
                    trans_std(0.0f),
                    no_trans(false),
                    output_dim(0),
                    part_size(0),
                    group_size(0),
                    spatial_bins_x(1),
                    spatial_bins_y(1) {}

    roi_pooling(const primitive_id& id,
                const input_info& input_data,
                const input_info& input_rois,
                pooling_mode mode,
                bool position_sensitive,
                int pooled_width,
                int pooled_height,
                float spatial_scale,
                int output_dim = 0,
                int spatial_bins_x = 1,
                int spatial_bins_y = 1)
        : primitive_base(id, {input_data, input_rois}),
          mode(mode),
          position_sensitive(position_sensitive),
          pooled_width(pooled_width),
          pooled_height(pooled_height),
          spatial_scale(spatial_scale),
          trans_std(0.0f),
          no_trans(false),
          output_dim(output_dim),
          part_size(0),
          group_size(0),
          spatial_bins_x(spatial_bins_x),
          spatial_bins_y(spatial_bins_y) {}

    roi_pooling(const primitive_id& id,
                const std::vector<input_info>& inputs,
                pooling_mode mode,
                bool position_sensitive,
                int pooled_width,
                int pooled_height,
                float spatial_scale,
                float trans_std,
                bool no_trans,
                int part_size,
                int group_size,
                int output_dim = 0,
                int spatial_bins_x = 1,
                int spatial_bins_y = 1)
        : primitive_base(id, inputs),
          mode(mode),
          position_sensitive(position_sensitive),
          pooled_width(pooled_width),
          pooled_height(pooled_height),
          spatial_scale(spatial_scale),
          trans_std(trans_std),
          no_trans(no_trans),
          output_dim(output_dim),
          part_size(part_size),
          group_size(group_size),
          spatial_bins_x(spatial_bins_x),
          spatial_bins_y(spatial_bins_y) {}

    pooling_mode mode;
    bool position_sensitive;
    int pooled_width;
    int pooled_height;
    float spatial_scale;
    float trans_std;
    bool no_trans;
    int output_dim;
    int part_size;
    int group_size;
    int spatial_bins_x;
    int spatial_bins_y;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mode);
        seed = hash_combine(seed, position_sensitive);
        seed = hash_combine(seed, pooled_width);
        seed = hash_combine(seed, pooled_height);
        seed = hash_combine(seed, spatial_scale);
        seed = hash_combine(seed, trans_std);
        seed = hash_combine(seed, no_trans);
        seed = hash_combine(seed, part_size);
        seed = hash_combine(seed, group_size);
        seed = hash_combine(seed, spatial_bins_x);
        seed = hash_combine(seed, spatial_bins_y);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const roi_pooling>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(mode) &&
               cmp_fields(position_sensitive) &&
               cmp_fields(pooled_width) &&
               cmp_fields(pooled_height) &&
               cmp_fields(spatial_scale) &&
               cmp_fields(trans_std) &&
               cmp_fields(no_trans) &&
               cmp_fields(output_dim) &&
               cmp_fields(part_size) &&
               cmp_fields(group_size) &&
               cmp_fields(spatial_bins_x) &&
               cmp_fields(spatial_bins_y);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<roi_pooling>::save(ob);
        ob << make_data(&mode, sizeof(pooling_mode));
        ob << position_sensitive;
        ob << pooled_width;
        ob << pooled_height;
        ob << spatial_scale;
        ob << trans_std;
        ob << no_trans;
        ob << output_dim;
        ob << part_size;
        ob << group_size;
        ob << spatial_bins_x;
        ob << spatial_bins_y;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<roi_pooling>::load(ib);
        ib >> make_data(&mode, sizeof(pooling_mode));
        ib >> position_sensitive;
        ib >> pooled_width;
        ib >> pooled_height;
        ib >> spatial_scale;
        ib >> trans_std;
        ib >> no_trans;
        ib >> output_dim;
        ib >> part_size;
        ib >> group_size;
        ib >> spatial_bins_x;
        ib >> spatial_bins_y;
    }
};

}  // namespace cldnn
