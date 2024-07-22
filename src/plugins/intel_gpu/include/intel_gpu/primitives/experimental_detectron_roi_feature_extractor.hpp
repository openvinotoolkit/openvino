// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief experimental detectron ROI feature extractor
struct experimental_detectron_roi_feature_extractor : public primitive_base<experimental_detectron_roi_feature_extractor> {
    CLDNN_DECLARE_PRIMITIVE(experimental_detectron_roi_feature_extractor)

    experimental_detectron_roi_feature_extractor() : primitive_base("", {}) {}

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
                                                 bool aligned) :
            primitive_base(id, inputs),
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

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, output_dim);
        seed = hash_combine(seed, pooled_height);
        seed = hash_combine(seed, pooled_width);
        seed = hash_range(seed, pyramid_scales.begin(), pyramid_scales.end());
        seed = hash_combine(seed, sampling_ratio);
        seed = hash_combine(seed, aligned);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const experimental_detectron_roi_feature_extractor>(rhs);

        return output_dim == rhs_casted.output_dim &&
               pooled_height == rhs_casted.pooled_height &&
               pooled_width == rhs_casted.pooled_width &&
               pyramid_scales == rhs_casted.pyramid_scales &&
               sampling_ratio == rhs_casted.sampling_ratio &&
               aligned == rhs_casted.aligned;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<experimental_detectron_roi_feature_extractor>::save(ob);
        ob << output_dim;
        ob << pooled_height;
        ob << pooled_width;
        ob << pyramid_scales;
        ob << sampling_ratio;
        ob << aligned;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<experimental_detectron_roi_feature_extractor>::load(ib);
        ib >> output_dim;
        ib >> pooled_height;
        ib >> pooled_width;
        ib >> pyramid_scales;
        ib >> sampling_ratio;
        ib >> aligned;
    }
};

}  // namespace cldnn
