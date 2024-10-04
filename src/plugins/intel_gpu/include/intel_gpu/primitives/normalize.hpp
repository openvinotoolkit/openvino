// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Normalizes the input using an L2 norm and multiplies the output with scale value.
/// The scale can be equal for all channels or one scale per channel.
/// @details The L2 norm is computed as:<br>
/// Across spatial mode (across_spatial=true)-<br>
/// norm(i,x,y) = sqrt( &Sigma;( in(f,w,h)^2 ) + epsilon ) where f in range (0,num_of_features), w in range (0,input_width), h in range (0,input_height).<br>
/// The summation is performed over all the pixels in the batch.<br>
/// Within spatial mode (across_spatial=false)-<br>
/// norm(i,x,y) = sqrt( &Sigma;( in(f,x,y)^2 ) + epsilon ) where f in range (0,num_of_features).<br>
/// The summation is performed over this (x,y) position on all the features.<br>
/// @par Algorithm:
///   out(i,x,y) = ( in(i,x,y) / norm(i,x,y) ) * scale(i)
/// @par Where:
///   @li out(i,x,y) : value at x, y from i-th feature map after normalization.
///   @li in(i,x,y) : value at x, y from i-th feature map before normalization.
///   @li norm(i,x,y) : L2 norm as described above.
///   @li scale(i) : the scale value of the i-th feature map.
struct normalize : public primitive_base<normalize> {
    CLDNN_DECLARE_PRIMITIVE(normalize)

    normalize() : primitive_base("", {}) {}

    /// @brief Constructs normalize primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale_input Scale input primitive id with values needed for scaling after the normalization.
    /// Scale x dimension should be 1 (if all channels have the same scale) or equal to input feature size (one scale per channel).
    /// All other dimensions should be 1.
    /// @param across_spatial Determines if the normalization is done across or within spatial (see documentation above).
    /// @param epsilon Epsilon for not dividing by zero while normalizing.
    normalize(const primitive_id& id,
              const input_info& input,
              const primitive_id& scale_input,
              const bool across_spatial = true,
              const float epsilon = 1e-10f)
        : primitive_base(id, {input}),
          scale_input(scale_input),
          across_spatial(across_spatial),
          epsilon(epsilon) {}

    /// @brief Scale input primitive id with values needed for scaling after the normalization.
    /// Scale x dimension should be 1 (if all channels have the same scale) or equal to input feature size (one scale per channel).
    /// All other dimensions should be 1.
    primitive_id scale_input;
    /// @brief Determines if the normalization is done across or within spatial (see documentation above).
    bool across_spatial = true;
    /// @brief Epsilon for not dividing by zero while normalizing.
    float epsilon = 1e-10f;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, across_spatial);
        seed = hash_combine(seed, epsilon);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const normalize>(rhs);

        return across_spatial == rhs_casted.across_spatial &&
               epsilon == rhs_casted.epsilon;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<normalize>::save(ob);
        ob << scale_input;
        ob << across_spatial;
        ob << epsilon;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<normalize>::load(ib);
        ib >> scale_input;
        ib >> across_spatial;
        ib >> epsilon;
    }

protected:
    std::vector<input_info> get_dependencies() const override { return {scale_input}; }
};
}  // namespace cldnn
