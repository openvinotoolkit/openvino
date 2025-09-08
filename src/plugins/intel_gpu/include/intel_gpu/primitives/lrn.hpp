// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {
typedef enum { /*:int32_t*/
    lrn_norm_region_across_channel,
    lrn_norm_region_within_channel
} lrn_norm_region;

/// @brief Local response normalization
/// @details LRN layer as described in chapter 3.3 of "ImageNet Classification with Deep Convolutional
/// Neural Networks" by Khrizevsky, Sutskever, Hinton. @n See: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
/// @par Alogrithm:
///   b(i,x,y) = a(i,x,y) / (k+alpha*sum(min(N-1, i+n/2); j=max(0,i-n/2); a(j,x,y)^2))
/// @par Where:
///   @li b(i,x,y) : value at x, y from i-th feature map after normalization
///   @li a(i,x,y) : value at x, y from i-th feature map before normalization
///   @li N : number of feature maps
///   @li n : size of normalization
///   @li k, alpha, beta : hyper parameters (equal to 2, 10e-4, 0.75 in paper).
struct lrn : public primitive_base<lrn> {
    CLDNN_DECLARE_PRIMITIVE(lrn)

    lrn() : primitive_base("", {}) {}

    /// @brief Constructs LRN primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param size Size of normalization.
    /// @param k Hyper parameter "k".
    /// @param alpha Hyper parameter "alpha".
    /// @param beta Hyper parameter "beta".
    /// @param lrn_norm_region Normalize across or within channel
    lrn(const primitive_id& id,
        const input_info& input,
        uint32_t size,
        float k,
        float alpha,
        float beta,
        lrn_norm_region lrn_norm_region)
        : primitive_base(id, {input}),
          size(size),
          k(k),
          alpha(alpha),
          beta(beta),
          norm_region(lrn_norm_region) {}

    /// @brief Size of normalization.
    uint32_t size = 0;
    /// @brief Hyper parameter "k".
    float k = 0.0f;
    /// @brief Hyper parameter "alpha".
    float alpha = 0.0f;
    /// @brief Hyper parameter "beta".
    float beta = 0.0f;
    /// @brief Normalize across or within channel
    lrn_norm_region norm_region = lrn_norm_region::lrn_norm_region_within_channel;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, size);
        seed = hash_combine(seed, k);
        seed = hash_combine(seed, alpha);
        seed = hash_combine(seed, beta);
        seed = hash_combine(seed, norm_region);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lrn>(rhs);

        return size == rhs_casted.size &&
               k == rhs_casted.k &&
               alpha == rhs_casted.alpha &&
               beta == rhs_casted.beta &&
               norm_region == rhs_casted.norm_region;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lrn>::save(ob);
        ob << size;
        ob << k;
        ob << alpha;
        ob << beta;
        ob << make_data(&norm_region, sizeof(lrn_norm_region));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lrn>::load(ib);
        ib >> size;
        ib >> k;
        ib >> alpha;
        ib >> beta;
        ib >> make_data(&norm_region, sizeof(lrn_norm_region));
    }
};
}  // namespace cldnn
