// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief The ExtractImagePatches operation collects patches from the input tensor, as if applying a convolution.
/// All extracted patches are stacked in the depth dimension of the output.
/// @details The ExtractImagePatches operation is similar to the TensorFlow*
/// operation ExtractImagePatches.
/// This op extracts patches of shape `sizes` which are `strides` apart in the
/// input image. The output elements are taken from the input at intervals
/// given by the `rate` argument, as in dilated convolutions.
/// The result is a 4D tensor containing image patches with size
/// `size[0] * size[1] * depth` vectorized in the "depth" dimension.
/// The "auto_pad" attribute has no effect on the size of each patch, it
/// determines how many patches are extracted.
struct extract_image_patches : public primitive_base<extract_image_patches> {
    CLDNN_DECLARE_PRIMITIVE(extract_image_patches)

    extract_image_patches() : primitive_base("", {}) {}

    /// @brief Constructs select primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id containing input 4-D tensor.
    /// @param sizes Vector with sizes.
    /// @param strides Vector with strides.
    /// @param rates Vector with rates.
    /// @param auto_pad How the padding is calculated.
    /// @param output_shape Tensor with shape of output layout
    extract_image_patches(const primitive_id& id,
                          const input_info& input,
                          const std::vector<unsigned int>& sizes,
                          const std::vector<unsigned int>& strides,
                          const std::vector<unsigned int>& rates,
                          const std::string& auto_pad,
                          const tensor& output_shape,
                          const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          sizes(sizes),
          strides(strides),
          rates(rates),
          auto_pad(auto_pad),
          output_shape(output_shape) {}

    /// @brief Vector with sizes
    std::vector<unsigned int> sizes;
    /// @brief Vector with strides
    std::vector<unsigned int> strides;
    /// @brief Vector with rates
    std::vector<unsigned int> rates;
    /// @brief Mode how the padding is calculated
    std::string auto_pad;
    /// @brief Shape of output layout
    tensor output_shape;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, sizes.begin(), sizes.end());
        seed = hash_range(seed, strides.begin(), strides.end());
        seed = hash_range(seed, rates.begin(), rates.end());
        seed = hash_combine(seed, auto_pad);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const extract_image_patches>(rhs);

        return sizes == rhs_casted.sizes &&
               strides == rhs_casted.strides &&
               rates == rhs_casted.rates &&
               auto_pad == rhs_casted.auto_pad;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<extract_image_patches>::save(ob);
        ob << sizes;
        ob << strides;
        ob << rates;
        ob << auto_pad;
        ob << output_shape;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<extract_image_patches>::load(ib);
        ib >> sizes;
        ib >> strides;
        ib >> rates;
        ib >> auto_pad;
        ib >> output_shape;
    }
};
}  // namespace cldnn
