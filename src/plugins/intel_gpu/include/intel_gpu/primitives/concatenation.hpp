// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @details Concatenation is used to concatenate multiple sources into one destination along specified dimension.
/// @notes
/// - all other dimensions (except the one along which concatenation take place) must have the same value in each source.
/// - order of arguments in primitive creation has impact on order of feature maps in output primitive.
///
/// @par Alogrithm:
/// \code
///     int outputIdx = 0
///     for(i : input)
///     {
///         for(f : i.features)
///         {
///             output[outputIdx] = f
///             outputIdx += 1
///         }
///     }
/// \endcode
/// @par Where:
///   @li input : data structure holding all source inputs for this primitive
///   @li output : data structure holding output data for this primitive
///   @li i.features : number of features in currently processed input
///   @li outputIdx : index of destination feature
struct concatenation : public primitive_base<concatenation> {
    CLDNN_DECLARE_PRIMITIVE(concatenation)

    concatenation() : primitive_base("", {}) {}

    /// @li Constructs concatenation primitive.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param axis Selected dimension for concatenation.
    concatenation(
        const primitive_id& id,
        const std::vector<input_info>& input,
        const int64_t axis)
        : primitive_base(id, {input}), axis(axis) {}

    /// @li Constructs concatenation primitive.
    /// @param id This primitive id.
    /// @param input Vector of input primitives ids.
    /// @param axis Selected dimension for concatenation.
    /// @param output_dt Data type of output tensor
    concatenation(
        const primitive_id& id,
        const std::vector<input_info>& input,
        const int64_t axis,
        const data_types output_dt)
        : primitive_base(id, {input}, 1, {optional_data_type{output_dt}}), axis(axis) {}

    /// @brief Dimension along which concatenation should take place
    int64_t axis = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const concatenation>(rhs);

        return axis == rhs_casted.axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<concatenation>::save(ob);
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<concatenation>::load(ib);
        ib >> axis;
    }
};
}  // namespace cldnn
