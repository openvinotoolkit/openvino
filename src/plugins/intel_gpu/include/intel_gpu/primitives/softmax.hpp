// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
///   b = e^a/sum(N-1; j=0; e^j)
/// @par Where:
///   @li N : number of values to normalize
///   @li b : value after normalization
///   @li a : value before normalization
struct softmax : public primitive_base<softmax> {
    CLDNN_DECLARE_PRIMITIVE(softmax)

    /// @brief Constructs softmax primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization
    softmax(const primitive_id& id,
            const input_info& input,
            const int64_t dimension = 1,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), dimension(dimension) {}

    /// @brief Defines a scope of a single softmax normalization.
    /// @details
    /// Being given a 4-dimensional input, which consists of b,f,y,x dimensions, softmax normalizes data which are divided into multiple independent sets.
    /// Specific behaviour is determined by this parameter, as follows:
    /// - when softmax dimension is set to 0 (b dim) each batch vector of input is normalized independently,
    /// - when softmax dimension is set to 1 (f dim) each in-depth vector of input is normalized independently,
    /// - when softmax dimension is set to 2 (y dim) each input column is normalized independently,
    /// - when softmax dimension is set to 3 (x dim) each input row is normalized independently.

    int64_t dimension;

    size_t hash() const override {
        if (!seed) {
            seed = hash_combine(seed, dimension);
        }
        return seed;
    }
};
}  // namespace cldnn
