// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Type of gemm that will be added to the input by border layer / primitive.

/// @brief Adds gemm  input.
///
/// @details General Matrix Multiplication witch batch support,
///          A(B,Z,X)xA2(B,Y,Z)=C(B,X,Y)
/// @n
/// @n@b Requirements:
/// @n - @c input - first matrix
/// @n - @c input2 - second matrix
/// @n - @c optional: input3 matrix, alpha, beta, transpose
/// @n - @c computations with optional params: output = alpha x (input3 x beta + input x input2)
/// @n - @c transpose params tranposing second matrix <-TODO

struct gemm : public primitive_base<gemm> {
    CLDNN_DECLARE_PRIMITIVE(gemm)

    gemm() : primitive_base("", {}) {}

    /// @brief Constructs gemm layer.
    /// @brief Primitive id containing first matrix
    /// @brief Primitive id containing second matrix
    /// @brief Flag for transposing first input matrix
    /// @brief Flag for transposing second input matrix
    /// @brief Variable containing ALPHA parameter
    /// @brief Variable containing BETA parameter

    gemm(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const data_types data_type,
         const bool transpose_input0 = false,
         const bool transpose_input1 = false,
         const float alpha = 1.0f,
         const float beta = 0.0f,
         const size_t input_rank = 4,
         const size_t weight_rank = 4,
         const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}, {optional_data_type{ data_type }}),
          transpose_input0(transpose_input0),
          transpose_input1(transpose_input1),
          alpha(alpha),
          beta(beta),
          input_rank(input_rank),
          weight_rank(weight_rank) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            throw std::invalid_argument("Invalid inputs count - gemm expects either two or three inputs");
        }
    }

    /// @brief Flag for transposing first input matrix
    bool transpose_input0 = false;
    /// @brief Flag for transposing second input matrix
    bool transpose_input1 = false;
    /// @brief Variable containing ALPHA parameter
    float alpha = 1.0f;
    /// @brief Variable containing BETA parameter
    float beta = 1.0f;
    /// @brief First matrix rank
    size_t input_rank = 4;
     /// @brief Second matrix rank
    size_t weight_rank = 4;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, transpose_input0);
        seed = hash_combine(seed, transpose_input1);
        seed = hash_combine(seed, alpha);
        seed = hash_combine(seed, beta);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gemm>(rhs);

        return transpose_input0 == rhs_casted.transpose_input0 &&
               transpose_input1 == rhs_casted.transpose_input1 &&
               alpha == rhs_casted.alpha &&
               beta == rhs_casted.beta &&
               input_rank == rhs_casted.input_rank &&
               weight_rank == rhs_casted.weight_rank;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gemm>::save(ob);
        ob << transpose_input0;
        ob << transpose_input1;
        ob << alpha;
        ob << beta;
        ob << input_rank;
        ob << weight_rank;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gemm>::load(ib);
        ib >> transpose_input0;
        ib >> transpose_input1;
        ib >> alpha;
        ib >> beta;
        ib >> input_rank;
        ib >> weight_rank;
    }
};

}  // namespace cldnn

/// @}
/// @}
/// @}
