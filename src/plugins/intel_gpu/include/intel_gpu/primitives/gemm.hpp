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

    typedef enum {
        X_LAST = 0,
        Y_LAST,
        OTHER,
    } TransposeType;

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
         const bool transpose_input0,
         const bool transpose_input1,
         const float alpha = 1.0f,
         const float beta = 0.0f,
         const size_t input_rank = 4,
         const size_t weight_rank = 4,
         const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}, {optional_data_type{ data_type }}),
          transpose_input0(transpose_input0 ? 1 : 0),
          transpose_input1(transpose_input1 ? 1 : 0),
          alpha(alpha),
          beta(beta),
          input_rank(input_rank),
          weight_rank(weight_rank) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            throw std::invalid_argument("Invalid inputs count - gemm expects either two or three inputs");
        }

        auto get_transposed_order = [] (size_t rank, bool transposed) {
            std::vector<int64_t> order(rank);
            std::iota(order.begin(), order.end(), 0);
            if (transposed)
                std::swap(order[rank - 1], order[rank - 2]);
            return order;
        };

        input0_order = get_transposed_order(input_rank, transpose_input0);
        input1_order = get_transposed_order(weight_rank, transpose_input1);
        output_order = {};
    }

    /// @brief Constructs gemm layer.
    /// @brief Primitive id containing first matrix
    /// @brief Primitive id containing second matrix
    /// @brief Transposed order of first input matrix
    /// @brief Transposed order of second input matrix
    /// @brief Transposed order of output matrix
    /// @brief Variable containing ALPHA parameter
    /// @brief Variable containing BETA parameter
    gemm(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const data_types data_type,
         const std::vector<int64_t>& input0_order = {0, 1, 2, 3},
         const std::vector<int64_t>& input1_order = {0, 1, 2, 3},
         const std::vector<int64_t>& output_order = {},
         const float alpha = 1.0f,
         const float beta = 0.0f,
         const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}, {optional_data_type{ data_type }}),
          input0_order(input0_order),
          input1_order(input1_order),
          output_order(output_order),
          alpha(alpha),
          beta(beta),
          input_rank(input0_order.size()),
          weight_rank(input1_order.size()) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            throw std::invalid_argument("Invalid inputs count - gemm expects either two or three inputs");
        }

        transpose_input0 = get_transpose_mode(input0_order);
        transpose_input1 = get_transpose_mode(input1_order);
    }

    gemm(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const input_info& beam_table,
         const data_types data_type,
         const std::vector<int64_t>& input0_order,
         const std::vector<int64_t>& input1_order,
         const std::vector<int64_t>& output_order,
         bool indirect_a,
         bool indirect_b,
         const float alpha = 1.0f,
         const float beta = 0.0f,
         const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}, {optional_data_type{ data_type }}),
          input0_order(input0_order),
          input1_order(input1_order),
          output_order(output_order),
          alpha(alpha),
          beta(beta),
          input_rank(input0_order.size()),
          weight_rank(input1_order.size()),
          beam_table(beam_table),
          indirect_a(indirect_a),
          indirect_b(indirect_b) {
        if (inputs.size() != 2 && inputs.size() != 3) {
            throw std::invalid_argument("Invalid inputs count - gemm expects either two or three inputs");
        }

        transpose_input0 = get_transpose_mode(input0_order);
        transpose_input1 = get_transpose_mode(input1_order);
    }

    /// @brief Flag for transposing first input matrix
    uint32_t transpose_input0 = 0;
    /// @brief Flag for transposing second input matrix
    uint32_t transpose_input1 = 0;
    /// @brief order of input 0
    std::vector<int64_t> input0_order;
    /// @brief order of input 1
    std::vector<int64_t> input1_order;
    /// @brief order of output
    std::vector<int64_t> output_order;
    /// @brief Variable containing ALPHA parameter
    float alpha = 1.0f;
    /// @brief Variable containing BETA parameter
    float beta = 1.0f;
    /// @brief First matrix rank
    size_t input_rank = 4;
    /// @brief Second matrix rank
    size_t weight_rank = 4;

    /// @brief Beam table input for indirect access for one of the inputs
    input_info beam_table = {};
    bool indirect_a = false;
    bool indirect_b = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, transpose_input0);
        seed = hash_combine(seed, transpose_input1);
        seed = hash_combine(seed, indirect_a);
        seed = hash_combine(seed, indirect_b);
        for (auto order : input0_order)
            seed = hash_combine(seed, order);
        for (auto order : input1_order)
            seed = hash_combine(seed, order);
        for (auto order : output_order)
            seed = hash_combine(seed, order);
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
               indirect_a == rhs_casted.indirect_a &&
               indirect_b == rhs_casted.indirect_b &&
               input_rank == rhs_casted.input_rank &&
               weight_rank == rhs_casted.weight_rank;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gemm>::save(ob);
        ob << transpose_input0;
        ob << transpose_input1;
        ob << input0_order;
        ob << input1_order;
        ob << output_order;
        ob << alpha;
        ob << beta;
        ob << input_rank;
        ob << weight_rank;
        ob << indirect_a;
        ob << indirect_b;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gemm>::load(ib);
        ib >> transpose_input0;
        ib >> transpose_input1;
        ib >> input0_order;
        ib >> input1_order;
        ib >> output_order;
        ib >> alpha;
        ib >> beta;
        ib >> input_rank;
        ib >> weight_rank;
        ib >> indirect_a;
        ib >> indirect_b;
    }

    std::vector<input_info> get_dependencies() const override {
        if (beam_table.is_valid())
            return { beam_table };
        return {};
    }

private:
    TransposeType get_transpose_mode(const std::vector<int64_t>& order_idx) {
        int64_t rank = order_idx.size() - 1;

        if (rank == order_idx[rank]) {
            // normal
            return TransposeType::X_LAST;
        } else if (rank == order_idx[rank - 1]) {
            // the second last dim is moved to the last
            return TransposeType::Y_LAST;
        } else {
            // other
            return TransposeType::OTHER;
        }
    }
};

}  // namespace cldnn

/// @}
/// @}
/// @}
