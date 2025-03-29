// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_multiply_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

static std::shared_ptr<Node> fuse_const_to_weights(const std::shared_ptr<Node>& matmul,
                                                   const Output<Node>& weights,
                                                   std::shared_ptr<ov::op::v0::Constant> mul_const) {
    auto const_shape = mul_const->get_shape();
    auto const_rank = static_cast<int64_t>(const_shape.size());
    const auto& weights_shape = weights.get_partial_shape();
    int64_t weights_rank = static_cast<int64_t>(weights_shape.rank().get_length());

    // Fuse if const is a scalar
    if (ov::is_scalar(const_shape)) {
        return std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
    }

    // Disallow consts that have rank greater than weights' rank when MatMul has dynamic rank.
    // Or if MatMul result rank is static - disallow constant that extends MatMul rank
    const auto& matmul_rank = matmul->get_output_partial_shape(0).rank();
    if (matmul_rank.is_dynamic()) {
        if (const_rank > weights_rank) {
            return nullptr;
        }
    } else if (matmul_rank.get_length() < const_rank) {
        return nullptr;
    }

    // Disallow const with shapes other than (a, b, ..., 1, z)
    if (const_rank > 1 && const_shape[const_rank - 2] != 1) {
        return nullptr;
    }

    // If weights is not a constant node - disallow Multiply constant
    // that extends weights rank. This is LPT requirement in case where
    // weights are meant to be quantized.
    if (const_rank > weights_rank && !ov::is_type<ov::op::v0::Constant>(weights.get_node())) {
        return nullptr;
    }

    auto matmul_casted = ov::as_type_ptr<ov::op::v0::MatMul>(matmul);
    if (!matmul_casted) {
        return nullptr;
    }

    // Check if const shape matches weights
    if (shape_size(const_shape) > 1) {
        if (const_shape.back() > 1) {
            // Check if const's last dimension matches last weights dimension
            if (matmul_casted->get_transpose_b()) {
                if (weights_shape[weights_rank - 2].is_dynamic() ||
                    (weights_rank > 1 &&
                     const_shape.back() != static_cast<size_t>(weights_shape[weights_rank - 2].get_length()))) {
                    return nullptr;
                }
            } else if (weights_shape[weights_rank - 1].is_dynamic() ||
                       const_shape.back() != static_cast<size_t>(weights_shape[weights_rank - 1].get_length())) {
                return nullptr;
            }
        }

        // Check if Multiply constant broadcasts MatMul input or weights.
        // If it broadcasts both, we're dealing with case like:
        // MatMul({1, 1, n, m}, const{1, 1, m, k}) -> mm{1, 1, n, k}
        // Multiply(mm{1, 1, n, k}, const{x, y, 1, k})
        //
        // After fusion, it'd look like:
        // MatMul({1, 1, n, m}, const{x, y, m, k}) -> mm{x, y, n, k}
        // In general, x * y elementwise multiples of size {n, k} should be cheaper than x * y matrix multiplies
        // of size {n, m} x {m, k}, so the fusion should be disallowed in that case.
        if (const_rank > 2) {
            bool const_broadcasts_weights = weights_rank < const_rank;
            for (int64_t i = 3; i <= const_rank; i++) {
                if (const_shape[const_rank - i] != 1) {
                    const_broadcasts_weights =
                        const_broadcasts_weights ||
                        ((weights_rank - i >= 0) && (weights_shape[weights_rank - i] != const_shape[const_rank - i]));
                }
            }
            bool const_broadcasts_input = true;
            const auto& input_shape = matmul->get_input_partial_shape(0);
            if (input_shape.rank().is_static()) {
                const auto& input_rank = input_shape.rank().get_length();
                const_broadcasts_input = input_rank < const_rank;
                for (int64_t i = 3; i <= const_rank; i++) {
                    if (const_shape[const_rank - i] != 1) {
                        const_broadcasts_input =
                            const_broadcasts_input ||
                            ((input_rank - i >= 0) && (input_shape[input_rank - i] != const_shape[const_rank - i]));
                    }
                }
            }
            if (const_broadcasts_input && const_broadcasts_weights) {
                return nullptr;
            }
        }
    }

    auto transpose_const = [](const std::shared_ptr<Node>& mul_const) -> std::shared_ptr<Node> {
        auto const_shape = mul_const->get_shape();
        auto const_rank = const_shape.size();
        if (shape_size(const_shape) == 1 ||
            (const_rank > 1 && const_shape[const_rank - 2] == 1 && const_shape[const_rank - 1] == 1)) {
            // Nothing to transpose - constant has shape (..., 1, 1)
            return mul_const;
        }
        std::shared_ptr<Node> new_const = mul_const;
        // Scalars were fused before, it suffices to check for 1D shape here
        if (const_rank == 1) {
            const_shape.insert(const_shape.begin(), 1);
            new_const = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(element::u64, Shape{const_shape.size()}, const_shape),
                false);
        }
        std::vector<int64_t> perm(const_shape.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(*(perm.end() - 1), *(perm.end() - 2));
        auto transpose = std::make_shared<ov::op::v1::Transpose>(
            new_const,
            ov::op::v0::Constant::create(element::i64, Shape{perm.size()}, perm));
        return ov::util::get_constant_from_source(transpose);
    };

    // If weights meant to be transposed - we need to also transpose constant
    if (matmul_casted->get_transpose_b()) {
        auto transpose = transpose_const(mul_const);
        if (!transpose)
            return nullptr;
        return std::make_shared<ov::op::v1::Multiply>(weights, transpose);
    }
    return std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
}

pass::MatMulMultiplyFusion::MatMulMultiplyFusion() {
    MATCHER_SCOPE(MatMulMultiplyFusion);
    auto input_pattern = pattern::any_input();
    auto weights_pattern = pattern::any_input(pattern::has_static_rank());
    auto mul_const_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto matmul_pattern =
        pattern::wrap_type<ov::op::v0::MatMul>({input_pattern, weights_pattern}, pattern::consumers_count(1));
    auto mul_pattern = pattern::wrap_type<ov::op::v1::Multiply>({matmul_pattern, mul_const_pattern});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& weights = pattern_map.at(weights_pattern);
        auto mul = pattern_map.at(mul_pattern).get_node_shared_ptr();
        auto mul_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(mul_const_pattern).get_node_shared_ptr());
        if (!mul_const)
            return false;
        auto matmul = pattern_map.at(matmul_pattern).get_node_shared_ptr();

        auto new_weights = fuse_const_to_weights(matmul, weights, mul_const);
        if (!new_weights)
            return false;

        // Constantfold new weights, only if old weights is a constant node.
        // To make sure that subgraphs with e.g. FakeQuantize don't get constant folded here.
        if (ov::is_type<ov::op::v0::Constant>(weights.get_node())) {
            if (auto constant = ov::util::get_constant_from_source(new_weights)) {
                new_weights = constant;
            }
        }

        const auto& input = pattern_map.at(input_pattern);
        auto new_mm = matmul->clone_with_new_inputs({input, new_weights});
        new_mm->set_friendly_name(mul->get_friendly_name());

        register_new_node(new_mm);
        copy_runtime_info({mul, weights.get_node_shared_ptr(), matmul}, {new_weights, new_mm});
        replace_node(mul, new_mm);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}
