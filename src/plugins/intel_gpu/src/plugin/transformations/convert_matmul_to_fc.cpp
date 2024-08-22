// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "convert_matmul_to_fc.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

ConvertMatMulToFullyConnected::ConvertMatMulToFullyConnected() {
    auto static_rank_gt_1 = [](const ov::Output<ov::Node>& output) {
        const auto& r = output.get_partial_shape().rank();
        return r.is_static() && r.get_length() > 1;
    };
    auto weights_path = [&static_rank_gt_1](const ov::Output<ov::Node>& output) {
        const auto& pshape = output.get_partial_shape();
        return ov::op::util::is_on_constant_path(output) &&
               static_rank_gt_1(output) &&
               pshape.is_static() &&
               std::count_if(pshape.begin(), pshape.end(), [](const ov::Dimension& x) { return x != 1; }) <= 2;
    };

    auto activations_m = ov::pass::pattern::any_input(static_rank_gt_1);
    auto weights_m = ov::pass::pattern::any_input(weights_path);
    auto matmul_m = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({ activations_m, weights_m }, ov::pass::pattern::has_static_rank());

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto fc_input_a = pattern_map.at(activations_m);
        auto fc_input_b = pattern_map.at(weights_m);

        // If 'fc_input_b' is shared with another matmul, transposing 'fc_input_b' is restricted.
        // If it is connected to the 'input_a' of another matmul, do not transpose
        // If it is connected to the 'input_b' of another matmul and the transpose option differs between the two matmuls, do not transpose.
        auto input_b = fc_input_b.get_node_shared_ptr();
        for (auto& user : input_b->get_users()) {
            if (user != matmul && ov::is_type<ov::op::v0::MatMul>(user) && ov::is_type<ov::op::v0::Convert>(input_b)) {
                auto other_matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(user);
                // Transpose for input_b generates invalid input for other sibling matmul
                if (input_b == other_matmul->get_input_node_shared_ptr(0) || fc_input_b == fc_input_a ||
                    (input_b == other_matmul->get_input_node_shared_ptr(1) && matmul->get_transpose_b() != other_matmul->get_transpose_b())) {
                    return false;
                }
            }
        }

        // fc_input_a and fc_input_b - are the final inputs that will be set to FullyConnected.
        // So in case of adding new operations that takes matmul inputs we need keep update fc_input_a and fc_input_b.
        bool is_convert = false;
        if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(fc_input_b.get_node_shared_ptr())) {
            is_convert = true;
            fc_input_b = convert_node->get_input_node_shared_ptr(0);
        }

        auto transpose_node = std::dynamic_pointer_cast<ov::op::v1::Transpose>(fc_input_b.get_node_shared_ptr());
        if (transpose_node) {
            fc_input_b = transpose_node->get_input_node_shared_ptr(0);
        }

        auto shape_a = fc_input_a.get_partial_shape();
        auto shape_b = fc_input_b.get_partial_shape();
        OPENVINO_ASSERT(shape_b.is_static());

        auto rank_a = shape_a.rank().get_length();
        auto rank_b = shape_b.rank().get_length();

        /*
         *  get_aligned_shapes function align two input shapes to have the same size and
         *  the same batch dimensions (last two dimensions are not comparable).
         *  It also checks that dimensions are compatible so in case with two shapes
         *  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
         */

        auto get_aligned_shapes = [&shape_a, &shape_b, &rank_a, &rank_b, &matmul]() -> std::tuple<bool, ov::PartialShape, ov::PartialShape> {
            ov::PartialShape shape_a_aligned(shape_a);
            ov::PartialShape shape_b_aligned(shape_b);
            size_t max_size = std::max(rank_a, rank_b);
            for (size_t i = 0, cnt = max_size - rank_a; i < cnt; ++i) {
                shape_a_aligned.insert(shape_a_aligned.begin(), 1);
            }
            for (size_t i = 0, cnt = max_size - rank_b; i < cnt; ++i) {
                shape_b_aligned.insert(shape_b_aligned.begin(), 1);
            }

            if (matmul->get_transpose_a()) {
                std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
            }
            if (matmul->get_transpose_b()) {
                std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
            }

            // check on per-batch MatMul which can't be converted to FC
            for (size_t i = 0; i < max_size - 2; ++i) {
                if (shape_b_aligned[i] == 1) {
                    shape_b_aligned[i] = shape_a_aligned[i];
                } else {
                    return std::make_tuple(false, std::move(shape_a_aligned), std::move(shape_b_aligned));
                }
            }
            return std::make_tuple(true, std::move(shape_a_aligned), std::move(shape_b_aligned));
        };

        /*
         *  create_transpose function return Transpose operation to replace transpose_a or transpose_b
         *  arguments with an operation. In other words in this function we create Transpose operation
         *  with order length equal to output_shape length of given node and fill order with increasing
         *  sequence starting from 0 and replace last two dimension. For example for length = 4  the
         *  order will be [0, 1, 3, 2] that emulates transpose_a or transpose_b attribute.
         */
        ov::NodeVector new_ops;

        auto create_transpose = [this, &new_ops ](const ov::Output<ov::Node>& node, const std::string& transpose_name) {
            std::vector<size_t> transpose_order(node.get_partial_shape().size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ transpose_order.size() }, transpose_order);
            auto transpose = std::make_shared<ov::op::v1::Transpose>(node, transpose_const);
            if (!ov::is_type<ov::op::v0::Constant>(transpose)) {
                new_ops.push_back(transpose_const);
                MatcherPass::register_new_node(transpose);
            }
            transpose->set_friendly_name(transpose_name);
            ov::disable_constant_folding(transpose);
            new_ops.push_back(transpose);
            return transpose;
        };

        bool success = true;
        ov::PartialShape shape_a_aligned;
        ov::PartialShape shape_b_aligned;
        std::tie(success, shape_a_aligned, shape_b_aligned) = get_aligned_shapes();
        if (!success) {
            return false;
        }

        auto aligned_a_rank = shape_a_aligned.rank();
        auto aligned_b_rank = shape_b_aligned.rank();
        if (aligned_a_rank.get_length() < 2 || aligned_b_rank.get_length() < 2) {
            OPENVINO_THROW("MatMul " + matmul->get_friendly_name() + " shapes are inconsistent.");
        }

        // Weights normalization
        bool can_reuse_transpose = false;
        if (!matmul->get_transpose_b()) {
            if (transpose_node && transpose_node->get_input_size() == 2) {
                auto order_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose_node->get_input_node_shared_ptr(1));
                if (order_constant) {
                    std::vector<size_t> order = order_constant->cast_vector<size_t>();

                    std::vector<size_t> expected_order(fc_input_b.get_partial_shape().size());
                    std::iota(expected_order.begin(), expected_order.end(), 0);
                    std::swap(*(expected_order.end() - 1), *(expected_order.end() - 2));

                    can_reuse_transpose = order == expected_order;
                }
            }

            fc_input_b = can_reuse_transpose ? transpose_node
                                             : create_transpose(fc_input_b, matmul->get_friendly_name() + "/transpose_b");
        }

        // Input normalization
        if (matmul->get_transpose_a()) {
            fc_input_a = create_transpose(fc_input_a, matmul->get_friendly_name() + "/transpose_a");
        }

        // Connect Convert to new input if needed
        if (is_convert && transpose_node && !can_reuse_transpose) {
            auto convert = pattern_map.at(weights_m).get_node_shared_ptr();
            auto new_convert = convert->clone_with_new_inputs({fc_input_b});
            new_ops.push_back(new_convert);
            new_convert->validate_and_infer_types();
            fc_input_b = new_convert;
        } else if (is_convert) {
            auto convert = pattern_map.at(weights_m).get_node_shared_ptr();
            convert->input(0).replace_source_output(fc_input_b);
            convert->validate_and_infer_types();
            fc_input_b = convert;
        }

        auto no_bias = std::make_shared<op::Placeholder>();

        // Create FullyConnected
        auto fc = std::make_shared<op::FullyConnected>(fc_input_a, fc_input_b, no_bias, matmul->get_output_element_type(0));
        fc->set_friendly_name(matmul->get_friendly_name());
        new_ops.push_back(fc);
        ov::copy_runtime_info(matmul, new_ops);
        ov::replace_node(matmul, fc);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "ConvertMatMulToFullyConnected");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
