// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/extract_reshapes_from_mha.hpp"

#include <algorithm>
#include <memory>
#include <openvino/opsets/opset1.hpp>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "snippets/itt.hpp"
#include "snippets/pass/mha_tokenization.hpp"

using namespace ov::pass;

ov::snippets::pass::ExtractPairsAfterMatmul::ExtractPairsAfterMatmul() {
    MATCHER_SCOPE(ExtractPairsAfterMatmul);
    auto static_shape_single_consumer = [](const ov::Output<ov::Node>& out) {
        return pattern::has_static_shape()(out) && pattern::consumers_count(1)(out);
    };
    auto matmul_m = pattern::wrap_type<opset1::MatMul>(static_shape_single_consumer);
    auto reshape_1_m = pattern::wrap_type<opset1::Reshape>({matmul_m, pattern::wrap_type<opset1::Constant>()},
                                                           static_shape_single_consumer);
    auto sparse_input_1_m = pattern::any_input(pattern::has_static_shape());
    auto sparse_input_2_m = pattern::any_input(pattern::has_static_shape());
    auto add_1_m = pattern::wrap_type<opset1::Add>({reshape_1_m, sparse_input_1_m}, static_shape_single_consumer);
    auto add_2_m = pattern::wrap_type<opset1::Add>({add_1_m, sparse_input_2_m}, static_shape_single_consumer);
    auto reshape_2_m = pattern::wrap_type<opset1::Reshape>({add_2_m, pattern::wrap_type<opset1::Constant>()},
                                                           pattern::has_static_shape());

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ExtractPairsAfterMatmul")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& matmul = pattern_map.at(matmul_m);
        const auto matmul_node = ov::as_type_ptr<opset1::MatMul>(matmul.get_node_shared_ptr());
        if (!ov::snippets::pass::TokenizeMHASnippets::is_matmul0_supported(matmul_node) ||
            transformation_callback(matmul_node)) {
            return false;
        }

        const auto& reshape_2 = pattern_map.at(reshape_2_m);
        const auto& matmul_shape = matmul.get_shape();
        const auto& output_shape = reshape_2.get_shape();
        if (matmul_shape != output_shape) {
            return false;
        }

        const auto add_1 = pattern_map.at(add_1_m).get_node_shared_ptr();
        const auto add_2 = pattern_map.at(add_2_m).get_node_shared_ptr();
        const auto& bcast_type = add_1->get_autob();
        if (bcast_type != ov::op::AutoBroadcastType::NUMPY || bcast_type != add_2->get_autob()) {
            return false;
        }

        const auto& sparse_input_1 = pattern_map.at(sparse_input_1_m);
        const auto& sparse_input_2 = pattern_map.at(sparse_input_2_m);
        auto broadcasted_shape = sparse_input_1.get_partial_shape();
        ov::PartialShape::broadcast_merge_into(broadcasted_shape, sparse_input_2.get_partial_shape(), bcast_type);
        if (ov::shape_size(matmul_shape) != ov::shape_size(broadcasted_shape.to_shape())) {
            return false;
        }

        const auto extracted_add = std::make_shared<ov::opset1::Add>(sparse_input_1, sparse_input_2);
        const auto target_shape = ov::opset1::Constant::create(ov::element::i32, {matmul_shape.size()}, matmul_shape);
        const auto extracted_reshape = std::make_shared<ov::opset1::Reshape>(extracted_add, target_shape, true);
        const auto new_add = std::make_shared<ov::opset1::Add>(matmul, extracted_reshape);

        const auto& old_reshape = pattern_map.at(reshape_2_m);
        return ov::replace_output_update_name(old_reshape, new_add);
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_2_m, matcher_name);
    register_matcher(m, callback);
}

ov::snippets::pass::RankUpgradeToRankReduction::RankUpgradeToRankReduction() {
    MATCHER_SCOPE(RankUpgradeToRankReduction);
    auto static_shape_single_consumer = [](const ov::Output<ov::Node>& out) {
        return pattern::has_static_shape()(out) && pattern::consumers_count(1)(out);
    };
    // reshape_1_m insert leading dimension of 1.
    auto rank_upgrade_reshape = [&](const ov::Output<ov::Node>& out) {
        if (!static_shape_single_consumer(out)) {
            return false;
        }
        auto out_shape = out.get_shape();
        if (out_shape.empty() || out_shape[0] != 1) {
            return false;
        }
        out_shape.erase(out_shape.begin());
        const auto& in_shape_partial = out.get_node_shared_ptr()->get_input_partial_shape(0);
        if (in_shape_partial.is_dynamic()) {
            return false;
        }
        const auto& in_shape = in_shape_partial.to_shape();
        return out_shape == in_shape;
    };
    // input_2_m has leading dimension of 1.
    auto has_leading_dimension_one = [&](const ov::Output<ov::Node>& out) {
        if (!static_shape_single_consumer(out)) {
            return false;
        }
        const auto& out_shape = out.get_shape();
        return (!out_shape.empty() && out_shape[0] == 1);
    };
    // reshape_2_m delete leading dimension of 1.
    auto rank_reduction_reshape = [&](const ov::Output<ov::Node>& out) {
        if (!static_shape_single_consumer(out)) {
            return false;
        }
        auto in_shape_partial = out.get_node_shared_ptr()->get_input_partial_shape(0);
        if (in_shape_partial.is_dynamic()) {
            return false;
        }
        auto in_shape = in_shape_partial.to_shape();
        if (in_shape.empty() || in_shape[0] != 1) {
            return false;
        }
        in_shape.erase(in_shape.begin());
        const auto& out_shape = out.get_shape();
        return out_shape == in_shape;
    };

    auto matmul_m = pattern::wrap_type<opset1::MatMul>(static_shape_single_consumer);
    auto input_1_m = pattern::any_input(pattern::has_static_shape());
    auto eltwise_1_m = pattern::optional<ov::op::util::BinaryElementwiseArithmetic>({matmul_m, input_1_m},
                                                                                    static_shape_single_consumer);
    auto reshape_1_m = pattern::wrap_type<opset1::Reshape>({eltwise_1_m, pattern::wrap_type<opset1::Constant>()},
                                                           rank_upgrade_reshape);
    auto input_2_m = pattern::any_input(has_leading_dimension_one);
    auto eltwise_2_m = pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({reshape_1_m, input_2_m},
                                                                                     static_shape_single_consumer);
    auto reshape_2_m = pattern::wrap_type<opset1::Reshape>({eltwise_2_m, pattern::wrap_type<opset1::Constant>()},
                                                           rank_reduction_reshape);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::RankUpgradeToRankReduction")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& matmul = pattern_map.at(matmul_m);
        const auto matmul_node = ov::as_type_ptr<opset1::MatMul>(matmul.get_node_shared_ptr());
        if (!ov::snippets::pass::TokenizeMHASnippets::is_matmul0_supported(matmul_node) ||
            transformation_callback(matmul_node)) {
            return false;
        }
        const auto& eltwise_2 = pattern_map.at(eltwise_2_m).get_node_shared_ptr();
        const auto& shapes = ov::util::get_node_input_partial_shapes(*eltwise_2);
        OPENVINO_ASSERT(!shapes.empty(), "Eltwise node should has at least one input.");
        auto equal_rank = [&](const PartialShape& p) {
            return p.size() == shapes[0].size();
        };
        if (!std::all_of(shapes.cbegin(), shapes.cend(), equal_rank)) {
            return false;
        }

        const auto& input_2 = pattern_map.at(input_2_m);
        auto input_2_shape = input_2.get_shape();
        input_2_shape.erase(input_2_shape.begin());
        const auto target_shape = ov::opset1::Constant::create(ov::element::i32, {input_2_shape.size()}, input_2_shape);
        const auto reshaped_input2 = std::make_shared<ov::opset1::Reshape>(input_2, target_shape, true);
        const auto& reshape_1 = pattern_map.at(reshape_1_m).get_node_shared_ptr();
        ov::copy_runtime_info(reshape_1, reshaped_input2);

        auto first_input = pattern_map.at(matmul_m);
        if (pattern_map.count(eltwise_1_m)) {
            first_input = pattern_map.at(eltwise_1_m);
        }

        OutputVector new_args({first_input, reshaped_input2});
        eltwise_2->set_arguments(new_args);
        const auto& old_reshape = pattern_map.at(reshape_2_m);
        return ov::replace_output_update_name(old_reshape, eltwise_2);
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_2_m, matcher_name);
    register_matcher(m, callback);
}
