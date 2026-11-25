// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_matmul.hpp"

#include <cstdint>
#include <functional>
#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;

ov::OutputVector split_compressed_weights(const ov::Output<ov::Node>& weights, int64_t num_groups) {
    auto source = weights.get_node_shared_ptr();
    bool has_decompression = false;
    while (ov::is_type<v0::Convert>(source) || ov::is_type<v1::Reshape>(source) || ov::is_type<v1::Multiply>(source) ||
           ov::is_type<v1::Subtract>(source)) {
        // Already marked by MarkDequantization: the clones would lose this non-copyable mark and be constant-folded.
        if (ov::pass::constant_folding_is_disabled(source)) {
            return {};
        }
        has_decompression |= ov::is_type<v1::Multiply>(source);
        source = source->input_value(0).get_node_shared_ptr();
    }
    const auto weight_type = source->get_output_element_type(0);
    if (!has_decompression || !ov::is_type<v0::Constant>(source) || num_groups <= 0 ||
        (weight_type != ov::element::u4 && weight_type != ov::element::i4 && weight_type != ov::element::u8 &&
         weight_type != ov::element::i8)) {
        return {};
    }

    const auto group_count = static_cast<size_t>(num_groups);
    std::function<bool(const ov::Output<ov::Node>&, size_t)> can_split;
    can_split = [&](const ov::Output<ov::Node>& value, size_t rank) {
        if (!value.get_partial_shape().is_static()) {
            return false;
        }
        const auto shape = value.get_shape();
        if (shape.size() > rank ||
            (shape.size() == rank && (shape.empty() || (shape[0] != 1 && shape[0] != group_count)))) {
            return false;
        }
        const auto node = value.get_node_shared_ptr();
        if (ov::is_type<v0::Constant>(node)) {
            if (shape.size() < rank) {
                return true;
            }
            const ov::Shape expert_shape(shape.begin() + 1, shape.end());
            return (ov::shape_size(expert_shape) * value.get_element_type().bitwidth()) % 8 == 0;
        }
        if (ov::is_type<v0::Convert>(node)) {
            return can_split(node->input_value(0), rank);
        }
        if (ov::is_type<v1::Reshape>(node) && shape.size() == rank && node->get_input_partial_shape(0).is_static()) {
            const auto input_shape = node->get_input_shape(0);
            return !input_shape.empty() && input_shape[0] == shape[0] &&
                   can_split(node->input_value(0), input_shape.size());
        }
        if (ov::is_type<v1::Multiply>(node) || ov::is_type<v1::Subtract>(node)) {
            return can_split(node->input_value(0), rank) && can_split(node->input_value(1), rank);
        }
        return false;
    };
    if (!weights.get_partial_shape().is_static() || !can_split(weights, weights.get_shape().size())) {
        return {};
    }

    ov::OutputVector experts;
    for (int64_t expert_index = 0; expert_index < num_groups; ++expert_index) {
        std::function<std::shared_ptr<ov::Node>(const ov::Output<ov::Node>&, size_t)> clone;
        clone = [&](const ov::Output<ov::Node>& value, size_t rank) -> std::shared_ptr<ov::Node> {
            const auto node = value.get_node_shared_ptr();
            const auto shape = value.get_shape();
            if (shape.size() < rank) {
                return node;
            }
            std::shared_ptr<ov::Node> result;
            if (const auto constant = ov::as_type_ptr<v0::Constant>(node)) {
                const ov::Shape expert_shape(shape.begin() + 1, shape.end());
                const auto bytes = ov::shape_size(expert_shape) * value.get_element_type().bitwidth() / 8;
                const auto index = shape[0] == 1 ? 0 : expert_index;
                const auto data = static_cast<const uint8_t*>(constant->get_data_ptr()) + index * bytes;
                // A view into the original buffer, kept alive by the original constant, so the weights are not copied.
                // WeightlessCacheAttribute is not copied: it describes the whole constant, not this slice.
                result = std::make_shared<v0::Constant>(value.get_element_type(), expert_shape, data, constant);
            } else if (ov::is_type<v1::Reshape>(node)) {
                const std::vector<int64_t> expert_shape(shape.begin() + 1, shape.end());
                auto pattern = v0::Constant::create(ov::element::i64, ov::Shape{expert_shape.size()}, expert_shape);
                result = std::make_shared<v1::Reshape>(clone(node->input_value(0), node->get_input_shape(0).size()),
                                                       pattern,
                                                       false);
            } else {
                ov::OutputVector inputs;
                for (const auto& input : node->input_values()) {
                    inputs.push_back(clone(input, rank));
                }
                result = node->clone_with_new_inputs(inputs);
            }
            result->set_friendly_name(node->get_friendly_name() + "/expert_" + std::to_string(expert_index));
            ov::copy_runtime_info(node, result);
            return result;
        };
        experts.push_back(clone(weights, weights.get_shape().size()));
    }
    return experts;
}

}  // namespace

ConvertGroupedMatMul3Dx3DToMatMul::ConvertGroupedMatMul3Dx3DToMatMul() {
    MATCHER_SCOPE(ConvertGroupedMatMul3Dx3DToMatMul);
    using namespace ov::pass::pattern;

    // 3Dx3D: A:[G,M,K] B:[G,N,K]
    auto matrix_a_3d = any_input(rank_equals(3));
    auto matrix_b_3d = any_input(rank_equals(3));
    auto gmm_3d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_3d, matrix_b_3d});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }
        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        NodeRegistry rg;
        // A:[G,M,K] B:[G,N,K] -> MatMul(A, B, transpose_b=true) -> [G,M,N]
        auto replacement = rg.make<v0::MatMul>(mat_a, mat_b, false, true);

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, rg.get());
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_3d_3d, matcher_name);
    register_matcher(matcher, callback);
}

ConvertGroupedMatMul2Dx3DToMatMul::ConvertGroupedMatMul2Dx3DToMatMul(bool only_compressed_split) {
    MATCHER_SCOPE(ConvertGroupedMatMul2Dx3DToMatMul);
    using namespace ov::pass::pattern;

    // 2Dx3D: A:[T,K] B:[G,N,K] offsets:[G]
    // A fixed MatMul per group is emitted, so the group count G (B.shape[0]) must be static.
    auto matrix_a_2d = any_input(rank_equals(2));
    auto matrix_b_3d = any_input(rank_equals(3) && has_static_dim(0));
    auto offsets = any_input(rank_equals(1) && type_matches_any({ov::element::i32, ov::element::i64}));
    auto gmm_2d_3d = wrap_type<v17::GroupedMatMul>({matrix_a_2d, matrix_b_3d, offsets});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gmm = ov::as_type_ptr<v17::GroupedMatMul>(m.get_match_root());
        if (!gmm || transformation_callback(gmm)) {
            return false;
        }
        const auto mat_a = gmm->input_value(0);
        const auto mat_b = gmm->input_value(1);

        const auto num_groups = mat_b.get_partial_shape()[0].get_length();
        const auto compressed_weights = split_compressed_weights(mat_b, num_groups);
        if (only_compressed_split && compressed_weights.empty()) {
            return false;
        }

        using ov::element::i32;

        NodeRegistry rg;

        const auto offsets = gmm->input_value(2);
        const auto offsets_et = offsets.get_element_type();

        auto axis_scalar = rg.make<v0::Constant>(i32, ov::Shape{}, 0);
        auto axis_1d = rg.make<v0::Constant>(i32, ov::Shape{1}, 0);

        // offsets are cumulative ends, so per-group row counts are offsets - [0, offsets[:-1]].
        ov::Output<ov::Node> split_lengths = offsets;
        if (num_groups > 1) {
            // Only Concat requires a matching type; Slice and VariadicSplit take any integral one.
            auto zero = rg.make<v0::Constant>(offsets_et, ov::Shape{1}, 0);
            auto head_stop = rg.make<v0::Constant>(i32, ov::Shape{1}, num_groups - 1);
            auto step = rg.make<v0::Constant>(i32, ov::Shape{1}, 1);
            auto offsets_head = rg.make<v8::Slice>(offsets, zero, head_stop, step, axis_1d);
            auto shifted = rg.make<v0::Concat>(ov::OutputVector{zero, offsets_head}, 0);
            split_lengths = rg.make<v1::Subtract>(offsets, shifted);
        }

        auto a_split = rg.make<v1::VariadicSplit>(mat_a, axis_scalar, split_lengths);  // G x [Mg, K]
        auto b_split = rg.make<v1::Split>(mat_b, axis_scalar, num_groups);             // G x [1, N, K]

        ov::OutputVector group_outputs;
        group_outputs.reserve(static_cast<size_t>(num_groups));

        for (int64_t g = 0; g < num_groups; ++g) {
            ov::Output<ov::Node> b_g = compressed_weights.empty()
                                           ? rg.make<v0::Squeeze>(b_split->output(g), axis_1d)->output(0)
                                           : compressed_weights[g];
            auto mm = rg.make<v0::MatMul>(a_split->output(g), b_g, false, true);  // [Mg, N]
            group_outputs.push_back(std::move(mm));
        }

        auto replacement = rg.make<v0::Concat>(group_outputs, 0);  // [T, N]

        replacement->set_friendly_name(gmm->get_friendly_name());
        ov::copy_runtime_info(gmm, rg.get());
        ov::replace_node(gmm, replacement);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gmm_2d_3d, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
