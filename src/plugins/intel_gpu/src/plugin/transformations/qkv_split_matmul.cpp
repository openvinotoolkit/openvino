// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_split_matmul.hpp"

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

using ov::op::util::GatherBase;

// All consumers are Gather(output, scalar constant index, axis 0) with distinct indices
bool consumers_are_part_gathers(const ov::Output<ov::Node>& output) {
    const auto& shape = output.get_partial_shape();
    if (shape.rank().is_dynamic() || shape[0].is_dynamic()) {
        return false;
    }
    const auto parts = shape[0].get_length();
    std::set<int64_t> indices;
    for (const auto& input : output.get_target_inputs()) {
        const auto* const gather = ov::as_type<const GatherBase>(input.get_node());
        if (gather == nullptr || input.get_index() != 0) {
            return false;
        }
        const auto* const index = ov::as_type<const ov::op::v0::Constant>(gather->get_input_node_ptr(1));
        const auto* const axis = ov::as_type<const ov::op::v0::Constant>(gather->get_input_node_ptr(2));
        if (index == nullptr || axis == nullptr || !index->get_shape().empty() || axis->cast_vector<int64_t>() != std::vector<int64_t>{0}) {
            return false;
        }
        auto k = index->cast_vector<int64_t>()[0];
        k = k < 0 ? k + parts : k;
        if (k < 0 || k >= parts || !indices.insert(k).second) {
            return false;
        }
    }
    return !indices.empty();
}

// The subgraph producing `output` doesn't depend on model inputs
bool is_constant_subgraph(const ov::Output<ov::Node>& output) {
    bool depends_on_input = false;
    std::unordered_set<ov::Node*> visited;
    ov::op::util::visit_path(
        output.get_node(),
        visited,
        [](ov::Node*) {},
        [&](ov::Node* node) {
            depends_on_input = depends_on_input || ov::is_type<ov::op::v0::Parameter>(node);
            return depends_on_input;
        });
    return !depends_on_input;
}

// Clones the constant weight subgraph keeping columns [start, start + len) of the last axis.
// Returns an empty output if the subgraph has an op other than Constant, Convert, element-wise or Reshape.
ov::Output<ov::Node> slice_weight(const ov::Output<ov::Node>& weight, int64_t n_full, int64_t start, int64_t len) {
    std::unordered_set<ov::Node*> visited;
    ov::NodeVector nodes;
    ov::op::util::visit_constant_path(weight.get_node(), visited, [&](ov::Node* node) {
        nodes.push_back(node->shared_from_this());
    });

    std::unordered_map<const ov::Node*, std::shared_ptr<ov::Node>> sliced;
    for (const auto& node : ov::topological_sort(nodes)) {
        const auto& shape = node->get_output_partial_shape(0);
        const bool has_n = shape.rank().is_static() && shape.size() > 0 && shape[shape.size() - 1] == n_full;
        std::shared_ptr<ov::Node> result;
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            result = node;
            if (has_n) {
                auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
                auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {start + len});
                auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
                result = ov::op::util::make_try_fold<ov::op::v8::Slice>(node, begin, end, step, axis);
            }
        } else if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            auto target = ov::as_type_ptr<ov::op::v0::Constant>(sliced.at(reshape->get_input_node_ptr(1)));
            if (!target) {
                return {};
            }
            auto dims = target->cast_vector<int64_t>();
            if (has_n) {
                dims.back() = len;
            }
            result = reshape->clone_with_new_inputs(
                {sliced.at(reshape->get_input_node_ptr(0)), ov::op::v0::Constant::create(target->get_element_type(), {dims.size()}, dims)});
        } else if (ov::is_type_any_of<ov::op::v0::Convert, ov::op::util::BinaryElementwiseArithmetic>(node)) {
            ov::OutputVector inputs;
            for (const auto& input : node->input_values()) {
                inputs.push_back(ov::Output<ov::Node>(sliced.at(input.get_node()), input.get_index()));
            }
            result = node->clone_with_new_inputs(inputs);
        } else {
            return {};
        }
        if (result != node) {
            ov::copy_runtime_info(node, result);
        }
        sliced[node.get()] = result;
    }
    return ov::Output<ov::Node>(sliced.at(weight.get_node()), weight.get_index());
}

}  // namespace

QKVSplitMatMul::QKVSplitMatMul() {
    using namespace ov::pass::pattern;

    auto weight_m = any_input(rank_equals(2) && has_static_shape() && is_constant_subgraph);
    auto matmul_m = wrap_type<ov::op::v0::MatMul>({any_input(), weight_m},
                                                  consumers_count(1) && shape_matches("[batch, seq_len, ?]"),
                                                  {{"transpose_a", false}, {"transpose_b", false}});
    auto reshape_m =
        wrap_type<ov::op::v1::Reshape>({matmul_m, any_input()},
                                       consumers_count(1) && has_static_dims({2, 3, 4}) && shape_matches("[batch, seq_len, parts, heads, head_size]"));
    auto transpose_m =
        wrap_type<ov::op::v1::Transpose>({reshape_m, wrap_type<ov::op::v0::Constant>(value_matches("2, 0, 3, 1, 4"))}, consumers_are_part_gathers);
    auto index_m = wrap_type<ov::op::v0::Constant>(rank_equals(0));
    auto gather_m =
        wrap_type<ov::op::v1::Gather, ov::op::v7::Gather, ov::op::v8::Gather>({transpose_m, index_m, wrap_type<ov::op::v0::Constant>(value_matches("0"))});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto matmul = pattern_map.at(matmul_m).get_node_shared_ptr();
        const auto gather = m.get_match_root();
        if (transformation_callback(matmul)) {
            return false;
        }

        const auto& symbols = m.get_symbols();
        const int64_t parts = symbols.at("parts").i();
        const int64_t heads = symbols.at("heads").i();
        const int64_t head_size = symbols.at("head_size").i();
        const int64_t n_part = heads * head_size;
        int64_t index = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(index_m).get_node_shared_ptr())->cast_vector<int64_t>()[0];
        if (index < 0) {
            index += parts;
        }

        auto weight = slice_weight(pattern_map.at(weight_m), parts * n_part, index * n_part, n_part);
        if (!weight.get_node()) {
            return false;
        }
        ov::pass::NodeRegistry rg;
        auto part_matmul = rg.make<ov::op::v0::MatMul>(matmul->input_value(0), weight, false, false);
        auto heads_shape = rg.add(ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, heads, head_size}));
        auto part_reshape = rg.make<ov::op::v1::Reshape>(part_matmul, heads_shape, true);
        auto heads_order = rg.add(ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3}));
        auto part_transpose = rg.make<ov::op::v1::Transpose>(part_reshape, heads_order);

        const auto name = matmul->get_friendly_name() + "/part" + std::to_string(index);
        part_matmul->set_friendly_name(name);
        part_reshape->set_friendly_name(name + "/reshape");
        part_transpose->set_friendly_name(gather->get_friendly_name());
        ov::copy_runtime_info({matmul, pattern_map.at(reshape_m).get_node_shared_ptr(), pattern_map.at(transpose_m).get_node_shared_ptr(), gather},
                              rg.get());
        ov::replace_node(gather, part_transpose);
        return true;
    };

    auto m = std::make_shared<Matcher>(gather_m, "QKVSplitMatMul");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu