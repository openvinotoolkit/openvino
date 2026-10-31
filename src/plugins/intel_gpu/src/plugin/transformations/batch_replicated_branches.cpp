// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_replicated_branches.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Predicate;

using NodePtr = std::shared_ptr<ov::Node>;
using NodeVector = std::vector<NodePtr>;

bool is_type_relaxed(const ov::Node* node) {
    return dynamic_cast<const ov::op::TypeRelaxedBase*>(node) != nullptr;
}

int64_t normalized_concat_axis(const ov::op::v0::Concat& concat) {
    const auto axis = concat.get_axis();
    return axis < 0 ? axis + concat.get_output_partial_shape(0).rank().get_length() : axis;
}

struct BranchPattern {
    BranchPattern() {
        // Branch nodes are cloned with [N, ...] input instead of [1, ...], so they must have batch 1
        // and a single consumer; TypeRelaxed and control dependencies are not preserved by cloning
        const auto branch_node = has_static_shape() && shape_matches("[1, ?, ?, ?]") && consumers_count(1) &&
                                 Predicate(
                                     [](const ov::Output<ov::Node>& output) {
                                         const auto node = output.get_node();
                                         return !is_type_relaxed(node) && node->get_control_dependencies().empty() && node->get_control_dependents().empty();
                                     },
                                     "batchable_branch_node");
        // MaxPool indices are not batched, so they must be unused
        const auto indices_unused = Predicate(
            [](const ov::Output<ov::Node>& output) {
                const auto node = output.get_node();
                return output.get_index() == 0 && (node->get_output_size() < 2 || node->output(1).get_target_inputs().empty());
            },
            "max_pool_indices_unused");

        data = any_input(has_static_shape());
        gather = wrap_type<ov::op::v8::Gather>({data, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()},
                                               branch_node && attrs_match({{"batch_dims", 0}}));

        weights_const = wrap_type<ov::op::v0::Constant>();
        weights_convert = wrap_type<ov::op::v0::Convert>({weights_const});
        const auto weights = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{weights_const, weights_convert});
        conv = wrap_type<ov::op::v1::Convolution>({gather, weights}, branch_node);
        pool = optional<ov::op::v1::MaxPool, ov::op::v8::MaxPool, ov::op::v14::MaxPool>({conv}, branch_node && indices_unused);
        // Bias broadcast to [1, C, H, W] is also valid for [N, C, H, W]; checked after cloning
        bias = wrap_type<ov::op::v0::Constant>();
        add = optional<ov::op::v1::Add>({pool, bias}, branch_node && attrs_match({{"auto_broadcast", "numpy"}}));
        relu = optional<ov::op::v0::Relu>({add}, branch_node);
    }

    std::shared_ptr<ov::Node> data, gather, weights_const, weights_convert, conv, pool, bias, add, relu;

    std::vector<std::shared_ptr<ov::Node>> chain() const {
        return {conv, pool, add, relu};
    }
};

bool constant_inputs_are_equal(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    if (lhs == rhs)
        return true;
    const auto lhs_node = lhs.get_node_shared_ptr();
    const auto rhs_node = rhs.get_node_shared_ptr();
    const auto lhs_constant = ov::as_type_ptr<ov::op::v0::Constant>(lhs_node);
    const auto rhs_constant = ov::as_type_ptr<ov::op::v0::Constant>(rhs_node);
    if (lhs_constant || rhs_constant)
        return lhs_constant && rhs_constant && ov::op::util::constants_are_equal(*lhs_constant, *rhs_constant);
    return ov::op::util::node_attributes_are_equal(lhs_node, rhs_node) && constant_inputs_are_equal(lhs_node->input_value(0), rhs_node->input_value(0));
}

bool branches_are_equal(const BranchPattern& pattern, const PatternValueMap& reference, const PatternValueMap& branch) {
    for (const auto& pattern_node : pattern.chain()) {
        if (reference.count(pattern_node) != branch.count(pattern_node))
            return false;
        if (!reference.count(pattern_node))
            continue;
        const auto reference_node = reference.at(pattern_node).get_node_shared_ptr();
        const auto branch_node = branch.at(pattern_node).get_node_shared_ptr();
        if (!ov::op::util::node_attributes_are_equal(reference_node, branch_node))
            return false;
    }
    const auto weights = [&pattern](const PatternValueMap& map) {
        return map.at(pattern.conv).get_node()->input_value(1);
    };
    if (!constant_inputs_are_equal(weights(reference), weights(branch)))
        return false;
    return !reference.count(pattern.add) || constant_inputs_are_equal(reference.at(pattern.bias), branch.at(pattern.bias));
}

// Checks relations the single-branch pattern can't express: branch i gathers slice i of the same
// tensor, the slices cover the whole replica dimension, and all branches are equal
std::optional<std::vector<PatternValueMap>> match_branches(const BranchPattern& pattern, const ov::Output<ov::Node>& concat_output) {
    const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(concat_output.get_node_shared_ptr());
    const auto replica_count = concat ? concat->get_input_size() : 0;
    // Replicas are stacked on axis 0, so Concat must be on another axis
    if (replica_count < 2 || normalized_concat_axis(*concat) <= 0)
        return std::nullopt;

    std::vector<PatternValueMap> branches;
    branches.reserve(replica_count);
    for (size_t i = 0; i < replica_count; ++i) {
        Matcher matcher(pattern.relu, "BatchReplicatedBranch");
        if (!matcher.match(concat->input_value(i)))
            return std::nullopt;
        branches.push_back(matcher.get_pattern_value_map());
        const auto& branch = branches.back();

        const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(branch.at(pattern.gather).get_node_shared_ptr());
        const auto index = ov::as_type_ptr<ov::op::v0::Constant>(gather->get_input_node_shared_ptr(1));
        if (index->cast_vector<int64_t>() != std::vector<int64_t>{static_cast<int64_t>(i)})
            return std::nullopt;

        const auto& reference = branches.front();
        const auto reference_gather = reference.at(pattern.gather).get_node_shared_ptr();
        if (i == 0) {
            const auto data_shape = gather->get_input_shape(0);
            const auto axis = static_cast<size_t>(gather->get_axis());
            if (axis >= data_shape.size() || data_shape[axis] != replica_count)
                return std::nullopt;
        } else if (branch.at(pattern.data) != reference.at(pattern.data) || !ov::op::util::node_attributes_are_equal(gather, reference_gather) ||
                   gather->get_axis() != ov::as_type_ptr<ov::op::v8::Gather>(reference_gather)->get_axis() || !branches_are_equal(pattern, reference, branch)) {
            return std::nullopt;
        }
    }
    return branches;
}

// rt_info of all replicas is merged (e.g. fused names) so query_model reports all of them
void copy_metadata(const NodeVector& replicas, const NodePtr& clone) {
    const auto& reference = replicas.front();
    clone->set_friendly_name(reference->get_friendly_name());
    ov::copy_runtime_info(replicas, clone);
    for (size_t i = 0; i < clone->get_input_size(); ++i)
        clone->input(i).get_rt_info() = reference->input(i).get_rt_info();
    for (size_t i = 0; i < clone->get_output_size(); ++i) {
        clone->output(i).get_rt_info() = reference->output(i).get_rt_info();
        clone->output(i).set_names(reference->output(i).get_names());
    }
}

bool batch_replicated_branches(const BranchPattern& pattern,
                               const std::shared_ptr<ov::op::v0::Concat>& output_concat,
                               const std::vector<PatternValueMap>& branches) {
    const auto& reference = branches.front();
    const auto replica_count = branches.size();
    const auto replicas_of = [&branches](const std::shared_ptr<ov::Node>& pattern_node) {
        NodeVector replicas;
        for (const auto& branch : branches)
            replicas.push_back(branch.at(pattern_node).get_node_shared_ptr());
        return replicas;
    };

    // [1, C, H, W] x N -> [N, C, H, W]
    ov::OutputVector slices;
    for (const auto& gather : replicas_of(pattern.gather))
        slices.push_back(gather);
    auto batched_input = std::make_shared<ov::op::v0::Concat>(slices, 0);
    batched_input->set_friendly_name(output_concat->get_friendly_name() + "/replica_batch");
    ov::copy_runtime_info(output_concat, batched_input);

    ov::Output<ov::Node> reference_output = reference.at(pattern.gather);
    ov::Output<ov::Node> batched = batched_input;
    for (const auto& pattern_node : pattern.chain()) {
        if (!reference.count(pattern_node))
            continue;
        const auto replicas = replicas_of(pattern_node);
        const auto& reference_node = replicas.front();
        ov::OutputVector inputs = reference_node->input_values();
        // Constant inputs are equal in all replicas, so the reference ones are reused as is with
        // their markers. Add is commutative, so the data input may be on any port
        for (auto& input : inputs) {
            if (input == reference_output)
                input = batched;
        }
        const auto clone = reference_node->clone_with_new_inputs(inputs);
        copy_metadata(replicas, clone);
        reference_output = reference_node->output(0);
        batched = clone->output(0);
    }

    const auto branch_shape = output_concat->get_input_shape(0);
    auto batched_shape = branch_shape;
    batched_shape[0] = replica_count;
    if (batched.get_shape() != batched_shape)
        return false;

    // Restore the Concat layout, e.g. for axis 3: [N, C, H, W] -> [C, H, N, W] -> [1, C, H, N * W]
    const auto rank = branch_shape.size();
    const auto concat_axis = static_cast<size_t>(normalized_concat_axis(*output_concat));
    std::vector<int64_t> transpose_order;
    std::vector<int64_t> output_shape{1};
    for (size_t axis = 1; axis < rank; ++axis) {
        if (axis == concat_axis)
            transpose_order.push_back(0);
        transpose_order.push_back(static_cast<int64_t>(axis));
        output_shape.push_back(static_cast<int64_t>(axis == concat_axis ? replica_count * branch_shape[axis] : branch_shape[axis]));
    }
    auto repack_transpose = std::make_shared<ov::op::v1::Transpose>(batched, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, transpose_order));
    repack_transpose->set_friendly_name(output_concat->get_friendly_name() + "/replica_repack_transpose");
    auto repacked_output =
        std::make_shared<ov::op::v1::Reshape>(repack_transpose, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank}, output_shape), false);
    ov::copy_runtime_info(output_concat, repack_transpose);
    ov::replace_node_update_name(output_concat, repacked_output);
    return true;
}

}  // namespace

BatchReplicatedBranches::BatchReplicatedBranches() {
    const auto branch_pattern = std::make_shared<BranchPattern>();
    auto output_concat = wrap_type<ov::op::v0::Concat>(has_static_shape() &&
                                                       Predicate(
                                                           [](const ov::Output<ov::Node>& output) {
                                                               return !is_type_relaxed(output.get_node());
                                                           },
                                                           "not_type_relaxed") &&
                                                       Predicate(
                                                           [branch_pattern](const ov::Output<ov::Node>& output) {
                                                               return match_branches(*branch_pattern, output).has_value();
                                                           },
                                                           "replicated_branches_match"));

    auto callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        if (transformation_callback(concat))
            return false;
        const auto branches = match_branches(*branch_pattern, concat->output(0));
        return branches && batch_replicated_branches(*branch_pattern, concat, *branches);
    };

    auto matcher = std::make_shared<Matcher>(output_concat, "BatchReplicatedBranches");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
