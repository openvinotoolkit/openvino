// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_replicated_branches.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
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
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Predicate;

bool is_type_relaxed(const ov::Node* node) {
    return dynamic_cast<const ov::op::TypeRelaxedBase*>(node) != nullptr;
}

// Branch i: Gather(data, i, axis) -> Convolution -> [MaxPool] -> [Add] -> [Relu]
class ReplicaBranch : public ov::pass::pattern::op::Block {
public:
    explicit ReplicaBranch(size_t index) : Block({}, {}, "ReplicaBranch" + std::to_string(index)) {
        // Branch nodes are cloned with [N, ...] input instead of [1, ...], so they must have batch 1
        // and a single consumer; TypeRelaxed and control dependencies are not preserved by cloning
        const auto branch_node = has_static_shape() && shape_matches("[1, ?, ?, ?]") && consumers_count(1) &&
                                 Predicate(
                                     [](const ov::Output<ov::Node>& output) {
                                         const auto* node = output.get_node();
                                         return !is_type_relaxed(node) && node->get_control_dependencies().empty() && node->get_control_dependents().empty();
                                     },
                                     "batchable_branch_node");
        // MaxPool indices are not batched, so they must be unused
        const auto indices_unused = Predicate(
            [](const ov::Output<ov::Node>& output) {
                const auto* node = output.get_node();
                return output.get_index() == 0 && (node->get_output_size() < 2 || node->output(1).get_target_inputs().empty());
            },
            "max_pool_indices_unused");

        auto data = any_input(has_static_shape());
        auto axis = wrap_type<ov::op::v0::Constant>();
        auto gather = wrap_type<ov::op::v8::Gather>({data, wrap_type<ov::op::v0::Constant>(value_matches(std::to_string(index))), axis},
                                                    branch_node && attrs_match({{"batch_dims", 0}}));
        auto weights = wrap_type<ov::op::v0::Constant>();
        auto weights_convert = optional<ov::op::v0::Convert>({weights});
        auto conv = wrap_type<ov::op::v1::Convolution>({gather, weights_convert}, branch_node);
        auto pool = optional<ov::op::v1::MaxPool, ov::op::v8::MaxPool, ov::op::v14::MaxPool>({conv}, branch_node && indices_unused);
        // Bias broadcast to [1, C, H, W] is also valid for [N, C, H, W]
        auto bias = wrap_type<ov::op::v0::Constant>();
        auto add = optional<ov::op::v1::Add>({pool, bias}, branch_node && attrs_match({{"auto_broadcast", "numpy"}}));
        auto relu = optional<ov::op::v0::Relu>({add}, branch_node);

        m_inputs = {data};
        m_outputs = {relu};
        REGISTER_ANCHORS(this, data, axis, gather, weights, weights_convert, conv, pool, bias, add, relu);
    }

    // Compared by value
    std::vector<std::optional<ov::Output<ov::Node>>> get_constants_to_compare(const PatternValueMap& pm) const {
        return {get_anchor("axis", pm), get_anchor("weights", pm), get_anchor("bias", pm)};
    }

    // Compared by type and attributes
    std::vector<std::optional<ov::Output<ov::Node>>> get_nodes_to_compare(const PatternValueMap& pm) const {
        return {get_anchor("gather", pm),
                get_anchor("weights_convert", pm),
                get_anchor("conv", pm),
                get_anchor("pool", pm),
                get_anchor("add", pm),
                get_anchor("relu", pm)};
    }
};

struct MatchedBranch {
    std::shared_ptr<ReplicaBranch> block;
    PatternValueMap pm;

    std::optional<ov::Output<ov::Node>> get(const std::string& name) const {
        return block->get_anchor(name, pm);
    }
};

// Optional nodes must be matched in both branches or in neither
template <typename Equal>
bool all_equal(const std::vector<std::optional<ov::Output<ov::Node>>>& lhs, const std::vector<std::optional<ov::Output<ov::Node>>>& rhs, Equal equal) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].has_value() != rhs[i].has_value() || (lhs[i] && !equal(lhs[i]->get_node_shared_ptr(), rhs[i]->get_node_shared_ptr())))
            return false;
    }
    return true;
}

bool branches_are_equal(const MatchedBranch& lhs, const MatchedBranch& rhs) {
    const auto constants_are_equal = [](const std::shared_ptr<ov::Node>& lhs, const std::shared_ptr<ov::Node>& rhs) {
        return ov::op::util::constants_are_equal(ov::as_type_ptr<ov::op::v0::Constant>(lhs), ov::as_type_ptr<ov::op::v0::Constant>(rhs));
    };
    // The pattern fixes the branch ops, and all of them have comparable attributes
    const auto attributes_are_equal = [](const std::shared_ptr<ov::Node>& lhs, const std::shared_ptr<ov::Node>& rhs) {
        return lhs->get_type_info() == rhs->get_type_info() && ov::op::util::get_comparable_attributes(lhs) == ov::op::util::get_comparable_attributes(rhs);
    };
    return all_equal(lhs.block->get_constants_to_compare(lhs.pm), rhs.block->get_constants_to_compare(rhs.pm), constants_are_equal) &&
           all_equal(lhs.block->get_nodes_to_compare(lhs.pm), rhs.block->get_nodes_to_compare(rhs.pm), attributes_are_equal);
}

// Checks relations the single-branch pattern can't express: all branches gather from the same
// tensor, the slices cover the whole replica dimension, and all branches are equal
std::optional<std::vector<MatchedBranch>> match_branches(const ov::Output<ov::Node>& concat_output) {
    const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(concat_output.get_node_shared_ptr());
    const auto replica_count = concat ? concat->get_input_size() : 0;
    // Replicas are stacked on axis 0, so Concat must be on another axis
    if (replica_count < 2 || ov::util::normalize(concat->get_axis(), concat->get_output_shape(0).size()) <= 0)
        return std::nullopt;

    std::vector<MatchedBranch> branches;
    branches.reserve(replica_count);
    for (size_t i = 0; i < replica_count; ++i) {
        const auto block = std::make_shared<ReplicaBranch>(i);
        Matcher matcher(block, "BatchReplicatedBranch");
        if (!matcher.match(concat->input_value(i)))
            return std::nullopt;
        const auto& branch = branches.emplace_back(MatchedBranch{block, matcher.get_pattern_value_map()});
        const auto& reference = branches.front();
        if (i == 0) {
            const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(branch.get("gather")->get_node_shared_ptr());
            if (gather->get_input_shape(0)[gather->get_axis()] != replica_count)
                return std::nullopt;
        } else if (branch.get("data") != reference.get("data") || !branches_are_equal(reference, branch)) {
            return std::nullopt;
        }
    }
    return branches;
}

// rt_info of all replicas is merged (e.g. fused names) so query_model reports all of them
void copy_metadata(const ov::NodeVector& replicas, const std::shared_ptr<ov::Node>& clone) {
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

bool batch_replicated_branches(const std::shared_ptr<ov::op::v0::Concat>& output_concat, const std::vector<MatchedBranch>& branches) {
    const auto& reference = branches.front();
    const auto replica_count = branches.size();
    const auto replicas_of = [&branches](const std::string& name) {
        ov::NodeVector replicas;
        for (const auto& branch : branches)
            replicas.push_back(branch.get(name)->get_node_shared_ptr());
        return replicas;
    };

    // Branch i gathers slice i of data, so the stacked slices are data with the replica axis moved to the front.
    // Transpose is needed only when a dim before the replica axis is not 1, otherwise Reshape keeps the order
    const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(reference.get("gather")->get_node_shared_ptr());
    ov::Output<ov::Node> data = gather->input_value(0);
    const auto& data_shape = data.get_shape();
    const auto replica_axis = static_cast<size_t>(gather->get_axis());
    ov::NodeVector batched_input_nodes;
    if (std::any_of(data_shape.begin(), data_shape.begin() + replica_axis, [](size_t dim) {
            return dim != 1;
        })) {
        std::vector<int64_t> order{static_cast<int64_t>(replica_axis)};
        for (size_t axis = 0; axis < data_shape.size(); ++axis) {
            if (axis != replica_axis)
                order.push_back(static_cast<int64_t>(axis));
        }
        data = std::make_shared<ov::op::v1::Transpose>(data, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{order.size()}, order));
        data.get_node()->set_friendly_name(output_concat->get_friendly_name() + "/replica_batch_transpose");
        batched_input_nodes.push_back(data.get_node_shared_ptr());
    }
    auto batched_input_shape = gather->get_output_shape(0);
    batched_input_shape[0] = replica_count;
    auto batched_input =
        std::make_shared<ov::op::v1::Reshape>(data,
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{batched_input_shape.size()}, batched_input_shape),
                                              false);
    batched_input->set_friendly_name(output_concat->get_friendly_name() + "/replica_batch");
    batched_input_nodes.push_back(batched_input);
    ov::copy_runtime_info(replicas_of("gather"), batched_input_nodes);

    ov::Output<ov::Node> reference_output = *reference.get("gather");
    ov::Output<ov::Node> batched = batched_input;
    for (const auto& name : {"conv", "pool", "add", "relu"}) {
        if (!reference.get(name))
            continue;
        const auto replicas = replicas_of(name);
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

    OPENVINO_ASSERT(batched.get_shape() == batched_shape, "Unexpected shape of the batched branch: ", batched.get_shape());

    // Restore the Concat layout, e.g. for axis 3: [N, C, H, W] -> [C, H, N, W] -> [1, C, H, N * W]
    const auto rank = branch_shape.size();
    const auto concat_axis = static_cast<size_t>(ov::util::normalize(output_concat->get_axis(), rank));
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
    auto output_concat = wrap_type<ov::op::v0::Concat>(has_static_shape() &&
                                                       Predicate(
                                                           [](const ov::Output<ov::Node>& output) {
                                                               return !is_type_relaxed(output.get_node());
                                                           },
                                                           "not_type_relaxed") &&
                                                       Predicate(
                                                           [](const ov::Output<ov::Node>& output) {
                                                               return match_branches(output).has_value();
                                                           },
                                                           "replicated_branches_match"));

    auto callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        if (transformation_callback(concat))
            return false;
        const auto branches = match_branches(concat->output(0));
        return branches && batch_replicated_branches(concat, *branches);
    };

    auto matcher = std::make_shared<Matcher>(output_concat, "BatchReplicatedBranches");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
