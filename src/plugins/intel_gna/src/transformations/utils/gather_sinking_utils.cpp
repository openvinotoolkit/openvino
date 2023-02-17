// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/gather_sinking_utils.hpp"

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"

namespace gather_sinking {

using namespace ov;
using namespace ov::intel_gna::rt_info;
using namespace ov::opset9;

using NodePtr = std::shared_ptr<Node>;

GatherInputsInfo GetFirstGatherInput(NodePtr node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto gather_node = as_type_ptr<Gather>(input_node);
        if (!gather_node)
            continue;
        auto indices_const_node = as_type_ptr<Constant>(gather_node->input_value(1).get_node_shared_ptr());
        if (!indices_const_node)
            continue;
        auto axes_const_node = as_type_ptr<Constant>(gather_node->input_value(2).get_node_shared_ptr());
        if (!axes_const_node)
            continue;
        {
            GatherInputsInfo input_info;
            input_info.gather = gather_node;
            input_info.indices_const = indices_const_node;
            input_info.axes_const = axes_const_node;
            input_info.input_idx = input_idx;
            return input_info;
        }
    }

    return GatherInputsInfo();
}

bool IfNodeHasGatherInputs(const Output<Node>& output) {
    GatherInputsInfo inputs_info = GetFirstGatherInput(output.get_node_shared_ptr());
    return !inputs_info.isEmpty();
}

namespace {

bool HasDynamicRankInput(NodePtr node) {
    for (auto& input_node : node->input_values()) {
        const Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return true;
    }
    return false;
}

Rank::value_type GetMaxInputRank(const NodePtr& node) {
    Rank::value_type max_input_rank = 0;
    for (auto& input_node : node->input_values()) {
        const Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return -1;
        const Rank::value_type output_rank_len = output_rank.get_length();
        if (output_rank_len > max_input_rank)
            max_input_rank = output_rank_len;
    }
    return max_input_rank;
}

NodePtr InsertUnsqueeze(Output<Node> node, size_t n_dims) {
    std::vector<size_t> dims(n_dims);
    std::iota(dims.begin(), dims.end(), 0);
    auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{dims.size()}, dims);
    auto unsqueeze = std::make_shared<Unsqueeze>(node, unsqueeze_const);
    copy_runtime_info(node.get_node_shared_ptr(), {unsqueeze, unsqueeze_const});
    return unsqueeze;
}

Output<Node> FixInputNodeRank(Output<Node> input_node, Rank::value_type required_rank) {
    const Rank::value_type output_rank = input_node.get_partial_shape().rank().get_length();
    if (output_rank >= required_rank)
        return input_node;
    return InsertUnsqueeze(input_node, required_rank - output_rank)->output(0);
}

}  // namespace

namespace sink_backward {

NodeVector InsertGatherBeforeNode(NodePtr main_node,
                                  const std::shared_ptr<Constant>& indices_const,
                                  const std::shared_ptr<Constant>& axes_const) {
    if (HasDynamicRankInput(main_node))
        return {};

    NodeVector new_nodes;

    const auto max_input_rank = GetMaxInputRank(main_node);
    if (max_input_rank < 0)
        return {};

    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = FixInputNodeRank(main_node->input_value(i), max_input_rank);

        auto new_indices_const = indices_const->clone_with_new_inputs({});
        auto new_axes_const = axes_const->clone_with_new_inputs({});
        auto new_gather = std::make_shared<Gather>(input_node, new_indices_const, new_axes_const);

        main_node->input(i).replace_source_output(new_gather->output(0));

        copy_runtime_info(input_node.get_node_shared_ptr(), {new_gather, new_indices_const, new_axes_const});

        new_nodes.push_back(new_gather);
    }

    return new_nodes;
}

}  // namespace sink_backward

namespace {
#define CHECK_GATHER_SINKING_SUPPORTED(TYPE, node) \
    if (dynamic_cast<TYPE*>(node)) {               \
        return true;                               \
    }

bool CanPropagateGatherForwardThrough(Node* node) {
    CHECK_GATHER_SINKING_SUPPORTED(ov::op::util::UnaryElementwiseArithmetic, node);
    CHECK_GATHER_SINKING_SUPPORTED(Clamp, node);
    CHECK_GATHER_SINKING_SUPPORTED(Elu, node);
    CHECK_GATHER_SINKING_SUPPORTED(SoftPlus, node);
    CHECK_GATHER_SINKING_SUPPORTED(LogicalNot, node);
    CHECK_GATHER_SINKING_SUPPORTED(Convert, node);
    return false;
}

#undef CHECK_GATHER_SINKING_SUPPORTED

bool CanGatherPropagateForward(NodePtr node) {
    for (auto output : node->outputs()) {
        for (auto& consumer_input : output.get_target_inputs()) {
            if (!CanPropagateGatherForwardThrough(consumer_input.get_node()))
                return false;
        }
    }

    return true;
}

}  // namespace

void UpdateForwardGatherSinkingAbility(NodePtr node) {
    if (!CanGatherPropagateForward(node))
        mark_as_no_gather_sinking_node(node);
}

namespace {

struct GatherInfo {
    bool isEmpty() const {
        return indices.empty();
    }
    bool operator==(const GatherInfo& another) {
        if (indices.size() != another.indices.size())
            return false;
        if (!std::equal(indices.begin(), indices.end(), another.indices.begin()))
            return false;
        return axis == another.axis;
    }
    bool operator!=(const GatherInfo& another) {
        return !(*this == another);
    }

    ov::AxisVector indices;
    int64_t axis = {};
};

GatherInfo GetGatherInfo(Node* node) {
    GatherInfo gather_info;

    auto gather_node = dynamic_cast<Gather*>(node);
    if (!gather_node)
        return {};

    auto constant_node = as_type_ptr<Constant>(gather_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return {};

    gather_info.indices = constant_node->get_axis_vector_val();

    constant_node = as_type_ptr<Constant>(gather_node->input_value(2).get_node_shared_ptr());
    if (!constant_node)
        return {};

    gather_info.axis = constant_node->get_axis_vector_val()[0];

    return gather_info;
}

Node* FindFirstConsumer(NodePtr node) {
    for (auto output : node->outputs()) {
        auto inputs = output.get_target_inputs();
        if (inputs.empty())
            continue;
        return inputs.begin()->get_node();
    }
    return nullptr;
}

bool HasSameOutputGatherNodes(NodePtr main_node) {
    GatherInfo first_gather_info;
    {
        Node* first_consumer = FindFirstConsumer(main_node);
        if (!first_consumer)
            return false;
        first_gather_info = GetGatherInfo(first_consumer);
        if (first_gather_info.isEmpty())
            return false;
    }

    for (size_t output_idx = 0; output_idx < main_node->get_output_size(); ++output_idx) {
        for (auto& input : main_node->get_output_target_inputs(output_idx)) {
            GatherInfo gather_info = GetGatherInfo(input.get_node());
            if (gather_info.isEmpty() || gather_info != first_gather_info)
                return false;
        }
    }

    return true;
}

}  // namespace

bool HasSameOutputGatherNodes(const Output<Node>& output) {
    return HasSameOutputGatherNodes(output.get_node_shared_ptr());
}

void RemoveSingleOutputConsumers(NodePtr node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            Node* consumer = input.get_node();
            if (consumer->get_output_size() != 1)
                continue;
            consumer->output(0).replace(node->output(output_idx));
        }
    }
}

}  // namespace gather_sinking
