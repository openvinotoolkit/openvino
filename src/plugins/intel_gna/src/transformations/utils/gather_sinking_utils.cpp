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
        auto axis_const_node = as_type_ptr<Constant>(gather_node->input_value(2).get_node_shared_ptr());
        if (!axis_const_node)
            continue;
        {
            GatherInputsInfo input_info;
            input_info.gather = gather_node;
            input_info.indices_const = indices_const_node;
            input_info.axis_const = axis_const_node;
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

/*
Converts gather indices to positive form
*/
std::vector<int64_t> NormalizeGatherIndices(const std::vector<int64_t>& indices) {
    std::vector<int64_t> normalized(indices.size());
    for (int i = 0; i < indices.size(); ++i) {
        int64_t index = indices[i];
        if (index < 0)
            index += indices.size();
        normalized[i] = index;
    }
    return normalized;
}

/*
Gets gather indices in positive form
*/
std::vector<int64_t> GetNormalizedGatherIndices(const std::shared_ptr<Constant>& indices) {
    return NormalizeGatherIndices(indices->cast_vector<int64_t>());
}

/*
Converts axis to negative form
*/
int64_t NormalizeNegativeGatherAxis(int64_t axis, ov::Rank::value_type gather_input_rank) {
    if (axis < 0)
        return axis;
    return axis - gather_input_rank;
}

/*
Gets gather axis in negative form
*/
int64_t GetNormalizedNegativeGatherAxis(const std::shared_ptr<Constant>& axis, ov::Rank::value_type gather_input_rank) {
    return NormalizeNegativeGatherAxis(axis->cast_vector<int64_t>()[0], gather_input_rank);
}

int64_t ConvertAxisToPositive(int64_t axis, ov::Rank::value_type rank) {
    if (axis >= 0)
        return axis;
    return axis + rank;
}

/*
Reverts gather indices in a such way that reverted and initial gather will do nothing if
stays after another.
Works only with positive form (no negative indices).
*/
std::vector<int64_t> ReverseGatherIndexes(const std::vector<int64_t>& indexes) {
    std::vector<int64_t> out(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        out.at(indexes[i]) = i;
    }
    return out;
}

size_t GetDimByAxis(const Shape& shape, int64_t axis) {
    if (axis < 0)
        axis += shape.size();
    return shape[axis];
}

Shape Broadcast(const Shape& shape, ov::Rank::value_type rank) {
    const int rank_delta = rank - shape.size();

    if (rank_delta <= 0)
        return shape;

    Shape broadcasted(rank);
    for (int i = 0; i < rank_delta; ++i) {
        broadcasted[i] = 1;
    }
    std::copy(shape.begin(), shape.end(), broadcasted.begin() + rank_delta);

    return broadcasted;
}

}  // namespace

void SwapOutputNames(Output<Node> output1, Output<Node> output2) {
    const auto node2_output_names = output2.get_names();
    output2.set_names(output1.get_names());
    output1.set_names(node2_output_names);
}

void SwapFriendlyNames(NodePtr node1, NodePtr node2) {
    const std::string node2_name = node2->get_friendly_name();
    node2->set_friendly_name(node1->get_friendly_name());
    node1->set_friendly_name(node2_name);
}

void SwapNames(NodePtr node1, NodePtr node2) {
    SwapFriendlyNames(node1, node2);
    SwapOutputNames(node1->output(0), node2->output(0));
}

namespace sink_forward {
/** @brief
  * Inserts inverted Gather layer on all @main_node inputs except input from GatherInputsInfo argument
  * Works only with 1D indices.
  * It's simpler to work with negative gather axis since it doesn't depend on shape broadcasting.
  * Converts gather axis to a negative form
  * Doesn't add Gather layer if input_node_shape[axis] == 1 since it is useless and causes an invalid result.
  * Input nodes can have different shapes. That shapes can have smaller or larger ranks. To manage it we need
  * to find max input shape rank and broadcast all input shapes to it.
  */
void UpdateInputGather(NodePtr main_node, const GatherInputsInfo& gather_input_info) {
    if (gather_input_info.isEmpty() || HasDynamicRankInput(main_node))
        return;

    const int64_t gather_negative_axis = GetNormalizedNegativeGatherAxis(gather_input_info.axis_const,
                                                                         gather_input_info.gather->get_input_partial_shape(0).rank().get_length());

    const std::vector<int64_t> gather_indices = GetNormalizedGatherIndices(gather_input_info.indices_const);
    const std::vector<int64_t> reversed_gather_indices = ReverseGatherIndexes(gather_indices);

    const auto indices_element_type = gather_input_info.indices_const->get_element_type();
    const auto axis_element_type = gather_input_info.axis_const->get_element_type();

    const auto max_input_rank = GetMaxInputRank(main_node);
    if (max_input_rank < 0)
        return;

    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = main_node->input_value(i);
        if (i == gather_input_info.input_idx) {
            auto gather_parent = input_node.get_node()->input_value(0);
            main_node->input(i).replace_source_output(gather_parent);
        } else {
            const Shape broadcasted_input_shape = Broadcast(input_node.get_shape(), max_input_rank);
            if (GetDimByAxis(broadcasted_input_shape, gather_negative_axis) == 1)
                continue;

            auto new_indices_const = std::make_shared<Constant>(indices_element_type,
                                                                 Shape{reversed_gather_indices.size()},
                                                                 reversed_gather_indices);

            const int64_t gather_positive_axis = ConvertAxisToPositive(gather_negative_axis,
                                                                       input_node.get_partial_shape().rank().get_length());
            auto new_axis_const = std::make_shared<Constant>(axis_element_type,
                                                             Shape{},
                                                             gather_positive_axis);

            auto new_gather = std::make_shared<Gather>(input_node, new_indices_const, new_axis_const);

            main_node->input(i).replace_source_output(new_gather->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_gather, new_indices_const, new_axis_const});
        }
    }
}

NodeVector InsertOutputGather(NodePtr main_node, const GatherInputsInfo& gather_input_info) {
    if (gather_input_info.isEmpty())
        return {};

    const int64_t gather_negative_axis = GetNormalizedNegativeGatherAxis(gather_input_info.axis_const,
                                                                         gather_input_info.gather->get_input_partial_shape(0).rank().get_length());
    const auto axis_element_type = gather_input_info.axis_const->get_element_type();

    NodeVector new_nodes;
    for (size_t i = 0; i < main_node->get_output_size(); ++i) {
        auto main_node_consumers = main_node->output(i).get_target_inputs();

        auto new_indices_const = gather_input_info.indices_const->clone_with_new_inputs({});

        const int64_t gather_positive_axis = ConvertAxisToPositive(gather_negative_axis,
                                                                   main_node->output(i).get_partial_shape().rank().get_length());
        auto new_axis_const = std::make_shared<Constant>(axis_element_type,
                                                         Shape{},
                                                         gather_positive_axis);
        auto new_gather = std::make_shared<Gather>(main_node->output(i), new_indices_const, new_axis_const);

        for (auto& consumer : main_node_consumers) {
            consumer.replace_source_output(new_gather);
        }

        copy_runtime_info(main_node, {new_gather, new_indices_const, new_axis_const});
        SwapOutputNames(main_node->output(i), new_gather->output(0));

        if (main_node->get_output_size() > 1)
            new_gather->set_friendly_name(main_node->get_friendly_name() + "." + std::to_string(i));
        else
            SwapFriendlyNames(new_gather, main_node);

        new_nodes.push_back(new_gather);
    }

    return new_nodes;
}

} // namespace sink_forward

namespace sink_backward {

NodeVector InsertGatherBeforeNode(NodePtr main_node,
                                  const std::shared_ptr<Constant>& indices_const,
                                  const std::shared_ptr<Constant>& axis_const,
                                  const std::shared_ptr<Gather>& gather_node) {
    if (HasDynamicRankInput(main_node))
        return {};

    const int64_t gather_negative_axis = GetNormalizedNegativeGatherAxis(axis_const,
                                                                         gather_node->get_input_partial_shape(0).rank().get_length());
    const auto axis_element_type = axis_const->get_element_type();

    const auto max_input_rank = GetMaxInputRank(main_node);
    if (max_input_rank < 0)
        return {};

    NodeVector new_nodes;
    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = main_node->input_value(i);

        const Shape broadcasted_input_shape = Broadcast(input_node.get_shape(), max_input_rank);
        if (GetDimByAxis(broadcasted_input_shape, gather_negative_axis) == 1)
            continue;

        auto new_indices_const = indices_const->clone_with_new_inputs({});

        const int64_t gather_positive_axis = ConvertAxisToPositive(gather_negative_axis,
                                                                   input_node.get_partial_shape().rank().get_length());
        auto new_axis_const = std::make_shared<Constant>(axis_element_type,
                                                         Shape{},
                                                         gather_positive_axis);

        auto new_gather = std::make_shared<Gather>(input_node, new_indices_const, new_axis_const);

        main_node->input(i).replace_source_output(new_gather->output(0));

        copy_runtime_info(input_node.get_node_shared_ptr(), {new_gather, new_indices_const, new_axis_const});

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

std::function<bool(Output<Node>)> rank_not_more_than(const ov::Rank::value_type expected_rank) {
    return [=](Output<Node> output) -> bool {
        const Rank rank = output.get_partial_shape().rank();
        return (rank.is_static() && (rank.get_length() <= expected_rank));
    };
}

bool constant_has_rank_not_more_than(const std::shared_ptr<Constant>& node, const ov::Rank::value_type expected_rank) {
    const Rank rank = node->get_output_partial_shape(0).rank();
    return (rank.is_static() && (rank.get_length() <= expected_rank)); 
}

}  // namespace gather_sinking
