// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/gather_sinking_utils.hpp"

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::intel_gna::graph_utils;
using namespace ov::intel_gna::rt_info;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::pass::helper;

namespace gather_sinking {
using NodePtr = std::shared_ptr<Node>;

GatherInputsInfo get_first_gather_input(NodePtr node) {
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

namespace sink_forward {
/** @brief
 * Inserts inverted Gather layer on all @main_node inputs except input from GatherInputsInfo argument
 * Works only with 1D indices.
 */
void update_input_gather(NodePtr main_node,
                         const GatherInputsInfo& gather_input_info,
                         const int64_t* a_gather_negative_axis) {
    if (gather_input_info.isEmpty() || has_dynamic_rank_input(main_node))
        return;

    /* It's simpler to work with negative gather axis since it doesn't depend on shape unsqueezeing.
     * Converts gather axis to a negative form
     */
    int64_t gather_negative_axis = {};
    if (a_gather_negative_axis)
        gather_negative_axis = *a_gather_negative_axis;
    else
        gather_negative_axis = get_normalized_negative_gather_axis(
            gather_input_info.axis_const,
            gather_input_info.gather->get_input_partial_shape(0).rank().get_length());

    const std::vector<int64_t> gather_indices = get_normalized_gather_indices(gather_input_info.indices_const);
    const std::vector<int64_t> reversed_gather_indices = reverse_gather_indexes(gather_indices);

    const auto indices_element_type = gather_input_info.indices_const->get_element_type();
    const auto axis_element_type = gather_input_info.axis_const->get_element_type();

    const auto max_input_rank = get_max_input_rank(main_node);
    if (max_input_rank < 0)
        return;

    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = main_node->input_value(i);
        if (i == gather_input_info.input_idx) {
            auto gather_parent = input_node.get_node()->input_value(0);
            main_node->input(i).replace_source_output(gather_parent);
        } else {
            /* Doesn't add Gather layer if input_node_shape[axis] == 1 since it is useless and causes an invalid result.
             * Input nodes can have different shapes. That shapes can have smaller or larger ranks. To manage it we need
             * to find max input shape rank and unsqueeze all input shapes to it.
             */
            const Shape unsqueezed_input_shape = unsqueeze_shape(input_node.get_shape(), max_input_rank);
            if (get_dim_by_axis(unsqueezed_input_shape, gather_negative_axis) == 1)
                continue;

            auto new_indices_const = std::make_shared<Constant>(indices_element_type,
                                                                Shape{reversed_gather_indices.size()},
                                                                reversed_gather_indices);

            const int64_t gather_positive_axis =
                convert_axis_to_positive(gather_negative_axis, input_node.get_partial_shape().rank().get_length());
            auto new_axis_const = std::make_shared<Constant>(axis_element_type, Shape{}, gather_positive_axis);

            auto new_gather = std::make_shared<Gather>(input_node, new_indices_const, new_axis_const);

            main_node->input(i).replace_source_output(new_gather->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_gather, new_indices_const, new_axis_const});
        }
    }
}

NodeVector insert_output_gather(NodePtr main_node, const GatherInputsInfo& gather_input_info) {
    if (gather_input_info.isEmpty())
        return {};

    const int64_t gather_negative_axis =
        get_normalized_negative_gather_axis(gather_input_info.axis_const,
                                            gather_input_info.gather->get_input_partial_shape(0).rank().get_length());
    const auto axis_element_type = gather_input_info.axis_const->get_element_type();

    NodeVector new_nodes;
    for (size_t i = 0; i < main_node->get_output_size(); ++i) {
        auto main_node_consumers = main_node->output(i).get_target_inputs();

        auto new_indices_const = gather_input_info.indices_const->clone_with_new_inputs({});

        const int64_t gather_positive_axis =
            convert_axis_to_positive(gather_negative_axis,
                                     main_node->output(i).get_partial_shape().rank().get_length());
        auto new_axis_const = std::make_shared<Constant>(axis_element_type, Shape{}, gather_positive_axis);
        auto new_gather = std::make_shared<Gather>(main_node->output(i), new_indices_const, new_axis_const);

        for (auto& consumer : main_node_consumers) {
            consumer.replace_source_output(new_gather);
        }

        copy_runtime_info(main_node, {new_gather, new_indices_const, new_axis_const});
        swap_output_names(main_node->output(i), new_gather->output(0));

        if (main_node->get_output_size() > 1)
            new_gather->set_friendly_name(main_node->get_friendly_name() + "." + std::to_string(i));
        else
            swap_friendly_names(new_gather, main_node);

        new_nodes.push_back(new_gather);
    }

    return new_nodes;
}

}  // namespace sink_forward

namespace sink_backward {

NodeVector insert_gather_before_node(NodePtr main_node,
                                     const std::shared_ptr<Constant>& indices_const,
                                     const std::shared_ptr<Constant>& axis_const,
                                     const std::shared_ptr<Gather>& gather_node,
                                     std::vector<int> input_indices) {
    if (has_dynamic_rank_input(main_node))
        return {};

    if (input_indices.empty()) {
        input_indices.resize(main_node->get_input_size());
        std::iota(input_indices.begin(), input_indices.end(), 0);
    }

    const int64_t gather_negative_axis =
        get_normalized_negative_gather_axis(axis_const, gather_node->get_input_partial_shape(0).rank().get_length());
    const auto axis_element_type = axis_const->get_element_type();

    const auto max_input_rank = get_max_input_rank(main_node);
    if (max_input_rank < 0)
        return {};

    NodeVector new_nodes;
    for (const auto& i : input_indices) {
        auto input_node = main_node->input_value(i);

        const Shape unsqueezed_input_shape = unsqueeze_shape(input_node.get_shape(), max_input_rank);
        if (get_dim_by_axis(unsqueezed_input_shape, gather_negative_axis) == 1)
            continue;

        auto new_indices_const = indices_const->clone_with_new_inputs({});

        const int64_t gather_positive_axis =
            convert_axis_to_positive(gather_negative_axis, input_node.get_partial_shape().rank().get_length());
        auto new_axis_const = std::make_shared<Constant>(axis_element_type, Shape{}, gather_positive_axis);

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
    CHECK_GATHER_SINKING_SUPPORTED(ov::op::util::BinaryElementwiseArithmetic, node);
    CHECK_GATHER_SINKING_SUPPORTED(Gather, node);
    CHECK_GATHER_SINKING_SUPPORTED(Reshape, node);
    CHECK_GATHER_SINKING_SUPPORTED(MatMul, node);
    return false;
}

#undef CHECK_GATHER_SINKING_SUPPORTED

bool can_gather_propagate_forward(NodePtr node) {
    for (const auto& output : node->outputs()) {
        for (auto& consumer_input : output.get_target_inputs()) {
            if (!CanPropagateGatherForwardThrough(consumer_input.get_node()))
                return false;
        }
    }

    return true;
}

}  // namespace

void update_forward_gather_sinking_ability(NodePtr node) {
    if (!can_gather_propagate_forward(node))
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
        if (!are_shapes_equal(indices, another.indices))
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

bool has_same_output_gather_nodes(NodePtr main_node) {
    GatherInfo first_gather_info;
    {
        Node* first_consumer = find_first_consumer(main_node);
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

bool has_same_output_gather_nodes(const Output<Node>& output) {
    return has_same_output_gather_nodes(output.get_node_shared_ptr());
}

void remove_single_output_consumers(NodePtr node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            Node* consumer = input.get_node();
            if (consumer->get_output_size() != 1)
                continue;
            consumer->output(0).replace(node->output(output_idx));
        }
    }
}

bool is_gather_sinking_enabled(const Output<Node>& output) {
    auto node = ov::as_type_ptr<Gather>(output.get_node_shared_ptr());
    if (!node)
        return false;
    return is_gather_sinking_node(output.get_node_shared_ptr());
}

bool is_split_sinked(const Output<Node>& output) {
    return find_first_input_node<Split>(output.get_node_shared_ptr()) && is_gather_sinking_node(output);
}

int64_t normalize_negative_gather_axis(int64_t axis, ov::Rank::value_type gather_input_rank) {
    if (axis < 0)
        return axis;
    return axis - gather_input_rank;
}

int64_t get_normalized_negative_gather_axis(const std::shared_ptr<Constant>& axis,
                                            ov::Rank::value_type gather_input_rank) {
    return normalize_negative_gather_axis(axis->cast_vector<int64_t>()[0], gather_input_rank);
}

bool get_gather_axis(const std::shared_ptr<ov::Node>& gather, int64_t& axis) {
    auto gather_node = as_type_ptr<Gather>(gather);
    if (!gather_node)
        return false;
    auto output_gather_axis_node = as_type_ptr<Constant>(gather->input_value(2).get_node_shared_ptr());
    if (!output_gather_axis_node)
        return false;
    axis = get_normalized_negative_gather_axis(output_gather_axis_node,
                                               gather->get_input_partial_shape(0).rank().get_length());
    return true;
}

}  // namespace gather_sinking
