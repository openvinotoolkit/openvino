#include "transformations/common_optimizations/transpose_sinking_utils.hpp"

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

namespace transpose_sinking {

using namespace ov;
using namespace ov::opset9;

using NodePtr = std::shared_ptr<Node>;

TransposeInputsInfo GetFirstTransposeInput(NodePtr node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto transpose_node = as_type_ptr<Transpose>(input_node);
        if (!transpose_node)
            continue;
        auto constant_node = as_type_ptr<Constant>(transpose_node->input_value(1).get_node_shared_ptr());
        if (!constant_node)
            continue;
        {
            TransposeInputsInfo input_info;
            input_info.transpose = transpose_node;
            input_info.transpose_const = constant_node;
            input_info.input_idx = input_idx;
            return input_info;
        }
    }

    return TransposeInputsInfo();
}

bool IfNodeHasTransposeInputs(const Output<Node>& output) {
    TransposeInputsInfo inputs_info = GetFirstTransposeInput(output.get_node_shared_ptr());
    return !inputs_info.isEmpty();
}

AxisVector ReverseTransposeOrder(const AxisVector& axis_order) {
    AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++) {
        out.at(axis_order[i]) = i;
    }
    return out;
}

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

// insert input reversed transposes, remove first input tranpose
void UpdateInputTransposes(NodePtr main_node, TransposeInputsInfo& transpose_input_info) {
    if (transpose_input_info.isEmpty())
        return;
    const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
    const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
    const size_t tranpose_input_index = transpose_input_info.input_idx;
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = main_node->input_value(i);
        if (i == tranpose_input_index) {
            auto transpose_parent = input_node.get_node()->input_value(0);
            main_node->input(i).replace_source_output(transpose_parent);
        } else {
            auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                                  Shape{reversed_traspose_axis_order.size()},
                                                                  reversed_traspose_axis_order);
            auto new_transpose = std::make_shared<Transpose>(input_node, new_transpose_const);

            main_node->input(i).replace_source_output(new_transpose->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});
        }
    }
}

void RemoveZeroInputNode(NodePtr main_node) {
    auto input_node = main_node->input_value(0);
    if (input_node.get_node()->get_input_size() < 1)
        return;
    auto parent_node = input_node.get_node()->input_value(0);
    main_node->input(0).replace_source_output(parent_node);
}

NodeVector InsertOutputTransposes(NodePtr main_node, TransposeInputsInfo& transpose_input_info) {
    if (transpose_input_info.isEmpty())
        return {};
    const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
    const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    NodeVector new_nodes;

    for (size_t i = 0; i < main_node->get_output_size(); ++i) {
        auto main_node_consumers = main_node->output(i).get_target_inputs();

        auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                              Shape{transpose_axis_order.size()},
                                                              transpose_axis_order);
        auto new_transpose = std::make_shared<Transpose>(main_node->output(i), new_transpose_const);

        for (auto& consumer : main_node_consumers) {
            consumer.replace_source_output(new_transpose);
        }

        copy_runtime_info(main_node, {new_transpose, new_transpose_const});
        SwapOutputNames(main_node->output(i), new_transpose->output(0));

        if (main_node->get_output_size() > 1)
            new_transpose->set_friendly_name(main_node->get_friendly_name() + "." + std::to_string(i));
        else
            SwapFriendlyNames(new_transpose, main_node);

        new_nodes.push_back(new_transpose);
    }

    return new_nodes;
}

}  // namespace sink_forward

namespace sink_backward {
NodeVector InsertTransposeBeforeNode(NodePtr main_node, std::shared_ptr<Constant> transpose_const) {
    const auto transpose_axis_order = transpose_const->get_axis_vector_val();
    const auto transpose_element_type = transpose_const->get_element_type();

    NodeVector new_nodes;

    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = main_node->input_value(i);
        auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                              Shape{transpose_axis_order.size()},
                                                              transpose_axis_order);
        auto new_transpose = std::make_shared<Transpose>(input_node, new_transpose_const);

        main_node->input(i).replace_source_output(new_transpose->output(0));

        copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

        new_nodes.push_back(new_transpose);
    }

    return new_nodes;
}
}  // namespace sink_backward

#define CHECK_TRANSPOSE_SINKING_SUPPORTED(TYPE, node) \
    if (dynamic_cast<TYPE*>(node)) {                  \
        return true;                                  \
    }

namespace {

bool CanPropagateForwardThrough(Node* node) {
    CHECK_TRANSPOSE_SINKING_SUPPORTED(ov::op::util::UnaryElementwiseArithmetic, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Clamp, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Elu, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(SoftPlus, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(LogicalNot, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Convert, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(ov::op::util::BinaryElementwiseArithmetic, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Concat, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Split, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Transpose, node);

    return false;
}

bool CanPropagateForward(NodePtr node) {
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        for (auto& consumer_input : node->output(i).get_target_inputs()) {
            if (!CanPropagateForwardThrough(consumer_input.get_node()))
                return false;
        }
    }

    return true;
}

}  // namespace

void UpdateForwardSinkingAbility(NodePtr node) {
    if (!CanPropagateForward(node))
        mark_as_no_sinking_node(node);
}

}  // namespace transpose_sinking
