#include "transformations/common_optimizations/transpose_sinking_binary.hpp"

#include <openvino/opsets/opset9.hpp>
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

using namespace ov::pass::pattern;
using namespace ov;
using namespace ov::opset9;

namespace {

using NodePtr = std::shared_ptr<Node>;

struct TransposeInputsInfo {
    std::shared_ptr<Transpose> transpose;
    std::shared_ptr<Constant> transpose_const;
    size_t input_idx;

    bool isEmpty() const {
        return !transpose || !transpose_const;
    }
};

TransposeInputsInfo GetFirstTransposeInput(NodePtr node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto transpose_node = as_type_ptr<Transpose>(input_node);
        if (!transpose_node)
            continue;
        auto constant_node =
            as_type_ptr<Constant>(transpose_node->input_value(1).get_node_shared_ptr());
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

}  // namespace

struct OutputTranspose {
    OutputTranspose() : transpose(nullptr), transpose_const(nullptr) {}
    Transpose* transpose;
    Constant* transpose_const;
};

OutputTranspose GetOutputTransposes(NodePtr node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            auto transpose_node = dynamic_cast<Transpose*>(input.get_node());
            if (!transpose_node)
                continue;
            auto constant_node = dynamic_cast<Constant*>(transpose_node->input_value(1).get_node());
            if (!constant_node)
                continue;
            {
                OutputTranspose output_transpose;
                output_transpose.transpose = transpose_node;
                output_transpose.transpose_const = constant_node;

                return output_transpose;
            }
        }
    }

    return OutputTranspose();
}

NodePtr FindSplitInput(Node* node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto split_node = as_type_ptr<Split>(input_node);
        if (split_node)
            return split_node;
    }
    return {};
}

std::shared_ptr<Constant> GetTransposeConstant(Input<Node> input) {
    auto transpose_node = dynamic_cast<Transpose*>(input.get_node());
    if (!transpose_node)
        return {};

    auto constant_node = as_type_ptr<Constant>(transpose_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return {};

    return constant_node;
}

bool HasInputSplitAndTransposeSiblings(const Output<Node>& output) {
    NodePtr split_node = FindSplitInput(output.get_node());
    if (!split_node) {
        return false;
    }

    AxisVector first_transpose_axis_order;
    // get first transpose axis
    {
        auto constant_node = GetTransposeConstant(*(split_node->get_output_target_inputs(0).begin()));
        if (!constant_node)
            return false;
        first_transpose_axis_order = constant_node->get_axis_vector_val();
    }

    for (size_t output_idx = 1; output_idx < split_node->get_output_size(); ++output_idx) {
        for (auto& input : split_node->get_output_target_inputs(output_idx)) {
            auto constant_node = GetTransposeConstant(input);
            if (!constant_node)
                return false;

            AxisVector transpose_axis_order = constant_node->get_axis_vector_val();
            if (transpose_axis_order.size() != first_transpose_axis_order.size())
                return false;
            if (!std::equal(transpose_axis_order.begin(),
                            transpose_axis_order.end(),
                            first_transpose_axis_order.begin()))
                return false;
        }
    }

    return true;
}

pass::TransposeSinkingSplitBackward::TransposeSinkingSplitBackward() {
    MATCHER_SCOPE(TransposeSinkingSplitBackward);

    auto transpose_const_label = wrap_type<Constant>(consumers_count(1));
    auto transpose_label =
        wrap_type<Transpose>({any_input(), transpose_const_label}, HasInputSplitAndTransposeSiblings);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_label_node = pattern_to_output.at(transpose_label).get_node();

        NodePtr split = FindSplitInput(transpose_label_node);
        auto split_axis_constant = as_type_ptr<Constant>(split->input_value(1).get_node_shared_ptr());
        OutputTranspose output_transpose = GetOutputTransposes(split);

        const auto transpose_axis_order = output_transpose.transpose_const->get_axis_vector_val();
        const auto transpose_element_type = output_transpose.transpose_const->get_element_type();

        const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);

        const size_t split_axis = split_axis_constant->get_axis_vector_val()[0];
        const size_t reversed_transposed_split_axis = reversed_traspose_axis_order[split_axis];

        // insert transpose before split
        {
            auto input_node = split->input_value(0);
            auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                                              Shape{transpose_axis_order.size()},
                                                                              transpose_axis_order);
            auto new_transpose = std::make_shared<Transpose>(input_node, new_transpose_const);

            split->input(0).replace_source_output(new_transpose->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

            register_new_node(new_transpose);
        }

        // update split axis
        auto new_split_axis_const = std::make_shared<Constant>(split_axis_constant->get_element_type(),
                                                                           Shape{},
                                                                           reversed_transposed_split_axis);
        split->input(1).replace_source_output(new_split_axis_const);

        // remove split output transposes
        for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
            for (auto& input : split->get_output_target_inputs(output_idx)) {
                input.get_node()->output(0).replace(split->output(output_idx));
            }
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

namespace sink_forward {

// insert input reversed transposes, remove first input tranpose
void UpdateInputTransposes(NodePtr main_node, TransposeInputsInfo & transpose_input_info) {
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
    auto parent_node = input_node.get_node()->input_value(0);
    main_node->input(0).replace_source_output(parent_node);
}

std::vector<NodePtr> InsertOutputTransposes(NodePtr main_node, TransposeInputsInfo & transpose_input_info) {
    const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
    const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    std::vector<NodePtr> new_nodes;

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

} // namespace sink_forward

pass::TransposeSinkingBinaryElementwiseForward::TransposeSinkingBinaryElementwiseForward() {
    MATCHER_SCOPE(TransposeSinkingBinaryElementwiseForward);

    auto main_node_label = wrap_type<op::util::BinaryElementwiseArithmetic>(
        IfNodeHasTransposeInputs);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();

        TransposeInputsInfo transpose_input_info = GetFirstTransposeInput(main_node);

        sink_forward::UpdateInputTransposes(main_node, transpose_input_info);
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

pass::TransposeSinkingConcatForward::TransposeSinkingConcatForward() {
    MATCHER_SCOPE(TransposeSinkingConcatForward);

    auto main_node_label = wrap_type<Concat>(IfNodeHasTransposeInputs);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();

        TransposeInputsInfo transpose_input_info = GetFirstTransposeInput(main_node);

        sink_forward::UpdateInputTransposes(main_node, transpose_input_info);
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
        }

        auto concat_node = as_type_ptr<Concat>(main_node);
        const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        const int64_t transposed_concat_axis = transpose_axis_order[concat_node->get_axis()];
        concat_node->set_concatenation_axis(transposed_concat_axis);

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

pass::TransposeSinkingSplitForward::TransposeSinkingSplitForward() {
    MATCHER_SCOPE(TransposeSinkingSplitForward);

    auto main_node_label = wrap_type<Split>(IfNodeHasTransposeInputs);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();

        TransposeInputsInfo transpose_input_info = GetFirstTransposeInput(main_node);

        sink_forward::RemoveZeroInputNode(main_node);
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
        }

        const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        auto split_node = as_type_ptr<Split>(main_node);
        auto split_axis_constant =
                as_type_ptr<Constant>(split_node->input_value(1).get_node_shared_ptr());
        const size_t split_axis = split_axis_constant->get_axis_vector_val()[0];
        const size_t transposed_split_axis = transpose_axis_order[split_axis];
        auto new_split_axis_const = std::make_shared<Constant>(split_axis_constant->get_element_type(),
                                                                               Shape{},
                                                                               transposed_split_axis);
        split_node->input(1).replace_source_output(new_split_axis_const);

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

namespace sink_backward {
std::vector<NodePtr> InsertTransposeBeforeNode(NodePtr main_node, std::shared_ptr<Constant> transpose_const) {
    const auto transpose_axis_order = transpose_const->get_axis_vector_val();
    const auto transpose_element_type = transpose_const->get_element_type();

    std::vector<NodePtr> new_nodes;

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

} // namespace sink_backward

pass::TransposeSinkingBinaryElementwiseBackward::TransposeSinkingBinaryElementwiseBackward() {
    MATCHER_SCOPE(TransposeSinkingBinaryElementwiseBackward);

    auto main_node_label = wrap_type<op::util::BinaryElementwiseArithmetic>(consumers_count(1));

    auto transpose_const_label = wrap_type<Constant>(consumers_count(1));
    auto transpose_label =
        wrap_type<Transpose>({main_node_label, transpose_const_label}, consumers_count(1));

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, transpose_const)) {
            register_new_node(new_node);
        }

        // remove transpose after main node
        transpose->output(0).replace(main_node);

        SwapNames(transpose, main_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

pass::TransposeSinkingConcatBackward::TransposeSinkingConcatBackward() {
    MATCHER_SCOPE(TransposeSinkingConcatBackward);

    auto main_node_label = wrap_type<Concat>(consumers_count(1));

    auto transpose_const_label = wrap_type<Constant>(consumers_count(1));
    auto transpose_label =
        wrap_type<Transpose>({main_node_label, transpose_const_label}, consumers_count(1));

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, transpose_const)) {
            register_new_node(new_node);
        }

        // remove transpose after main node
        transpose->output(0).replace(main_node);

        SwapNames(transpose, main_node);

        auto concat_node = as_type_ptr<Concat>(main_node);
        const auto transpose_axis_order = transpose_const->get_axis_vector_val();
        const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const int64_t transposed_concat_axis = reversed_traspose_axis_order[concat_node->get_axis()];
        concat_node->set_concatenation_axis(transposed_concat_axis);

        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
