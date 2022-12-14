#include "transformations/common_optimizations/transpose_sinking_split.hpp"

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
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov::pass::pattern;
using namespace ov;
using namespace ov::opset9;
using namespace transpose_sinking;

namespace {

using NodePtr = std::shared_ptr<Node>;

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

    if (!is_sinking_node(input.get_node()))
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
}  // namespace

ov::pass::TransposeSinkingSplitBackward::TransposeSinkingSplitBackward() {
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

ov::pass::TransposeSinkingSplitForward::TransposeSinkingSplitForward() {
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
            transpose_sinking::UpdateForwardSinkingAbility(new_node);
        }

        const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        auto split_node = as_type_ptr<Split>(main_node);
        auto split_axis_constant = as_type_ptr<Constant>(split_node->input_value(1).get_node_shared_ptr());
        const size_t split_axis = split_axis_constant->get_axis_vector_val()[0];
        const size_t transposed_split_axis = transpose_axis_order[split_axis];
        auto new_split_axis_const =
            std::make_shared<Constant>(split_axis_constant->get_element_type(), Shape{}, transposed_split_axis);
        split_node->input(1).replace_source_output(new_split_axis_const);

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
