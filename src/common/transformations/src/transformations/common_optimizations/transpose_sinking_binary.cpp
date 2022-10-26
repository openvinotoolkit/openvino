#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "itt.hpp"
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>

#include "transformations/common_optimizations/transpose_sinking_binary.hpp"

#include <utility>

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using Nodes = std::vector<NodePtr>;

ov::OutputVector GetOutputs(const Nodes& nodes) {
    ov::OutputVector outputs;

    for (auto& node : nodes)
        for (auto& output : node->outputs())
            outputs.push_back(output);

    return outputs;
}

// --------------------------------------------------------------------------------------

struct TrasposeInputsInfo {
    std::shared_ptr<ov::opset9::Transpose> transpose;
    std::shared_ptr<ov::opset9::Constant> transpose_const;
    size_t input_idx;
};

TrasposeInputsInfo GetFirstTransposeInput(NodePtr node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto transpose_node = ov::as_type_ptr<ov::opset9::Transpose>(input_node);
        if (!transpose_node)
            continue;
        auto constant_node = ov::as_type_ptr<ov::opset9::Constant>(transpose_node->input_value(1).get_node_shared_ptr());
        if (!constant_node)
            continue;
        {
            TrasposeInputsInfo input_info;
            input_info.transpose = transpose_node;
            input_info.transpose_const = constant_node;
            input_info.input_idx = input_idx;
            return input_info;
        }
    }

    return TrasposeInputsInfo();
}

bool IfNodeHasTransposeInputs(const ov::Output<ov::Node>& output) {
    for (size_t input_idx = 0; input_idx < output.get_node()->get_input_size(); ++input_idx) {
        NodePtr input_node = output.get_node()->get_input_node_shared_ptr(input_idx);
        auto transpose_node = ov::as_type_ptr<ov::opset9::Transpose>(input_node);
        if (!transpose_node)
            continue;
        auto constant_node = ov::as_type_ptr<ov::opset9::Constant>(transpose_node->input_value(1).get_node_shared_ptr());
        if (!constant_node)
            continue;
        return true;
    }

    return false;
}

// --------------------------------------------------------------------------------------

ov::AxisVector ReverseTransposeOrder(const ov::AxisVector& axis_order) {
    ov::AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++) {
        out.at(axis_order[i]) = i;
    }
    return out;
}

}

void SwapOutputNames(ov::Output<ov::Node> output1, ov::Output<ov::Node> output2) {
    const auto node2_output_names = output2.get_names();
    output2.set_names(output1.get_names());
    output1.set_names(node2_output_names);
}

void SwapNames(NodePtr node1, NodePtr node2) {
    const std::string node2_name = node2->get_friendly_name();
    node2->set_friendly_name(node1->get_friendly_name());
    node1->set_friendly_name(node2_name);

    SwapOutputNames(node1->output(0), node2->output(0));
}

// --------------------------------------------------------------------------------------

ngraph::pass::TransposeSinkingBinaryForward::TransposeSinkingBinaryForward() {
    MATCHER_SCOPE(TransposeSinkingBinaryForward);

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({ov::pass::pattern::any_input(), ngraph::pattern::wrap_type<ov::opset9::Constant>()},
                                                             ov::pass::pattern::consumers_count(1));
    auto binary_label_left = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({transpose_label,
                                                                                                      ov::pass::pattern::any_input()},
                                                                                                      ov::pass::pattern::consumers_count(1));

    auto binary_label_right = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({ov::pass::pattern::any_input(),
                                                                                                       transpose_label},
                                                                                                       ov::pass::pattern::consumers_count(1));
    auto binary_label = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{binary_label_left, binary_label_right});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto binary = m.get_match_root();
        TrasposeInputsInfo transpose_input_info = GetFirstTransposeInput(binary);

        const ov::AxisVector transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const size_t tranpose_input_index = transpose_input_info.input_idx;
        const ov::element::Type transpose_element_type = transpose_input_info.transpose_const->get_element_type();

        for (size_t i = 0; i < binary->get_input_size(); ++i) {
            auto input_node = binary->input_value(i);
            if (i == tranpose_input_index) {
                auto transpose_parent = input_node.get_node()->input_value(0);
                binary->input(i).replace_source_output(transpose_parent);
            } else {
                auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                                  ov::Shape{reversed_traspose_axis_order.size()},
                                                                                  reversed_traspose_axis_order);
                auto new_transpose = std::make_shared<ov::opset9::Transpose>(input_node, new_transpose_const);

                binary->input(i).replace_source_output(new_transpose->output(0));

                ov::copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});
            }
        }

        auto binary_consumers = binary->output(0).get_target_inputs();

        auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                          ov::Shape{transpose_axis_order.size()},
                                                                          transpose_axis_order);
        auto new_transpose = std::make_shared<ov::opset9::Transpose>(binary, new_transpose_const);

        for (auto& consumer: binary_consumers) {
            consumer.replace_source_output(new_transpose);
        }

        ov::copy_runtime_info(binary, {new_transpose, new_transpose_const});

        SwapNames(new_transpose, binary);

        register_new_node(new_transpose);

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(binary_label, matcher_name);
    register_matcher(matcher, matcher_pass_callback);
}

ngraph::pass::TransposeSinkingBinaryBackward::TransposeSinkingBinaryBackward() {
    MATCHER_SCOPE(TransposeSinkingBinaryBackward);

    auto binary_label = ov::pass::pattern::wrap_type<ov::op::util::BinaryElementwiseArithmetic>({ov::pass::pattern::any_input(),
                                                                                                 ov::pass::pattern::any_input()},
                                                                                                 ov::pass::pattern::consumers_count(1));

    auto transpose_const_label = ov::pass::pattern::wrap_type<ov::opset9::Constant>(ov::pass::pattern::consumers_count(1));
    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({binary_label, transpose_const_label},
                                                            ov::pass::pattern::consumers_count(1));

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = ov::as_type_ptr<ov::opset9::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto binary = pattern_to_output.at(binary_label).get_node_shared_ptr();

        const ov::AxisVector transpose_axis_order = transpose_const->get_axis_vector_val();
        const ov::element::Type transpose_element_type = transpose_const->get_element_type();

        for (size_t i = 0; i < binary->get_input_size(); ++i) {
            auto input_node = binary->input_value(i);
            auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                              ov::Shape{transpose_axis_order.size()},
                                                                              transpose_axis_order);
            auto new_transpose = std::make_shared<ov::opset9::Transpose>(input_node, new_transpose_const);

            binary->input(i).replace_source_output(new_transpose->output(0));

            ov::copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

            register_new_node(new_transpose);
        }

        auto transpose_consumers = transpose->output(0).get_target_inputs();
        for (auto& consumer: transpose_consumers) {
            consumer.replace_source_output(binary);
        }

        SwapNames(transpose, binary);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

// --------------------------------------------------------------------------------------

namespace {

// get new axis for Concat/Split according to transpose order
template <typename T>
T TransposeAxis(T axis, const ov::AxisVector& transpose_order) {
    return transpose_order[axis];
}

} // namespace

ngraph::pass::TransposeSinkingConcatForward::TransposeSinkingConcatForward() {
    MATCHER_SCOPE(TransposeSinkingConcatForward);

    auto concat_label = ov::pass::pattern::wrap_type<ov::opset9::Concat>(IfNodeHasTransposeInputs);

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto& concat_output = pattern_to_output.at(concat_label);
        auto concat = ov::as_type_ptr<ov::opset9::Concat>(concat_output.get_node_shared_ptr());

        TrasposeInputsInfo transpose_input_info = GetFirstTransposeInput(concat);

        const ov::AxisVector transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const size_t tranpose_input_index = transpose_input_info.input_idx;
        const ov::element::Type transpose_element_type = transpose_input_info.transpose_const->get_element_type();
        const int64_t transposed_concat_axis = TransposeAxis(concat->get_axis(), transpose_axis_order);

        for (size_t i = 0; i < concat->get_input_size(); ++i) {
            auto input_node = concat->input_value(i);
            if (i == tranpose_input_index) {
                auto transpose_parent = input_node.get_node()->input_value(0);
                concat->input(i).replace_source_output(transpose_parent);
            }
            else {
                auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                                  ov::Shape{reversed_traspose_axis_order.size()},
                                                                                  reversed_traspose_axis_order);
                auto new_transpose = std::make_shared<ov::opset9::Transpose>(input_node, new_transpose_const);

                concat->input(i).replace_source_output(new_transpose->output(0));

                ov::copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});
            }
        }

        auto binary_consumers = concat->output(0).get_target_inputs();

        auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                          ov::Shape{transpose_axis_order.size()},
                                                                          transpose_axis_order);
        auto new_transpose = std::make_shared<ov::opset9::Transpose>(concat, new_transpose_const);

        for (auto& consumer: binary_consumers) {
            consumer.replace_source_output(new_transpose);
        }

        ov::copy_runtime_info(concat, {new_transpose, new_transpose_const});

        SwapNames(new_transpose, concat);

        register_new_node(new_transpose);

        //
        concat->set_concatenation_axis(transposed_concat_axis);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::TransposeSinkingConcatBackward::TransposeSinkingConcatBackward() {
    MATCHER_SCOPE(TransposeSinkingConcatBackward);

    auto concat_label = ov::pass::pattern::wrap_type<ov::opset9::Concat>(ov::pass::pattern::consumers_count(1));

    auto transpose_const_label = ov::pass::pattern::wrap_type<ov::opset9::Constant>(ov::pass::pattern::consumers_count(1));
    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({concat_label, transpose_const_label},
                                                            ov::pass::pattern::consumers_count(1));

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = ov::as_type_ptr<ov::opset9::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto concat = ov::as_type_ptr<ov::opset9::Concat>(pattern_to_output.at(concat_label).get_node_shared_ptr());

        const ov::AxisVector transpose_axis_order = transpose_const->get_axis_vector_val();
        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        const ov::element::Type transpose_element_type = transpose_const->get_element_type();
        const int64_t transposed_concat_axis = TransposeAxis(concat->get_axis(), reversed_traspose_axis_order);

        for (size_t i = 0; i < concat->get_input_size(); ++i) {
            auto input_node = concat->input_value(i);
            auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                              ov::Shape{transpose_axis_order.size()},
                                                                              transpose_axis_order);
            auto new_transpose = std::make_shared<ov::opset9::Transpose>(input_node, new_transpose_const);

            concat->input(i).replace_source_output(new_transpose->output(0));

            ov::copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

            register_new_node(new_transpose);
        }

        auto transpose_consumers = transpose->output(0).get_target_inputs();
        for (auto& consumer: transpose_consumers) {
            consumer.replace_source_output(concat);
        }

        SwapNames(transpose, concat);

        //
        concat->set_concatenation_axis(transposed_concat_axis);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

// --------------------------------------------------------------------------------------

ngraph::pass::TransposeSinkingSplitForward::TransposeSinkingSplitForward() {
    MATCHER_SCOPE(TransposeSinkingSplitForward);

    auto transpose_const_label = ov::pass::pattern::wrap_type<ov::opset9::Constant>(ov::pass::pattern::consumers_count(1));
    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({ov::pass::pattern::any_input(),
                                                             transpose_const_label},
                                                             ov::pass::pattern::consumers_count(1));

    auto split_label = ov::pass::pattern::wrap_type<ov::opset9::Split>({transpose_label,
                                                                        ov::pass::pattern::any_input()},
                                                                        ov::pass::pattern::consumers_count(1));

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = ov::as_type_ptr<ov::opset9::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto split = ov::as_type_ptr<ov::opset9::Split>(pattern_to_output.at(split_label).get_node_shared_ptr());
        auto split_axis_constant = ov::as_type_ptr<ov::opset9::Constant>(split->input_value(1).get_node_shared_ptr());

        const ov::AxisVector transpose_axis_order = transpose_const->get_axis_vector_val();
        const ov::element::Type transpose_element_type = transpose_const->get_element_type();

        const size_t split_axis = split_axis_constant->get_axis_vector_val()[0];
        const size_t transposed_split_axis = TransposeAxis(split_axis, transpose_axis_order);

        // remove input transpose
        auto transpose_parent = split->input_value(0).get_node()->input_value(0);
        split->input(0).replace_source_output(transpose_parent);

        // insert output transposes
        for (size_t i = 0; i < split->get_output_size(); ++i) {
            auto split_consumers = split->output(i).get_target_inputs();

            auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                              ov::Shape{transpose_axis_order.size()},
                                                                              transpose_axis_order);
            auto new_transpose = std::make_shared<ov::opset9::Transpose>(split->output(i), new_transpose_const);


            for (auto& consumer: split_consumers) {
                consumer.replace_source_output(new_transpose);
            }

            ov::copy_runtime_info(split, {new_transpose, new_transpose_const});
            SwapOutputNames(split->output(i), new_transpose->output(0));

            new_transpose->set_friendly_name(split->get_friendly_name() + "." + std::to_string(i));

            register_new_node(new_transpose);
        }

        // update split axis
        auto new_split_axis_const = std::make_shared<ov::opset9::Constant>(split_axis_constant->get_element_type(), ov::Shape{}, transposed_split_axis);
        split->input(1).replace_source_output(new_split_axis_const);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(split_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

struct OutputTranspose {
    ov::opset9::Transpose * transpose;
    ov::opset9::Constant * transpose_const;
};

OutputTranspose GetOutputTransposes(NodePtr node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            auto transpose_node = dynamic_cast<ov::opset9::Transpose*>(input.get_node());
            if (!transpose_node)
                continue;
            auto constant_node = dynamic_cast<ov::opset9::Constant*>(transpose_node->input_value(1).get_node());
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

NodePtr FindSplitInput(ov::Node * node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto split_node = ov::as_type_ptr<ov::opset9::Split>(input_node);
            if (!split_node)
                continue;
    }
    return {};
}

std::shared_ptr<ov::opset9::Constant> GetTransposeConstant(ov::Input<ov::Node> input) {
    auto transpose_node = dynamic_cast<ov::opset9::Transpose*>(input.get_node());
    if (!transpose_node)
        return {};
    
    auto constant_node = ov::as_type_ptr<ov::opset9::Constant>(transpose_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return {};

    return constant_node;
}

bool HasInputSplitAndTransposeSiblings(const ov::Output<ov::Node>& output) {
    NodePtr split_node = FindSplitInput(output.get_node());
    if (!split_node)
        return false;

    ov::AxisVector first_transpose_axis_order;
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

            ov::AxisVector transpose_axis_order = constant_node->get_axis_vector_val();
            if (transpose_axis_order.size() != first_transpose_axis_order.size())
                return false;
            if (!std::equal(transpose_axis_order.begin(), transpose_axis_order.end(), first_transpose_axis_order.begin()))
                return false;
        }
    }

    return true;
}

#define EMUTEX_DEBUG_CHECKPOINT std::cout << "[EMUTEX DEBUG] CHECKPOINT " << __FILE__ << ":" << __LINE__ << std::endl;

ngraph::pass::TransposeSinkingSplitBackward::TransposeSinkingSplitBackward() {
    MATCHER_SCOPE(TransposeSinkingSplitBackward);

    auto transpose_const_label = ov::pass::pattern::wrap_type<ov::opset9::Constant>(ov::pass::pattern::consumers_count(1));
    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({ov::pass::pattern::any_input(),
                                                             transpose_const_label},
                                                             HasInputSplitAndTransposeSiblings);

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        EMUTEX_DEBUG_CHECKPOINT;

        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_label_node = pattern_to_output.at(transpose_label).get_node();
        
        NodePtr split = FindSplitInput(transpose_label_node);
        auto split_axis_constant = ov::as_type_ptr<ov::opset9::Constant>(split->input_value(1).get_node_shared_ptr());
        OutputTranspose output_transpose = GetOutputTransposes(split);

        const ov::AxisVector transpose_axis_order = output_transpose.transpose_const->get_axis_vector_val();
        const ov::element::Type transpose_element_type = output_transpose.transpose_const->get_element_type();

        const ov::AxisVector reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);

        const size_t split_axis = split_axis_constant->get_axis_vector_val()[0];
        const size_t reversed_transposed_split_axis = TransposeAxis(split_axis, reversed_traspose_axis_order);

        // insert transpose before split
        {
            auto input_node = split->input_value(0);
            auto new_transpose_const = std::make_shared<ov::opset9::Constant>(transpose_element_type,
                                                                              ov::Shape{transpose_axis_order.size()},
                                                                              transpose_axis_order);
            auto new_transpose = std::make_shared<ov::opset9::Transpose>(input_node, new_transpose_const);

            split->input(0).replace_source_output(new_transpose->output(0));

            ov::copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

            register_new_node(new_transpose);
        }

        // update split axis
        auto new_split_axis_const = std::make_shared<ov::opset9::Constant>(split_axis_constant->get_element_type(),
                                                                           ov::Shape{},
                                                                           reversed_transposed_split_axis);
        split->input(1).replace_source_output(new_split_axis_const);

        // remote split output transposes
        for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
            for (auto& input : split->get_output_target_inputs(output_idx)) {
                auto transpose_consumers = input.get_node()->output(0).get_target_inputs();
                for (auto& consumer: transpose_consumers) {
                    consumer.replace_source_output(split->output(output_idx));
                }
            }
        }
        EMUTEX_DEBUG_CHECKPOINT;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
