#include "openvino/opsets/opset9.hpp"
#include "itt.hpp"
#include <transformations/utils/utils.hpp>

#include "transformations/common_optimizations/transpose_sinking_unary.hpp"

#include <utility>

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

void SwapNames(NodePtr first_node, NodePtr second_node) {
    const std::string first_name = first_node->get_friendly_name();
    first_node->set_friendly_name(second_node->get_friendly_name());
    second_node->set_friendly_name(first_name);
}

void SwapNodes(NodePtr first_node, NodePtr second_node) {
    const auto first_node_output_names = first_node->output(0).get_names();
    const auto second_node_output_names = second_node->output(0).get_names();

    auto out_1 = first_node->input_value(0);
    second_node->input(0).replace_source_output(out_1);
    
    auto out_2 = second_node->output(0);
    second_node->output(0).replace(first_node->output(0));

    first_node->input(0).replace_source_output(out_2);

    SwapNames(first_node, second_node);

    first_node->output(0).set_names(second_node_output_names);
    second_node->output(0).set_names(first_node_output_names);
}

} // namespace

ov::pass::TransposeSinkingUnaryForward::TransposeSinkingUnaryForward() {
    MATCHER_SCOPE(TransposeSinkingUnaryForward);

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({ov::pass::pattern::any_input(),
                                                             ov::pass::pattern::any_input()},
                                                             ov::pass::pattern::consumers_count(1));
    auto unary_label = ov::pass::pattern::wrap_type<ov::op::util::UnaryElementwiseArithmetic,
                                 ov::opset9::Clamp,
                                 ov::opset9::Elu,
                                 ov::opset9::SoftPlus,
                                 ov::opset9::LogicalNot,
                                 ov::opset9::Convert>({transpose_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();

        SwapNodes(transpose, unary);

        register_new_node(unary);
        register_new_node(transpose);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        unary_label,
        "ov::pass::TransposeSinkingUnaryForward");
    register_matcher(m, matcher_pass_callback);
}

ov::pass::TransposeSinkingUnaryBackward::TransposeSinkingUnaryBackward() {
    MATCHER_SCOPE(TransposeSinkingUnaryBackward);

    auto unary_label = ov::pass::pattern::wrap_type<ov::op::util::UnaryElementwiseArithmetic,
                                 ov::opset9::Clamp,
                                 ov::opset9::Elu,
                                 ov::opset9::SoftPlus,
                                 ov::opset9::LogicalNot,
                                 ov::opset9::Convert>({ov::pass::pattern::any_input()},
                                 ov::pass::pattern::consumers_count(1));

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({unary_label,
                                                             ov::pass::pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();

        SwapNodes(unary, transpose);

        register_new_node(transpose);
        register_new_node(unary);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        transpose_label,
        "ov::pass::TransposeSinkingUnaryBackward");
    register_matcher(m, matcher_pass_callback);
}
