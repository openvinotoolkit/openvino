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

#include "transformations/common_optimizations/transpose_sinking_unary.hpp"

#include <utility>

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

NodePair SwapNodes(NodePtr first_node, NodePtr second_node) {
    auto second_node_inputs = second_node->input_values();
    second_node_inputs[0] = first_node->input_value(0);
    auto new_first_node = second_node->clone_with_new_inputs(second_node_inputs);
    
    auto first_node_inputs = first_node->input_values();
    first_node_inputs[0] = new_first_node;
    auto new_second_node = first_node->clone_with_new_inputs(first_node_inputs);

    new_second_node->set_friendly_name(second_node->get_friendly_name());
    ov::copy_runtime_info({first_node, second_node}, {new_first_node, new_second_node});

    ov::replace_node(second_node, new_second_node);

    return std::make_pair(new_first_node, new_second_node);
}

} // namespace

ov::pass::TransposeSinkingUnaryForward::TransposeSinkingUnaryForward() {
    MATCHER_SCOPE(TransposeSinkingUnaryForward);

    auto transpose_label =
        ov::pass::pattern::wrap_type<ov::opset9::Transpose>({ov::pass::pattern::any_input(),
                                                             ov::pass::pattern::wrap_type<ov::opset9::Constant>()},
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

        auto new_nodes = SwapNodes(transpose, unary);
    
        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        unary_label,
        "ov::pass::TransposeSinkingUnaryForward" /* matcher_name */);
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
                                                             ov::pass::pattern::wrap_type<ov::opset9::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();

        auto new_nodes = SwapNodes(unary, transpose);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        transpose_label,
        "ov::pass::TransposeSinkingUnaryBackward" /* matcher_name */);
    register_matcher(m, matcher_pass_callback);
}
