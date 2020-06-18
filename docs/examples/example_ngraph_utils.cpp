// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

// ! [ngraph_utils:simple_function]
std::shared_ptr<ngraph::Function> create_simple_function() {
    // Basic example with explicit Result operation creation
    // Create opset3::Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
    // Create opset3::Constant operation with value
    auto divide_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
    // Create opset3::Power operation that takes two opset3::Constant operations as input
    auto pow = std::make_shared<ngraph::opset3::Power>(divide_constant,
                                                       ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1}));
    auto mul = std::make_shared<ngraph::opset3::Multiply>(data, pow);
    auto res = std::make_shared<ngraph::opset3::Result>(mul);

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
}
// ! [ngraph_utils:simple_function]

// ! [ngraph_utils:advanced_function]
std::shared_ptr<ngraph::Function> create_advanced_function() {
    // Advanced example with multioutput operation. Results operation will be created automatically.
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
    auto axis_const = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}/*scalar shape*/, {1});
    // Create opset3::Split operation that splits input to three slices across 1st dimension
    auto split = std::make_shared<ngraph::opset3::Split>(data, axis_const, 3);
    auto relu = std::make_shared<ngraph::opset3::Relu>(split->output(1)/*specify explicit output*/);
    // Results operations will be created automatically based on provided OutputVector
    return std::make_shared<ngraph::Function>(ngraph::OutputVector{split->output(0), relu, split->output(2)}, ngraph::ParameterVector{data});
}
// ! [ngraph_utils:advanced_function]

void pattern_matcher_examples() {
{
// ! [pattern:label_example]
// Detect Multiply with arbitrary first input and second as Constant
auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1});
auto value = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
auto mul = std::make_shared<ngraph::opset3::Multiply>(input, value);
auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "MultiplyMatcher");
// ! [pattern:label_example]
}

{
// ! [pattern:concat_example]
// Detect Concat operation with arbitrary number of inputs
auto concat = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset3::Concat>());
auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "ConcatMatcher");
// ! [pattern:concat_example]
}

{
// ! [pattern:predicate_example]
// Detect Multiply or Add operation
auto lin_op = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{},
              [](const std::shared_ptr<ngraph::Node> & node) -> bool {
                    return std::dynamic_pointer_cast<ngraph::opset3::Multiply>(node) ||
                    std::dynamic_pointer_cast<ngraph::opset3::Add>(node);
              });
auto m = std::make_shared<ngraph::pattern::Matcher>(lin_op, "MultiplyOrAddMatcher");
// ! [pattern:predicate_example]
}

}

void ngraph_api_examples(std::shared_ptr<Node> node) {
{
// ! [ngraph:ports_example]
// Let's supose that node is opset3::Convolution operation
// as we know opset3::Convolution has two input ports (data, weights) and one output port
Input <Node> data = node->input(0);
Input <Node> weights = node->input(1);
Output <Node> output = node->output(0);
// Getting shape and type
auto pshape = data.get_partial_shape();
auto el_type = data.get_element_type();
// Ggetting parent for input port
Output <Node> parent_output;
parent_output = data.get_source_output();
// Another short way to get partent for output port
parent_output = node->input_value(0);
// Getting all consumers for output port
auto consumers = output.get_target_inputs();
// ! [ngraph:ports_example]
}
}

// ! [ngraph:replace_node]
bool ngraph_replace_node(std::shared_ptr<Node> node) {
    auto neg = std::dynamic_pointer_cast<ngraph::opset3::Negative>(node);
    if (!neg) {
        return false;
    }

    auto mul = std::make_shared<ngraph::opset3::Multiply>(neg->input_value(0),
                                                          opset3::Constant::create(neg->get_element_type(), Shape{1}, {-1}));
    mul->set_friendly_name(neg->get_friendly_name());
    ngraph::copy_runtime_info(neg, mul);
    // Replaces Negative operation with Multiply operation
    ngraph::replace_node(neg, mul);
}
// ! [ngraph:replace_node]

// ! [ngraph:insert_node]
// Lets suppose that we have a node with single output port and we want to insert additional operation new_node after it
void insert_example(std::shared_ptr<ngraph::Node> node) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();
    // Create new node. Let it be opset1::Relu.
    auto new_node = std::make_shared<ngraph::opset3::Relu>(node);
    // Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(new_node);
    }
}
// ! [ngraph:insert_node]

// ! [ngraph:insert_node_with_copy]
void insert_example_with_copy(std::shared_ptr<ngraph::Node> node) {
    // Make a node copy
    auto node_copy = node->clone_with_new_inputs(node->input_values());
    // Create new node
    auto new_node = std::make_shared<ngraph::opset3::Relu>(node_copy);
    ngraph::replace_node(node, new_node);
}
// ! [ngraph:insert_node_with_copy]