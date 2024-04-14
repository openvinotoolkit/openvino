// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ! [ov:include]
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
// ! [ov:include]

#include <openvino/pass/visualize_tree.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/core/rt_info.hpp>


// ! [ov:create_simple_model]
std::shared_ptr<ov::Model> create_simple_model() {
    // This example shows how to create ov::Model
    //
    // Parameter--->Multiply--->Add--->Result
    //    Constant---'          /
    //              Constant---'

    // Create opset8::Parameter operation with static shape
    auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});

    auto mul_constant = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {1.5});
    auto mul = std::make_shared<ov::opset8::Multiply>(data, mul_constant);

    auto add_constant = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {0.5});
    auto add = std::make_shared<ov::opset8::Add>(mul, add_constant);

    // Create opset8::Result operation
    auto res = std::make_shared<ov::opset8::Result>(mul);

    // Create OpenVINO function
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data});
}
// ! [ov:create_simple_model]

// ! [ov:create_advanced_model]
std::shared_ptr<ov::Model> create_advanced_model() {
    // Advanced example with multi output operation
    //
    // Parameter->Split---0-->Result
    //               | `--1-->Relu-->Result
    //               `----2-->Result

    auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});

    // Create Constant for axis value
    auto axis_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{} /*scalar shape*/, {1});

    // Create opset8::Split operation that splits input to three slices across 1st dimension
    auto split = std::make_shared<ov::opset8::Split>(data, axis_const, 3);

    // Create opset8::Relu operation that takes 1st Split output as input
    auto relu = std::make_shared<ov::opset8::Relu>(split->output(1) /*specify explicit output*/);

    // Results operations will be created automatically based on provided OutputVector
    return std::make_shared<ov::Model>(ov::OutputVector{split->output(0), relu, split->output(2)},
                                       ov::ParameterVector{data});
}
// ! [ov:create_advanced_model]

void ov_api_examples() {
    std::shared_ptr<ov::Node> node = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 3, 64, 64});

    // ! [ov:partial_shape]
    ov::Shape static_shape;
    ov::PartialShape partial_shape = node->output(0).get_partial_shape(); // get zero output partial shape
    if (!partial_shape.is_dynamic() /* or partial_shape.is_static() */) {
        static_shape = partial_shape.get_shape();
    }
    // ! [ov:partial_shape]
}

// ! [ov:serialize]
void serialize_example(const std::shared_ptr<ov::Model>& model) {
    ov::serialize(model, "/path/to/file/model.xml", "/path/to/file/model.bin");
}
// ! [ov:serialize]

// ! [ov:visualize]
void visualize_example(const std::shared_ptr<ov::Model>& m) {
    // Need include:
    // * openvino/pass/manager.hpp
    // * openvino/pass/visualize_tree.hpp
    ov::pass::Manager manager;

    // Serialize ov::Model to before.svg file before transformation
    manager.register_pass<ov::pass::VisualizeTree>("image.svg");

    manager.run_passes(m);
}
// ! [ov:visualize]

void model_inputs() {
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("model.xml");
//! [all_inputs_ouputs]
/* Take information about all topology inputs */
auto inputs = model->inputs();
/* Take information about all topology outputs */
auto outputs = model->outputs();
//! [all_inputs_ouputs]
}

void pattern_matcher_examples(std::shared_ptr<ov::Node> node) {
{
// ! [pattern:simple_example]
// Pattern example
auto input = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{1});
auto shapeof = std::make_shared<ov::opset8::ShapeOf>(input);

// Create Matcher with Parameter->ShapeOf pattern
auto m = std::make_shared<ov::pass::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
// ! [pattern:simple_example]

// ! [pattern:callback_example]
ov::graph_rewrite_callback callback = [](ov::pass::pattern::Matcher& m) {
    // Get root node
    std::shared_ptr<ov::Node> root_node = m.get_match_root();

    // Get all nodes matched by pattern
    ov::NodeVector nodes = m.get_matched_nodes();

    // Transformation code
    return false;
};
// ! [pattern:callback_example]
}

{
// ! [pattern:label_example]
// Detect Multiply with arbitrary first input and second as Constant
// ov::pattern::op::Label - represent arbitrary input
auto input = ov::pass::pattern::any_input();
auto value = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {0.5});
auto mul = std::make_shared<ov::opset8::Multiply>(input, value);
auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, "MultiplyMatcher");
// ! [pattern:label_example]
}

{
// ! [pattern:concat_example]
// Detect Concat operation with arbitrary number of inputs
auto concat = ov::pass::pattern::wrap_type<ov::opset8::Concat>();
auto m = std::make_shared<ov::pass::pattern::Matcher>(concat, "ConcatMatcher");
// ! [pattern:concat_example]
}

{
// ! [pattern:predicate_example]
// Detect Multiply->Add sequence where mul has exactly one consumer
auto mul = ov::pass::pattern::wrap_type<ov::opset8::Multiply>(ov::pass::pattern::consumers_count(1)/*сheck consumers count*/);
auto add = ov::pass::pattern::wrap_type<ov::opset8::Add>({mul, ov::pass::pattern::any_input()});
auto m = std::make_shared<ov::pass::pattern::Matcher>(add, "MultiplyAddMatcher");
// Matcher can be used to match pattern manually on given node
if (m->match(node->output(0))) {
    // Successfully matched
}
// ! [pattern:predicate_example]
}

}

bool openvino_api_examples(std::shared_ptr<ov::Node> node) {
{
// ! [ov:ports_example]
// Let's suppose that node is of ov::op::v0::Convolution type.
// As we know ov::op::v0::Convolution has two input ports (data, weights) and one output port.
ov::Input<ov::Node> data = node->input(0);
ov::Input<ov::Node> weights = node->input(1);
ov::Output<ov::Node> output = node->output(0);

// Getting shape and type
auto pshape = data.get_partial_shape();
auto el_type = data.get_element_type();

// Getting parent for input port i.e. Output mapped by the input
ov::Output<ov::Node> parent_output;
parent_output = data.get_source_output();

// Another short way to get partent for output port
parent_output = node->input_value(0);

// Getting all consumers for output port
auto consumers = output.get_target_inputs();
// ! [ov:ports_example]
}

{
// ! [ngraph:shape_check]
auto partial_shape = node->input(0).get_partial_shape(); // get zero input partial shape

// Check that input shape rank is static
if (!partial_shape.rank().is_static()) {
    return false;
}
auto rank_size = partial_shape.rank().get_length();

// Check that second dimension is not dynamic
if (rank_size < 2 || partial_shape[1].is_dynamic()) {
    return false;
}
auto dim = partial_shape[1].get_length();
// ! [ngraph:shape_check]
}

return true;
}

// ! [ov:replace_node]
bool ov_replace_node(std::shared_ptr<ov::Node> node) {
    // Step 1. Verify that node is of type ov::op::v0::Negative
    auto neg = std::dynamic_pointer_cast<ov::op::v0::Negative>(node);
    if (!neg) {
        return false;
    }

    // Step 2. Create ov::op::v1::Multiply operation with the first input being the output going into Negative and second as Constant with -1 value
    auto mul = std::make_shared<ov::op::v1::Multiply>(neg->input_value(0),
                                                      ov::op::v0::Constant::create(neg->get_element_type(), ov::Shape{1}, {-1}));

    mul->set_friendly_name(neg->get_friendly_name());
    ov::copy_runtime_info(neg, mul);

    // Step 3. Replace Negative operation with Multiply operation
    ov::replace_node(neg, mul);
    return true;

    // Step 4. Negative operation will be removed automatically because all consumers were moved to Multiply operation
}
// ! [ov:replace_node]

bool ov_manual_replace_node(std::shared_ptr<ov::Node> node) {
auto neg = std::dynamic_pointer_cast<ov::op::v0::Negative>(node);
if (!neg) {
    return false;
}

auto mul = std::make_shared<ov::op::v1::Multiply>(neg->input_value(0),
                                                  ov::op::v0::Constant::create(neg->get_element_type(), ov::Shape{1}, {-1}));

mul->set_friendly_name(neg->get_friendly_name());
ov::copy_runtime_info(neg, mul);

// ! [ov:manual_replace]
// All neg->output(0) consumers will be moved to mul->output(0) port
neg->output(0).replace(mul->output(0));
// ! [ov:manual_replace]
return true;
}

// ! [ov:insert_node]
// Step 1. Lets suppose that we have a node with single output port and we want to insert additional operation new_node after it
void insert_example(std::shared_ptr<ov::Node> node) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();

    // Step 2. Create new node ov::op::v0::Relu.
    auto new_node = std::make_shared<ov::op::v0::Relu>(node);

    // Step 3. Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(new_node);
    }
}
// ! [ov:insert_node]

// ! [ov:insert_node_with_copy]
void insert_example_with_copy(std::shared_ptr<ov::Node> node) {
    // Make a node copy
    auto node_copy = node->clone_with_new_inputs(node->input_values());
    // Create new node
    auto new_node = std::make_shared<ov::op::v0::Relu>(node_copy);
    ov::replace_node(node, new_node);
}
// ! [ov:insert_node_with_copy]

void eliminate_example(std::shared_ptr<ov::Node> node) {
// ! [ov:eliminate_node]
// Suppose we have a node that we want to remove
bool success = ov::replace_output_update_name(node->output(0), node->input_value(0));
// ! [ov:eliminate_node]
}

void replace_friendly_name() {
auto div = std::make_shared<ov::op::v1::Divide>();
// ! [ov:replace_friendly_name]
// Replace Div operation with Power and Multiply sub-graph and set original friendly name to Multiply operation
auto pow = std::make_shared<ov::op::v1::Power>(div->input(1).get_source_output(),
                                               ov::op::v0::Constant::create(div->get_input_element_type(1), ov::Shape{1}, {-1}));
auto mul = std::make_shared<ov::op::v1::Multiply>(div->input(0).get_source_output(), pow);
mul->set_friendly_name(div->get_friendly_name());
ov::replace_node(div, mul);
// ! [ov:replace_friendly_name]
}

void constant_subgraph() {
// ! [ov:constant_subgraph]
// After ConstantFolding pass Power will be replaced with Constant
auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
auto pow = std::make_shared<ov::op::v1::Power>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2}),
                                               ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {3}));
auto mul = std::make_shared<ov::op::v1::Multiply>(input /* not constant input */, pow);
// ! [ov:constant_subgraph]
}

void copy_runtime_info_snippet() {
std::shared_ptr<ov::Node> transpose, reshape, div, pow, mul, conv, bias, conv_fused, a, b, c, e, f;
// ! [ov:copy_runtime_info]
// Replace Transpose with Reshape operation (1:1)
ov::copy_runtime_info(transpose, reshape);

// Replace Div operation with Power and Multiply sub-graph (1:N)
ov::copy_runtime_info(div, {pow, mul});

// Fuse Convolution with Add operation (N:1)
ov::copy_runtime_info({conv, bias}, {conv_fused});

// Any other transformation that replaces one sub-graph with another sub-graph (N:M)
ov::copy_runtime_info({a, b, c}, {e, f});
// ! [ov:copy_runtime_info]
}
