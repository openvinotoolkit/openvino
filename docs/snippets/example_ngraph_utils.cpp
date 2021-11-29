// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/convert_space_to_depth.hpp>
#include <transformations/op_conversions/convert_depth_to_space.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>

// ! [ngraph:include]
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
// ! [ngraph:include]

using namespace ngraph;

// ! [ngraph_utils:simple_function]
std::shared_ptr<ngraph::Function> create_simple_function() {
    // This example shows how to create ngraph::Function
    //
    // Parameter--->Multiply--->Add--->Result
    //    Constant---'          /
    //              Constant---'

    // Create opset3::Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});

    auto mul_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
    auto mul = std::make_shared<ngraph::opset3::Multiply>(data, mul_constant);

    auto add_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
    auto add = std::make_shared<ngraph::opset3::Add>(mul, add_constant);

    // Create opset3::Result operation
    auto res = std::make_shared<ngraph::opset3::Result>(mul);

    // Create nGraph function
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
}
// ! [ngraph_utils:simple_function]

// ! [ngraph_utils:advanced_function]
std::shared_ptr<ngraph::Function> create_advanced_function() {
    // Advanced example with multi output operation
    //
    // Parameter->Split---0-->Result
    //               | `--1-->Relu-->Result
    //               `----2-->Result

    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});

    // Create Constant for axis value
    auto axis_const = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}/*scalar shape*/, {1});

    // Create opset3::Split operation that splits input to three slices across 1st dimension
    auto split = std::make_shared<ngraph::opset3::Split>(data, axis_const, 3);

    // Create opset3::Relu operation that takes 1st Split output as input
    auto relu = std::make_shared<ngraph::opset3::Relu>(split->output(1)/*specify explicit output*/);

    // Results operations will be created automatically based on provided OutputVector
    return std::make_shared<ngraph::Function>(ngraph::OutputVector{split->output(0), relu, split->output(2)}, ngraph::ParameterVector{data});
}
// ! [ngraph_utils:advanced_function]

void pattern_matcher_examples(std::shared_ptr<Node> node) {
{
// ! [pattern:simple_example]
// Pattern example
auto input = std::make_shared<ngraph::opset3::Parameter>(element::i64, Shape{1});
auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input);

// Create Matcher with Parameter->ShapeOf pattern
auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
// ! [pattern:simple_example]

// ! [pattern:callback_example]
ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
    // Get root node
    std::shared_ptr<Node> root_node = m.get_match_root();

    // Get all nodes matched by pattern
    NodeVector nodes = m.get_matched_nodes();

    // Transformation code
    return false;
};
// ! [pattern:callback_example]
}

{
// ! [pattern:label_example]
// Detect Multiply with arbitrary first input and second as Constant
// ngraph::pattern::op::Label - represent arbitrary input
auto input = ngraph::pattern::any_input();
auto value = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
auto mul = std::make_shared<ngraph::opset3::Multiply>(input, value);
auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "MultiplyMatcher");
// ! [pattern:label_example]
}

{
// ! [pattern:concat_example]
// Detect Concat operation with arbitrary number of inputs
auto concat = ngraph::pattern::wrap_type<ngraph::opset3::Concat>();
auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "ConcatMatcher");
// ! [pattern:concat_example]
}

{
// ! [pattern:predicate_example]
// Detect Multiply->Add sequence where mul has exactly one consumer
auto mul = ngraph::pattern::wrap_type<ngraph::opset3::Multiply>(ngraph::pattern::consumers_count(1)/*сheck consumers count*/);
auto add = ngraph::pattern::wrap_type<ngraph::opset3::Add>({mul, ngraph::pattern::any_input()});
auto m = std::make_shared<ngraph::pattern::Matcher>(add, "MultiplyAddMatcher");
// Matcher can be used to match pattern manually on given node
if (m->match(node->output(0))) {
    // Successfully matched
}
// ! [pattern:predicate_example]
}

}

bool ngraph_api_examples(std::shared_ptr<Node> node) {
{
// ! [ngraph:ports_example]
// Let's suppose that node is opset3::Convolution operation
// as we know opset3::Convolution has two input ports (data, weights) and one output port
Input <Node> data = node->input(0);
Input <Node> weights = node->input(1);
Output <Node> output = node->output(0);

// Getting shape and type
auto pshape = data.get_partial_shape();
auto el_type = data.get_element_type();

// Getting parent for input port
Output <Node> parent_output;
parent_output = data.get_source_output();

// Another short way to get partent for output port
parent_output = node->input_value(0);

// Getting all consumers for output port
auto consumers = output.get_target_inputs();
// ! [ngraph:ports_example]
}

{
// ! [ngraph:shape]
auto partial_shape = node->input(0).get_partial_shape(); // get zero input partial shape
if (partial_shape.is_dynamic() /* or !partial_shape.is_static() */) {
    return false;
}
auto static_shape = partial_shape.get_shape();
// ! [ngraph:shape]
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

// ! [ngraph:replace_node]
bool ngraph_replace_node(std::shared_ptr<Node> node) {
    // Step 1. Verify that node has opset3::Negative type
    auto neg = std::dynamic_pointer_cast<ngraph::opset3::Negative>(node);
    if (!neg) {
        return false;
    }

    // Step 2. Create opset3::Multiply operation where the first input is negative operation input and second as Constant with -1 value
    auto mul = std::make_shared<ngraph::opset3::Multiply>(neg->input_value(0),
                                                          opset3::Constant::create(neg->get_element_type(), Shape{1}, {-1}));

    mul->set_friendly_name(neg->get_friendly_name());
    ngraph::copy_runtime_info(neg, mul);

    // Step 3. Replace Negative operation with Multiply operation
    ngraph::replace_node(neg, mul);
    return true;

    // Step 4. Negative operation will be removed automatically because all consumers was moved to Multiply operation
}
// ! [ngraph:replace_node]

// ! [ngraph:insert_node]
// Step 1. Lets suppose that we have a node with single output port and we want to insert additional operation new_node after it
void insert_example(std::shared_ptr<ngraph::Node> node) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();

    // Step 2. Create new node. Let it be opset1::Relu.
    auto new_node = std::make_shared<ngraph::opset3::Relu>(node);

    // Step 3. Reconnect all consumers to new_node
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

void eliminate_example(std::shared_ptr<ngraph::Node> node) {
// ! [ngraph:eliminate_node]
// Suppose we have a node that we want to remove
bool success = replace_output_update_name(node->output(0), node->input_value(0));
// ! [ngraph:eliminate_node]
}

// ! [ngraph:visualize]
void visualization_example(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;

    // Serialize ngraph::Function to before.svg file before transformation
    manager.register_pass<ngraph::pass::VisualizeTree>("/path/to/file/before.svg");

    // Run your transformation
    // manager.register_pass<ngraph::pass::MyTransformation>();

    // Serialize ngraph::Function to after.svg file after transformation
    manager.register_pass<ngraph::pass::VisualizeTree>("/path/to/file/after.svg");

    manager.run_passes(f);
}
// ! [ngraph:visualize]

void pass_manager_example1(std::shared_ptr<ngraph::Function> f) {
// ! [ngraph:disable_gelu]
ngraph::pass::Manager manager;
manager.register_pass<ngraph::pass::CommonOptimizations>();

auto pass_config = manager.get_pass_config();
pass_config->disable<ngraph::pass::ConvertGELU>();

manager.run_passes(f);
// ! [ngraph:disable_gelu]
}

void pass_manager_example2(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;
    std::function<bool(const std::shared_ptr<const Node>)> transformation_callback;
// ! [ngraph:disable_callback]
// Set callback to particular transformation with specific condition
auto pass_config = manager.get_pass_config();
pass_config->set_callback<ngraph::pass::ConvertSpaceToDepth,
                          ngraph::pass::ConvertDepthToSpace>(
        [](const std::shared_ptr<const Node> &node) -> bool {
            return node->input_value(0).get_shape().size() <= 5lu &&
                   node->input_value(0).get_shape().size() == node->get_output_shape(0).size();
        });

// Update transformation to call callback
ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
    auto node = m.get_match_root();
    if (transformation_callback(node)) {
        return false;
    }
    // transformation code
    return false;
};
// ! [ngraph:disable_callback]
}

void pass_manager_example3(std::shared_ptr<ngraph::Function> f) {
    std::function<bool(const std::shared_ptr<const Node>)> transformation_callback;
// ! [ngraph:disabled_by_default]
// Example of disabled by default transformation
{
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::ConvertPadToGroupConvolution, false>();
    manager.run_passes(f);
}

// Enable disabled by default transformation inside plugin
{
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    auto pass_config = manager.get_pass_config();
    pass_config->enable<ngraph::pass::ConvertPadToGroupConvolution>();
    manager.run_passes(f);
}
// ! [ngraph:disabled_by_default]
}
