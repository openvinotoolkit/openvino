# Writing ngraph transformations {#new_ngraph_transformation}

This guide contains all necessary information that helps you to start writing nGraph transformations.

First of all before writing transformation make sure that there is no the same existing transformation in [transformation library].
To start writing transformation it's good to know how [transformation library] is structured, how transformations are organized and where you should store your transformation code.

Lets start from reviewing transformations library structure.
Transformations library is independent from InferenceEngine target library and locates in `inference-engine/src/transformations` directory.
On the top it contains two folders:
1. ngraph_ops - legacy opset operations needed for nGraph to CNNNetwork conversion.
2. transformations - includes all transformations, utils, runtime info attributes and pass managers for conversion between opsets.

Transformation flow in transformation library has several layers:
1. Pass managers - executes list of transformations using `*_tbl.hpp` file. For example conversion form OpSetX to OpSetY.
2. Transformations - performs particular transformation algorithm on `ngraph::Funcion` (find more about transformations in [Transformations types]).
3. Low level functions that takes set of nodes and performs some transformation action. They are not mandatory and all transformation code can be located inside transformation but if some transformation parts can potentially be resued in other transformations we suggest to keep them as a separate functions.

To decide where to store your transformation code please follow this rules:
1. If it's plugin specific transformation and can't be reused by other plugins keep source code inside plugin.
2. If this transformation relates to OpSetXToOpSetY conversion or it's common optimization then keep sources inside transformation library.

After you decided where to store your transformation code you can start develop your own nGraph transformation.

## Table of Contents:

1. `ngraph::Function` and graph representation
2. Transformations types
3. Pattern matching
4. Working with ngraph::Function
5. How to debug transformations
6. Using pass manager
7. Transformation writing essentials
8. Disabling/Enabling specific transformations for plugin X
9. Custom attributes in nodes
10. Common mistakes in transformations
11. Transformations testing

## ngraph::Function and graph representation

nGraph function is a very simple thing: it stores shared pointers to [Result] and [Parameter] operations that are inputs and outputs of the graph. 
All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore.

Below you can find examples how `ngraph::Function` can be created:

~~~~~~~~~~~~~{.cpp}
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

auto f = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
~~~~~~~~~~~~~

~~~~~~~~~~~~~{.cpp}
// Advanced example with multioutput operation. Results operation will be created automatically.
auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});
auto axis_const = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}/*scalar shape*/, {1});
// Create opset3::Split operation that splits input to three slices across 1st dimension
auto split = std::make_shared<ngraph::opset3::Split>(data, axis_const, 3);
auto relu = std::make_shared<ngraph::opset3::Relu>(split->output(1)/*specify explicit output*/);
// Results operations will be created automatically based on provided OutputVector
auto f = std::make_shared<ngraph::Function>(ngraph::OutputVector{split->output(0), relu, split->output(2)}, ngraph::ParameterVector{data});
~~~~~~~~~~~~~

## Transformations types

There are two main transformation types:

`1.` ngraph::pass::FunctionalPass is used for transformations that take entire `ngraph::Function` as input and process it.

~~~~~~~~~~~~~{.cpp}
// my_transformation.hpp
// Template for FunctionPass transformation class
class MyFunctionTransformation: public ngraph::pass::FunctionPass {
public:
    MyFunctionTransformation() : FunctionPass() {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

// my_transformation.cpp
#include "my_transformation.hpp"

bool ngraph::pass::MyFunctionTransformation::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Transformation code
    return false;
}
~~~~~~~~~~~~~

Using ngraph::FunctionPass you need to override `run_on_function` method where you will write transformation code. Return value must be `true` if original function was changes during transformation otherwise it must be `false`. For transformation API please follow [Working with ngraph::Function] section.

`2.` `ngraph::pass::GraphRewrite` is used for pattern based transformations.

~~~~~~~~~~~~~{.cpp}
// my_transformation.hpp
// Template for GraphRewrite transformation class
class MyPatternBasedTransformation: public ngraph::pass::GraphRewrite {
public:
    MyPatternBasedTransformation() : GraphRewrite() {
        transform();
    }

private:
    void transform();
};

// my_transformation.cpp
#include "my_transformation.hpp"

void ngraph::pass::MyPatternBasedTransformation::transform() {
    // Pattern example
    auto input = std::make_shared<pattern::opset3::Parameter>(element::i64, Shape{1});
    auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        // Transformation code
        return false;
    };

    // Register Pattern and Matcher
    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
~~~~~~~~~~~~~

Using `ngraph::GraphRewrite` you need to complete three steps:
1. Create pattern using nGraph operations.
2. Implement callback. 
3. Register pattern and Matcher.

Pattern is a single root ngraph::Function. But the only difference is that you don't need to create function object, you just create and connect nGraph operations then take the last one and put it as a root of the pattern.

~~~~~~~~~~~~~{.cpp}
// Pattern example
auto input = std::make_shared<pattern::opset3::Parameter>(element::i64, Shape{1});
auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input);

// Create Matcher with Parameter->ShapeOf pattern
auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
~~~~~~~~~~~~~

You may notice that `Parameter` operation in example has type and shape specified. These attributes are needed only to create Parameter operation class and not used in pattern matching. 
But what if we want to match pattern where `ShapeOf` takes any operation as input? To find an answer for this question please follow [Pattern matching] section.

What is callback? Callback is an action applied to every pattern entrance. In general callback is lambda function that takes Matcher object with detected sub-graph.

~~~~~~~~~~~~~{.cpp}
ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
    // Get root node
    std::shared_ptr<Node> root_node = m.get_match_root();
    
    // Get all nodes mathched by pattern
    NodeVector nodes = m.get_matched_nodes();
    
    // Transformation code
    return false;
};
~~~~~~~~~~~~~

Example above shows callback structure and how Matcher can be used for accessing nodes detected by pattern.
Callback return value must be `true` if something has happened to nodes (replacing/reconnection) otherwise it must be `false`.

And the last step is to register Matcher and callback inside GraphRewrite pass. And to do this you need to call `add_matcher` method. 

~~~~~~~~~~~~~{.cpp}
// Register matcher and callback
this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
~~~~~~~~~~~~~

Also you can have multiple matchers and callbacks and they can be registered in single Graphrewrite pass. In this case all registered patterns will be applied in a singe graph traversal. 

~~~~~~~~~~~~~{.cpp}
// Multiple matchers example
this->add_matcher(m1, callback1, PassProperty::CHANGE_DYNAMIC_STATE);
this->add_matcher(m2, callback2, PassProperty::CHANGE_DYNAMIC_STATE);
~~~~~~~~~~~~~

The last argument `PassProperty::CHANGE_DYNAMIC_STATE` says that callback can be applied for ngraph::Function with dynamic shapes. In case if callback do not support dynamic shapes `PassProperty::REQUIRE_STATIC_SHAPE` can be used.

To run any transformation you need to call `ngraph::pass::MyTransformationClass().run_on_function(f)` where `f` is `ngraph::Function`.
  
## Pattern matching

Sometimes patterns can't be expressed via regular nGraph operations. For example if you want to detect Convolution->Add sub-graph without specifying particular input type for Convolution operation or you want to create pattern where some of operations can have different types.
And for this cases nGraph provides additional helpers to construct patterns for GraphRewrite transformations. 
There are two main helpers:
1. `ngraph::pattern::op::Label` - helps to express inputs if their type is undefined.
2. `ngraph::pattern::op::Any` - helps to express intermediate nodes of pattern if their type unknown.

Lets go through example to have better understanding how it works:
Note: node attributes do not participate in pattern matching and needed only for operations creation. Only operation types participate in pattern matching.

Example below show basic usage of `pattern::op::Label` class.
Here we construct Multiply pattern with arbitrary first input and Constant as a second input.

~~~~~~~~~~~~~{.cpp}
auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
auto value = ngraph::opset3::Constant::create(element::f32, Shape{1}, {0.5});
auto mul = std::make_shared<opset3::Multiply>(input, value);
auto m = std::make_shared<pattern::Matcher>(mul, "MultiplyMatcher");
~~~~~~~~~~~~~

This example show how we can construct pattern when operation has arbitrary number of inputs.

~~~~~~~~~~~~~{.cpp}
// Detect Concat operation with arbitrary number of inputs
auto concat = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Concat>());
auto m = std::make_shared<pattern::Matcher>(gelu, "ConcatMatcher");
~~~~~~~~~~~~~

This example shows how to use predicate to construct pattern where operation has two different types.

~~~~~~~~~~~~~{.cpp}
// Detect Multiply or Add operation
auto lin_op = std::make_shared<pattern::op::Label>(element::f32, Shape{}, 
        [](const std::shared_ptr<Node> & node) -> bool {
            return std::dynamic_pointer_cast<opset3::Multiply>(node) || 
                   std::dynamic_pointer_cast<opset3::Add>(node);
        });
auto m = std::make_shared<pattern::Matcher>(lin_op, "MultiplyOrAddMatcher");
~~~~~~~~~~~~~

TODO: add examples for ngraph::pattern::op::Any

## Working with ngraph::Function

In this chapter we will review nGraph API that allows us to manipulate with `ngraph::Function`.

`1.` ngraph::Node input and output ports

First of all let's talk about `ngraph::Node` input/output ports. Each nGraph operation has input and output ports except cases when operation has Result, Parameter or Constant type.

Every port belongs to its node so via port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port and get producer node in case of input port.

Lets look at code example.
~~~~~~~~~~~~~{.cpp}
// Let's supose that node is opset3::Convolution operation
// as we know opset3::Convolution has two input ports (data, weights) and one output port

Input<Node> data = node->input(0);
Input<Node> weights = node->input(1);
Output<Node> output = node->output(0);

// Getting shape and type
auto pshape = data.get_partial_shape();
auto el_type = data.get_element_type();

// Ggetting parent for input port
Output<Node> parent_output = data.get_source_output();
// Another short way to get partent for output port
Output<Node> parent_output = node->input_value(0);

// Getting all consumers for output port
auto consumers = output.get_target_inputs();
~~~~~~~~~~~~~

You may notice that we usually construct operations in this way:
~~~~~~~~~~~~~{.cpp}
std::shared_ptr<Node> neg_const = opset1::Constant::create(sub->get_input_element_type(1), Shape{1}, {-1}));
Output<Node> data = node->input_value(0);
auto neg = std::make_shared<ngraph::opset1::Multiply>(data, neg_const);
~~~~~~~~~~~~~
In this example `opset3::Multiply` operation takes `Output<Node>` and `std::shared_ptr<Node>` as inputs. But constructor takes both as `Output<Node>`. 
In this case `std::shared_ptr<Node>` will be automatically converted to `Output<Node>` if node has exactly one output port otherwise conversion will raise an exception.   

`2.` ngraph::Node replacement

nGraph provides two ways for node replacement: via ngraph helper function and directly via port methods. We are going to review both of them.

Lets start with nGraph helper functions. The most popular function is `ngraph::replace_node(old_node, new_node)`.

Usage example:
~~~~~~~~~~~~~{.cpp}
auto neg = std::dynamic_pointer_cast<ngraph::opset1::Negative> (m.get_match_root());
if (!neg) {
    return false;
}

auto mul = std::make_shared<ngraph::opset1::Multiply>(neg->input_value(0),
                                                      opset1::Constant::create(neg->get_element_type(), Shape{1}, {-1}));
mul->set_friendly_name(neg->get_friendly_name());
ngraph::copy_runtime_info(neg, mul);
// Replaces Negative operation with Multiply operation
ngraph::replace_node(neg, mul);
~~~~~~~~~~~~~ 
`ngraph::replace_node` has a constraint that number of output ports for both of ops must be the same otherwise it will raise an exception.


The alternative way to do the same replacement is next:
~~~~~~~~~~~~~{.cpp}
// All neg->output(0) consumers will be moved to mul->output(0) port
neg->output(0).replace(mul->output(0));
~~~~~~~~~~~~~

Another transformation example is insertion.
~~~~~~~~~~~~~{.cpp}
// Lets suppose that we have a node with single output port and we want to insert additional operation new_node after it
void insert_example(std::shared_ptr<ngraph::Node> node) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();
    // Create new node. Let it be opset1::Relu.
    auto new_node = std::make_shared<ngraph::opset1::Relu>(node);
    // Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(new_node);
    }
}
~~~~~~~~~~~~~ 

The alternative way to insert operation is to make a node copy and use `replace_node`:
~~~~~~~~~~~~~{.cpp} 
void insert_example(std::shared_ptr<ngraph::Node> node) {
    // Make a node copy 
    auto node_copy = node->clone_with_new_inputs(node->input_values());
    // Create new node
    auto new_node = std::make_shared<ngraph::opset1::Relu>(node_copy);
    ngraph::replace_node(node, new_node);
}
~~~~~~~~~~~~~

`3.` ngraph::Node elimination

Another type of node replacement is its elimination.

To eliminate operation nGraph has special method that consider all limitations related to InferenceEngine.
~~~~~~~~~~~~~{.cpp}
// Suppose we have a node that we want to remove
bool success = replace_output_update_name(node->output(0), node->input_value(0));
~~~~~~~~~~~~~ 

`replace_output_update_name` in case of successful replacement it automatically preserves friendly name and runtime info.
  

## Transformation writing essentials

When developing transformation we need to follow next transformation rules:

`1.` dynamic shape and rank: 

nGraph has two types for shape representation: 
`ngraph::Shape` - represents static shape.
`ngraph::PartialShape` - represents dynamic shape. That means that rank or some of dimensions are undefined.
`ngraph::PartialShape` can be converted to `ngraph::Shape` using `get_shape()` method if all dimensions are static otherwise conversion will raise an exception.

~~~~~~~~~~~~~{.cpp}
auto partial_shape = node->input(0).get_partial_shape(); // get zero input partial shape
if (partial_shape.is_dynamic() /* or !partial_shape.is_staic() */) {
    return false;
}
auto static_shape = partial_shape.get_shape();
~~~~~~~~~~~~~

But in most cases before getting static shape using `get_shape()` method you need to check that shape is static.  

Also if your transformation requires only input shape rank or particular dimension value for some reason please do not use `get_shape()` method. See example below how not to use `get_shape()` method. 

~~~~~~~~~~~~~{.cpp}
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
~~~~~~~~~~~~~

Not using `get_shape()` method makes your transformation more flexible and applicable for more cases.

`2.` friendly names:

Each `ngraph::Node` has unique name (is used for nGraph internals) and friendly name. In transformations we care only about friendly name because it represents name from IR. 
Also friendly name is used as output tensor name (until we do not have other way to represent output tensor name) and user code that requests intermediate outputs based on this names.
So not to loose friendly name when replacing node with other node or sub-graph we need to set original friendly name to the latest node in replacing sub-garph. See example below. 

~~~~~~~~~~~~~{.cpp}
// Replace Div operation with Power and Multiply sub-graph and set original friendly name to Multiply operation
auto pow = std::make_shared<ngraph::opset1::Power>(div->input(1).get_source_output(),
                                                           op::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));
auto mul = std::make_shared<ngraph::opset1::Multiply>(div->input(0).get_source_output(), pow);
mul->set_friendly_name(div->get_friendly_name());
ngraph::replace_node(div, mul);
~~~~~~~~~~~~~

In more advanced cases when replaced operation has several outputs and we add additional consumers to its outputs we make decision how to set friendly name by arrangement.

`3.` runtime info:

Runtime info is a map `std::map<std::string, std::shared_ptr<Variant>>` located inside `ngraph::Node` class. It represents additional attributes in `ngraph::Node`. Find more information about runtime info in [Custom attributes in nodes] chapter.
This attributes can be set by users or by plugins and when executing transformation that changes `ngraph::Function` we need to preserve this attributes as they won't be automatically propagated.
In most cases transformations has next types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently there is no mechanism that automatically detects transformation types so we need to propagate this runtime information manually. See examples below.

~~~~~~~~~~~~~{.cpp}
// Replace Transpose with Reshape operation (1:1)
ngraph::copy_runtime_info(transpose, reshape);
~~~~~~~~~~~~~

~~~~~~~~~~~~~{.cpp}
// Replace Div operation with Power and Multiply sub-graph (1:N)
ngraph::copy_runtime_info(div, {pow, mul});
~~~~~~~~~~~~~

~~~~~~~~~~~~~{.cpp}
// Fuse Convolution with Add operation (N:1)
ngraph::copy_runtime_info({conv, bias}, {conv_ie});
~~~~~~~~~~~~~

~~~~~~~~~~~~~{.cpp}
// Any other transformation that replaces one sub-graph with another sub-graph (N:M)
ngraph::copy_runtime_info({a, b, c}, {e, f});
~~~~~~~~~~~~~

When transformation has multiple fusions or decompositions `ngraph::copy_runtime_info` must be called multiple times for each case. 

`4.` constant folding:

If your transformation inserts constant sub-graphs that needs to be folded do not forget to use `ngraph::pass::ConstantFolding()` after your transformation.
Example below shows how constant sub-graph can be constructed.

~~~~~~~~~~~~~{.cpp}
// After ConstantFolding pass Power will be replaced with Constant 
auto pow = std::make_shared<ngraph::opset3::Power>(
                    opset3::Constant::create(element::f32, Shape{1}, {2})
                    opset3::Constant::create(element::f32, Shape{1}, {3}));
auto mul = std::make_shared<ngraph::opset3::Multiply>(input /* not constant input */, pow);
~~~~~~~~~~~~~ 

## Common mistakes in transformations	 

TODO: deprecated API, duplicates, CF, shape inference, opset versions, get_node_shared_ptr() as input

## Using pass manager

`ngraph::pass::Manager` is a container class that can store list of transformations and execute them. The main idea of this class is to have high-level representation for grouped list of transformations.
For example [ConmmonOptimizations] pass manager register list of transformation related to common optimizations.

Below you can find basic usage of `ngraph::pass::Manager`
~~~~~~~~~~~~~{.cpp}
ngraph::pass::Manager pass_manager;
pass_manager.register_pass<pass::Opset1Upgrade>();
pass_manager.run_passes(f);
~~~~~~~~~~~~~

TODO: Advanced pass manager usage.

## How to debug transformations

The most popular tool for transformations debugging is `ngraph::pass::VisualizeTree` transformation that visualize ngraph::Function.

Usage example:

~~~~~~~~~~~~~{.cpp}
std::vector<std::shared_ptr<ngraph::Function> > g{f};
// Serialize ngraph::Function to before.svg file before transformation
ngraph::pass::VisualizeTree("/path/to/file/before.svg").run_on_module(g);
// Run your transformation
ngraph::pass::MyTransformation().run_on_function();
// Serialize ngraph::Function to after.svg file after transformation
ngraph::pass::VisualizeTree("/path/to/file/after.svg").run_on_module(g);
~~~~~~~~~~~~~

ngraph::pass::VisualizeTree can be parametrized via environment variables:

`NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES=1` - visualize shapes

`NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES=1`  - visualize types

Note: current VisualTree has not user friendly interface and it will be changed in nearest future. The intention is to move visualize abilities inside transformations.

If you are using `ngraph::pass::Manager` to run sequence of transformations you can get additional debug capabilities by using next environment variables:

`NGRAPH_PROFILE_PASS_ENABLE=1` - enables performance measurement for each transformation and prints execution status

`NGRAPH_ENABLE_VISUALIZE_TRACING=1` -  enables visualization after each transformation. By default it saves dot and svg files.

Note: make sure that you have dot installed on your machine otherwise it will silently save only dot file without svg file.

## Disabling/Enabling specific transformations for plugin X	 

This topic mostly related to conversion to legacy opset and plugins that based on CNNNetwork but still this mechanism can be applied for other cases.
Let's suppose that plugin X starts support `opset3::StridedSlice` operation and you want to disable `ConvertStridedSliceToCrop` transformation for plugin X.
To do this you need to extend transformation class with `ngraph::pass::PassParam` class. This class extends transformations class with `transformation_callback` that can be set by plugin that uses legacy conversion. 

~~~~~~~~~~~~~{.cpp}
// Extend transformation class with PassParam
class ngraph::pass::ConvertStridedSliceToCrop: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
    ...
}

// Update callback to be able to use transformation_callback if this transformation based on GraphRewrite.
ngraph::graph_rewrite_callback callback = [this](pattern::Matcher &m) {
    ...
}

// Use transformation_callback not to execute transformation
if (transformation_callback(node)) {
    return false;
}
~~~~~~~~~~~~~

TODO: link to existing example

## Transformations testing

We have two types of tests: nGraph reader tests located in `inference-engine/tests/functional/inference_engine/ngraph_reader` and transformation tests located  in `inference-engine/tests/functional/inference_engine/transformations`
Reader tests are based on IR and tests end to end conversion from IR to CNNNetwork. Transformation tests tests single ngraph transformations or low level functiont that are used inside transformations.

The basic transformation test looks like this:
~~~~~~~~~~~~~{.cpp}
TEST(TransformationTests, MyTransformationTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        // Create nGraph function that will be processed by transformation
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
        
        // This pass inits runtime info attributes
        ngraph::pass::InitNodeInfo().run_on_function(f);
        // Run transformation
        ngraph::pass::MyTransformation().run_on_function(f);
        // Check that runtime info attributes was correctly processed
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        // Create reference nGraph function that is excpeted after applying MyTransformation
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto pow = std::make_shared<ngraph::opset1::Power>(divide_constant,
                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1}));
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
    }

    // Comparing processed function with expected
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
~~~~~~~~~~~~~

TODO: insert advanced transformation tests