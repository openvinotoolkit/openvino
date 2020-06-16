# Writing ngraph transformations {#new_ngraph_transformation}

Table of Сontents:

1. ngraph::Function and graph representation	 
2. Transformations types	 
3. Pattern matching
4. Working with ngraph::Function	 
5. Transformation developing essentials	 
6. How to debug transformations	 
7. Disabling/Enabling specific transformations for plugin X	 
8. Custom attributes in nodes	 
9. Common mistakes in transformations	 
10. Transformations testing

## ngraph::Function and graph representation

nGraph function is a very simple thing: it stores shared pointers to Result and Parameter operations that are inputs and outputs of the graph. 
All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore.

## Transformations types

There are two main transformation types:

`1.` ngraph::pass::FunctionalPass is used for transformations that take entire ngraph::Function as input and process it.

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

`2.` ngraph::pass::GraphRewrite is used for pattern based transformations.

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

Using ngraph::GraphRewrite you need to complete three steps:
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

You may notice that `Parameter` operation in example has type and shape specified. This attributes is needed only to create Parameter operation class and not used in pattern matching. 
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
  
## Pattern matching

Sometimes patterns can't be expressed via regular nGraph operations. For example if you want to detect Convolution->Add sub-graph without specifying particular input type for Convolution operation or you want to create pattern where some of operations can have different types.
And for this cases nGraph provides additional helpers to construct patterns for GraphRewrite transformations. 
There are two main helpers:
1. ngraph::pattern::op::Label - helps to express inputs if their type is undefined.
2. ngraph::pattern::op::Any - helps to express intermediate nodes of pattern if their type unknown.

Lets go through example to have better understanding how it works:
Note: node attributes do not participate in pattern matching and needed only for operations creation. Only operation types participate in pattern matching.

Example below show basic usage of pattern::op::Label class.
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

1. Nodes and input/output ports
2. Node replacement
3. Node elimination
4. Sub-graph elimination

## Transformation developing essentials	 

## How to debug transformations

## Disabling/Enabling specific transformations for plugin X	 

## Custom attributes in nodes

## Common mistakes in transformations	 

## Transformations testing