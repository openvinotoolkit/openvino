// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ! [ov:include]
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
// ! [ov:include]

#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>

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

    // Create nGraph function
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
void serialize_example(const std::shared_ptr<ov::Model>& f) {
    // Need include:
    // * openvino/pass/manager.hpp
    // * openvino/pass/serialize.hpp
    ov::pass::Manager manager;

    // Serialize ov::Model to IR
    manager.register_pass<ov::pass::Serialize>("/path/to/file/model.xml", "/path/to/file/model.bin");

    manager.run_passes(f);
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
