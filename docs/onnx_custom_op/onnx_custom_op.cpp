// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [onnx_custom_op:headers]
// onnx_import/onnx_utils.hpp provides ngraph::onnx_import::register_operator function, that registers operator in ONNX importer's set.
#include <onnx_import/onnx_utils.hpp>
// ngraph/opsets/opset5.hpp provides the declaration of predefined nGraph operator set
#include <ngraph/opsets/opset5.hpp>
//! [onnx_custom_op:headers]


std::string custom_relu_model() {
    return
//! [onnx_custom_op:model]
R"ONNX(
ir_version: 3
producer_name: "nGraph ONNX Importer"
graph {
  node {
    input: "in"
    output: "out"
    name: "customrelu"
    op_type: "CustomRelu"
    domain: "com.example"
    attribute {
        name: "alpha"
        type: FLOAT
        f: 2
    }
    attribute {
        name: "beta"
        type: FLOAT
        f: 3
    }
  }
  name: "custom relu graph"
  input {
    name: "in"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 8
          }
        }
      }
    }
  }
  output {
    name: "out"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 8
          }
        }
      }
    }
  }
}
opset_import {
  domain: "com.example"
  version: 1
}
)ONNX";
//! [onnx_custom_op:model]
}


void register_custom_relu_operator() {
    // CustomRelu is defined as follows:
    // x >= 0 => f(x) = x * alpha
    // x < 0  => f(x) = x * beta

//! [onnx_custom_op:register_operator]
    ngraph::onnx_import::register_operator(
        "CustomRelu", 1, "com.example", [](const ngraph::onnx_import::Node& onnx_node) -> ngraph::OutputVector {
            namespace opset = ngraph::opset5;

            ngraph::OutputVector ng_inputs{onnx_node.get_ng_inputs()};
            const ngraph::Output<ngraph::Node>& data = ng_inputs.at(0);
            // create constant node with a single element that's equal to zero
            std::shared_ptr<ngraph::Node> zero_node = opset::Constant::create(data.get_element_type(), ngraph::Shape{}, {0});
            // create a negative map for 'data' node, 1 for negative values , 0 for positive values or zero
            // then convert it from boolean type to `data.get_element_type()`
            std::shared_ptr<ngraph::Node> negative_map = std::make_shared<opset::Convert>(
                std::make_shared<opset::Less>(data, zero_node), data.get_element_type());
            // create a positive map for 'data' node, 0 for negative values , 1 for positive values or zero
            // then convert it from boolean type to `data.get_element_type()`
            std::shared_ptr<ngraph::Node> positive_map = std::make_shared<opset::Convert>(
                std::make_shared<opset::GreaterEqual>(data, zero_node), data.get_element_type());

            // fetch alpha and beta attributes from ONNX node
            float alpha = onnx_node.get_attribute_value<float>("alpha", 1); // if 'alpha' attribute is not provided in the model, then the default value is 1
            float beta = onnx_node.get_attribute_value<float>("beta");
            // create constant node with a single element 'alpha' with type f32
            std::shared_ptr<ngraph::Node> alpha_node = opset::Constant::create(ngraph::element::f32, ngraph::Shape{}, {alpha});
            // create constant node with a single element 'beta' with type f32
            std::shared_ptr<ngraph::Node> beta_node = opset::Constant::create(ngraph::element::f32, ngraph::Shape{}, {beta});

            return {
                std::make_shared<opset::Add>(
                    std::make_shared<opset::Multiply>(alpha_node, std::make_shared<opset::Multiply>(data, positive_map)),
                    std::make_shared<opset::Multiply>(beta_node, std::make_shared<opset::Multiply>(data, negative_map))
                )
            };
    });
//! [onnx_custom_op:register_operator]
}

void unregister_custom_relu_operator() {
//! [onnx_custom_op:unregister_operator]
    ngraph::onnx_import::unregister_operator("CustomRelu", 1, "com.example");
//! [onnx_custom_op:unregister_operator]
}
