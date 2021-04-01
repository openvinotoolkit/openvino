// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <fstream>

#include <ie_blob.h>
#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>

TEST(ONNX_Reader_Tests, ImportBasicModelToCore) {
    std::string model = R"V0G0N(
ir_version: 3
producer_name: "nGraph ONNX Importer"
graph {
  node {
    output: "B"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 2
        dims: 2
        data_type: 1
        float_data: 1
        float_data: 2
        float_data: 3
        float_data: 4
        name: "const_tensor"
      }
      type: TENSOR
    }
  }
  node {
    input: "A"
    input: "B"
    output: "X"
    name: "add_node1"
    op_type: "Add"
  }
  node {
    input: "X"
    input: "C"
    output: "Y"
    name: "add_node2"
    op_type: "Add"
  }
  name: "test_graph"
  initializer {
    dims: 2
    dims: 2
    data_type: 1
    name: "A"
    raw_data: "\000\000\200?\000\000\000@\000\000@@\000\000\200@"
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "C"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
}
opset_import {
  version: 4
}
)V0G0N";
    InferenceEngine::Core ie;
    InferenceEngine::Blob::CPtr weights;
    auto cnnNetwork = ie.ReadNetwork(model, weights);
    auto function = cnnNetwork.getFunction();

    int count_additions = 0;
    int count_constants = 0;
    int count_parameters = 0;

    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        count_additions += (op_type == "Add" ? 1 : 0);
        count_constants += (op_type == "Constant" ? 1 : 0);
        count_parameters += (op_type == "Parameter" ? 1 : 0);
    }

    ASSERT_EQ(function->get_output_size(), 1);
    ASSERT_EQ(std::string(function->get_output_op(0)->get_type_name()), "Result");
    ASSERT_EQ(function->get_output_element_type(0), ngraph::element::f32);
    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({2, 2}));
    ASSERT_EQ(count_additions, 2);
    ASSERT_EQ(count_constants, 2);
    ASSERT_EQ(count_parameters, 1);
}

