// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/remove_parameter.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphRemoveParameterNode::getTestCaseName(
    testing::TestParamInfo<std::string> obj) {
  std::string targetDevice = obj.param;
  return "Dev=" + targetDevice;
}

/**
 * Replacing parameter by another node change indexing for other parameters,
 * check that we still can correctly process changed network.
 */
TEST_P(ExecGraphRemoveParameterNode, RemoveParameterNode) {
  SKIP_IF_CURRENT_TEST_IS_DISABLED()

  auto device_name = this->GetParam();
  ov::Shape shape = {3, 2};
  float in_data_2[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float in_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  ov::element::Type type = ov::element::f32;

  // Some simple graph with 2 Parameters
  //    in2 in1    //
  //     \  / |    //
  //      mul |    //
  //       \  |    //
  //         sum   //
  //          |    //
  //         out   //
  auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
  auto input2 = std::make_shared<ov::op::v0::Parameter>(type, shape);
  auto mul = std::make_shared<ov::op::v1::Multiply>(input2, input);
  auto sum = std::make_shared<ov::op::v1::Add>(mul, input);

  auto function = std::make_shared<ov::Model>(
      ov::NodeVector{sum}, ov::ParameterVector{input2, input},
      "SimpleNet");

  // Load into plugin and get exec graph
  auto core = ov::Core();
  auto compiled_model = core.compile_model(function, device_name);
  auto infer_req = compiled_model.create_infer_request();

  ov::Tensor tensor2 {ov::element::f32, shape, in_data_2};
  infer_req.set_tensor(input2, tensor2);
  ov::Tensor tensor {ov::element::f32, shape, in_data};
  infer_req.set_tensor(input, tensor);

  infer_req.infer();

  auto out_tensor = infer_req.get_tensor(function->output(0));
  auto ref_result = out_tensor.data<float>();

  ASSERT_EQ(function->get_parameter_index(input2), 0);
  ASSERT_EQ(function->get_parameter_index(input), 1);

  // Replace input2 by constant
  auto const_in =
      std::make_shared<ov::op::v0::Constant>(type, shape, std::vector<float>(6, 1.0));
  mul->input(0).replace_source_output(const_in->output(0));
  function->remove_parameter(input2);

  ASSERT_EQ(function->get_parameters().size(), 1);
  ASSERT_EQ(function->get_parameter_index(input), 0);

  // Load new function into plugin and get exec graph
  auto new_compiled_model = core.compile_model(function, device_name);

  // infer new graph
  auto new_infer_req = new_compiled_model.create_infer_request();
  new_infer_req.set_tensor(input, tensor);

  new_infer_req.infer();

  auto new_out_tensor = new_infer_req.get_tensor(function->output(0));
  auto result = new_out_tensor.data<float>();

  for (int i = 0; i < 6; i++) {
    ASSERT_NEAR(result[i], ref_result[i], 1e-5);
  }
}

} // namespace ExecutionGraphTests
