// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/remove_parameter.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

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
  ngraph::Shape shape = {3, 2};
  float in_data_2[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float in_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  ngraph::element::Type type = ngraph::element::f32;

  using std::make_shared;
  using namespace ngraph::op;

  // Some simple graph with 2 Parameters
  //    in2 in1    //
  //     \  / |    //
  //      mul |    //
  //       \  |    //
  //         sum   //
  //          |    //
  //         out   //
  auto input = make_shared<Parameter>(type, shape);
  auto input2 = make_shared<Parameter>(type, shape);
  auto mul = make_shared<ngraph::op::v1::Multiply>(input2, input);
  auto sum = make_shared<ngraph::op::v1::Add>(mul, input);

  auto function = std::make_shared<ngraph::Function>(
      ngraph::NodeVector{sum}, ngraph::ParameterVector{input2, input},
      "SimpleNet");

  // Load into plugin and get exec graph
  auto ie = InferenceEngine::Core();
  auto net = InferenceEngine::CNNNetwork(function);
  auto exec_net = ie.LoadNetwork(net, device_name);
  auto exec_graph = exec_net.GetExecGraphInfo();
  auto infer_req = exec_net.CreateInferRequest();
  InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, shape,
                                    InferenceEngine::Layout::NC);
  InferenceEngine::Blob::Ptr inBlob2 =
      InferenceEngine::make_shared_blob<float>(tDesc, in_data_2);
  infer_req.SetBlob(input2->get_name(), inBlob2);

  InferenceEngine::Blob::Ptr inBlob =
      InferenceEngine::make_shared_blob<float>(tDesc, in_data);
  infer_req.SetBlob(input->get_name(), inBlob);

  infer_req.Infer();

  auto outBlob = infer_req.GetBlob(sum->get_name());
  InferenceEngine::MemoryBlob::CPtr output =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(outBlob);
  auto outputHolder = output->rmap();
  const auto ref_result = outputHolder.as<float *>();

  ASSERT_EQ(function->get_parameter_index(input2), 0);
  ASSERT_EQ(function->get_parameter_index(input), 1);

  // Replace input2 by constant
  auto const_in =
      make_shared<Constant>(type, shape, std::vector<float>(6, 1.0));
  mul->input(0).replace_source_output(const_in->output(0));
  function->remove_parameter(input2);

  ASSERT_EQ(function->get_parameters().size(), 1);
  ASSERT_EQ(function->get_parameter_index(input), 0);

  // Load new function into plugin and get exec graph
  auto new_net = InferenceEngine::CNNNetwork(function);
  auto new_exec_net = ie.LoadNetwork(new_net, device_name);
  auto new_exec_graph = new_exec_net.GetExecGraphInfo();

  // infer new graph
  auto new_infer_req = new_exec_net.CreateInferRequest();
  new_infer_req.SetBlob(input->get_name(), inBlob);

  new_infer_req.Infer();

  auto new_outBlob = new_infer_req.GetBlob(sum->get_name());
  InferenceEngine::MemoryBlob::CPtr new_output =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(new_outBlob);
  auto new_outputHolder = new_output->rmap();
  const auto result = new_outputHolder.as<float *>();

  for (int i = 0; i < 6; i++) {
    ASSERT_NEAR(result[i], ref_result[i], 1e-5);
  }
}

} // namespace ExecutionGraphTests
