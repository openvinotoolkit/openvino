// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/nms_transformation_for_last_node.hpp"

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <inference_engine.hpp>

#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/ov_test_utils.hpp"

#include <memory>
#include <string>
#include <algorithm>
#include <utility>

namespace ExecutionGraphTests {

std::string ExecGraphNmsTransformLastNode::getTestCaseName(
    testing::TestParamInfo<std::string> obj) {
  std::string targetDevice = obj.param;
  return "Dev=" + targetDevice;
}

/**
 * Infer simple graph with just NMS node.
 * Verify that after NMS transformation network can be inferred
 * especially, that NMS transformation does not change name
 * of the output (Result) node
 */
TEST_P(ExecGraphNmsTransformLastNode, CheckIfCanBeInfered) {
  SKIP_IF_CURRENT_TEST_IS_DISABLED()

  using namespace ngraph;

  auto device_name = this->GetParam();
  ngraph::Shape boxes_shape = {1, 2, 4};
  ngraph::Shape scores_shape = {1, 1, 2};
  float in_boxes[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float in_scores[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  auto boxes = std::make_shared<opset5::Parameter>(element::f32, boxes_shape);
  auto scores = std::make_shared<opset5::Parameter>(element::f32, scores_shape);
  auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
  auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
  auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
  auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                                  iou_threshold, score_threshold,
                                                                                  opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i64);
  ngraph::ResultVector results {
      std::make_shared<opset5::Result>(nms->output(0)),
  };

  auto f = std::make_shared<Function>(results, ParameterVector{boxes, scores}, "NMS");

  auto ie = InferenceEngine::Core();
  auto net = InferenceEngine::CNNNetwork(f);
  auto exec_net = ie.LoadNetwork(net, device_name);
  auto infer_req = exec_net.CreateInferRequest();

  InferenceEngine::TensorDesc tDesc1(InferenceEngine::Precision::FP32, boxes_shape,
                                    InferenceEngine::Layout::CHW);
  InferenceEngine::TensorDesc tDesc2(InferenceEngine::Precision::FP32, scores_shape,
                                    InferenceEngine::Layout::CHW);

  InferenceEngine::Blob::Ptr inBlob1 = InferenceEngine::make_shared_blob<float>(tDesc1, in_boxes);
  infer_req.SetBlob(boxes->get_name(), inBlob1);
  InferenceEngine::Blob::Ptr inBlob2 = InferenceEngine::make_shared_blob<float>(tDesc2, in_scores);
  infer_req.SetBlob(scores->get_name(), inBlob2);

  infer_req.Infer();

  const auto& initial_outputs = net.getOutputsInfo();
  const auto& final_outputs = exec_net.GetOutputsInfo();

  auto compareOutputNames = [] (const std::pair<std::string, InferenceEngine::CDataPtr>& lhs,
                                const std::pair<std::string, InferenceEngine::CDataPtr>& rhs)
  { return lhs.first == rhs.first; };

  ASSERT_TRUE(std::equal(initial_outputs.begin(), initial_outputs.end(), final_outputs.begin(), compareOutputNames));
}

} // namespace ExecutionGraphTests
