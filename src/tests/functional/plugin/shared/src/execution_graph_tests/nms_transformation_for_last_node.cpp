// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/nms_transformation_for_last_node.hpp"

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/runtime/core.hpp"

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

  auto device_name = this->GetParam();
  ov::Shape boxes_shape = {1, 2, 4};
  ov::Shape scores_shape = {1, 1, 2};
  float in_boxes[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float in_scores[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, boxes_shape);
  auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
  auto max_output_boxes_per_class = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {10});
  auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.75});
  auto score_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.7});
  auto nms = std::make_shared<ov::op::v5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                             iou_threshold, score_threshold,
                                                             ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER, true, ov::element::i64);
  nms->output(0).set_names({"nms"});
  ov::ResultVector results {
      std::make_shared<ov::op::v0::Result>(nms->output(0)),
  };

  auto f = std::make_shared<ov::Model>(results, ov::ParameterVector{boxes, scores}, "NMS");

  auto core = ov::Core();
  auto exec_net = core.compile_model(f, device_name);
  auto infer_req = exec_net.create_infer_request();
  ov::Tensor boxes_tensor(ov::element::f32, boxes_shape, in_boxes);
  ov::Tensor scores_tensor(ov::element::f32, scores_shape, in_scores);
  infer_req.set_tensor(boxes, boxes_tensor);
  infer_req.set_tensor(scores, scores_tensor);
  infer_req.infer();

  const auto& initial_outputs = f->outputs();
  const auto& final_outputs = exec_net.outputs();

  auto compareOutputNames = [] (const ov::Output<ov::Node>& lhs,
                                const ov::Output<const ov::Node>& rhs)
  { return lhs.get_any_name() == rhs.get_any_name(); };

  ASSERT_TRUE(std::equal(initial_outputs.begin(), initial_outputs.end(), final_outputs.begin(), compareOutputNames));
}

} // namespace ExecutionGraphTests
