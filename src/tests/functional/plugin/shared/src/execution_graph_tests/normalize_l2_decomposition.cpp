// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <gtest/gtest.h>
#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset9.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "execution_graph_tests/normalize_l2_decomposition.hpp"

namespace ExecutionGraphTests {

std::string ExecGrapDecomposeNormalizeL2::getTestCaseName(
    testing::TestParamInfo<std::string> obj) {
  std::string targetDevice = obj.param;
  return "Dev=" + targetDevice;
}

TEST_P(ExecGrapDecomposeNormalizeL2, CheckIfDecomposeAppliedForNonContiguousAxes) {
      SKIP_IF_CURRENT_TEST_IS_DISABLED()
      auto device_name = this->GetParam();

      const float eps_value = 0.000099f;
      const auto input = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::PartialShape{3, 4, 5});
      const auto axes_const = ov::opset9::Constant::create(ov::element::i64, ov::Shape{2}, {0, 2});
      const auto normalize_l2 = std::make_shared<ov::opset9::NormalizeL2>(input, axes_const, eps_value, ov::op::EpsMode::MAX);

      const auto model = std::make_shared<ov::Model>(ov::NodeVector{normalize_l2}, ov::ParameterVector{input});

      auto core = ov::Core();
      ov::AnyMap config;
      if (device_name == ov::test::utils::DEVICE_GPU)
        config.insert(ov::hint::inference_precision(ov::element::f32));
      const auto compiled_model = core.compile_model(model, device_name, config);

      ASSERT_TRUE(model->get_ops().size() < compiled_model.get_runtime_model()->get_ops().size()); // decomposition applied
}

TEST_P(ExecGrapDecomposeNormalizeL2, CheckIfDecomposeAppliedForNormalizeOverAllAxes) {
      SKIP_IF_CURRENT_TEST_IS_DISABLED()
      auto device_name = this->GetParam();

      const float eps_value = 0.000099f;
      const auto input = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::PartialShape{3, 4, 5});
      const auto axes_const = ov::opset9::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
      const auto normalize_l2 = std::make_shared<ov::opset9::NormalizeL2>(input, axes_const, eps_value, ov::op::EpsMode::MAX);

      const auto model = std::make_shared<ov::Model>(ov::NodeVector{normalize_l2}, ov::ParameterVector{input});

      auto core = ov::Core();
      ov::AnyMap config;
      if (device_name == ov::test::utils::DEVICE_GPU)
        config.insert(ov::hint::inference_precision(ov::element::f32));
      const auto compiled_model = core.compile_model(model, device_name, config);

      ASSERT_TRUE(model->get_ops().size() < compiled_model.get_runtime_model()->get_ops().size()); // decomposition applied
}

TEST_P(ExecGrapDecomposeNormalizeL2, CheckIfDecomposeNotAppliedForNotSorted) {
      SKIP_IF_CURRENT_TEST_IS_DISABLED()
      auto device_name = this->GetParam();

      const float eps_value = 0.000099f;
      const auto input = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::PartialShape{2, 1});
      const auto axes_const = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {1});
      const auto normalize_l2 = std::make_shared<ov::opset9::NormalizeL2>(input, axes_const, eps_value, ov::op::EpsMode::ADD);

      const auto model = std::make_shared<ov::Model>(ov::NodeVector{normalize_l2}, ov::ParameterVector{input});

      auto core = ov::Core();
      ov::AnyMap config;
      if (device_name == ov::test::utils::DEVICE_GPU)
        config.insert(ov::hint::inference_precision(ov::element::f32));
      const auto compiled_model = core.compile_model(model, device_name, config);

      ASSERT_TRUE(model->get_ops().size() >= compiled_model.get_runtime_model()->get_ops().size()); // decomposition not applied
}

TEST_P(ExecGrapDecomposeNormalizeL2, CheckIfDecomposeNotAppliedForSingleAxis) {
      SKIP_IF_CURRENT_TEST_IS_DISABLED()
      auto device_name = this->GetParam();

      const float eps_value = 0.000099f;
      const auto input = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
      const auto axes_const = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {1});
      const auto normalize_l2 = std::make_shared<ov::opset9::NormalizeL2>(input, axes_const, eps_value, ov::op::EpsMode::ADD);

      const auto model = std::make_shared<ov::Model>(ov::NodeVector{normalize_l2}, ov::ParameterVector{input});

      auto core = ov::Core();
      ov::AnyMap config;
      if (device_name == ov::test::utils::DEVICE_GPU)
        config.insert(ov::hint::inference_precision(ov::element::f32));
      const auto compiled_model = core.compile_model(model, device_name, config);

      ASSERT_TRUE(model->get_ops().size() >= compiled_model.get_runtime_model()->get_ops().size()); // decomposition not applied
}

} // namespace ExecutionGraphTests
