// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <memory>
#include <tuple>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/plugin_cache.hpp"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>

namespace LayerTestsDefinitions {

using FQMulFusionParams = std::tuple<int, std::string>;

class FQMulFusion : public testing::WithParamInterface<FQMulFusionParams>,
                    public CommonTestUtils::TestsCommon {
public:
  void SetUp() override {
    const auto data = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{64, 3, 7, 7}, {0.0f});
    const auto in_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {-0.5f});
    const auto in_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.5f});
    const auto out_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.0f});
    const auto out_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {100.0f});
    const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, out_low, out_high, 255);

    // ngraph::builder::makeConstant<float>(ngPrc, outFormShapes2, {}, true);
    const auto mul_value = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{64, 1, 1, 1}, {3.14f});
    const auto mul = std::make_shared<ngraph::opset4::Multiply>(fq, mul_value);

    m_function = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{mul}, ngraph::ParameterVector{}, "FQMulFusion");
  }

  std::shared_ptr<ngraph::Function> m_function;
};

TEST_P(FQMulFusion, ValidateFusedSubgraph) {
  ngraph::pass::Manager manager;
  manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

  auto cloned_function = ngraph::clone_function(*m_function);
  manager.run_passes(cloned_function);

  // additional Mul node should be added
  ASSERT_EQ(cloned_function->get_ops().size(), m_function->get_ops().size());
};

namespace {
INSTANTIATE_TEST_CASE_P(FQMulFusionTests, FQMulFusion,
                        ::testing::Combine(::testing::Values(1),
                                           ::testing::Values("test1")));
} // namespace

} // namespace LayerTestsDefinitions
