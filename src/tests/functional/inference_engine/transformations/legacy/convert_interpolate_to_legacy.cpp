// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertInterpolateDynamic) {
    auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {30, 60});
    auto interp = std::make_shared<ngraph::opset1::Interpolate>(data, shape, ngraph::op::v0::InterpolateAttrs());

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{interp}, ngraph::ParameterVector{data});

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher>();
    ASSERT_NO_THROW(m.run_passes(f));
}