// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertPadToLegacyDynamic) {
    auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto pad_begin = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
    auto pad_end = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
    auto pad = std::make_shared<ngraph::opset1::Pad>(data, pad_begin, pad_end, ngraph::op::PadMode::SYMMETRIC);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{pad}, ngraph::ParameterVector{data});

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertPadToLegacyMatcher>();
    ASSERT_NO_THROW(m.run_passes(f));
}