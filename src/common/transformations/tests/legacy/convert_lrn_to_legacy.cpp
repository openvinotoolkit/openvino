// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertLRNToLegacyDynamic) {
    auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
    auto lrn = std::make_shared<ngraph::opset1::LRN>(data, axis, 1, 2, 3, 4);

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{lrn}, ngraph::ParameterVector{data});

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertLRNToLegacyMatcher>();
    ASSERT_NO_THROW(m.run_passes(f));
}