// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertInterpolateDynamic) {
    auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto shape = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {30, 60});
    auto interp = std::make_shared<ov::opset1::Interpolate>(data, shape, ov::op::v0::Interpolate::Attributes());

    auto f = std::make_shared<ov::Model>(ov::NodeVector{interp}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher>();
    ASSERT_NO_THROW(m.run_passes(f));
}
