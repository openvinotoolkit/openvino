// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertPadToLegacyDynamic) {
    auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto pad_begin = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto pad_end = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto pad = std::make_shared<ov::opset1::Pad>(data, pad_begin, pad_end, ngraph::op::PadMode::SYMMETRIC);

    auto f = std::make_shared<ov::Model>(ov::NodeVector{pad}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertPadToLegacyMatcher>();
    ASSERT_NO_THROW(m.run_passes(f));
}
