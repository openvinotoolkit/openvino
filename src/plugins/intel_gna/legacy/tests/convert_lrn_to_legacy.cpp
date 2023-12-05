// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertLRNToLegacyDynamic) {
    auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto axis = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto lrn = std::make_shared<ov::opset1::LRN>(data, axis, 1, 2, 3, 4);

    auto f = std::make_shared<ov::Model>(ov::NodeVector{lrn}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ngraph::pass::ConvertLRNToLegacyMatcher>();
    ASSERT_NO_THROW(m.run_passes(f));
}
