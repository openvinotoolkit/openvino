// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConcatFusedToConcat) {
    {
        auto data = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto data2 = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 3, 7, 14});
        auto concat1 = std::make_shared<opset13::Concat>(OutputVector{data, data}, 1);
        auto concat2 = std::make_shared<opset13::Concat>(OutputVector{data2, data2}, 2);
        auto concat3 = std::make_shared<opset13::Concat>(OutputVector{concat1, concat2, data}, 1);
        auto result = std::make_shared<opset13::Result>(concat3);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data, data2});
        manager.register_pass<pass::ConcatFusion>();
    }
    {
        auto data = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 3, 14, 14});
        auto data2 = std::make_shared<opset13::Parameter>(element::f32, PartialShape{1, 3, 7, 14});
        auto concat2 = std::make_shared<opset13::Concat>(OutputVector{data2, data2}, 2);
        auto concat3 = std::make_shared<opset13::Concat>(OutputVector{data, data, concat2, data}, 1);
        auto result = std::make_shared<opset13::Result>(concat3);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data, data2});
    }
}
