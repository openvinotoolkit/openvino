// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/enable_shapeof_constant_folding.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, EnableShapeOfV0ConstantFolding) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<op::v0::ShapeOf>(data);
        pass::disable_constant_folding(shape_of);
        auto abs = std::make_shared<op::v0::Abs>(shape_of);
        auto reshape = std::make_shared<op::v1::Reshape>(data, abs, false);
        model = std::make_shared<Model>(reshape, ParameterVector{data});

        manager.register_pass<pass::EnableShapeOfConstantFolding>();
        manager.register_pass<pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape = op::v0::Constant::create(element::i64, Shape{4}, {1, 4, 10, 10});
        auto reshape = std::make_shared<op::v1::Reshape>(data, shape, false);
        model_ref = std::make_shared<Model>(reshape, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EnableShapeOfV3ConstantFolding) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<op::v3::ShapeOf>(data);
        pass::disable_constant_folding(shape_of);
        auto abs = std::make_shared<op::v0::Abs>(shape_of);
        auto reshape = std::make_shared<op::v1::Reshape>(data, abs, false);
        model = std::make_shared<Model>(reshape, ParameterVector{data});

        manager.register_pass<pass::EnableShapeOfConstantFolding>();
        manager.register_pass<pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape = op::v0::Constant::create(element::i64, Shape{4}, {1, 4, 10, 10});
        auto reshape = std::make_shared<op::v1::Reshape>(data, shape, false);
        model_ref = std::make_shared<Model>(reshape, ParameterVector{data});
    }
}
