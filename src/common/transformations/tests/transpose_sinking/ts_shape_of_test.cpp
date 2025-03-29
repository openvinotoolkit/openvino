// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_shape_of.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "transformations/transpose_sinking/ts_fuse.hpp"

using namespace ov;

using TSShapeOfForward = TransformationTestsF;

TEST_F(TSShapeOfForward, v0ShapeOf) {
    {
        auto param = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto order1 = op::v0::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<op::v1::Transpose>(param, order1);
        auto shape_of = std::make_shared<op::v0::ShapeOf>(transpose1);
        auto order2 = op::v0::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1});
        auto transpose2 = std::make_shared<op::v1::Transpose>(transpose1, order2);
        auto abs = std::make_shared<op::v0::Abs>(transpose2);
        model = std::make_shared<Model>(NodeVector{shape_of, abs}, ParameterVector{param});

        manager.register_pass<ov::pass::transpose_sinking::TSShapeOfForward>();
        manager.register_pass<ov::pass::transpose_sinking::TSFuse>();
    }

    {
        auto param = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto order1 = op::v0::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2});
        auto shape_of = std::make_shared<op::v0::ShapeOf>(param);
        auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto gather = std::make_shared<op::v8::Gather>(shape_of, order1, axis);
        auto abs = std::make_shared<op::v0::Abs>(param);
        model_ref = std::make_shared<Model>(NodeVector{gather, abs}, ParameterVector{param});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TSShapeOfForward, v3ShapeOf) {
    {
        auto param = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto order1 = op::v0::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2});
        auto transpose1 = std::make_shared<op::v1::Transpose>(param, order1);
        auto shape_of = std::make_shared<op::v3::ShapeOf>(transpose1);
        auto order2 = op::v0::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1});
        auto transpose2 = std::make_shared<op::v1::Transpose>(transpose1, order2);
        auto abs = std::make_shared<op::v0::Abs>(transpose2);
        model = std::make_shared<Model>(NodeVector{shape_of, abs}, ParameterVector{param});

        manager.register_pass<ov::pass::transpose_sinking::TSShapeOfForward>();
        manager.register_pass<ov::pass::transpose_sinking::TSFuse>();
    }

    {
        auto param = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto order1 = op::v0::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2});
        auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
        auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto gather = std::make_shared<op::v8::Gather>(shape_of, order1, axis);
        auto abs = std::make_shared<op::v0::Abs>(param);
        model_ref = std::make_shared<Model>(NodeVector{gather, abs}, ParameterVector{param});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
