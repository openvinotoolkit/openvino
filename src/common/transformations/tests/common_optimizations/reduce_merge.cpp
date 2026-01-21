// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reduce_merge.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset9_decl.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;

TEST_F(TransformationTestsF, ReduceMergeReduceL1) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceL2) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceLogicalAnd) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::boolean, Shape{3, 2});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalAnd>(data, reduce1_axis, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceLogicalAnd>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::boolean, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceLogicalAnd>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceLogicalOr) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::boolean, Shape{1});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceLogicalOr>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceLogicalOr>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::boolean, Shape{1});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceLogicalOr>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceMax) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMax>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceMax>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceMax>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceMean) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceMean>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceMin) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMin>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceMin>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceMin>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceProd) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceProd>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceProd>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceProd>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceSum) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceSum>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceSum>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNoReduceDiffKeepDims) {
    {
        auto A = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(A, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{A});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    { model_ref = model->clone(); }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeKeepDimsFalse) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, false);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicShapesKeepDimsTrue) {
    {
        auto data =
            std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data =
            std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicShapesKeepDimsFalse) {
    auto data_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, data_shape);
        auto axis1 = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, axis1, false);
        auto axis2 = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, axis2, false);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, data_shape);
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, false);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicRankConcatKeepDimsTrue) {
    auto data_shape = PartialShape::dynamic(ov::Rank::dynamic());
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, data_shape);
        auto axis1 = op::v0::Constant::create(element::i64, Shape{}, {-1});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, axis1, true);
        auto axis2 = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, axis2, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    { model_ref = model->clone(); }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicRankKeepDimsTrue) {
    auto data_shape = PartialShape::dynamic(ov::Rank::dynamic());
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, data_shape);
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, data_shape);
        auto axis1 = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto axis2 = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto axes = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, true);
        model = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicRankKeepDimsFalse) {
    auto data_shape = PartialShape::dynamic(ov::Rank::dynamic());
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, data_shape);
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    { model_ref = model->clone(); }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMerge3ReducesL1) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axis, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, true);
        auto reduce3_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce3 = std::make_shared<opset9::ReduceL1>(reduce2, reduce3_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce3}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axes = op::v0::Constant::create(element::i64, Shape{3}, {0, 1, 2});
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeConcatAxes) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, axis1, true);
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, axis2, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data, axis1, axis2});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto axes = std::make_shared<opset9::Concat>(OutputVector{axis1, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data, axis1, axis2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeConcatAxesDifferentTypes) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, axis1, true);
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, axis2, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data, axis1, axis2});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
        auto convert = std::make_shared<opset9::Convert>(axis2, element::i64);
        auto axes = std::make_shared<opset9::Concat>(OutputVector{axis1, convert}, 0);
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data, axis1, axis2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeConcatAxesScalars) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, axis1, true);
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, axis2, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data, axis1, axis2});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        auto axis1_unsq =
            std::make_shared<opset9::Unsqueeze>(axis1, op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto axis2_unsq =
            std::make_shared<opset9::Unsqueeze>(axis2, op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto axes = std::make_shared<opset9::Concat>(OutputVector{axis1_unsq, axis2_unsq}, 0);
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data, axis1, axis2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeConcatAxesScalarAnd1DTensor) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, axis1, true);
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, axis2, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data, axis1, axis2});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        auto axis1_unsq =
            std::make_shared<opset9::Unsqueeze>(axis1, op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto axes = std::make_shared<opset9::Concat>(OutputVector{axis1_unsq, axis2}, 0);
        auto reduce = std::make_shared<opset9::ReduceL1>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data, axis1, axis2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDifferentShapesAndTypes) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i32, Shape{1}, {0});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce = std::make_shared<opset9::ReduceMean>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNegativeAxisKeepDimsFalse) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 2, 3, 4});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(data, reduce1_axes, false);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce2 = std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, false);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 2, 3, 4});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {2, 3});
        auto reduce = std::make_shared<opset9::ReduceMean>(data, axes, false);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNegativeAxisKeepDimsTrue) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 2, 3, 4});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce1 = std::make_shared<opset9::ReduceMean>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce2 = std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 2, 3, 4});
        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {3});
        auto reduce = std::make_shared<opset9::ReduceMean>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNotPassParamKeepDimsFalse) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2, 4});
        auto axis1 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, axis1, false);
        auto axis2 = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reduce2 = std::make_shared<opset9::ReduceL1>(reduce1, axis2, false);
        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data, axis1, axis2});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    { model_ref = model->clone(); }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

struct ReduceMergeKeepDimsFalseDifferentAxesParams {
    const Shape data_shape;
    const std::vector<int64_t> reduce1_axes;
    const std::vector<int64_t> reduce2_axes;
    const std::vector<int64_t> expected_axes;
    const std::string name;
};

class ReduceMergeKeepDimsFalseDifferentAxes
    : public TransformationTestsF,
      public ::testing::WithParamInterface<ReduceMergeKeepDimsFalseDifferentAxesParams> {};

TEST_P(ReduceMergeKeepDimsFalseDifferentAxes, CompareFunctions) {
    const auto& p = GetParam();
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, p.data_shape);

        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{p.reduce1_axes.size()}, p.reduce1_axes);
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);

        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{p.reduce2_axes.size()}, p.reduce2_axes);
        auto reduce2 = std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false);

        model = std::make_shared<Model>(OutputVector{reduce2}, ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, p.data_shape);
        auto axes = op::v0::Constant::create(element::i64, Shape{p.expected_axes.size()}, p.expected_axes);
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, false);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(
    TransformationTestsF,
    ReduceMergeKeepDimsFalseDifferentAxes,
    ::testing::Values(
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5}, {0, 1}, {0, 1}, {0, 1, 2, 3}, "equal_axes"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5}, {-1}, {0, 1}, {0, 1, 4}, "negative_axes1"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5}, {0, 1}, {-1}, {0, 1, 4}, "negative_axes2"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5, 6},
                                                    {0, 1},
                                                    {2, 3},
                                                    {0, 1, 4, 5},
                                                    "diff_axes_all_greater"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5, 6},
                                                    {2, 3},
                                                    {0, 1},
                                                    {0, 1, 2, 3},
                                                    "diff_axes_all_less"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5, 6},
                                                    {1, 3},
                                                    {0, 2},
                                                    {0, 1, 3, 4},
                                                    "diff_axes_interleaved"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4, 5, 6},
                                                    {1, 2},
                                                    {0, 3},
                                                    {0, 1, 2, 5},
                                                    "diff_axes_all_less_and_greater"},
        // 2D -> scalar (single-axis then remaining-axis)
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{3, 4}, {1}, {0}, {0, 1}, "2D_to_scalar"},
        // 3D: reduce last, then first of remaining axes
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{2, 3, 4}, {2}, {0}, {0, 2}, "3D_last_then_first"},
        // 4D: reduce middle, then reduce remaining tail
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{2, 3, 4, 5}, {1}, {1, 2}, {1, 2, 3}, "4D_middle_then_tail"},
        // Non-sorted reduce1 axes (tests normalization/ordering)
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{2, 3, 4, 5}, {2, 0}, {1}, {0, 2, 3}, "unsorted_reduce1_axes"},
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{2, 3, 4, 5}, {1}, {1, 0}, {0, 1, 2}, "unsorted_reduce2_axes"},
        // Reduce many axes first, then reduce the last remaining axis
        ReduceMergeKeepDimsFalseDifferentAxesParams{Shape{1, 2, 3, 4}, {0, 1, 2}, {0}, {0, 1, 2, 3}, "reduce_all"}),
    [](const ::testing::TestParamInfo<ReduceMergeKeepDimsFalseDifferentAxesParams>& info) {
        return info.param.name;
    });