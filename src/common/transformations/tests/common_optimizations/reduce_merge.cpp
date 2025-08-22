// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reduce_merge.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
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
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(
            OutputVector{std::make_shared<opset9::ReduceLogicalAnd>(reduce1, reduce2_axis, true)},
            ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::boolean, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(
            OutputVector{std::make_shared<opset9::ReduceLogicalOr>(reduce1, reduce2_axis, true)},
            ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::boolean, Shape{1});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceMax>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceMean>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceMin>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceProd>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
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
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                        ParameterVector{A});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::i64, Shape{3, 2});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL1>(data, reduce1_axes, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        model_ref =
            std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL1>(reduce1, reduce2_axis, false)},
                                    ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeNotPassInvalidAxes) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 2});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, false);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        model_ref =
            std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, false)},
                                    ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeDynamicShapes) {
    {
        auto data =
            std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto reduce1_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto reduce1 = std::make_shared<opset9::ReduceL2>(data, reduce1_axis, true);
        auto reduce2_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceL2>(reduce1, reduce2_axis, true)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        auto data =
            std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceL2>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
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
        auto axes = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
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
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
        auto reduce = std::make_shared<opset9::ReduceMean>(data, axes, true);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceSumKeepDimsFalse) {
    {
        // Test case for the bug described in issue #31833
        // Input shape: [1, 256, 64, 16]
        // First ReduceSum on axis -1 (16) with keep_dims=false -> output shape: [1, 256, 64]
        // Second ReduceSum on axis -1 (64) with keep_dims=false -> output shape: [1, 256]
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 256, 64, 16});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto reduce1 = std::make_shared<opset9::ReduceSum>(data, reduce1_axes, false);
        auto reduce2_axes = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axes, false)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        // Expected result after fusion:
        // Fused ReduceSum with axes [-1, -2] on original input shape [1, 256, 64, 16]
        // This will reduce both the last two dimensions (16 and 64) to get output shape [1, 256]
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 256, 64, 16});
        auto axes = op::v0::Constant::create(element::i64, Shape{2}, {-1, -2});
        auto reduce = std::make_shared<opset9::ReduceSum>(data, axes, false);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMergeReduceSumKeepDimsFalseComplex) {
    {
        // Test case with more complex axis combinations
        // Input shape: [2, 3, 4, 5, 6]
        // First ReduceSum on axes [0, 2] with keep_dims=false -> output shape: [3, 5, 6]
        // Second ReduceSum on axis -1 (6) with keep_dims=false -> output shape: [3, 5]
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
        auto reduce1_axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 2});
        auto reduce1 = std::make_shared<opset9::ReduceSum>(data, reduce1_axes, false);
        auto reduce2_axes = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        model = std::make_shared<Model>(OutputVector{std::make_shared<opset9::ReduceSum>(reduce1, reduce2_axes, false)},
                                        ParameterVector{data});
        manager.register_pass<ov::pass::ReduceMerge>();
    }
    {
        // Expected result after fusion:
        // Fused ReduceSum with axes [0, 2, 4] on original input shape [2, 3, 4, 5, 6]
        // This will reduce dimensions 0, 2, and 4 to get output shape [3, 5]
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
        auto axes = op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 4});
        auto reduce = std::make_shared<opset9::ReduceSum>(data, axes, false);
        model_ref = std::make_shared<Model>(OutputVector{reduce}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
