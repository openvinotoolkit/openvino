// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/chained_maximum.hpp"

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/maximum.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/bound_evaluation_util.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

void label_shape(ov::PartialShape& shape, size_t start_label = 42) {
    auto table = std::make_shared<ov::TableOfEquivalence>(start_label);
    auto tracker = ov::DimensionTracker(table);
    tracker.set_up_for_tracking(shape);
}

TEST_F(TransformationTestsF, ChainedMaximumAC) {
    // A == C
    // Maximum(Maximum(A, B), C) -> Maximum(B, C)
    {
        auto shape = PartialShape::dynamic(4);
        label_shape(shape); // we label shape with consecutive labels: 42, 43, 44, 45
        auto input = make_shared<v0::Parameter>(element::f32, shape);

        auto A = make_shared<v3::ShapeOf>(input);
        auto B = v0::Constant::create(element::i64, {}, {1});
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);

        ov::evaluate_both_bounds(maximum_1); // we request value, but more importantly label propagation for this graph

        model = make_shared<Model>(NodeVector{maximum_1}, ParameterVector{input});
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
    {
        auto shape = PartialShape::dynamic(4);
        label_shape(shape); // we label shape with consecutive labels: 42, 43, 44, 45
        auto input = make_shared<v0::Parameter>(element::f32, shape);

        auto B = v0::Constant::create(element::i64, {}, {1});
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum = make_shared<v1::Maximum>(B, C);

        model_ref = make_shared<Model>(NodeVector{maximum}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ChainedMaximumBC) {
    // B == C
    // Maximum(Maximum(A, B), C) -> Maximum(A, C)
    {
        auto shape = PartialShape::dynamic(4);
        label_shape(shape); // we label shape with consecutive labels: 42, 43, 44, 45
        auto input = make_shared<v0::Parameter>(element::f32, shape);

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto B = make_shared<v3::ShapeOf>(input);
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);

        ov::evaluate_both_bounds(maximum_1); // we request value, but more importantly label propagation for this graph

        model = make_shared<Model>(NodeVector{maximum_1}, ParameterVector{input});
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
    {
        auto shape = PartialShape::dynamic(4);
        label_shape(shape); // we label shape with consecutive labels: 42, 43, 44, 45
        auto input = make_shared<v0::Parameter>(element::f32, shape);

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum = make_shared<v1::Maximum>(A, C);

        model_ref = make_shared<Model>(NodeVector{maximum}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ChainedMaximumNegativeNoLabels) {
    {
        auto shape = PartialShape::dynamic(4);
        auto input = make_shared<v0::Parameter>(element::f32, shape);

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto B = make_shared<v3::ShapeOf>(input);
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);

        model = make_shared<Model>(NodeVector{maximum_1}, ParameterVector{input});
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
}

TEST_F(TransformationTestsF, ChainedMaximumNegativeDifferentLabels) {
    {
        auto shape_0 = PartialShape::dynamic(4);
        label_shape(shape_0, 42); // we label shape with consecutive labels: 42, 43, 44, 45
        auto shape_1 = PartialShape::dynamic(4);
        label_shape(shape_1, 47); // we label shape with consecutive labels: 47, 48, 49, 50
        auto input_0 = make_shared<v0::Parameter>(element::f32, shape_0);
        auto input_1 = make_shared<v0::Parameter>(element::f32, shape_1);

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto B = make_shared<v3::ShapeOf>(input_0);
        auto C = make_shared<v3::ShapeOf>(input_1);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);

        ov::evaluate_both_bounds(maximum_1); // we request value, but more importantly label propagation for this graph

        model = make_shared<Model>(NodeVector{maximum_1}, ParameterVector{input_0, input_1});
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
}
