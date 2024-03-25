// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/chained_maximum.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

TEST_F(TransformationTestsF, ChainedMaximumAC) {
    // A == C
    // Maximum(Maximum(A, B), C) -> Maximum(B, C)
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto A = make_shared<v3::ShapeOf>(input);
        auto B = v0::Constant::create(element::i64, {}, {1});
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);

        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto broadcast = make_shared<v1::Broadcast>(data, maximum_1);

        model = make_shared<Model>(NodeVector{broadcast}, ParameterVector{input, data});
        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto B = v0::Constant::create(element::i64, {}, {1});
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum = make_shared<v1::Maximum>(B, C);

        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto broadcast = make_shared<v1::Broadcast>(data, maximum);

        model_ref = make_shared<Model>(NodeVector{broadcast}, ParameterVector{input, data});
    }
}

TEST_F(TransformationTestsF, ChainedMaximumBC) {
    // B == C
    // Maximum(Maximum(A, B), C) -> Maximum(A, C)
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto B = make_shared<v3::ShapeOf>(input);
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);
        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto broadcast = make_shared<v1::Broadcast>(data, maximum_1);

        model = make_shared<Model>(NodeVector{broadcast}, ParameterVector{input, data});
        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
    {
        auto input = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto C = make_shared<v3::ShapeOf>(input);

        auto maximum = make_shared<v1::Maximum>(A, C);

        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto broadcast = make_shared<v1::Broadcast>(data, maximum);

        model_ref = make_shared<Model>(NodeVector{broadcast}, ParameterVector{input, data});
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

        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto broadcast = make_shared<v1::Broadcast>(data, maximum_1);

        model = make_shared<Model>(NodeVector{broadcast}, ParameterVector{input, data});
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
}

TEST_F(TransformationTestsF, ChainedMaximumNegativeDifferentLabels) {
    {
        auto input_0 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));

        auto A = v0::Constant::create(element::i64, {}, {1});
        auto B = make_shared<v3::ShapeOf>(input_0);
        auto C = make_shared<v3::ShapeOf>(input_1);

        auto maximum_0 = make_shared<v1::Maximum>(A, B);
        auto maximum_1 = make_shared<v1::Maximum>(maximum_0, C);

        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto broadcast = make_shared<v1::Broadcast>(data, maximum_1);

        model = make_shared<Model>(NodeVector{broadcast}, ParameterVector{input_0, input_1, data});
        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::ChainedMaximumOptimization>();
    }
}
