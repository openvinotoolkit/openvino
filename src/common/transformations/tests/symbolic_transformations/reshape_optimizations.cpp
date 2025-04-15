// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/reshape_optimizations.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

TEST_F(TransformationTestsF, FlattenOptimization) {
    // [A, B, C, D] -> [A, B, C*D]

    auto shape = PartialShape::dynamic(4);
    set_shape_symbols(shape);  // we set unique symbols to the shape: A, B, C, D

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto shape_of = make_shared<v3::ShapeOf>(data);
        auto indices = ov::op::v0::Constant::create(element::i64, {2}, {0, 1});
        auto axis = ov::op::v0::Constant::create(element::i64, {}, {0});

        auto as_is_dims = make_shared<v1::Gather>(shape_of, indices, axis);

        auto merged_dim = make_shared<v1::Multiply>(
            make_shared<v1::Gather>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {2}), axis),
            make_shared<v1::Gather>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {3}), axis));

        auto pattern = make_shared<v0::Concat>(OutputVector{as_is_dims, merged_dim}, 0);

        auto reshape = make_shared<v1::Reshape>(data, pattern, false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<pass::ReshapeOptimizations>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto pattern = ov::op::v0::Constant::create(element::i64, {3}, {0, 0, -1});

        auto reshape = make_shared<v1::Reshape>(data, pattern, true);

        model_ref = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, LastDimSplitStaticLast) {
    // [A, B, C, D] -> [A, B, C, D/8, 8]

    auto shape = PartialShape::dynamic(4);
    set_shape_symbols(shape);  // we set unique symbols to the shape: A, B, C, D

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto shape_of = make_shared<v3::ShapeOf>(data);
        auto indices = ov::op::v0::Constant::create(element::i64, {3}, {0, 1, 2});
        auto axis = ov::op::v0::Constant::create(element::i64, {}, {0});

        auto as_is_dims = make_shared<v1::Gather>(shape_of, indices, axis);
        auto splited_dim = ov::op::v0::Constant::create(element::i64, {2}, {-1, 8});

        auto pattern = make_shared<v0::Concat>(OutputVector{as_is_dims, splited_dim}, 0);

        auto reshape = make_shared<v1::Reshape>(data, pattern, false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<pass::ReshapeOptimizations>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto pattern = ov::op::v0::Constant::create(element::i64, {5}, {0, 0, 0, -1, 8});

        auto reshape = make_shared<v1::Reshape>(data, pattern, true);

        model_ref = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, LastDimSplitDymanicLast) {
    // [A, B, C, D] -> [A, B, C, 8, D/8]

    auto shape = PartialShape::dynamic(4);
    set_shape_symbols(shape);  // we set unique symbols to the shape: A, B, C, D

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto shape_of = make_shared<v3::ShapeOf>(data);
        auto indices = ov::op::v0::Constant::create(element::i64, {3}, {0, 1, 2});
        auto axis = ov::op::v0::Constant::create(element::i64, {}, {0});

        auto as_is_dims = make_shared<v1::Gather>(shape_of, indices, axis);
        auto splited_dim = ov::op::v0::Constant::create(element::i64, {2}, {8, -1});

        auto pattern = make_shared<v0::Concat>(OutputVector{as_is_dims, splited_dim}, 0);

        auto reshape = make_shared<v1::Reshape>(data, pattern, false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<pass::ReshapeOptimizations>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto pattern = ov::op::v0::Constant::create(element::i64, {5}, {0, 0, 0, 8, -1});

        auto reshape = make_shared<v1::Reshape>(data, pattern, true);

        model_ref = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeTest) {
    // [A, B, C, D] -> [A, B, C, D/2, D/3, 6]
    auto shape = PartialShape::dynamic(4);
    set_shape_symbols(shape);  // we set unique symbols to the shape: A, B, C, D

    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto shape_of = make_shared<v3::ShapeOf>(data);
        auto indices = ov::op::v0::Constant::create(element::i64, {3}, {0, 1, 2});
        auto axis = ov::op::v0::Constant::create(element::i64, {}, {0});
        auto as_is_dims = make_shared<v1::Gather>(shape_of, indices, axis);

        auto D = make_shared<v1::Gather>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {3}), axis);
        auto D_2 = make_shared<v1::Divide>(D, ov::op::v0::Constant::create(element::i64, {}, {2}));
        auto D_3 = make_shared<v1::Divide>(D, ov::op::v0::Constant::create(element::i64, {}, {3}));
        auto six = ov::op::v0::Constant::create(element::i64, {1}, {6});

        auto pattern = make_shared<v0::Concat>(OutputVector{as_is_dims, D_2, D_3, six}, 0);

        auto reshape = make_shared<v1::Reshape>(data, pattern, false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<pass::ReshapeOptimizations>();
    }
}

TEST_F(TransformationTestsF, ZeroDimsInOutputShape) {
    // [A, B]
    auto shape = PartialShape{0, 0};
    {
        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto b = make_shared<v0::Parameter>(element::f32, PartialShape{-1});

        auto a = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});

        auto shape_of = make_shared<v3::ShapeOf>(b);
        auto indices = ov::op::v0::Constant::create(element::i64, {1}, {0});
        auto axis = ov::op::v0::Constant::create(element::i64, {}, {0});
        auto b_dim = make_shared<v1::Gather>(shape_of, indices, axis);

        auto pattern = make_shared<v0::Concat>(OutputVector{a, b_dim}, 0);

        auto reshape = make_shared<v1::Reshape>(data, pattern, false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{data, b});
        manager.register_pass<pass::ReshapeOptimizations>();
    }
}
