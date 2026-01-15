// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_reduce_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConcatReduceMaxFusionDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<ov::op::v1::ReduceMax>(concat,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<ov::op::v1::Maximum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionKeepDimsDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<ov::op::v1::ReduceMax>(concat,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}),
                                                    true);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto maximum = std::make_shared<ov::op::v1::Maximum>(left_unsqueeze, right_unsqueeze);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<ov::op::v1::ReduceMax>(concat,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<ov::op::v1::Maximum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<ov::op::v1::ReduceMin>(concat,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<ov::op::v1::Minimum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<ov::op::v1::ReduceMin>(concat,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<ov::op::v1::Minimum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseStaticShape) {
    PartialShape shape{224, 224, 1, 1};
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<ov::op::v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze =
            std::make_shared<ov::op::v0::Squeeze>(add, ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::PullSqueezeThroughEltwise>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto left_squeeze =
            std::make_shared<ov::op::v0::Squeeze>(left_unsqueeze,
                                                  ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_squeeze =
            std::make_shared<ov::op::v0::Squeeze>(right_unsqueeze,
                                                  ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<ov::op::v1::Add>(left_squeeze, right_squeeze);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationStaticShape) {
    PartialShape shape{224, 224, 1, 1};
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<ov::op::v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze =
            std::make_shared<ov::op::v0::Squeeze>(add, ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto add = std::make_shared<ov::op::v1::Add>(left_input, right_input);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<ov::op::v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze =
            std::make_shared<ov::op::v0::Squeeze>(add, ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto add = std::make_shared<ov::op::v1::Add>(left_input, right_input);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(left_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(right_input,
                                                    ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<ov::op::v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze =
            std::make_shared<ov::op::v0::Squeeze>(add, ov::op::v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);

        auto add = std::make_shared<ov::op::v1::Add>(left_input, right_input);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionDifferentShapes1D) {
    // Test case for GitHub issue #33374
    // Concat of 1D tensors with different shapes followed by ReduceMax should NOT be optimized
    // Input A: (21,), Input B: (1,), Concat -> (22,), ReduceMax(axis=0) -> scalar ()
    // The transformation should be skipped because inputs have different shapes
    PartialShape shape_a{21};
    PartialShape shape_b{1};
    std::int64_t concat_axis = 0;
    std::int64_t reduce_axis = 0;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(element::f64, shape_a);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(element::f64, shape_b);

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{input_a, input_b}, concat_axis);

        auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(
            concat,
            ov::op::v0::Constant::create(element::i64, Shape{1}, {reduce_axis}),
            false);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{input_a, input_b});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        // Model should remain unchanged because inputs have different shapes
        auto input_a = std::make_shared<ov::op::v0::Parameter>(element::f64, shape_a);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(element::f64, shape_b);

        auto concat = std::make_shared<ov::op::v0::Concat>(NodeVector{input_a, input_b}, concat_axis);

        auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(
            concat,
            ov::op::v0::Constant::create(element::i64, Shape{1}, {reduce_axis}),
            false);

        model_ref = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{input_a, input_b});
    }
}
