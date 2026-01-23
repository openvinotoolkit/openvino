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

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
TEST_F(TransformationTestsF, ConcatReduceMaxFusionDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<v1::Maximum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionKeepDimsDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}), true);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto maximum = std::make_shared<v1::Maximum>(left_unsqueeze, right_unsqueeze);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<v1::Maximum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<v1::Minimum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<v1::Minimum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseStaticShape) {
    PartialShape shape{224, 224, 1, 1};
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<v0::Squeeze>(add, v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::PullSqueezeThroughEltwise>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {0}));
        auto left_squeeze =
            std::make_shared<v0::Squeeze>(left_unsqueeze, v0::Constant::create(element::i64, Shape{}, {0}));

        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_squeeze =
            std::make_shared<v0::Squeeze>(right_unsqueeze, v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<v1::Add>(left_squeeze, right_squeeze);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationStaticShape) {
    PartialShape shape{224, 224, 1, 1};
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<v0::Squeeze>(add, v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto add = std::make_shared<v1::Add>(left_input, right_input);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<v0::Squeeze>(add, v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto add = std::make_shared<v1::Add>(left_input, right_input);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<v0::Unsqueeze>(left_input, v0::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<v0::Unsqueeze>(right_input, v0::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<v1::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<v0::Squeeze>(add, v0::Constant::create(element::i64, Shape{}, {0}));

        model = std::make_shared<Model>(OutputVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        auto add = std::make_shared<v1::Add>(left_input, right_input);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{left_input, right_input});
    }
}

// ConcatReduceFusion should NOT apply when Concat inputs have different shapes.
// When inputs have shapes [2] and [1], reducing along concat axis should produce scalar [],
// but the incorrect transformation would produce Minimum([2], [1]) -> [2] (broadcast) -> Squeeze -> [2],
// which breaks downstream operations expecting scalar output.
TEST_F(TransformationTestsF, ConcatReduceMinFusionDifferentInputShapes_NotApplied) {
    // This test verifies that the transformation is NOT applied when Concat inputs have different shapes.
    // The bug was that the transformation would incorrectly replace:
    //   Concat([2], [1]) -> ReduceMin(axis=0, keepdims=false) -> scalar []
    // With:
    //   Minimum([2], [1]) -> Squeeze(axis=0) -> [2]  (WRONG! should be scalar)
    PartialShape left_shape{2};
    PartialShape right_shape{1};
    std::int64_t reduce_axis = 0;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, left_shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, right_shape);

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, reduce_axis);

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}), false);

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // model_ref is not set - transformation should NOT be applied, so model should remain unchanged
}
