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
    // With dynamic rank, we cannot verify that concat inputs have size 1 along the concat axis,
    // so the transformation should NOT be applied (model stays unchanged).
    // This is necessary to avoid incorrect results when inputs have size > 1 along concat axis.
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
    // model_ref is not set, so the test expects the model to remain unchanged
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
    // With dynamic rank, we cannot verify that concat inputs have size 1 along the concat axis,
    // so the transformation should NOT be applied (model stays unchanged).
    // This is necessary to avoid incorrect results when inputs have size > 1 along concat axis.
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

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // model_ref is not set, so the test expects the model to remain unchanged
}

// Test that transformation is NOT applied when concat inputs have different sizes
// along the concat axis, which would produce incorrect broadcast results.
TEST_F(TransformationTestsF, ConcatReduceMaxFusionDifferentSizesShouldNotApply) {
    // Concat([1], [53], axis=0) -> ReduceMax(axis=0) should produce scalar []
    // But Maximum([1], [53]) would broadcast to [53] - incorrect!
    // The transformation must NOT be applied in this case.
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{1});
        auto right_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{53});

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // model_ref is not set - transformation should NOT be applied, model stays unchanged
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDifferentSizesShouldNotApply) {
    // Same test for ReduceMin
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{1});
        auto right_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{53});

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // model_ref is not set - transformation should NOT be applied
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionSameLargerSizeShouldNotApply) {
    // Concat([5], [5], axis=0) -> ReduceMax(axis=0) should produce scalar []
    // Maximum([5], [5]) would produce [5] - incorrect!
    // The transformation must NOT be applied even when both inputs have the same size > 1.
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{5});
        auto right_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{5});

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // model_ref is not set - transformation should NOT be applied
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
