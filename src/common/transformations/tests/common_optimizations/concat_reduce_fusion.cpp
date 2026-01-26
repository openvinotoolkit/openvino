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
    // With dynamic rank inputs, Unsqueeze output is also dynamic rank,
    // so the transformation cannot verify concat dim sizes and is correctly skipped.
    // model_ref not set -> framework expects model to remain unchanged.
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
    // With dynamic rank inputs, Unsqueeze output is also dynamic rank,
    // so the transformation cannot verify concat dim sizes and is correctly skipped.
    // model_ref not set -> framework expects model to remain unchanged.
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

// Test that transformation is NOT applied when concat inputs have different sizes
// along the concat axis (would cause shape mismatch due to Maximum/Minimum broadcasting)
TEST_F(TransformationTestsF, ConcatReduceMaxFusionSkippedForDifferentConcatDimSizes) {
    // This test verifies that ReplaceConcatReduceByMinOrMax is NOT applied when
    // concat inputs have different sizes along the concat axis.
    // In this case, replacing Concat+ReduceMax with Maximum would change the output shape
    // due to numpy broadcasting semantics.
    {
        // Create concat with inputs [7] and [1] along axis 0 -> output [8]
        // ReduceMax along axis 0 -> output [] (scalar)
        // If transformed to Maximum([7], [1]), would broadcast to [7] - wrong shape!
        auto left_input = v0::Constant::create(element::f32, Shape{7}, std::vector<float>(7, 1.0f));
        auto right_input = v0::Constant::create(element::f32, Shape{1}, {2.0f});

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{});
        manager.register_pass<ov::pass::ReplaceConcatReduceByMinOrMax>();
    }
    {
        // Model should remain unchanged - no transformation applied
        auto left_input = v0::Constant::create(element::f32, Shape{7}, std::vector<float>(7, 1.0f));
        auto right_input = v0::Constant::create(element::f32, Shape{1}, {2.0f});

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model_ref = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionSkippedForDifferentConcatDimSizes) {
    // Same test but for ReduceMin
    {
        auto left_input = v0::Constant::create(element::f32, Shape{5}, std::vector<float>(5, 3.0f));
        auto right_input = v0::Constant::create(element::f32, Shape{3}, std::vector<float>(3, 1.0f));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{});
        manager.register_pass<ov::pass::ReplaceConcatReduceByMinOrMax>();
    }
    {
        // Model should remain unchanged
        auto left_input = v0::Constant::create(element::f32, Shape{5}, std::vector<float>(5, 3.0f));
        auto right_input = v0::Constant::create(element::f32, Shape{3}, std::vector<float>(3, 1.0f));

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, 0);

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {0}), false);

        model_ref = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{});
    }
}

// Test that transformation is NOT applied when concat dimension is dynamic
// (static rank but unknown size along the concat axis)
TEST_F(TransformationTestsF, ConcatReduceMaxFusionSkippedForDynamicConcatDim) {
    std::int64_t reduce_axis = 0;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});
        auto right_input = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}), false);

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ReplaceConcatReduceByMinOrMax>();
    }
    // model_ref not set -> framework expects model to remain unchanged.
}
