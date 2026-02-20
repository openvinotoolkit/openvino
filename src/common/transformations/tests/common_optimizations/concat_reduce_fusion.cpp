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
#include "openvino/op/reshape.hpp"
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
    // With dynamic rank, transformation should NOT apply - model remains unchanged
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
    // With dynamic rank, transformation should NOT apply - model remains unchanged
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

// Test: Right input has dimension > 1 along concat axis - no fusion should occur
TEST_F(TransformationTestsF, ConcatReduceNoFusionNoBroadcast) {
    PartialShape left_shape{8, 1, 5};
    PartialShape right_shape{8, 4, 5};
    std::int64_t concat_axis = 1;
    std::int64_t reduce_axis = 1;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f64, left_shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f64, right_shape);

        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, concat_axis);
        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}), false);

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // Reference model should remain unchanged - no fusion should occur due to right input dim=4
}

// Test: ReduceMin with keep_dims=false on inputs with singleton dimension
TEST_F(TransformationTestsF, ConcatReduceMinFusionNoUnsqueezeSingleDimStaticShape) {
    PartialShape shape{2, 5, 3, 1, 4};
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, reduce_axis);

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto minimum = std::make_shared<v1::Minimum>(left_input, right_input);
        // Keep dims is false - so Squeeze has been inserted after Minimum
        // Then Squeeze is replaced with Reshape by EliminateSqueeze pass
        auto reshape = std::make_shared<v1::Reshape>(
            minimum,
            v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{2, 5, 3, 4}),
            false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{left_input, right_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Test: ReduceMin with keep_dims=true on inputs with singleton dimension
TEST_F(TransformationTestsF, ConcatReduceMinFusionNoUnsqueezeSingleDimStaticShapeKeepDims) {
    PartialShape shape{2, 5, 3, 1, 4};
    std::int64_t reduce_axis = 3;
    bool keep_dims = true;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, reduce_axis);

        auto reduce_min = std::make_shared<v1::ReduceMin>(concat,
                                                          v0::Constant::create(element::i64, Shape{}, {reduce_axis}),
                                                          keep_dims);
        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto minimum = std::make_shared<v1::Minimum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{minimum}, ParameterVector{left_input, right_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Test: ReduceMax with keep_dims=false on inputs with singleton dimension
TEST_F(TransformationTestsF, ConcatReduceMaxFusionNoUnsqueezeSingleDimStaticShape) {
    PartialShape shape{2, 5, 3, 1, 4};
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, reduce_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {reduce_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<v1::Maximum>(left_input, right_input);
        // Keep dims is false - so Squeeze has been inserted after Maximum
        // Then Squeeze is replaced with Reshape by EliminateSqueeze pass
        auto reshape = std::make_shared<v1::Reshape>(
            maximum,
            v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{2, 5, 3, 4}),
            false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{left_input, right_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Test: ReduceMax with keep_dims=true on inputs with singleton dimension
TEST_F(TransformationTestsF, ConcatReduceMaxFusionNoUnsqueezeSingleDimStaticShapeKeepDims) {
    PartialShape shape{2, 5, 3, 1, 4};
    std::int64_t reduce_axis = 3;
    bool keep_dims = true;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, reduce_axis);

        auto reduce_max = std::make_shared<v1::ReduceMax>(concat,
                                                          v0::Constant::create(element::i64, Shape{}, {reduce_axis}),
                                                          keep_dims);
        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<v1::Maximum>(left_input, right_input);
        model_ref = std::make_shared<Model>(OutputVector{maximum}, ParameterVector{left_input, right_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Test: Static rank with dynamic dimension along concat axis - no fusion should occur
// This verifies that transformation is skipped when we cannot statically confirm dim=1
TEST_F(TransformationTestsF, ConcatReduceMaxNoFusionDynamicDimAlongConcatAxis) {
    // Shape has static rank but dynamic dimension at the concat axis
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t concat_axis = 2;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        // Direct concat without Unsqueeze - dimension along axis 2 is unknown
        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, concat_axis);

        auto reduce_max =
            std::make_shared<v1::ReduceMax>(concat, v0::Constant::create(element::i64, Shape{}, {concat_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // No model_ref - transformation should NOT apply because we cannot confirm dim=1
    // Model should remain unchanged
}

// Test: Static rank with dynamic dimension along concat axis for ReduceMin - no fusion should occur
TEST_F(TransformationTestsF, ConcatReduceMinNoFusionDynamicDimAlongConcatAxis) {
    // Shape has static rank but dynamic dimension at the concat axis
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t concat_axis = 2;
    {
        auto left_input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<v0::Parameter>(element::f32, shape);

        // Direct concat without Unsqueeze - dimension along axis 2 is unknown
        auto concat = std::make_shared<v0::Concat>(NodeVector{left_input, right_input}, concat_axis);

        auto reduce_min =
            std::make_shared<v1::ReduceMin>(concat, v0::Constant::create(element::i64, Shape{}, {concat_axis}));

        model = std::make_shared<Model>(OutputVector{reduce_min}, ParameterVector{left_input, right_input});
        manager.register_pass<ov::pass::ConcatReduceFusion>();
    }
    // No model_ref - transformation should NOT apply because we cannot confirm dim=1
    // Model should remain unchanged
}
