// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/concat_reduce_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConcatReduceMaxFusionDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<opset8::ReduceMax>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        function = std::make_shared<Model>(NodeVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<opset8::Maximum>(left_input, right_input);
        function_ref = std::make_shared<Model>(NodeVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionKeepDimsDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<opset8::ReduceMax>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}), true);

        function = std::make_shared<Model>(NodeVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto maximum = std::make_shared<opset8::Maximum>(left_unsqueeze, right_unsqueeze);
        function_ref = std::make_shared<Model>(NodeVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMaxFusionDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<opset8::ReduceMax>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        function = std::make_shared<Model>(NodeVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<opset8::Maximum>(left_input, right_input);
        function_ref = std::make_shared<Model>(NodeVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    std::int64_t reduce_axis = 2;
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<opset8::ReduceMin>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        function = std::make_shared<Model>(NodeVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<opset8::Minimum>(left_input, right_input);
        function_ref = std::make_shared<Model>(NodeVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, ConcatReduceMinFusionDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    std::int64_t reduce_axis = 3;
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input,
                                                opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max =
            std::make_shared<opset8::ReduceMin>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        function = std::make_shared<Model>(NodeVector{reduce_max}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<opset8::Minimum>(left_input, right_input);
        function_ref = std::make_shared<Model>(NodeVector{maximum}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseStaticShape) {
    PartialShape shape{224, 224, 1, 1};
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<opset8::Squeeze>(add, opset8::Constant::create(element::i64, Shape{}, {0}));

        function = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::PullSqueezeThroughEltwise>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto left_squeeze =
            std::make_shared<opset8::Squeeze>(left_unsqueeze, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_squeeze =
            std::make_shared<opset8::Squeeze>(right_unsqueeze, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_squeeze, right_squeeze);

        function_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationStaticShape) {
    PartialShape shape{224, 224, 1, 1};
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<opset8::Squeeze>(add, opset8::Constant::create(element::i64, Shape{}, {0}));

        function = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto add = std::make_shared<opset8::Add>(left_input, right_input);

        function_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationDynamicShape) {
    PartialShape shape{-1, -1, -1, -1};
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<opset8::Squeeze>(add, opset8::Constant::create(element::i64, Shape{}, {0}));

        function = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto add = std::make_shared<opset8::Add>(left_input, right_input);

        function_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{left_input, right_input});
    }
}

TEST_F(TransformationTestsF, PullSqueezeThroughEltwiseSqueezeEliminationDynamicRank) {
    PartialShape shape = PartialShape::dynamic();
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze =
            std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<opset8::Squeeze>(add, opset8::Constant::create(element::i64, Shape{}, {0}));

        function = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{left_input, right_input});
        manager.register_pass<ngraph::pass::ConcatReduceFusion>();
    }
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto add = std::make_shared<opset8::Add>(left_input, right_input);

        function_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{left_input, right_input});
    }
}
