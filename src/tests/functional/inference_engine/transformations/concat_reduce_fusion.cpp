// Copyright (C) 2021 Intel Corporation
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

using ConcatReduceParams = std::tuple<ov::PartialShape, int64_t>;
class ConcatReduceFixture : public ::testing::TestWithParam<ConcatReduceParams> {
};

TEST_P(ConcatReduceFixture, ConcatReduceMaxFusion) {
    PartialShape shape;
    int64_t reduce_axis;
    std::tie(shape, reduce_axis) = GetParam();
    std::shared_ptr<Model> f(nullptr);
    {
        auto left_input = std::make_shared<ov::opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze = std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze = std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_max = std::make_shared<opset8::ReduceMax>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        f = std::make_shared<Model>(NodeVector{reduce_max}, ParameterVector{left_input, right_input});
    }
    pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ngraph::pass::ConcatReduceFusion>();
    m.register_pass<ngraph::pass::NopElimination>();
    m.register_pass<pass::ConstantFolding>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<Model> f_ref(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto maximum = std::make_shared<opset8::Maximum>(left_input, right_input);
        f_ref = std::make_shared<Model>(NodeVector{maximum}, ParameterVector{left_input, right_input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST_P(ConcatReduceFixture, ConcatReduceMinFusion) {
    PartialShape shape;
    int64_t reduce_axis;
    std::tie(shape, reduce_axis) = GetParam();
    std::shared_ptr<Model> f(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze = std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));
        auto right_unsqueeze = std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        auto concat = std::make_shared<opset8::Concat>(NodeVector{left_unsqueeze, right_unsqueeze}, reduce_axis);

        auto reduce_min = std::make_shared<opset8::ReduceMin>(concat, opset8::Constant::create(element::i64, Shape{}, {reduce_axis}));

        f = std::make_shared<Model>(NodeVector{reduce_min}, ParameterVector{left_input, right_input});
    }
    pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ngraph::pass::ConcatReduceFusion>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<Model> f_ref(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto minimum = std::make_shared<opset8::Minimum                                                          >(left_input, right_input);
        f_ref = std::make_shared<Model>(NodeVector{minimum}, ParameterVector{left_input, right_input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

class PullSqueezeThroughEltwiseTest : public ::testing::Test {
};

TEST(PullSqueezeThroughEltwiseTest, Simple) {
    Shape shape{224, 224, 1, 1};
    std::shared_ptr<Model> f(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze = std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze = std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<opset8::Squeeze>(add, opset8::Constant::create(element::i64, Shape{}, {0}));

        f = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{left_input, right_input});
    }

    pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ngraph::pass::PullSqueezeThroughEltwise>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<Model> f_ref(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze = std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto left_squeeze = std::make_shared<opset8::Squeeze>(left_unsqueeze, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto right_unsqueeze = std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_squeeze = std::make_shared<opset8::Squeeze>(right_unsqueeze, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_squeeze, right_squeeze);

        f_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{left_input, right_input});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(PullSqueezeThroughEltwiseTest, SimpleSqueezeFolded) {
    Shape shape{224, 224, 1, 1};
    std::shared_ptr<Model> f(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto left_unsqueeze = std::make_shared<opset8::Unsqueeze>(left_input, opset8::Constant::create(element::i64, Shape{}, {0}));
        auto right_unsqueeze = std::make_shared<opset8::Unsqueeze>(right_input, opset8::Constant::create(element::i64, Shape{}, {0}));

        auto add = std::make_shared<opset8::Add>(left_unsqueeze, right_unsqueeze);

        auto squeeze = std::make_shared<opset8::Squeeze>(add, opset8::Constant::create(element::i64, Shape{}, {0}));

        f = std::make_shared<Model>(NodeVector{squeeze}, ParameterVector{left_input, right_input});
    }

    pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ngraph::pass::PullSqueezeThroughEltwise>();
    m.register_pass<ngraph::pass::NopElimination>();
    m.register_pass<pass::ConstantFolding>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    std::shared_ptr<Model> f_ref(nullptr);
    {
        auto left_input = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto right_input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto add = std::make_shared<opset8::Add>(left_input, right_input);

        f_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{left_input, right_input});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<int64_t> axes = {2, 3};
const std::vector<ov::PartialShape> inputShapes = {
    {224, 224, 1, 1},
    {224, 224, -1, -1},
    PartialShape::dynamic()
};

const auto concatReduceParams = ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(axes));
INSTANTIATE_TEST_SUITE_P(ConcatReduceFusionTests, ConcatReduceFixture, concatReduceParams);
