// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, ShuffleChannelsFusion) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{1, 4, 24, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 2, 1, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{1, 96, 3, 3}), false);
        f = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ShuffleChannelsFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto sc = std::make_shared<opset7::ShuffleChannels>(data, 1, 4);
        f_ref = std::make_shared<Function>(NodeVector{sc}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeShuffleChannelsFusionInvalidTranspose) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{1, 4, 24, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 1, 2, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{1, 96, 3, 3}), false);
        f = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ShuffleChannelsFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{1, 4, 24, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 1, 2, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{1, 96, 3, 3}), false);
        f_ref = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeShuffleChannelsFusionInvalidFirstReshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{4, 24, 1, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 2, 1, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{1, 96, 3, 3}), false);
        f = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ShuffleChannelsFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{4, 24, 1, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 2, 1, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{1, 96, 3, 3}), false);
        f_ref = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeShuffleChannelsFusionInvalidSecondReshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{1, 4, 24, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 2, 1, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{4, 24, 3, 3}), false);
        f = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ShuffleChannelsFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 96, 3, 3});
        auto reshape1 = std::make_shared<opset7::Reshape>(data, op::Constant::create(element::i64, Shape{5}, Shape{1, 4, 24, 3, 3}), false);
        auto transpose = std::make_shared<opset7::Transpose>(reshape1, op::Constant::create(element::i64, Shape{5}, Shape{0, 2, 1, 3, 4}));
        auto reshape2 = std::make_shared<opset7::Reshape>(transpose, op::Constant::create(element::i64, Shape{4}, Shape{4, 24, 3, 3}), false);
        f_ref = std::make_shared<Function>(NodeVector{reshape2}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
