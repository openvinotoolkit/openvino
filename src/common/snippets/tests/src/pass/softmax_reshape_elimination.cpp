// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/softmax_reshape_elimination.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, SoftmaxV1ReshapeElimination) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{6, 240});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v1::Softmax>(reshape0, 1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{2, 3, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        f = std::make_shared<Function>(NodeVector{reshape1}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::SoftmaxReshapeElimination>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3, 240});
        auto softmax_v1 = std::make_shared<ov::op::v1::Softmax>(data, 2);
        f_ref = std::make_shared<Function>(NodeVector{softmax_v1}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SoftmaxV8ReshapeElimination) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{680, 240});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(reshape0, -1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 2, 340, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        f = std::make_shared<Function>(NodeVector{reshape1}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::SoftmaxReshapeElimination>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(data, 3);
        f_ref = std::make_shared<Function>(NodeVector{softmax_v1}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SoftmaxReshapeElimination_IncorrectReshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{2, 81600});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(reshape0, -1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 2, 340, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        f = std::make_shared<Function>(NodeVector{reshape1}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::SoftmaxReshapeElimination>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 340, 240});
        auto shape0 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{2, 81600});
        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(data, shape0, false);
        auto softmax_v1 = std::make_shared<ov::op::v8::Softmax>(reshape0, -1);
        auto shape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{1, 2, 340, 240});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax_v1, shape1, false);
        f_ref = std::make_shared<Function>(NodeVector{reshape1}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
