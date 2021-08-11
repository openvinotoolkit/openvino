// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/common_optimizations/softmax_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


class SoftmaxFusionFixture : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
};

TEST_P(SoftmaxFusionFixture, SoftmaxFusion) {
    Shape shape{1, 1, 256};
    auto params = GetParam();
    auto reduce_max_axis_val = std::get<0>(params);
    auto reduce_sum_axis_val = std::get<1>(params);
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, shape);
        auto reduce_max_axis = opset6::Constant::create(element::i64, Shape{}, {reduce_max_axis_val});
        auto reduce_max = std::make_shared<opset6::ReduceMax>(data, reduce_max_axis);
        auto sub = std::make_shared<opset6::Subtract>(data, reduce_max);
        auto exp = std::make_shared<opset6::Exp>(sub);
        auto reduce_sum_axis = opset6::Constant::create(element::i64, Shape{}, {reduce_sum_axis_val});
        auto reduce_sum = std::make_shared<opset6::ReduceSum>(exp, reduce_sum_axis);
        auto div = std::make_shared<opset6::Divide>(exp, reduce_sum);
        f = std::make_shared<Function>(NodeVector{div}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SoftmaxFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, shape);
        if (reduce_max_axis_val < 0)
            reduce_max_axis_val += shape.size();
        auto softmax = std::make_shared<opset6::Softmax>(data, reduce_max_axis_val);
        f_ref = std::make_shared<Function>(NodeVector{softmax}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(SoftmaxFusionTests, SoftmaxFusionFixture,
        ::testing::Values(
            std::make_tuple(1, 1),
            std::make_tuple(1, -2),
            std::make_tuple(-1, -1),
            std::make_tuple(-1, 2),
            std::make_tuple(2, -1),
            std::make_tuple(2, 2)
        )
);


class NegativeSoftmaxFusionFixture : public ::testing::TestWithParam<std::tuple<std::vector<int64_t>, std::vector<int64_t>>> {
};

TEST_P(NegativeSoftmaxFusionFixture, NegativeSoftmaxFusion) {
    Shape shape{1, 1, 256};
    auto params = GetParam();
    auto reduce_max_axes_val = std::get<0>(params);
    auto reduce_sum_axes_val = std::get<1>(params);
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, shape);
        auto reduce_max_axes = opset6::Constant::create(element::i64, Shape{reduce_max_axes_val.size()}, reduce_max_axes_val);
        auto reduce_max = std::make_shared<opset6::ReduceMax>(data, reduce_max_axes);
        auto sub = std::make_shared<opset6::Subtract>(data, reduce_max);
        auto exp = std::make_shared<opset6::Exp>(sub);
        auto reduce_sum_axes = opset6::Constant::create(element::i64, Shape{reduce_sum_axes_val.size()}, reduce_sum_axes_val);
        auto reduce_sum = std::make_shared<opset6::ReduceSum>(exp, reduce_sum_axes);
        auto div = std::make_shared<opset6::Divide>(exp, reduce_sum);
        f = std::make_shared<Function>(NodeVector{div}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SoftmaxFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, shape);
        auto reduce_max_axes = opset6::Constant::create(element::i64, Shape{reduce_max_axes_val.size()}, reduce_max_axes_val);
        auto reduce_max = std::make_shared<opset6::ReduceMax>(data, reduce_max_axes);
        auto sub = std::make_shared<opset6::Subtract>(data, reduce_max);
        auto exp = std::make_shared<opset6::Exp>(sub);
        auto reduce_sum_axes = opset6::Constant::create(element::i64, Shape{reduce_sum_axes_val.size()}, reduce_sum_axes_val);
        auto reduce_sum = std::make_shared<opset6::ReduceSum>(exp, reduce_sum_axes);
        auto div = std::make_shared<opset6::Divide>(exp, reduce_sum);
        f_ref = std::make_shared<Function>(NodeVector{div}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(NegativeSoftmaxFusionTests, NegativeSoftmaxFusionFixture,
        ::testing::ValuesIn(std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>>>{
                                std::make_tuple<std::vector<int64_t>, std::vector<int64_t>>({2}, {1}),
                                std::make_tuple<std::vector<int64_t>, std::vector<int64_t>>({1}, {-1}),
                                std::make_tuple<std::vector<int64_t>, std::vector<int64_t>>({0, 1}, {0, 1})
                            }
        )
);
