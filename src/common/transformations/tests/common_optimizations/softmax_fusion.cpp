// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/softmax_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/init_node_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"

using namespace testing;
using namespace ov;

class SoftmaxFusionFixture : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {};
class SoftmaxFusionSimplePatternFixture : public ::testing::TestWithParam<std::tuple<int64_t>> {};

TEST_P(SoftmaxFusionFixture, SoftmaxFusion) {
    Shape shape{1, 1, 256};
    auto params = GetParam();
    auto reduce_max_axis_val = std::get<0>(params);
    auto reduce_sum_axis_val = std::get<1>(params);
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto reduce_max_axis = op::v0::Constant::create(element::i64, Shape{}, {reduce_max_axis_val});
        auto reduce_max = std::make_shared<op::v1::ReduceMax>(data, reduce_max_axis);
        auto sub = std::make_shared<op::v1::Subtract>(data, reduce_max);
        auto exp = std::make_shared<op::v0::Exp>(sub);
        auto reduce_sum_axis = op::v0::Constant::create(element::i64, Shape{}, {reduce_sum_axis_val});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(exp, reduce_sum_axis);
        auto div = std::make_shared<op::v1::Divide>(exp, reduce_sum);
        f = std::make_shared<Model>(NodeVector{div}, ParameterVector{data});

        auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
        pass::Manager m;
        m.register_pass<ov::pass::InitUniqueNames>(unh);
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::SoftmaxFusion>();
        m.register_pass<ov::pass::CheckUniqueNames>(unh);
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        if (reduce_max_axis_val < 0)
            reduce_max_axis_val += shape.size();
        auto softmax = std::make_shared<op::v1::Softmax>(data, reduce_max_axis_val);
        f_ref = std::make_shared<Model>(NodeVector{softmax}, ParameterVector{data});
    }

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::PRECISIONS).enable(FunctionsComparator::NODES);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

INSTANTIATE_TEST_SUITE_P(SoftmaxFusionTests,
                         SoftmaxFusionFixture,
                         ::testing::Values(std::make_tuple(1, 1),
                                           std::make_tuple(1, -2),
                                           std::make_tuple(-1, -1),
                                           std::make_tuple(-1, 2),
                                           std::make_tuple(2, -1),
                                           std::make_tuple(2, 2)));

TEST_P(SoftmaxFusionSimplePatternFixture, SoftmaxFusionSimplePatternTest) {
    Shape shape{1, 3, 256, 256};
    auto params = GetParam();
    auto reduce_axis_val = std::get<0>(params);
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto exp = std::make_shared<op::v0::Exp>(data);
        auto reduce_axis = op::v0::Constant::create(element::i64, Shape{}, {reduce_axis_val});
        auto reduce_sum = std::make_shared<op::v1::ReduceSum>(exp, reduce_axis, true);
        auto div = std::make_shared<op::v1::Divide>(exp, reduce_sum);
        f = std::make_shared<Model>(NodeVector{div}, ParameterVector{data});

        auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
        pass::Manager m;
        m.register_pass<ov::pass::InitUniqueNames>(unh);
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::SoftmaxFusion>();
        m.register_pass<ov::pass::CheckUniqueNames>(unh);
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        if (reduce_axis_val < 0)
            reduce_axis_val += shape.size();
        auto softmax = std::make_shared<op::v1::Softmax>(data, reduce_axis_val);
        f_ref = std::make_shared<Model>(NodeVector{softmax}, ParameterVector{data});
    }

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::PRECISIONS).enable(FunctionsComparator::NODES);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

INSTANTIATE_TEST_SUITE_P(SoftmaxFusionSimplePatternTests,
                         SoftmaxFusionSimplePatternFixture,
                         ::testing::Values(std::make_tuple(0),
                                           std::make_tuple(1),
                                           std::make_tuple(2),
                                           std::make_tuple(-1),
                                           std::make_tuple(-2)));

class NegativeSoftmaxFusionFixture
    : public ::testing::TestWithParam<std::tuple<std::vector<int64_t>, std::vector<int64_t>>> {};

TEST_P(NegativeSoftmaxFusionFixture, NegativeSoftmaxFusion) {
    // ReduceMax arguments do not match conditions, therefore these nodes
    // are not included into final SoftMax node
    Shape shape{1, 1, 256};
    auto params = GetParam();
    auto reduce_max_axes_val = std::get<0>(params);
    auto reduce_sum_axes_val = std::get<1>(params);
    std::shared_ptr<Model> f(nullptr);

    auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto reduce_max_axes =
        op::v0::Constant::create(element::i64, Shape{reduce_max_axes_val.size()}, reduce_max_axes_val);
    auto reduce_max = std::make_shared<op::v1::ReduceMax>(data, reduce_max_axes);
    auto sub = std::make_shared<op::v1::Subtract>(data, reduce_max);
    auto exp = std::make_shared<op::v0::Exp>(sub);
    auto reduce_sum_axes =
        op::v0::Constant::create(element::i64, Shape{reduce_sum_axes_val.size()}, reduce_sum_axes_val);
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(exp, reduce_sum_axes);
    auto div = std::make_shared<op::v1::Divide>(exp, reduce_sum);
    f = std::make_shared<Model>(NodeVector{div}, ParameterVector{data});

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager m;
    m.register_pass<ov::pass::InitUniqueNames>(unh);
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::SoftmaxFusion>();
    m.register_pass<ov::pass::CheckUniqueNames>(unh);
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));
    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::v1::Subtract>(f), 1);
}

INSTANTIATE_TEST_SUITE_P(NegativeSoftmaxFusionTests,
                         NegativeSoftmaxFusionFixture,
                         ::testing::ValuesIn(std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>>>{
                             std::make_tuple<std::vector<int64_t>, std::vector<int64_t>>({2}, {1}),
                             std::make_tuple<std::vector<int64_t>, std::vector<int64_t>>({1}, {-1}),
                             std::make_tuple<std::vector<int64_t>, std::vector<int64_t>>({0, 1}, {0, 1})}));
