// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/dimensions_tracking.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST(TransformationTests, BatchForRegularConvNetwork) {
    const auto& data = std::make_shared<opset6::Parameter>(element::f32, Shape{4, 1, 10, 10});

    const auto& order = std::make_shared<opset6::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    const auto& transpose = std::make_shared<opset6::Transpose>(data, order);

    const auto& filters = std::make_shared<opset6::Constant>(element::f32, Shape{1, 4, 3, 3});
    const auto& conv = std::make_shared<opset6::Convolution>(
            transpose, filters, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    const auto& f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::FindBatch>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    ASSERT_TRUE(data->get_partial_shape()[0].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[1].get_name() == "BATCH_DIM_1") << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[2].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[3].get_name().empty()) << data->get_partial_shape();
}

TEST(TransformationTests, BatchForMulNetwork) {
    const auto& data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});

    const auto& constant = std::make_shared<opset6::Constant>(element::f32, Shape{1, 4, 1, 1});
    const auto& mul = std::make_shared<opset6::Multiply>(data, constant);

    const auto& f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::FindBatch>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    ASSERT_TRUE(data->get_partial_shape()[0].get_name() == "BATCH_DIM_0") << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[1].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[2].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[3].get_name().empty()) << data->get_partial_shape();
}

TEST(TransformationTests, BatchForSplittedNetwork_0) {
    const auto& data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 1, 10, 10});

    const auto& order = std::make_shared<opset6::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    const auto& transpose = std::make_shared<opset6::Transpose>(data, order);

    const auto& filters = std::make_shared<opset6::Constant>(element::f32, Shape{1, 1, 3, 3});
    const auto& conv = std::make_shared<opset6::Convolution>(
            data, filters, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    const auto& f = std::make_shared<Function>(NodeVector{conv, transpose}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::FindBatch>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    ASSERT_TRUE(data->get_partial_shape()[0].get_name() == "BATCH_DIM_0") << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[1].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[2].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[3].get_name().empty()) << data->get_partial_shape();
}

TEST(TransformationTests, BatchForSplittedNetwork_1) {
    const auto& data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 1, 10, 10});

    const auto& filters = std::make_shared<opset6::Constant>(element::f32, Shape{1, 1, 3, 3});
    const auto& conv = std::make_shared<opset6::Convolution>(
            data, filters, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    const auto& order = std::make_shared<opset6::Constant>(element::i64, Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    const auto& transpose = std::make_shared<opset6::Transpose>(data, order);

    const auto& f = std::make_shared<Function>(NodeVector{transpose, conv}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::FindBatch>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    ASSERT_TRUE(data->get_partial_shape()[0].get_name() == "BATCH_DIM_0") << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[1].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[2].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[3].get_name().empty()) << data->get_partial_shape();
}

TEST(TransformationTests, BatchForSplittedNetwork_2) {
    const auto& data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});

    const auto& constant_0 = std::make_shared<opset6::Constant>(element::f32, Shape{1, 1, 1, 1});
    const auto& mul_0 = std::make_shared<opset6::Multiply>(data, constant_0);

    const auto& constant_1 = std::make_shared<opset6::Constant>(element::f32, Shape{1, 1, 1, 1});
    const auto& mul_1 = std::make_shared<opset6::Multiply>(data, constant_1);

    const auto& filters = std::make_shared<opset6::Constant>(element::f32, Shape{1, 4, 1, 1});
    const auto& conv = std::make_shared<opset6::Convolution>(
            mul_0, filters, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    const auto& concat = std::make_shared<opset6::Concat>(NodeVector{conv, mul_1}, 1);

    const auto& f = std::make_shared<Function>(NodeVector{concat}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::FindBatch>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    ASSERT_TRUE(data->get_partial_shape()[0].get_name() == "BATCH_DIM_0") << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[1].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[2].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[3].get_name().empty()) << data->get_partial_shape();
}

TEST(TransformationTests, BatchForTwoConvNetwork) {
    const auto& data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});

    const auto& filters = std::make_shared<opset6::Constant>(element::f32, Shape{1, 4, 3, 3});
    const auto& conv_0 = std::make_shared<opset6::Convolution>(
            data, filters, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    const auto& conv_1 = std::make_shared<opset6::Convolution>(
            data, filters, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    const auto& f = std::make_shared<Function>(NodeVector{conv_0, conv_1}, ParameterVector{data});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::FindBatch>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    ASSERT_TRUE(data->get_partial_shape()[0].get_name() == "BATCH_DIM_0") << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[1].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[2].get_name().empty()) << data->get_partial_shape();
    ASSERT_TRUE(data->get_partial_shape()[3].get_name().empty()) << data->get_partial_shape();
}
