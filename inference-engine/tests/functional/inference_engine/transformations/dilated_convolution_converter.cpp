// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST(TransformationTests, DilatedConvolutionConverter) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch, filters,
                Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto batch_to_space = std::make_shared<opset6::BatchToSpace>(conv,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::DilatedConvolutionConverter>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 3, 3});
        auto conv = std::make_shared<opset6::Convolution>(data, filters,
                Strides{1, 1}, CoordinateDiff{1, 1}, CoordinateDiff{-1, -2}, Strides{2, 2}, op::PadType::EXPLICIT);
        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NegativeDilatedConvolutionConverterNonZeroPadsForNC) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 5, 3, 3});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch, filters,
                Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto batch_to_space = std::make_shared<opset6::BatchToSpace>(conv,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 3}));
        f = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::DilatedConvolutionConverter>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto filters = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 5, 3, 3});
        auto space_to_batch = std::make_shared<opset6::SpaceToBatch>(data,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));
        auto conv = std::make_shared<opset6::Convolution>(space_to_batch, filters,
                Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto batch_to_space = std::make_shared<opset6::BatchToSpace>(conv,
                op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0}),
                op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 3}));
        f_ref = std::make_shared<Function>(NodeVector{batch_to_space}, ParameterVector{data, filters});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
