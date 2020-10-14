// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertPadToConv) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        f = std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});

        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node) != nullptr;
        };

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertPadToGroupConvolution>();
        manager.set_callback(transformations_callback);
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto weights = opset4::Constant::create(element::f32, Shape{3, 1, 1, 1, 1}, {1});
        Strides stride{1, 1};
        CoordinateDiff pad_begin{1, 0}, pad_end{0, 1};
        auto conv = std::make_shared<opset4::GroupConvolution>(input, weights, stride, pad_begin, pad_end, stride);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertPadToConvNeg1) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {1, 0, 1, 0}); // Batch dim padding
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return !!std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node);
    };

    std::shared_ptr<Function> f(get_function()), f_ref(get_function());
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
    manager.set_callback(transformations_callback);
    manager.run_passes(f);

    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertPadToConvNeg2) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 1, 0, 1}); // Channel dim padding
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return !!std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node);
    };

    std::shared_ptr<Function> f(get_function()), f_ref(get_function());
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
    manager.set_callback(transformations_callback);
    manager.run_passes(f);

    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertPadToConvNeg3) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::SYMMETRIC; // Unsupported mode
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return !!std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node);
    };

    std::shared_ptr<Function> f(get_function()), f_ref(get_function());
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
    manager.set_callback(transformations_callback);
    manager.run_passes(f);

    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, ConvertPadToConvNeg4) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {1.}); // Unsupported value
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            return !!std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node);
    };

    std::shared_ptr<Function> f(get_function()), f_ref(get_function());
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
    manager.set_callback(transformations_callback);
    manager.run_passes(f);

    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}