// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset7.hpp>
#include <transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(ov::Shape param_shape, ov::Shape weights_shape) {
        auto type = ov::element::f32;
        auto param = std::make_shared<ov::opset1::Parameter>(type, param_shape);
        auto weights = ov::opset1::Constant::create(type, weights_shape, { 1 });
        bool is1Dinput = param_shape.size() == 3;
        auto conv = std::make_shared<T>(param,
                                        weights,
                                        is1Dinput ? ov::Strides{1} :        ov::Strides{1, 1},
                                        is1Dinput ? ov::CoordinateDiff{0} : ov::CoordinateDiff{0, 0},
                                        is1Dinput ? ov::CoordinateDiff{0} : ov::CoordinateDiff{0, 0},
                                        is1Dinput ? ov::Strides{1} :        ov::Strides{1, 1});

        return std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ param });
}

template <class T>
static std::shared_ptr<ov::Model> createTransformedGraph(ov::Shape param_shape, ov::Shape weights_shape) {
        auto getUnsqueeze = [&](const ov::Output<ov::Node>& node) {
            auto rank = node.get_partial_shape().rank().get_length();
            return std::make_shared<ov::op::v0::Unsqueeze>(node,
                                                           ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {rank}));
        };
        auto type = ov::element::f32;
        auto param = std::make_shared<ov::opset1::Parameter>(type, param_shape);
        auto weights = ov::opset1::Constant::create(type, weights_shape, { 1 });
        auto input2d = getUnsqueeze(param);
        auto weights2d = getUnsqueeze(weights);
        auto conv2d = std::make_shared<T>(input2d,
                                          weights2d,
                                          ov::Strides{1, 1},
                                          ov::CoordinateDiff{0, 0},
                                          ov::CoordinateDiff{0, 0},
                                          ov::Strides{1, 1});

        auto reshape = std::make_shared<ov::opset1::Squeeze>(conv2d,
            ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {3}));
        return std::make_shared<ov::Model>(ov::NodeVector{ reshape }, ov::ParameterVector{ param });
}

TEST(TransformationTests, CheckConvertConv1DIsAppliedFor1DShapes) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        model = createInitGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7}, ov::Shape{ 30, 64, 1 });
        ov::pass::Manager manager;
        manager.register_pass<ConvertConv1D>();
        manager.run_passes(model);
    }
    {
        model_ref = createTransformedGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7}, ov::Shape{30, 64, 1});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertConv1DIsNotAppliedFor2DShapes) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        model = createInitGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7, 1}, ov::Shape{30, 64, 1, 1});
        ov::pass::Manager manager;
        manager.register_pass<ConvertConv1D>();
        manager.run_passes(model);
    }
    {
        model_ref = createInitGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7, 1}, ov::Shape{30, 64, 1, 1});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConv1DIsAppliedFor1dShapes) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        model = createInitGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64}, ov::Shape{4, 1, 3, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConv1D>();
        manager.run_passes(model);
    }
    {
        model_ref = createTransformedGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64}, ov::Shape{4, 1, 3, 5});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConv1DIsNotAppliedFor2DShapes) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        model = createInitGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64, 1}, ov::Shape{4, 1, 3, 5, 1});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConv1D>();
        manager.run_passes(model);
    }
    {
        model_ref = createInitGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64, 1}, ov::Shape{4, 1, 3, 5, 1});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}
