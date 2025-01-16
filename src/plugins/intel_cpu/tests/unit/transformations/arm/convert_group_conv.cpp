// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset7.hpp>
#include <transformations/cpu_opset/arm/pass/convert_group_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ov::opset1::Parameter> param, ov::Shape weights_shape) {
        auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, { 1 });
        auto conv = std::make_shared<T>(param,
                                        weights,
                                        ov::Strides{1},
                                        ov::CoordinateDiff{0},
                                        ov::CoordinateDiff{0},
                                        ov::Strides{1});

        return std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ param });
}

TEST(TransformationTests, CheckConvertGroupConvIsApplied) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 6, 224});
        model = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 3, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConvolution>();
        manager.run_passes(model);
    }
    {
        const unsigned int groups = 2;
        const unsigned int channel_axis = 1;
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 6, 224});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{groups, 1, 3, 5}, { 1 });
        auto split_weights = std::make_shared<ov::opset1::Split>(weights,
                                                                     ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                                                     groups);
        auto axis  = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {channel_axis});
        auto split = std::make_shared<ov::opset1::Split>(param, axis, groups);
        ov::NodeVector concat_inputs;
        for (size_t g = 0; g < groups; g++) {
            auto out = split->output(g);
            auto filter = std::make_shared<ov::opset1::Squeeze>(split_weights->output(g),
                                                                    ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {0}));
            auto conv = std::make_shared<ov::opset1::Convolution>(out,
                                                                      filter,
                                                                      ov::Strides{1},
                                                                      ov::CoordinateDiff{0},
                                                                      ov::CoordinateDiff{0},
                                                                      ov::Strides{1});
            concat_inputs.push_back(conv);
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 1);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ concat }, ov::ParameterVector{ param });
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConvIsNotAppliedForDepthwiseCase) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 224});
        model = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 1, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConvolution>();
        manager.run_passes(model);
    }
    {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 224});
        model_ref = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 1, 5});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CheckConvertGroupConvIsNotAppliedForDynamicShapes) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 224});
        model = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 1, 5});
        ov::pass::Manager manager;
        manager.register_pass<ConvertGroupConvolution>();
        manager.run_passes(model);
    }
    {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 224});
        model_ref = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 1, 5});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}