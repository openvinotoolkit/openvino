// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <transformations/cpu_opset/arm/pass/convert_group_conv1d.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset1_decl.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(ov::Shape param_shape, ov::Shape weights_shape) {
        auto type = ov::element::f32;
        auto param = std::make_shared<ov::opset1::Parameter>(type, param_shape);
        auto weights = ov::opset1::Constant::create(type, weights_shape, {1});
        bool is1Dinput = param_shape.size() == 3;
        auto conv = std::make_shared<T>(param,
                                        weights,
                                        is1Dinput ? ov::Strides{1} :        ov::Strides{1, 1},
                                        is1Dinput ? ov::CoordinateDiff{0} : ov::CoordinateDiff{0, 0},
                                        is1Dinput ? ov::CoordinateDiff{0} : ov::CoordinateDiff{0, 0},
                                        is1Dinput ? ov::Strides{1} :        ov::Strides{1, 1});

        return std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{param});
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
        auto weights = ov::opset1::Constant::create(type, weights_shape, {1});
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
        return std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{param});
}

class ConvertConv1DTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ConvertConv1D>();
    }
};

class ConvertGroupConv1DTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ConvertGroupConv1D>();
    }
};

TEST_F(ConvertConv1DTests, Applied_1DShapes) {
    model = createInitGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7}, ov::Shape{30, 64, 1});
    model_ref = createTransformedGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7}, ov::Shape{30, 64, 1});
}

TEST_F(ConvertConv1DTests, Negative_2DShapes) {
    model = createInitGraph<ov::opset1::Convolution>(ov::Shape{2, 64, 7, 1}, ov::Shape{30, 64, 1, 1});
}

TEST_F(ConvertGroupConv1DTests, Applied_1DShapes) {
    model = createInitGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64}, ov::Shape{4, 1, 3, 5});
    model_ref = createTransformedGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64}, ov::Shape{4, 1, 3, 5});
}

TEST_F(ConvertGroupConv1DTests, Negative_2DShapes) {
    model = createInitGraph<ov::opset1::GroupConvolution>(ov::Shape{1, 12, 64, 1}, ov::Shape{4, 1, 3, 5, 1});
}
