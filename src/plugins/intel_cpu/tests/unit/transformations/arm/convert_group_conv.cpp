// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <transformations/cpu_opset/arm/pass/convert_group_conv.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/opsets/opset1_decl.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ov::opset1::Parameter> param, ov::Shape weights_shape) {
        auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});
        auto conv = std::make_shared<T>(param,
                                        weights,
                                        ov::Strides{1},
                                        ov::CoordinateDiff{0},
                                        ov::CoordinateDiff{0},
                                        ov::Strides{1});

        return std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{param});
}

static std::shared_ptr<ov::Model> createRefGraph(std::shared_ptr<ov::opset1::Parameter> param, ov::Shape weights_shape) {
    const unsigned int groups = static_cast<unsigned int>(weights_shape[0]);
    const unsigned int channel_axis = 1;
    auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});
    auto split_weights = std::make_shared<ov::opset1::Split>(weights,
                                                             ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                                             groups);
    auto axis = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {channel_axis});
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
    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{param});
}

class ConvertGroupConvTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ConvertGroupConvolution>();
    }
};

TEST_F(ConvertGroupConvTests, Applied) {
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 6, 224});
    model = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 3, 5});
    model_ref = createRefGraph(
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 6, 224}),
        ov::Shape{2, 1, 3, 5});
}

TEST_F(ConvertGroupConvTests, Negative_DepthwiseCase) {
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 224});
    model = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 1, 5});
    // model_ref intentionally omitted — transformation should not fire
}

TEST_F(ConvertGroupConvTests, Negative_DynamicShapes) {
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 224});
    model = createInitGraph<ov::opset1::GroupConvolution>(param, ov::Shape{2, 1, 1, 5});
    // model_ref intentionally omitted — transformation should not fire
}
