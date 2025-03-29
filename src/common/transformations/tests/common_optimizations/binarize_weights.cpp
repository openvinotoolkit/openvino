// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/binarize_weights.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, BinarizeWeightsActivationsOutputLowZero) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
        manager.register_pass<ov::pass::BinarizeWeights>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto mul =
            std::make_shared<opset5::Multiply>(conv, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.7f}));
        auto mul2 =
            std::make_shared<opset5::Multiply>(mul, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.2f}));

        model_ref = std::make_shared<Model>(NodeVector{mul2}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, BinarizeWeightsActivationsOutputLowNegative) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.7f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
        manager.register_pass<ov::pass::BinarizeWeights>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 1, 1});
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto mul =
            std::make_shared<opset5::Multiply>(conv, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.7f}));
        auto mul2 =
            std::make_shared<opset5::Multiply>(mul, opset5::Constant::create(element::f32, Shape{1, 1, 1}, {0.2f}));

        model_ref = std::make_shared<Model>(NodeVector{mul2}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeBinarizeWeightsInvalidLevels) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.7f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
        manager.register_pass<ov::pass::BinarizeWeights>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.7f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 3);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeBinarizeWeightsInvalidActivationsOutputLowHigh) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
        manager.register_pass<ov::pass::BinarizeWeights>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {-0.2f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeBinarizeWeightsInvalidOutputLowHigh) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
        manager.register_pass<ov::pass::BinarizeWeights>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto act_in_low = opset5::Constant::create(element::f32, Shape{1}, {1.0f});
        auto act_in_high = opset5::Constant::create(element::f32, Shape{1}, {3.0f});
        auto act_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto act_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.7f});
        auto act_fq =
            std::make_shared<opset5::FakeQuantize>(data, act_in_low, act_in_high, act_out_low, act_out_high, 2);
        auto weights = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-1, 0, 2});
        auto weights_in_low = opset5::Constant::create(element::f32, Shape{1}, {-1.0f});
        auto weights_in_high = opset5::Constant::create(element::f32, Shape{1}, {2.0f});
        auto weights_out_low = opset5::Constant::create(element::f32, Shape{1}, {0.0f});
        auto weights_out_high = opset5::Constant::create(element::f32, Shape{1}, {0.2f});
        auto weights_fq = std::make_shared<opset5::FakeQuantize>(weights,
                                                                 weights_in_low,
                                                                 weights_in_high,
                                                                 weights_out_low,
                                                                 weights_out_high,
                                                                 2);
        auto conv = std::make_shared<opset5::Convolution>(act_fq,
                                                          weights_fq,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data});
    }
}
