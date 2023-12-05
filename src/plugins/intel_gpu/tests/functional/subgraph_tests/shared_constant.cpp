// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph/opsets/opset8.hpp>

namespace {

using namespace ngraph;

// Validate scenario where a single Constant has multiple users (like one constant is used for Convolution, ConvolutionBackpropData, Multiply, etc.)
class SharedConstant : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = element::f32;
        Shape constShape{4, 1, 3, 3};
        Shape convInputShape{1, 1, 5, 5};
        Shape convBackpropInputShape{1, 4, 5, 5};
        Shape constGroupConvBackpropShape{2, 2, 3, 3, 3};
        auto constant = opset8::Constant::create(type, constShape, {1});
        auto input1 = std::make_shared<opset8::Parameter>(type, convInputShape);
        auto conv = std::make_shared<opset8::Convolution>(input1, constant, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto input2 = std::make_shared<opset8::Parameter>(type, convBackpropInputShape);
        auto convBprop = std::make_shared<opset8::ConvolutionBackpropData>(input2, constant, Strides{1, 1},
                CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto input3 = std::make_shared<opset8::Parameter>(type, convBackpropInputShape);
        auto constantGroupConv = opset8::Constant::create(type, constGroupConvBackpropShape, {1});
        auto groupConvBprop = std::make_shared<opset8::GroupConvolutionBackpropData>(input3, constantGroupConv, Strides{1, 1},
                CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto input4 = std::make_shared<opset8::Parameter>(type, constShape);
        auto mul = std::make_shared<opset8::Multiply>(input4, constant);
        auto input5 = std::make_shared<opset8::Parameter>(type, constGroupConvBackpropShape);
        auto mul2 = std::make_shared<opset8::Multiply>(input5, constantGroupConv);
        // explicitly set the output name, to avoid global conflict
        mul2->set_friendly_name("Multiply_0");
        mul->set_friendly_name("Multiply_1");
        function = std::make_shared<ngraph::Function>(NodeVector{convBprop, conv, groupConvBprop, mul2, mul},
                ParameterVector{input1, input2, input3, input4, input5});
    }
};

TEST_F(SharedConstant, smoke_SharedConstant) {
    Run();
}

}  // namespace
