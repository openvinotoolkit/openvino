// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph/opsets/opset8.hpp>

namespace {

using namespace ngraph;

// Validate scenario where a single Constant has multiple users (like one constant is used for Convolution, ConvolutionBackpropData, Multiply, etc.)
class GatherU8InputBlob : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        inPrc = InferenceEngine::Precision::U8;
        outPrc = InferenceEngine::Precision::FP32;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        auto input = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto indexes = opset8::Constant::create(element::i32, Shape{3}, {2, 1, 0});
        auto axis = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto gather = std::make_shared<opset8::Gather>(input, indexes, axis);
        function = std::make_shared<ngraph::Function>(gather, ParameterVector{input});
    }
};

TEST_F(GatherU8InputBlob, smoke_GatherU8InputBlob) {
    Run();
}

} // namespace
