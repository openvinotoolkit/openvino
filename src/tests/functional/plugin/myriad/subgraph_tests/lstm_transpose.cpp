// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph/opsets/opset8.hpp>

namespace {

using namespace ngraph;

class LSTMTranspose : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_MYRIAD;

        auto X = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 26, 64});
        auto hidden_state = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 32});
        auto cell_state = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 32});
        auto seq_len = opset8::Constant::create(element::i32, Shape{1}, {26});
        auto W = opset8::Constant::create(element::f32, Shape{1, 128, 64}, {1});
        auto R = opset8::Constant::create(element::f32, Shape{1, 128, 32}, {1});
        auto B = opset8::Constant::create(element::f32, Shape{1, 128}, {1});
        auto lstm = std::make_shared<opset8::LSTMSequence>(X, hidden_state, cell_state, seq_len, W, R, B, 32, op::RecurrentSequenceDirection::FORWARD);
        auto transpose = std::make_shared<opset8::Transpose>(lstm->output(0), opset8::Constant::create(element::i32, Shape{4}, {2, 0, 1, 3}));
        function = std::make_shared<ngraph::Function>(transpose, ParameterVector{X, hidden_state, cell_state});
    }
};

TEST_F(LSTMTranspose, CompareWithRefs) {
    Run();
}

}  // namespace
