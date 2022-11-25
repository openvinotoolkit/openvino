// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include <openvino/opsets/opset10.hpp>

namespace {

using namespace ov;

class StridedSliceMatMul : virtual public test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;
        auto type = element::f32;
        Shape input_shape{1, 128, 768};
        auto input = std::make_shared<opset10::Parameter>(type, input_shape);
        auto begin = opset10::Constant::create(element::i32, Shape{3}, {0, 0, 0});
        auto end = opset10::Constant::create(element::i32, Shape{3}, {0, 1, 0});
        auto stride = opset10::Constant::create(element::i32, Shape{3}, {1, 1, 1});
        std::vector<int64_t> begin_mask{1, 0, 1};
        std::vector<int64_t> end_mask{1, 0, 1};
        std::vector<int64_t> new_axis_mask{0, 0, 0};
        std::vector<int64_t> shrink_axis_mask{0, 1, 0};
        auto strided_slice = std::make_shared<opset10::StridedSlice>(input, begin, end, stride, begin_mask, end_mask, new_axis_mask, shrink_axis_mask);
        Shape weights_shape{768, 768};
        auto weights = opset10::Constant::create(type, weights_shape, {1});
        auto matmul = std::make_shared<opset10::MatMul>(strided_slice, weights);
        function = std::make_shared<Model>(matmul, ParameterVector{input});

        targetStaticShapes.push_back({input_shape});
    }
};

TEST_F(StridedSliceMatMul, smoke_StridedSliceMatMul) {
    run();
}

}  // namespace
