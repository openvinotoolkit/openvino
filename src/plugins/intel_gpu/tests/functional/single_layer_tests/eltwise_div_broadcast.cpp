#include <gtest/gtest.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/openvino.hpp>

TEST(GPU_EltwiseBroadcast, DivNumpyBroadcastShouldCompile) {
    // Create input tensors:
    // A: [1, 4, 1, 8]
    // B: [1, 1, 1, 8]  (NumPy-broadcastable)
    auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 8});

    auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 8});

    // Elementwise Div with NUMPY broadcasting
    auto div = std::make_shared<ov::op::v1::Divide>(input_a, input_b);
    div->set_autob(ov::op::AutoBroadcastType::NUMPY);

    auto result = std::make_shared<ov::op::v0::Result>(div);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_a, input_b}, "EltwiseDivBroadcast");

    ov::Core core;

    // The test: GPU compilation must NOT throw
    EXPECT_NO_THROW({ core.compile_model(model, "GPU"); });
}
