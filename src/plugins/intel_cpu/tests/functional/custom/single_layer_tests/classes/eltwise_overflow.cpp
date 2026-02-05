// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_overflow.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {

std::string EltwiseOverflowLayerCPUTest::getTestCaseName(const testing::TestParamInfo<EltwiseOverflowTestParams>& obj) {
    const auto& [kind, shape] = obj.param;
    std::ostringstream result;
    result << "kind=" << (kind == EltwiseOverflowKind::UNDERFLOW ? "UNDERFLOW" : "OVERFLOW");
    result << "_shape=" << shape;
    return result.str();
}

void EltwiseOverflowLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    abs_threshold = 0;

    const auto& [kind, shape] = GetParam();
    overflowKind = kind;

    InputShape inShape = {{}, {shape}};
    init_input_shapes({inShape, inShape});

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape(shape));
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape(shape));

    std::shared_ptr<ov::Node> op;
    if (kind == EltwiseOverflowKind::UNDERFLOW) {
        op = std::make_shared<ov::op::v1::Subtract>(a, b);
    } else {
        op = std::make_shared<ov::op::v1::Add>(a, b);
    }

    auto result = std::make_shared<ov::op::v0::Result>(op);
    function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{a, b}, "EltwiseOverflow");
}

void EltwiseOverflowLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& modelInputs = function->inputs();
    ASSERT_EQ(modelInputs.size(), 2u);
    ASSERT_EQ(targetInputStaticShapes.size(), 2u);

    const size_t size = ov::shape_size(targetInputStaticShapes[0]);

    // Hardcoded values that guarantee underflow/overflow regardless of shape size.
    // Pattern repeats to fill any shape.
    static const std::vector<uint8_t> underflow_a = {3, 0, 1, 5, 10, 0, 100, 50};
    static const std::vector<uint8_t> underflow_b = {4, 1, 2, 3, 20, 1, 200, 51};
    // Expected results (wrap): 255, 255, 255, 2, 246, 255, 156, 255

    static const std::vector<uint8_t> overflow_a = {255, 254, 200, 128, 255, 250, 255, 1};
    static const std::vector<uint8_t> overflow_b = {1, 2, 100, 128, 255, 10, 128, 255};
    // Expected results (wrap): 0, 0, 44, 0, 254, 4, 127, 0

    const auto& src_a = (overflowKind == EltwiseOverflowKind::UNDERFLOW) ? underflow_a : overflow_a;
    const auto& src_b = (overflowKind == EltwiseOverflowKind::UNDERFLOW) ? underflow_b : overflow_b;

    std::vector<uint8_t> data0(size);
    std::vector<uint8_t> data1(size);

    for (size_t i = 0; i < size; ++i) {
        data0[i] = src_a[i % src_a.size()];
        data1[i] = src_b[i % src_b.size()];
    }

    auto t0 = ov::Tensor(ov::element::u8, targetInputStaticShapes[0]);
    auto t1 = ov::Tensor(ov::element::u8, targetInputStaticShapes[1]);
    std::copy(data0.begin(), data0.end(), t0.data<uint8_t>());
    std::copy(data1.begin(), data1.end(), t1.data<uint8_t>());

    inputs.insert({modelInputs[0].get_node_shared_ptr(), t0});
    inputs.insert({modelInputs[1].get_node_shared_ptr(), t1});
}

TEST_P(EltwiseOverflowLayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
