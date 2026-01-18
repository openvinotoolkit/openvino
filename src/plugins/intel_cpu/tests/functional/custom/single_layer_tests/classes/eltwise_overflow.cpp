// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_overflow.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
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

    function = makeNgraphFunction(ov::element::u8, {a, b}, op, "EltwiseOverflow");
}

void EltwiseOverflowLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& modelInputs = function->inputs();
    ASSERT_EQ(modelInputs.size(), 2u);
    ASSERT_EQ(targetInputStaticShapes.size(), 2u);

    const size_t size = ov::shape_size(targetInputStaticShapes[0]);

    std::vector<uint8_t> data0(size);
    std::vector<uint8_t> data1(size);

    if (overflowKind == EltwiseOverflowKind::UNDERFLOW) {
        // u8 subtract underflow: should wrap, not saturate.
        // E.g., 3 - 4 = 255 (not 0)
        for (size_t i = 0; i < size; ++i) {
            data0[i] = static_cast<uint8_t>(i % 10);        // 0-9 repeating
            data1[i] = static_cast<uint8_t>((i % 10) + 1);  // 1-10 repeating
        }
    } else {
        // u8 add overflow: should wrap (255 + 1 = 0).
        for (size_t i = 0; i < size; ++i) {
            data0[i] = static_cast<uint8_t>(250 + (i % 6));  // 250-255 repeating
            data1[i] = static_cast<uint8_t>((i % 10) + 1);   // 1-10 repeating
        }
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
