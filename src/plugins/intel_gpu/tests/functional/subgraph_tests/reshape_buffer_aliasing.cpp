// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"

namespace {

// Test 1: Single consumer after noop reshape
// Param -> Softmax -> Reshape(noop) -> MatMul -> Result
// Verifies that the reshape buffer aliasing doesn't corrupt data
// when the reshape is optimized away and buffers are shared.
class OVReshapeBufferAliasingSingleConsumerTest : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = ov::element::f32;
        ov::Shape input_shape{2, 4, 8};

        auto input = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(input, -1);

        // Noop reshape: {2, 4, 8} -> {2, 4, 8}
        auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 4, 8});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(softmax, target_shape, false);

        auto weights = ov::op::v0::Constant::create(type, {8, 8}, {0.1f});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, weights, false, true);

        function = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
    }
};

TEST_F(OVReshapeBufferAliasingSingleConsumerTest, Inference) {
    run();
}

// Test 2: Multiple consumers sharing reshape output
// Param_A -> Relu -> Reshape(noop) -> Add -> Result
// Param_B -> Relu ------------------>
// Verifies memory deps are correct when reshape output feeds into an op
// alongside another branch.
class OVReshapeBufferAliasingMultiConsumerTest : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = ov::element::f32;
        ov::Shape shape{2, 4};

        auto param_a = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto relu_a = std::make_shared<ov::op::v0::Relu>(param_a);

        // Noop reshape: {2, 4} -> {2, 4}
        auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 4});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(relu_a, target_shape, false);

        auto param_b = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto relu_b = std::make_shared<ov::op::v0::Relu>(param_b);

        auto add = std::make_shared<ov::op::v1::Add>(reshape, relu_b);

        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{param_a, param_b});
    }
};

TEST_F(OVReshapeBufferAliasingMultiConsumerTest, Inference) {
    run();
}

// Test 3: Chained reshapes
// Param -> Softmax -> Reshape -> Reshape -> Add -> Result
// Param_B ------------------------------>
// Verifies correct buffer aliasing through a chain of optimized reshapes.
class OVReshapeBufferAliasingChainedReshapesTest : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = ov::element::f32;
        ov::Shape shape{2, 8};

        auto param_a = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(param_a, -1);

        // Reshape 1: {2, 8} -> {2, 8} (noop)
        auto target1 = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 8});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(softmax, target1, false);

        // Reshape 2: {2, 8} -> {2, 8} (noop)
        auto target2 = ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 8});
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(reshape1, target2, false);

        auto param_b = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto add = std::make_shared<ov::op::v1::Add>(reshape2, param_b);

        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{param_a, param_b});
    }
};

TEST_F(OVReshapeBufferAliasingChainedReshapesTest, Inference) {
    run();
}

}  // namespace
