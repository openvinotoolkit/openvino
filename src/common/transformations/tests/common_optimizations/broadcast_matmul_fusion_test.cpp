// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_matmul_fusion.hpp"

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;

namespace {

namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;

std::shared_ptr<v0::Constant> make_const(const Shape& shape) {
    return v0::Constant::create(element::f32, shape, std::vector<float>(shape_size(shape), 1.0f));
}

// Constant -> Broadcast(static target) -> MatMul. Broadcast sits on the left or right MatMul input.
std::shared_ptr<ov::Model> getModel(const Shape& const_shape,
                                    const std::vector<int64_t>& target,
                                    const PartialShape& other_shape,
                                    bool broadcast_on_lhs,
                                    bool transpose_b = false) {
    auto other = std::make_shared<v0::Parameter>(element::f32, other_shape);
    auto data = make_const(const_shape);
    auto target_shape = v0::Constant::create(element::i64, Shape{target.size()}, target);
    auto broadcast = std::make_shared<v3::Broadcast>(data, target_shape, op::BroadcastType::BIDIRECTIONAL);
    std::shared_ptr<Node> matmul =
        broadcast_on_lhs ? std::make_shared<v0::MatMul>(broadcast, other, false, transpose_b)
                         : std::make_shared<v0::MatMul>(other, broadcast, false, transpose_b);
    auto result = std::make_shared<v0::Result>(matmul);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{other});
}

// Reference: the Constant feeds the MatMul directly, no Broadcast.
std::shared_ptr<ov::Model> getModelRef(const Shape& const_shape,
                                       const PartialShape& other_shape,
                                       bool broadcast_on_lhs,
                                       bool transpose_b = false) {
    auto other = std::make_shared<v0::Parameter>(element::f32, other_shape);
    auto data = make_const(const_shape);
    std::shared_ptr<Node> matmul =
        broadcast_on_lhs ? std::make_shared<v0::MatMul>(data, other, false, transpose_b)
                         : std::make_shared<v0::MatMul>(other, data, false, transpose_b);
    auto result = std::make_shared<v0::Result>(matmul);
    return std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{other});
}

struct PositiveCase {
    std::string name;
    Shape const_shape;
    std::vector<int64_t> target;
    PartialShape other_shape;
    bool broadcast_on_lhs;
};

}  // namespace

class BroadcastMatMulFusionTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::BroadcastMatMulFusion>();
    }
};

// ----------------------------- Positive: parametrized static-shape cases -----------------------------

class BroadcastMatMulFusionPositive : public TransformationTestsF, public testing::WithParamInterface<PositiveCase> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::BroadcastMatMulFusion>();
    }
};

TEST_P(BroadcastMatMulFusionPositive, RemovesBroadcast) {
    const auto& p = GetParam();
    model = getModel(p.const_shape, p.target, p.other_shape, p.broadcast_on_lhs);
    model_ref = getModelRef(p.const_shape, p.other_shape, p.broadcast_on_lhs);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastMatMulFusion,
    BroadcastMatMulFusionPositive,
    testing::Values(
        // Broadcast on the left input; other operand carries the expanded batch (4).
        PositiveCase{"lhs_static_batch", Shape{1, 32, 8}, {4, 32, 8}, PartialShape{4, 8, 16}, true},
        // Broadcast on the right input; contraction dim on the shared axis (32).
        PositiveCase{"rhs_static_batch", Shape{1, 32, 8}, {4, 32, 8}, PartialShape{4, 16, 32}, false},
        // 4D: two batch dims (2,4), both carried by the other operand.
        PositiveCase{"lhs_4d_two_batch_dims", Shape{1, 1, 32, 8}, {2, 4, 32, 8}, PartialShape{2, 4, 8, 16}, true}),
    [](const testing::TestParamInfo<PositiveCase>& info) {
        return info.param.name;
    });

// ----------------------------- Positive: standalone cases -----------------------------

TEST_F(BroadcastMatMulFusionTest, RemovesBroadcastWhenOtherBatchDynamic) {
    // Matrix dims stay static/equal; both batch dims are dynamic, so the other operand is
    // assumed runtime-compatible and the Broadcast is removed.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 8, 16});
    auto data = make_const(Shape{1, 32, 8});

    {
        auto batch = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(other, element::i64),
                                                  v0::Constant::create(element::i64, Shape{1}, {0}),
                                                  v0::Constant::create(element::i64, Shape{}, {0}));
        auto target = std::make_shared<v0::Concat>(
            OutputVector{batch,
                         v0::Constant::create(element::i64, Shape{1}, {32}),
                         v0::Constant::create(element::i64, Shape{1}, {8})},
            0);
        auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastType::BIDIRECTIONAL);
        auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
        model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul)},
                                            ParameterVector{other});
    }

    auto other_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 8, 16});
    auto data_ref = make_const(Shape{1, 32, 8});
    auto matmul_ref = std::make_shared<v0::MatMul>(data_ref, other_ref);
    model_ref = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul_ref)},
                                            ParameterVector{other_ref});
}

TEST_F(BroadcastMatMulFusionTest, RemovesBroadcastWithTransposedMatMul) {
    // transpose_b only reinterprets the (unchanged) matrix dims, so removal is still valid.
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    model = getModel(Shape{1, 32, 8}, {4, 32, 8}, PartialShape{4, 16, 8}, /*broadcast_on_lhs=*/true, /*transpose_b=*/true);
    model_ref = getModelRef(Shape{1, 32, 8}, PartialShape{4, 16, 8}, /*broadcast_on_lhs=*/true, /*transpose_b=*/true);
}

TEST_F(BroadcastMatMulFusionTest, RemovesBroadcastWhenDataAlreadyCarriesBatch) {
    // Broadcast does not change the batch dim (already 4) — still a no-op that can be detached.
    model = getModel(Shape{4, 32, 8}, {4, 32, 8}, PartialShape{4, 8, 16}, /*broadcast_on_lhs=*/true);
    model_ref = getModelRef(Shape{4, 32, 8}, PartialShape{4, 8, 16}, /*broadcast_on_lhs=*/true);
}

// ----------------------------- Negative: transformation must NOT fire -----------------------------

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenBatchNotCarriedByOther) {
    // Other operand batch is a static 1, which cannot reproduce the broadcast batch of 4.
    model = getModel(Shape{1, 32, 8}, {4, 32, 8}, PartialShape{1, 8, 16}, /*broadcast_on_lhs=*/true);
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenMatrixDimChanged) {
    // Broadcast expands a matrix dim (1 -> 8): removing it would change the contraction.
    model = getModel(Shape{4, 1, 8}, {4, 8, 8}, PartialShape{4, 8, 16}, /*broadcast_on_lhs=*/true);
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenDataNotConstant) {
    // The Broadcast data input is a Parameter, not a Constant: the pattern must not match.
    auto param_data = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 32, 8});
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast = std::make_shared<v3::Broadcast>(param_data, target, op::BroadcastType::BIDIRECTIONAL);
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul)},
                                        ParameterVector{param_data, other});
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWithMultipleConsumers) {
    // The Broadcast feeds the MatMul and a second Result: consumers_count(1) must reject it.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto data = make_const(Shape{1, 32, 8});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastType::BIDIRECTIONAL);
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(
        ResultVector{std::make_shared<v0::Result>(matmul), std::make_shared<v0::Result>(broadcast)},
        ParameterVector{other});
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenConsumerNotMatMul) {
    // The Broadcast feeds an elementwise Add, not a MatMul: the pattern must not match.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 32, 8});
    auto data = make_const(Shape{1, 32, 8});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastType::BIDIRECTIONAL);
    auto add = std::make_shared<ov::op::v1::Add>(broadcast, other);
    model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(add)}, ParameterVector{other});
}
