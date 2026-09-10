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
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;

std::shared_ptr<v0::Constant> make_const(const Shape& shape) {
    return v0::Constant::create(element::f32, shape, std::vector<float>(shape_size(shape), 1.0f));
}

// Data [-> Broadcast(static target)] -> MatMul. Broadcast sits on the left or right MatMul
// input; passing an empty `target` omits the Broadcast, producing the fused reference graph.
// Data is a Constant unless `data_is_parameter` is set, in which case it is an extra Parameter,
// exercising that the pass is not limited to constant data inputs.
std::shared_ptr<ov::Model> buildModel(const Shape& data_shape,
                                      const std::vector<int64_t>& target,
                                      const PartialShape& other_shape,
                                      bool broadcast_on_lhs,
                                      bool transpose_b = false,
                                      bool data_is_parameter = false) {
    auto other = std::make_shared<v0::Parameter>(element::f32, other_shape);
    ParameterVector params{other};
    std::shared_ptr<Node> data_input;
    if (data_is_parameter) {
        auto data_param = std::make_shared<v0::Parameter>(element::f32, data_shape);
        params.push_back(data_param);
        data_input = data_param;
    } else {
        data_input = make_const(data_shape);
    }
    if (!target.empty()) {
        auto target_shape = v0::Constant::create(element::i64, Shape{target.size()}, target);
        data_input = std::make_shared<v3::Broadcast>(data_input, target_shape, op::BroadcastType::BIDIRECTIONAL);
    }
    std::shared_ptr<Node> matmul = broadcast_on_lhs
                                       ? std::make_shared<v0::MatMul>(data_input, other, false, transpose_b)
                                       : std::make_shared<v0::MatMul>(other, data_input, false, transpose_b);
    auto result = std::make_shared<v0::Result>(matmul);
    return std::make_shared<ov::Model>(ResultVector{result}, params);
}

struct PositiveCase {
    std::string name;
    Shape const_shape;
    std::vector<int64_t> target;
    PartialShape other_shape;
    bool broadcast_on_lhs;
    bool transpose_b = false;
    bool data_is_parameter = false;
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
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        manager.register_pass<ov::pass::BroadcastMatMulFusion>();
    }
};

TEST_P(BroadcastMatMulFusionPositive, RemovesBroadcast) {
    const auto& p = GetParam();
    model = buildModel(p.const_shape, p.target, p.other_shape, p.broadcast_on_lhs, p.transpose_b, p.data_is_parameter);
    model_ref = buildModel(p.const_shape, {}, p.other_shape, p.broadcast_on_lhs, p.transpose_b, p.data_is_parameter);
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
        PositiveCase{"lhs_4d_two_batch_dims", Shape{1, 1, 32, 8}, {2, 4, 32, 8}, PartialShape{2, 4, 8, 16}, true},
        // transpose_b only reinterprets the (unchanged) matrix dims, so removal is still valid.
        PositiveCase{"transposed_matmul", Shape{1, 32, 8}, {4, 32, 8}, PartialShape{4, 16, 8}, true, true},
        // Broadcast does not change the batch dim (already 4) — still a no-op that can be detached.
        PositiveCase{"data_already_carries_batch", Shape{4, 32, 8}, {4, 32, 8}, PartialShape{4, 8, 16}, true},
        // Broadcast target already matches data's batch (4): a no-op broadcast.
        PositiveCase{"no_op_bcast", Shape{4, 32, 8}, {4, 32, 8}, PartialShape{1, 8, 16}, true},
        // Other operand carries an extra leading batch dim (2) beyond what the Broadcast expands (4).
        PositiveCase{"other_has_extra_leading_batch_dim", Shape{1, 32, 8}, {4, 32, 8}, PartialShape{2, 4, 8, 16}, true},
        // Data is a Parameter, not a Constant: the pass is not limited to constant data inputs.
        PositiveCase{"non_constant_data", Shape{1, 32, 8}, {4, 32, 8}, PartialShape{4, 8, 16}, true, false, true}),
    [](const testing::TestParamInfo<PositiveCase>& info) {
        return info.param.name;
    });

// ----------------------------- Positive: standalone cases -----------------------------

TEST_F(BroadcastMatMulFusionTest, RemovesBroadcastFromMatMulInputButKeepsBroadcastForOtherConsumer) {
    // The Broadcast feeds both the MatMul and a second Result: only the MatMul input edge is
    // rewired to the Constant directly, the Broadcast node itself stays for the other consumer.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto data = make_const(Shape{1, 32, 8});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastType::BIDIRECTIONAL);
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(
        ResultVector{std::make_shared<v0::Result>(matmul), std::make_shared<v0::Result>(broadcast)},
        ParameterVector{other});

    auto other_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto data_ref = make_const(Shape{1, 32, 8});
    auto target_ref = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast_ref = std::make_shared<v3::Broadcast>(data_ref, target_ref, op::BroadcastType::BIDIRECTIONAL);
    auto matmul_ref = std::make_shared<v0::MatMul>(data_ref, other_ref);
    model_ref = std::make_shared<ov::Model>(
        ResultVector{std::make_shared<v0::Result>(matmul_ref), std::make_shared<v0::Result>(broadcast_ref)},
        ParameterVector{other_ref});
}

TEST_F(BroadcastMatMulFusionTest, RemovesV1BroadcastWithAxesMapping) {
    // v1::Broadcast in NUMPY mode always carries a 3rd (mocked) axes_mapping input, so the
    // 3-input BroadcastBase form must be matched too, not just the 2-input v3::Broadcast one.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto data = make_const(Shape{1, 32, 8});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto axes_mapping = v0::Constant::create(element::i64, Shape{3}, {0, 1, 2});
    auto broadcast = std::make_shared<v1::Broadcast>(data, target, axes_mapping, op::AutoBroadcastType::NUMPY);
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul)}, ParameterVector{other});

    auto other_ref = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto data_ref = make_const(Shape{1, 32, 8});
    auto matmul_ref = std::make_shared<v0::MatMul>(data_ref, other_ref);
    model_ref =
        std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul_ref)}, ParameterVector{other_ref});
}

// ----------------------------- Negative: transformation must NOT fire -----------------------------

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenOtherBatchDynamicWithoutSymbols) {
    // Matrix dims stay static/equal, but both batch dims are dynamic and carry no shape symbol
    // (no symbol propagation ran): equality is not provable, so the Broadcast must be kept.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 8, 16});
    auto data = make_const(Shape{1, 32, 8});

    auto batch = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(other, element::i64),
                                              v0::Constant::create(element::i64, Shape{1}, {0}),
                                              v0::Constant::create(element::i64, Shape{}, {0}));
    auto target = std::make_shared<v0::Concat>(OutputVector{batch,
                                                            v0::Constant::create(element::i64, Shape{1}, {32}),
                                                            v0::Constant::create(element::i64, Shape{1}, {8})},
                                               0);
    auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastType::BIDIRECTIONAL);
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul)}, ParameterVector{other});
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenBatchNotCarriedByOther) {
    // Other operand batch is a static 1, which cannot reproduce the broadcast batch of 4.
    model = buildModel(Shape{1, 32, 8}, {4, 32, 8}, PartialShape{1, 8, 16}, /*broadcast_on_lhs=*/true);
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenMatrixDimChanged) {
    // Broadcast expands a matrix dim (1 -> 8): removing it would change the contraction.
    model = buildModel(Shape{4, 1, 8}, {4, 8, 8}, PartialShape{4, 8, 16}, /*broadcast_on_lhs=*/true);
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

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWithPdpdMode) {
    // PDPD broadcasting aligns dimensions differently from MatMul's NumPy-style batch
    // broadcasting, so such a Broadcast must not be detached.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, 8, 16});
    auto data = make_const(Shape{1, 32, 8});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastModeSpec(op::BroadcastType::PDPD, 0));
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul)}, ParameterVector{other});
}

TEST_F(BroadcastMatMulFusionTest, KeepsBroadcastWhenFixedExtentAndOtherDynamic) {
    // Broadcast batch is a fixed 4, the other operand batch is dynamic: not provably equal,
    // so removing the Broadcast could hide a runtime batch mismatch it would have rejected.
    auto other = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 8, 16});
    auto data = make_const(Shape{1, 32, 8});
    auto target = v0::Constant::create(element::i64, Shape{3}, {4, 32, 8});
    auto broadcast = std::make_shared<v3::Broadcast>(data, target, op::BroadcastType::BIDIRECTIONAL);
    auto matmul = std::make_shared<v0::MatMul>(broadcast, other);
    model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(matmul)}, ParameterVector{other});
}
