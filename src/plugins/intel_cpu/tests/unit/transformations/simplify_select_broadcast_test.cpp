// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "transformations/cpu_opset/common/pass/simplify_select_broadcast.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;

/**
 * Tests for SimplifySelectBroadcast transformation.
 *
 * Pattern being simplified:
 *   Select(condition, on_true_scalar_const, Broadcast(on_false_scalar_const, target_shape))
 *   → Select(condition, on_true_scalar_const, on_false_scalar_const)
 *
 * Motivation: In hybrid SSM+attention models (e.g. qwen3_5_text), the attention mask is
 * constructed as:
 *   Select(bool_mask, -65504, Broadcast(0.0, [B,1,Q,Q]))
 * where Q comes from SSM output length, but bool_mask has shape [B,1,kv_len,Q].
 * When kv_len ≠ Q and both > 1, EltwiseShapeInfer raises a dim-mismatch error.
 */
class SimplifySelectBroadcastTest : public TransformationTestsF {
public:
    SimplifySelectBroadcastTest() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CmpValues::NAMES);
    }
};

// ---------------------------------------------------------
// Test 1: Basic case — on_false is a Broadcast of a scalar constant.
//         Select(cond, scalar_A, Broadcast(scalar_B, shape)) → Select(cond, scalar_A, scalar_B)
// ---------------------------------------------------------
TEST_F(SimplifySelectBroadcastTest, OnFalseBroadcastScalar) {
    const ov::element::Type et = ov::element::f32;
    const ov::Shape cond_shape{1, 1, 4, 4};

    {
        // Input graph
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-65504.0f});
        auto on_false_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

        // Broadcast(0.0, shape_of_cond)
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(cond);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            on_false_src,
            shape_of,
            ov::op::BroadcastType::NUMPY);

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true, broadcast);
        select->set_friendly_name("Select");

        model = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{cond});
        manager.register_pass<ov::intel_cpu::pass::SimplifySelectBroadcast>();
    }
    {
        // Expected graph after transformation
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-65504.0f});
        auto on_false_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true, on_false_src);
        select->set_friendly_name("Select");

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{cond});
    }
}

// ---------------------------------------------------------
// Test 2: on_true is a Broadcast of a scalar constant.
//         Select(cond, Broadcast(scalar_A, shape), scalar_B) → Select(cond, scalar_A, scalar_B)
// ---------------------------------------------------------
TEST_F(SimplifySelectBroadcastTest, OnTrueBroadcastScalar) {
    const ov::element::Type et = ov::element::f32;
    const ov::Shape cond_shape{2, 1, 3, 3};

    {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{1.0f});
        auto on_false = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(cond);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            on_true_src,
            shape_of,
            ov::op::BroadcastType::NUMPY);

        auto select = std::make_shared<ov::op::v1::Select>(cond, broadcast, on_false);
        select->set_friendly_name("Select");

        model = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{cond});
        manager.register_pass<ov::intel_cpu::pass::SimplifySelectBroadcast>();
    }
    {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{1.0f});
        auto on_false = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true_src, on_false);
        select->set_friendly_name("Select");

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{cond});
    }
}

// ---------------------------------------------------------
// Test 3: Negative — Broadcast of a non-scalar constant should NOT be simplified.
// ---------------------------------------------------------
TEST_F(SimplifySelectBroadcastTest, NoSimplifyNonScalarBroadcast) {
    const ov::element::Type et = ov::element::f32;
    const ov::Shape cond_shape{1, 1, 4, 4};

    {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-65504.0f});

        // Non-scalar broadcast source: shape {4}
        auto on_false_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{4}, std::vector<float>{0.f, 1.f, 2.f, 3.f});
        auto target_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{1, 1, 4, 4});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            on_false_src,
            target_shape,
            ov::op::BroadcastType::NUMPY);

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true, broadcast);
        select->set_friendly_name("Select");

        model = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{cond});
        manager.register_pass<ov::intel_cpu::pass::SimplifySelectBroadcast>();
    }
    {
        // Reference is unchanged (no simplification expected).
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-65504.0f});

        auto on_false_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{4}, std::vector<float>{0.f, 1.f, 2.f, 3.f});
        auto target_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{1, 1, 4, 4});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            on_false_src,
            target_shape,
            ov::op::BroadcastType::NUMPY);

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true, broadcast);
        select->set_friendly_name("Select");

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{cond});
    }
}

// ---------------------------------------------------------
// Test 4: Negative — Broadcast output with multiple consumers: do NOT simplify.
// ---------------------------------------------------------
TEST_F(SimplifySelectBroadcastTest, NoSimplifyMultipleConsumers) {
    const ov::element::Type et = ov::element::f32;
    const ov::Shape cond_shape{1, 1, 4, 4};

    {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-65504.0f});
        auto on_false_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

        auto target_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{1, 1, 4, 4});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            on_false_src,
            target_shape,
            ov::op::BroadcastType::NUMPY);

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true, broadcast);
        select->set_friendly_name("Select");

        // Second consumer of broadcast — prevents simplification.
        auto second_consumer = std::make_shared<ov::op::v1::Select>(cond, broadcast, on_true);
        second_consumer->set_friendly_name("Select2");

        model = std::make_shared<ov::Model>(ov::OutputVector{select, second_consumer}, ov::ParameterVector{cond});
        manager.register_pass<ov::intel_cpu::pass::SimplifySelectBroadcast>();
    }
    {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_shape);
        auto on_true = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-65504.0f});
        auto on_false_src = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

        auto target_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{1, 1, 4, 4});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            on_false_src,
            target_shape,
            ov::op::BroadcastType::NUMPY);

        auto select = std::make_shared<ov::op::v1::Select>(cond, on_true, broadcast);
        select->set_friendly_name("Select");

        auto second_consumer = std::make_shared<ov::op::v1::Select>(cond, broadcast, on_true);
        second_consumer->set_friendly_name("Select2");

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select, second_consumer}, ov::ParameterVector{cond});
    }
}
