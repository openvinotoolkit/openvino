// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for ov::frontend::gguf::pass::PushScaleBeforeMatMul, the opt-in rewrite
// MatMul(A, W) * c -> MatMul(A * c, W) that keeps f16 plugins in range.

#include <functional>
#include <memory>
#include <vector>

#include "op_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/push_scale_before_matmul.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov_gguf_test;

namespace {

using ov::Output;
using ov::op::v0::Constant;

// Parameter a [1, 3, 8] and a constant weight [8, 6]; `build` wires the MatMul(a, w) into a result.
std::shared_ptr<ov::Model> make_model(
    const std::function<Output<ov::Node>(const Output<ov::Node>& a, const Output<ov::Node>& matmul)>& build) {
    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8});
    a->output(0).set_names({"a"});
    std::vector<float> w(8 * 6);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.01f * static_cast<float>(i % 13) - 0.05f;
    }
    auto matmul = std::make_shared<ov::op::v0::MatMul>(a, Constant::create(ov::element::f32, {8, 6}, w));
    return std::make_shared<ov::Model>(ov::OutputVector{build(a, matmul)}, ov::ParameterVector{a});
}

void run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::frontend::gguf::pass::PushScaleBeforeMatMul>();
    manager.run_passes(model);
}

ov::Tensor run(const std::shared_ptr<ov::Model>& model) {
    ov::Tensor input(ov::element::f32, ov::Shape{1, 3, 8});
    for (size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = 0.1f * static_cast<float>(i) - 1.0f;
    }
    return run_on_cpu(model, {{"a", input}});
}

std::vector<float> run_vector(const std::shared_ptr<ov::Model>& model) {
    const auto out = run(model);
    return {out.data<float>(), out.data<float>() + out.get_size()};
}

std::shared_ptr<ov::Node> result_producer(const std::shared_ptr<ov::Model>& model) {
    return model->get_results()[0]->input_value(0).get_node_shared_ptr();
}

}  // namespace

TEST(PushScaleBeforeMatMul, ScaleMovesInFrontOfMatMul) {
    const auto build = [](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        return std::make_shared<ov::op::v1::Multiply>(matmul, Constant::create(ov::element::f32, {}, {0.25f}));
    };
    auto reference = make_model(build);
    auto model = make_model(build);
    run_pass(model);

    const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(result_producer(model));
    ASSERT_NE(matmul, nullptr) << "the MatMul is now the last op";
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(matmul->get_input_node_shared_ptr(0)));
    expect_near(run(model), run_vector(reference), 1e-6f);
}

TEST(PushScaleBeforeMatMul, ScaleBehindUnsqueezeIsFound) {
    // The residual multiplier reaches the Multiply as Unsqueeze(scalar), not as a bare scalar.
    const auto build = [](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        auto scale = std::make_shared<ov::op::v0::Unsqueeze>(Constant::create(ov::element::f32, {}, {0.263f}),
                                                             Constant::create(ov::element::i64, {1}, {0}));
        return std::make_shared<ov::op::v1::Multiply>(matmul, scale);
    };
    auto reference = make_model(build);
    auto model = make_model(build);
    run_pass(model);

    EXPECT_TRUE(ov::is_type<ov::op::v0::MatMul>(result_producer(model)));
    expect_near(run(model), run_vector(reference), 1e-6f);
}

TEST(PushScaleBeforeMatMul, GatedProductScalesTheNonSwishFactor) {
    // silu(gate) * up -> MatMul -> * c: the scale goes onto `up`, since silu(c * x) != c * silu(x).
    const auto build = [](const Output<ov::Node>& a, const Output<ov::Node>&) -> Output<ov::Node> {
        auto gate = std::make_shared<ov::op::v4::Swish>(a);
        auto up = std::make_shared<ov::op::v1::Multiply>(a, Constant::create(ov::element::f32, {}, {1.5f}));
        auto product = std::make_shared<ov::op::v1::Multiply>(gate, up);
        std::vector<float> w(8 * 6, 0.02f);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(product, Constant::create(ov::element::f32, {8, 6}, w));
        return std::make_shared<ov::op::v1::Multiply>(matmul, Constant::create(ov::element::f32, {}, {0.5f}));
    };
    auto reference = make_model(build);
    auto model = make_model(build);
    run_pass(model);

    const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(result_producer(model));
    ASSERT_NE(matmul, nullptr);
    const auto product = ov::as_type_ptr<ov::op::v1::Multiply>(matmul->get_input_node_shared_ptr(0));
    ASSERT_NE(product, nullptr);
    // The Swish branch is untouched; the other factor carries the scale.
    const bool swish_first = ov::is_type<ov::op::v4::Swish>(product->get_input_node_shared_ptr(0));
    EXPECT_TRUE(ov::is_type<ov::op::v4::Swish>(product->get_input_node_shared_ptr(swish_first ? 0 : 1)));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(product->get_input_node_shared_ptr(swish_first ? 1 : 0)));
    expect_near(run(model), run_vector(reference), 1e-5f);
}

TEST(PushScaleBeforeMatMul, ScalesOutsideTheRangeAreLeftAlone) {
    for (const float scale : {2.0f, 1.0f, 0.001f}) {
        auto model = make_model([scale](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
            return std::make_shared<ov::op::v1::Multiply>(matmul, Constant::create(ov::element::f32, {}, {scale}));
        });
        run_pass(model);
        EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(result_producer(model))) << "scale " << scale;
    }
}

TEST(PushScaleBeforeMatMul, MatMulWithAnotherConsumerIsLeftAlone) {
    auto model = make_model([](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        auto scaled = std::make_shared<ov::op::v1::Multiply>(matmul, Constant::create(ov::element::f32, {}, {0.25f}));
        return std::make_shared<ov::op::v1::Multiply>(scaled, matmul);
    });
    run_pass(model);
    const auto outer = result_producer(model);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(outer));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(outer->get_input_node_shared_ptr(0)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::MatMul>(outer->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0)));
}
