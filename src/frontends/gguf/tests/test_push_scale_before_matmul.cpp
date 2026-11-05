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
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov_gguf_test;

namespace ov_gguf_test {

class CountingConvert : public ov::op::v0::Convert {
public:
    OPENVINO_OP("CountingConvert", "test", ov::op::v0::Convert);

    CountingConvert(const ov::Output<ov::Node>& input, size_t& evaluations)
        : ov::op::v0::Convert(input, ov::element::f32),
          m_evaluations(evaluations) {}

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        ++m_evaluations;
        return ov::op::v0::Convert::evaluate(outputs, inputs);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CountingConvert>(inputs.at(0), m_evaluations);
    }

private:
    size_t& m_evaluations;
};

}  // namespace ov_gguf_test

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

TEST(PushScaleBeforeMatMul, ScaleUsesTheConvertedValue) {
    const auto build = [](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        auto as_integer =
            std::make_shared<ov::op::v0::Convert>(Constant::create(ov::element::f32, {}, {0.25f}), ov::element::i32);
        auto scale = std::make_shared<ov::op::v0::Convert>(as_integer, ov::element::f32);
        return std::make_shared<ov::op::v1::Multiply>(matmul, scale);
    };
    auto reference = make_model(build);
    auto model = make_model(build);
    run_pass(model);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(result_producer(model)));
    expect_near(run(model), run_vector(reference), 0.0f);
}

TEST(PushScaleBeforeMatMul, ScaleConvertLikePreservesEffectivePrecision) {
    for (const bool unsqueeze : {false, true}) {
        const auto build = [unsqueeze](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
            auto rounded = std::make_shared<ov::op::v0::Convert>(Constant::create(ov::element::f32, {}, {0.263f}),
                                                                 ov::element::f16);
            Output<ov::Node> scale = std::make_shared<ov::op::v1::ConvertLike>(rounded, matmul);
            if (unsqueeze) {
                scale = std::make_shared<ov::op::v0::Unsqueeze>(scale, Constant::create(ov::element::i64, {1}, {0}));
            }
            return std::make_shared<ov::op::v1::Multiply>(matmul, scale);
        };
        auto reference = make_model(build);
        auto model = make_model(build);
        run_pass(model);
        EXPECT_TRUE(ov::is_type<ov::op::v0::MatMul>(result_producer(model))) << "unsqueeze " << unsqueeze;
        expect_near(run(model), run_vector(reference), 1e-6f);
    }
}

TEST(PushScaleBeforeMatMul, NonScaleOperandsAreNotConstantFolded) {
    size_t evaluations = 0;
    auto model = make_model([&](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        auto weights = Constant::create(ov::element::f16, {6}, {0.25f});
        auto converted = std::make_shared<CountingConvert>(weights, evaluations);
        auto gate = std::make_shared<ov::op::v4::Swish>(std::make_shared<ov::op::v1::Add>(converted, matmul));
        return std::make_shared<ov::op::v1::Multiply>(matmul, gate);
    });
    evaluations = 0;
    run_pass(model);
    EXPECT_EQ(evaluations, 0);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(result_producer(model)));
}

TEST(PushScaleBeforeMatMul, NonScalarConstantConvertsAreNotFolded) {
    size_t evaluations = 0;
    auto model = make_model([&](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        auto weights = Constant::create(ov::element::f16, {6}, {0.25f});
        auto converted = std::make_shared<CountingConvert>(weights, evaluations);
        return std::make_shared<ov::op::v1::Multiply>(matmul, converted);
    });
    evaluations = 0;
    run_pass(model);
    EXPECT_EQ(evaluations, 0);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(result_producer(model)));
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

TEST(PushScaleBeforeMatMul, PreservesPdpdProductBroadcastAndInputOrder) {
    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    const ov::op::AutoBroadcastSpec broadcast(ov::op::AutoBroadcastType::PDPD, 0);
    auto product = std::make_shared<ov::op::v1::Multiply>(a, b, broadcast);
    auto matmul =
        std::make_shared<ov::op::v0::MatMul>(product, Constant::create(ov::element::f32, {3, 2}, {1, 0, 0, 1, 1, 1}));
    auto output = std::make_shared<ov::op::v1::Multiply>(matmul, Constant::create(ov::element::f32, {}, {0.5f}));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{a, b});
    ov::TensorVector inputs{make_f32_tensor({2, 3}, {1, 2, 3, 4, 5, 6}), make_f32_tensor({2}, {2, 3})};
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, {2, 2})};
    ASSERT_TRUE(model->evaluate(outputs, inputs));
    expect_near(outputs[0], {4, 5, 15, 16.5f}, 0.0f, 0.0f);

    run_pass(model);
    const auto new_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(result_producer(model));
    ASSERT_NE(new_matmul, nullptr);
    const auto new_product = ov::as_type_ptr<ov::op::v1::Multiply>(new_matmul->get_input_node_shared_ptr(0));
    ASSERT_NE(new_product, nullptr);
    EXPECT_EQ(new_product->get_autob(), broadcast);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(new_product->get_input_node_shared_ptr(0)));
    EXPECT_EQ(new_product->get_input_node_shared_ptr(1), b);
    ASSERT_TRUE(model->evaluate(outputs, inputs));
    expect_near(outputs[0], {4, 5, 15, 16.5f}, 0.0f, 0.0f);
}

TEST(PushScaleBeforeMatMul, CopiesScaleRuntimeInfo) {
    for (const bool gated : {false, true}) {
        SCOPED_TRACE(gated);
        auto model = make_model([gated](const Output<ov::Node>& a, const Output<ov::Node>& matmul) -> Output<ov::Node> {
            if (!gated) {
                return std::make_shared<ov::op::v1::Multiply>(matmul, Constant::create(ov::element::f32, {}, {0.5f}));
            }
            auto gate = std::make_shared<ov::op::v4::Swish>(a);
            auto up = std::make_shared<ov::op::v1::Multiply>(a, Constant::create(ov::element::f32, {}, {1.5f}));
            auto product = std::make_shared<ov::op::v1::Multiply>(gate, up);
            product->get_rt_info()["product_marker"] = 7;
            auto gated_matmul =
                std::make_shared<ov::op::v0::MatMul>(product, Constant::create(ov::element::f32, {8, 6}, {0.02f}));
            return std::make_shared<ov::op::v1::Multiply>(gated_matmul, Constant::create(ov::element::f32, {}, {0.5f}));
        });
        result_producer(model)->get_rt_info()["scale_marker"] = 42;
        run_pass(model);
        const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(result_producer(model));
        ASSERT_NE(matmul, nullptr);
        auto scaled = ov::as_type_ptr<ov::op::v1::Multiply>(matmul->get_input_node_shared_ptr(0));
        ASSERT_NE(scaled, nullptr);
        if (gated) {
            ASSERT_EQ(scaled->get_rt_info().count("product_marker"), 1u);
            ASSERT_EQ(scaled->get_rt_info().at("product_marker").as<int>(), 7);
            scaled = ov::as_type_ptr<ov::op::v1::Multiply>(scaled->get_input_node_shared_ptr(1));
            ASSERT_NE(scaled, nullptr);
        }
        ASSERT_EQ(scaled->get_rt_info().count("scale_marker"), 1u);
        EXPECT_EQ(scaled->get_rt_info().at("scale_marker").as<int>(), 42);
        const auto constant = ov::as_type_ptr<Constant>(scaled->get_input_node_shared_ptr(1));
        ASSERT_NE(constant, nullptr);
        ASSERT_EQ(constant->get_rt_info().count("scale_marker"), 1u);
        EXPECT_EQ(constant->get_rt_info().at("scale_marker").as<int>(), 42);
    }
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

TEST(PushScaleBeforeMatMul, RankExpandingScaleIsLeftAlone) {
    const auto build = [](const Output<ov::Node>&, const Output<ov::Node>& matmul) -> Output<ov::Node> {
        return std::make_shared<ov::op::v1::Multiply>(matmul,
                                                      Constant::create(ov::element::f32, {1, 1, 1, 1}, {0.25f}));
    };
    auto reference = make_model(build);
    auto model = make_model(build);
    run_pass(model);
    ASSERT_EQ(model->get_output_partial_shape(0), reference->get_output_partial_shape(0));
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(result_producer(model)));
    expect_near(run(model), run_vector(reference), 1e-6f);
}
