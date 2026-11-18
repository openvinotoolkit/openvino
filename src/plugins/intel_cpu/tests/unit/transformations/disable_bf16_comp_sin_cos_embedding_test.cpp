// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/disable_bf16_comp_sin_cos_embedding.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace ov::intel_cpu;

namespace sin_cos_embedding_test {
namespace {

const std::string name_scale_mul = "scale_mul";
const std::string name_unsqueeze = "unsqueeze";
const std::string name_freq_mul = "freq_mul";
const std::string name_sin = "sin";
const std::string name_cos = "cos";
const std::string name_concat = "concat";
const std::string name_relu = "relu";

std::shared_ptr<ov::Model> create_timestep_embedding_model(bool freq_on_first_input) {
    auto timestep = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});

    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1000.0f});
    auto scale_mul = std::make_shared<ov::op::v1::Multiply>(timestep, scale);
    scale_mul->set_friendly_name(name_scale_mul);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(scale_mul, axis);
    unsqueeze->set_friendly_name(name_unsqueeze);

    auto freqs = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 128}, {0.5f});
    auto freq_mul = freq_on_first_input ? std::make_shared<ov::op::v1::Multiply>(freqs, unsqueeze)
                                        : std::make_shared<ov::op::v1::Multiply>(unsqueeze, freqs);
    freq_mul->set_friendly_name(name_freq_mul);

    auto sin = std::make_shared<ov::op::v0::Sin>(freq_mul);
    sin->set_friendly_name(name_sin);
    auto cos = std::make_shared<ov::op::v0::Cos>(freq_mul);
    cos->set_friendly_name(name_cos);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin, cos}, -1);
    concat->set_friendly_name(name_concat);

    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{timestep});
}

// Sin is fed by a Multiply of two runtime tensors without a frequency Constant, so it must not match.
std::shared_ptr<ov::Model> create_model_mul_without_constant() {
    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 128});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 128});

    auto freq_mul = std::make_shared<ov::op::v1::Multiply>(a, b);
    freq_mul->set_friendly_name(name_freq_mul);

    auto sin = std::make_shared<ov::op::v0::Sin>(freq_mul);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{a, b});
}

// Sin is fed by a non-Multiply op, so it must not match.
std::shared_ptr<ov::Model> create_model_sin_after_relu() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 128});

    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    relu->set_friendly_name(name_relu);

    auto sin = std::make_shared<ov::op::v0::Sin>(relu);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

}  // namespace

class DisableBF16CompSinCosEmbeddingTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<DisableBF16CompSinCosEmbedding>();
    }

    void TearDown() override {
        TransformationTestsF::TearDown();
        if (HasFatalFailure()) {
            return;
        }
        for (const auto& op : model->get_ops()) {
            EXPECT_EQ(ov::is_conversion_disabled(op, ov::element::f32, ov::element::bf16),
                      expected_marked.count(op->get_friendly_name()) > 0)
                << "Unexpected BF16 conversion mark on node: " << op->get_friendly_name();
        }
    }

    void expect_marked(const std::unordered_set<std::string>& names) {
        expected_marked = names;
        for (const auto& op : model_ref->get_ops()) {
            if (names.count(op->get_friendly_name())) {
                ov::disable_conversion(op, ov::element::f32, ov::element::bf16);
            }
        }
    }

    std::unordered_set<std::string> expected_marked;
};

TEST_F(DisableBF16CompSinCosEmbeddingTest, TimestepEmbedding) {
    model = create_timestep_embedding_model(false);
    model_ref = create_timestep_embedding_model(false);
    expect_marked({name_freq_mul, name_sin, name_cos});
}

TEST_F(DisableBF16CompSinCosEmbeddingTest, ConstantOnFirstInput) {
    model = create_timestep_embedding_model(true);
    model_ref = create_timestep_embedding_model(true);
    expect_marked({name_freq_mul, name_sin, name_cos});
}

TEST_F(DisableBF16CompSinCosEmbeddingTest, MultiplyWithoutConstant_NoOp) {
    model = create_model_mul_without_constant();
}

TEST_F(DisableBF16CompSinCosEmbeddingTest, SinAfterRelu_NoOp) {
    model = create_model_sin_after_relu();
}

}  // namespace sin_cos_embedding_test
