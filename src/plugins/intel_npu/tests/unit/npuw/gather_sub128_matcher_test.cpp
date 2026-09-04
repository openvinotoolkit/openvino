// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "npuw_transformations/insert_vocab_sub128.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/opt.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace {

std::shared_ptr<ov::Model> make_gather_model(float shift_value) {
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1});
    auto weights = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 2}, std::vector<uint8_t>(8, 200));
    auto zero_point = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 1}, std::vector<uint8_t>(4, 128));
    auto scale = ov::opset10::Constant::create(ov::element::f16, ov::Shape{4, 1}, std::vector<float>(4, 1.0f));
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, ov::element::f16);
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f16);
    auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {shift_value});
    auto shifted_weight = std::make_shared<ov::opset10::Subtract>(weight_convert, shift);
    auto shifted_zero_point = std::make_shared<ov::opset10::Subtract>(zero_point_convert, shift);

    auto dequantized = std::make_shared<ov::opset10::Subtract>(shifted_weight, shifted_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    auto gathered = std::make_shared<ov::opset10::Gather>(converted, ids, axis);
    auto result = std::make_shared<ov::opset10::Result>(gathered);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids});
}

std::size_t count_gathers(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Gather>(node)) {
            ++count;
        }
    }
    return count;
}

bool run_lift(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::DQLiftGatherAsymCW>();
    return rewrite.run_on_model(model);
}

std::shared_ptr<ov::Model> make_parameter_gather_model(std::optional<float> weight_shift,
                                                       std::optional<float> zero_point_shift) {
    constexpr std::size_t vocab_size = 4096;
    constexpr std::size_t hidden_size = 2048;
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto weights = std::make_shared<ov::opset10::Parameter>(ov::element::u8, ov::Shape{vocab_size, hidden_size});
    auto zero_point = std::make_shared<ov::opset10::Parameter>(ov::element::u8, ov::Shape{vocab_size, hidden_size});
    auto scale = std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{vocab_size, hidden_size});
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto gathered_weights = std::make_shared<ov::opset10::Gather>(weights, ids, axis);
    auto gathered_zero_point = std::make_shared<ov::opset10::Gather>(zero_point, ids, axis);
    auto gathered_scale = std::make_shared<ov::opset10::Gather>(scale, ids, axis);
    auto weight_convert = std::make_shared<ov::opset10::Convert>(gathered_weights, ov::element::f16);
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(gathered_zero_point, ov::element::f16);
    ov::Output<ov::Node> dequantized_weight = weight_convert;
    ov::Output<ov::Node> dequantized_zero_point = zero_point_convert;
    if (weight_shift.has_value()) {
        auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {weight_shift.value()});
        dequantized_weight = std::make_shared<ov::opset10::Subtract>(weight_convert, shift);
    }
    if (zero_point_shift.has_value()) {
        auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {zero_point_shift.value()});
        dequantized_zero_point = std::make_shared<ov::opset10::Subtract>(zero_point_convert, shift);
    }

    auto dequantized = std::make_shared<ov::opset10::Subtract>(dequantized_weight, dequantized_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, gathered_scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f16);
    auto result = std::make_shared<ov::opset10::Result>(converted);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids, weights, zero_point, scale});
}

std::shared_ptr<ov::Model> make_parameter_gather_model(float shift_value) {
    return make_parameter_gather_model(shift_value, shift_value);
}

bool run_host_gather(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& context) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::HostGatherQuantAsymm<>>(std::ref(context));
    return rewrite.run_on_model(model);
}

std::shared_ptr<ov::Model> make_vocab_matmul_model() {
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto weights = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 2}, std::vector<uint8_t>(8, 200));
    auto zero_point = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 1}, std::vector<uint8_t>(4, 128));
    auto scale = ov::opset10::Constant::create(ov::element::f16, ov::Shape{4, 1}, std::vector<float>(4, 1.0f));
    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, ov::element::f16);
    weight_convert->set_friendly_name("vocab_weight_convert");
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f16);
    zero_point_convert->set_friendly_name("vocab_zero_point_convert");
    auto dequantized = std::make_shared<ov::opset10::Subtract>(weight_convert, zero_point_convert);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    auto matmul = std::make_shared<ov::opset10::MatMul>(hidden, converted, false, true);
    return std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{hidden});
}

bool contains_node(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto nodes = model->get_ordered_ops();
    return std::any_of(nodes.begin(), nodes.end(), [&](const auto& node) {
        return node->get_friendly_name() == name;
    });
}

}  // namespace

TEST(DQLiftGatherAsymCWTest, LiftsPairedSub128Shifts) {
    const auto model = make_gather_model(128.0f);

    EXPECT_TRUE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 3);
}

TEST(DQLiftGatherAsymCWTest, RejectsNon128Subtractions) {
    const auto model = make_gather_model(127.0f);

    EXPECT_FALSE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 1);
}

TEST(HostGatherQuantAsymmTest, AcceptsPairedSub128Shifts) {
    ov::npuw::patterns::opt::Context context;
    EXPECT_TRUE(run_host_gather(make_parameter_gather_model(128.0f), context));
    ASSERT_TRUE(context.params_to_quant_gather_unpack.has_value());
    ASSERT_EQ(context.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1);
    EXPECT_TRUE(context.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.begin()->second.apply_sub128);
}

TEST(HostGatherQuantAsymmTest, RejectsNon128Subtractions) {
    ov::npuw::patterns::opt::Context context;
    EXPECT_FALSE(run_host_gather(make_parameter_gather_model(127.0f), context));
}

TEST(InsertVocabSub128PrePostProcessingTest, PreservesVocabularyConverts) {
    const auto model = make_vocab_matmul_model();
    ov::npuw::InsertVocabSub128().run_on_model(model);

    const auto nodes = model->get_ordered_ops();
    const auto weight_convert = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
        return node->get_friendly_name() == "vocab_weight_convert";
    });
    const auto zero_point_convert = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
        return node->get_friendly_name() == "vocab_zero_point_convert";
    });
    ASSERT_NE(weight_convert, nodes.end());
    ASSERT_NE(zero_point_convert, nodes.end());
    EXPECT_TRUE(ov::is_decompression(*weight_convert));
    EXPECT_TRUE(ov::is_decompression(*zero_point_convert));

    ov::preprocess::PrePostProcessor(model).build();

    EXPECT_TRUE(contains_node(model, "vocab_weight_convert"));
    EXPECT_TRUE(contains_node(model, "vocab_zero_point_convert"));
}
