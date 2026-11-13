// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "lazy_tensor.hpp"
#include "npuw_transformations/insert_vocab_sub128.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/tensor.hpp"
#include "partitioning/partitioning.hpp"
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
    if (shift_value == 128.0f) {
        shifted_weight->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        shifted_zero_point->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    }

    auto dequantized = std::make_shared<ov::opset10::Subtract>(shifted_weight, shifted_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    auto gathered = std::make_shared<ov::opset10::Gather>(converted, ids, axis);
    auto result = std::make_shared<ov::opset10::Result>(gathered);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids});
}

std::shared_ptr<ov::Model> make_vocab_sharing_model(bool shared,
                                                    bool symmetric = false,
                                                    const ov::Shape& scale_shape = {4, 1}) {
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto embedding_weights = symmetric
                                 ? ov::opset10::Constant::create(ov::element::i4, ov::Shape{4, 2},
                                                                 std::vector<int8_t>(8, 1))
                                 : ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 2},
                                                                 std::vector<uint8_t>(8, 200));
    auto embedding_zero_point = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 1},
                                                              std::vector<uint8_t>(4, 128));
    auto embedding_scale = ov::opset10::Constant::create(ov::element::f16, scale_shape,
                                                         std::vector<float>(ov::shape_size(scale_shape), 1.0f));

    auto lm_head_weights = shared
                               ? embedding_weights
                               : (symmetric ? ov::opset10::Constant::create(ov::element::i4, ov::Shape{4, 2},
                                                                             std::vector<int8_t>(8, 1))
                                            : ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 2},
                                                                            std::vector<uint8_t>(8, 200)));
    auto lm_head_zero_point = shared
                                  ? embedding_zero_point
                                  : ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 1},
                                                                  std::vector<uint8_t>(4, 128));
    auto lm_head_scale = shared ? embedding_scale : ov::opset10::Constant::create(ov::element::f16,
                                                                                    scale_shape,
                                                                                    std::vector<float>(ov::shape_size(scale_shape), 1.0f));

    auto embedding_convert = std::make_shared<ov::opset10::Convert>(embedding_weights, ov::element::f16);
    ov::Output<ov::Node> embedding_dequantized = embedding_convert;
    if (!symmetric) {
        auto embedding_zero_point_convert =
            std::make_shared<ov::opset10::Convert>(embedding_zero_point, ov::element::f16);
        embedding_dequantized =
            std::make_shared<ov::opset10::Subtract>(embedding_convert, embedding_zero_point_convert);
    }
    auto embedding_scaled = std::make_shared<ov::opset10::Multiply>(embedding_dequantized, embedding_scale);
    auto embedding_gather = std::make_shared<ov::opset10::Gather>(
        std::make_shared<ov::opset10::Convert>(embedding_scaled, ov::element::f32), ids, axis);

    auto lm_head_convert = std::make_shared<ov::opset10::Convert>(lm_head_weights, ov::element::f16);
    ov::Output<ov::Node> lm_head_dequantized = lm_head_convert;
    if (!symmetric) {
        auto lm_head_zero_point_convert =
            std::make_shared<ov::opset10::Convert>(lm_head_zero_point, ov::element::f16);
        lm_head_dequantized = std::make_shared<ov::opset10::Subtract>(lm_head_convert, lm_head_zero_point_convert);
    }
    auto lm_head_scaled = std::make_shared<ov::opset10::Multiply>(lm_head_dequantized, lm_head_scale);
    auto lm_head_matmul = std::make_shared<ov::opset10::MatMul>(
        hidden,
        std::make_shared<ov::opset10::Convert>(lm_head_scaled, ov::element::f32),
        false,
        true);

    auto embedding_result = std::make_shared<ov::opset10::Result>(embedding_gather);
    auto lm_head_result = std::make_shared<ov::opset10::Result>(lm_head_matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{embedding_result, lm_head_result},
                                       ov::ParameterVector{ids, hidden});
}

std::shared_ptr<ov::Model> make_full_precision_vocab_sharing_model(bool shared) {
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto embedding_weights = ov::opset10::Constant::create(ov::element::f16,
                                                            ov::Shape{4, 2},
                                                            std::vector<float>(8, 1.0f));
    auto lm_head_weights = shared
                               ? embedding_weights
                               : ov::opset10::Constant::create(ov::element::f16,
                                                               ov::Shape{4, 2},
                                                               std::vector<float>(8, 1.0f));
    auto embedding = std::make_shared<ov::opset10::Gather>(
        std::make_shared<ov::opset10::Convert>(embedding_weights, ov::element::f32), ids, axis);
    auto lm_head = std::make_shared<ov::opset10::MatMul>(
        hidden,
        std::make_shared<ov::opset10::Convert>(lm_head_weights, ov::element::f32),
        false,
        true);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::opset10::Result>(embedding),
                                                        std::make_shared<ov::opset10::Result>(lm_head)},
                                       ov::ParameterVector{ids, hidden});
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
                                                       std::optional<float> zero_point_shift,
                                                       ov::element::Type weight_type = ov::element::u8,
                                                       ov::element::Type zero_point_type = ov::element::u8) {
    constexpr std::size_t vocab_size = 4096;
    constexpr std::size_t hidden_size = 2048;
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto weights = std::make_shared<ov::opset10::Parameter>(weight_type, ov::Shape{vocab_size, hidden_size});
    auto zero_point = std::make_shared<ov::opset10::Parameter>(zero_point_type, ov::Shape{vocab_size, hidden_size});
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
        if (weight_shift.value() == 128.0f) {
            dequantized_weight.get_node_shared_ptr()->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        }
    }
    if (zero_point_shift.has_value()) {
        auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {zero_point_shift.value()});
        dequantized_zero_point = std::make_shared<ov::opset10::Subtract>(zero_point_convert, shift);
        if (zero_point_shift.value() == 128.0f) {
            dequantized_zero_point.get_node_shared_ptr()->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        }
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

std::shared_ptr<ov::Model> make_parameter_vocab_dq_model() {
    auto weights = std::make_shared<ov::opset10::Parameter>(ov::element::u8, ov::Shape{4, 2});
    auto zero_point = std::make_shared<ov::opset10::Parameter>(ov::element::u8, ov::Shape{4, 1});
    auto scale = std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{4, 1});

    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, ov::element::f16);
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f16);
    auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {128.0f});
    auto shifted_weight = std::make_shared<ov::opset10::Subtract>(weight_convert, shift);
    auto shifted_zero_point = std::make_shared<ov::opset10::Subtract>(zero_point_convert, shift);
    shifted_weight->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    shifted_zero_point->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;

    auto dequantized = std::make_shared<ov::opset10::Subtract>(shifted_weight, shifted_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    auto result = std::make_shared<ov::opset10::Result>(scaled);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weights, zero_point, scale});
}

bool run_host_gather(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& context) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::HostGatherQuantAsymm<>>(std::ref(context));
    return rewrite.run_on_model(model);
}

enum class VocabMatMulTerminal { Add, Transpose, Convert, Gated, Divide };

std::shared_ptr<ov::Model> make_vocab_matmul_model(bool convert_before_matmul,
                                                   std::optional<VocabMatMulTerminal> terminal = std::nullopt,
                                                   std::size_t vocab_size = 4) {
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto weights = ov::opset10::Constant::create(ov::element::u8,
                                                 ov::Shape{vocab_size, 2},
                                                 std::vector<uint8_t>(vocab_size * 2, 200));
    auto zero_point =
        ov::opset10::Constant::create(ov::element::u8, ov::Shape{vocab_size, 1}, std::vector<uint8_t>(vocab_size, 128));
    const auto scale_type = convert_before_matmul ? ov::element::f16 : ov::element::f32;
    auto scale =
        ov::opset10::Constant::create(scale_type, ov::Shape{vocab_size, 1}, std::vector<float>(vocab_size, 1.0f));
    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, scale_type);
    weight_convert->set_friendly_name("vocab_weight_convert");
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, scale_type);
    zero_point_convert->set_friendly_name("vocab_zero_point_convert");
    auto dequantized = std::make_shared<ov::opset10::Subtract>(weight_convert, zero_point_convert);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    ov::Output<ov::Node> matmul_weights = scaled;
    if (convert_before_matmul) {
        matmul_weights = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    }
    auto matmul = std::make_shared<ov::opset10::MatMul>(hidden, matmul_weights, false, true);
    ov::Output<ov::Node> output = matmul;
    if (terminal == VocabMatMulTerminal::Add) {
        auto bias = ov::opset10::Constant::create(ov::element::f32,
                                                  ov::Shape{1, vocab_size},
                                                  std::vector<float>(vocab_size, 1.0f));
        output = std::make_shared<ov::opset10::Add>(matmul, bias);
    } else if (terminal == VocabMatMulTerminal::Transpose) {
        auto order = ov::opset10::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        output = std::make_shared<ov::opset10::Transpose>(matmul, order);
    } else if (terminal == VocabMatMulTerminal::Convert) {
        output = std::make_shared<ov::opset10::Convert>(matmul, ov::element::f32);
    } else if (terminal == VocabMatMulTerminal::Gated) {
        auto divisor = ov::opset10::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        auto div = std::make_shared<ov::opset10::Divide>(matmul, divisor);
        auto tanh = std::make_shared<ov::opset10::Tanh>(div);
        auto multiplier = ov::opset10::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        output = std::make_shared<ov::opset10::Multiply>(tanh, multiplier);
    } else if (terminal == VocabMatMulTerminal::Divide) {
        auto divisor = ov::opset10::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        output = std::make_shared<ov::opset10::Divide>(matmul, divisor);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{hidden});
}

std::shared_ptr<ov::Model> make_pretransposed_vocab_matmul_model(bool convert_before_matmul,
                                                                 std::size_t vocab_size = 4) {
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto weights = ov::opset10::Constant::create(ov::element::u8,
                                                 ov::Shape{2, vocab_size},
                                                 std::vector<uint8_t>(vocab_size * 2, 200));
    auto zero_point =
        ov::opset10::Constant::create(ov::element::u8, ov::Shape{1, vocab_size}, std::vector<uint8_t>(vocab_size, 128));
    const auto scale_type = convert_before_matmul ? ov::element::f16 : ov::element::f32;
    auto scale =
        ov::opset10::Constant::create(scale_type, ov::Shape{1, vocab_size}, std::vector<float>(vocab_size, 1.0f));
    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, scale_type);
    weight_convert->set_friendly_name("vocab_weight_convert");
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, scale_type);
    zero_point_convert->set_friendly_name("vocab_zero_point_convert");
    auto dequantized = std::make_shared<ov::opset10::Subtract>(weight_convert, zero_point_convert);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    ov::Output<ov::Node> matmul_weights = scaled;
    if (convert_before_matmul) {
        matmul_weights = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    }
    auto matmul = std::make_shared<ov::opset10::MatMul>(hidden, matmul_weights, false, false);
    return std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{hidden});
}

std::size_t count_subtracts(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node)) {
            ++count;
        }
    }
    return count;
}

std::size_t count_sub128_shifts(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node) &&
            node->get_rt_info().count(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO) > 0) {
            ++count;
        }
    }
    return count;
}

void clear_sub128_shift_markers(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node)) {
            node->get_rt_info().erase(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO);
        }
    }
}

bool contains_node(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto nodes = model->get_ordered_ops();
    return std::any_of(nodes.begin(), nodes.end(), [&](const auto& node) {
        return node->get_friendly_name() == name;
    });
}

struct TwoLmHeadModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Node> tied_weight;
    std::shared_ptr<ov::Node> tied_zero_point;  // nullptr for the f16 tied vocabulary
    std::shared_ptr<ov::Node> extra_weight;
    std::shared_ptr<ov::Node> extra_zero_point;
};

ov::Output<ov::Node> make_u8_dequantization(const std::shared_ptr<ov::Node>& weight,
                                            const std::shared_ptr<ov::Node>& zero_point,
                                            const std::shared_ptr<ov::Node>& scale) {
    auto weight_convert = std::make_shared<ov::opset10::Convert>(weight, ov::element::f32);
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f32);
    auto dequantized = std::make_shared<ov::opset10::Subtract>(weight_convert, zero_point_convert);
    return std::make_shared<ov::opset10::Multiply>(dequantized, scale);
}

// Embedding Gather + main LM head share one vocabulary (tied). An extra LM head (for example an
// MTP or auxiliary head) has its own, separate u8 asymmetric vocabulary. As in a real IR, each
// consumer of the tied vocabulary has its own dequantization nodes over the same constants.
TwoLmHeadModel make_two_lm_head_model(bool tied_u8, bool extra_pretransposed = false) {
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1});
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    std::shared_ptr<ov::Node> tied_weight;
    std::shared_ptr<ov::Node> tied_zero_point;
    ov::Output<ov::Node> embedding_vocab;
    ov::Output<ov::Node> main_head_vocab;
    if (tied_u8) {
        tied_weight = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 2}, std::vector<uint8_t>(8, 200));
        tied_zero_point = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 1}, std::vector<uint8_t>(4, 128));
        auto scale = ov::opset10::Constant::create(ov::element::f32, ov::Shape{4, 1}, std::vector<float>(4, 1.0f));
        embedding_vocab = make_u8_dequantization(tied_weight, tied_zero_point, scale);
        main_head_vocab = make_u8_dequantization(tied_weight, tied_zero_point, scale);
    } else {
        tied_weight = ov::opset10::Constant::create(ov::element::f16, ov::Shape{4, 2}, std::vector<float>(8, 1.0f));
        embedding_vocab = std::make_shared<ov::opset10::Convert>(tied_weight, ov::element::f32);
        main_head_vocab = std::make_shared<ov::opset10::Convert>(tied_weight, ov::element::f32);
    }

    // Pretransposed extra head: weight [H, V], zero point and scale [1, V], MatMul without transpose_b.
    const auto extra_weight_shape = extra_pretransposed ? ov::Shape{2, 4} : ov::Shape{4, 2};
    const auto extra_channel_shape = extra_pretransposed ? ov::Shape{1, 4} : ov::Shape{4, 1};
    auto extra_weight =
        ov::opset10::Constant::create(ov::element::u8, extra_weight_shape, std::vector<uint8_t>(8, 100));
    auto extra_zero_point =
        ov::opset10::Constant::create(ov::element::u8, extra_channel_shape, std::vector<uint8_t>(4, 120));
    auto extra_scale =
        ov::opset10::Constant::create(ov::element::f32, extra_channel_shape, std::vector<float>(4, 0.5f));
    auto extra_head_vocab = make_u8_dequantization(extra_weight, extra_zero_point, extra_scale);

    auto embedding = std::make_shared<ov::opset10::Gather>(embedding_vocab, ids, axis);
    auto main_head = std::make_shared<ov::opset10::MatMul>(hidden, main_head_vocab, false, true);
    auto extra_head = std::make_shared<ov::opset10::MatMul>(hidden, extra_head_vocab, false, !extra_pretransposed);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::opset10::Result>(embedding),
                                                              std::make_shared<ov::opset10::Result>(main_head),
                                                              std::make_shared<ov::opset10::Result>(extra_head)},
                                             ov::ParameterVector{ids, hidden});
    return {model, tied_weight, tied_zero_point, extra_weight, extra_zero_point};
}

// Same gate as in LLMCompiledModel when NPUW_LLM_VOCAB_ASYM_SHARED=YES. Returns the detection result.
bool run_vocab_asym_shared_gate(const std::shared_ptr<ov::Model>& model) {
    bool shared = false;
    ov::npuw::DetectVocabSharing(shared).run_on_model(model);
    if (!shared) {
        return false;
    }
    ov::npuw::InsertVocabSub128().run_on_model(model);
    return true;
}

// Source constants (behind the Convert) of all Subtracts marked as Sub128 shifts.
std::set<std::shared_ptr<ov::Node>> sub128_shift_sources(const std::shared_ptr<ov::Model>& model) {
    std::set<std::shared_ptr<ov::Node>> sources;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node) &&
            node->get_rt_info().count(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO) > 0) {
            const auto convert = node->input_value(0).get_node_shared_ptr();
            sources.insert(convert->input_value(0).get_node_shared_ptr());
        }
    }
    return sources;
}

}  // namespace

TEST(DQLiftGatherAsymCWTest, LiftsPairedSub128Shifts) {
    const auto model = make_gather_model(128.0f);

    EXPECT_TRUE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 3);
}

TEST(DetectVocabSharingTest, DetectsSharedQuantizedVocabulary) {
    bool shared = false;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(make_vocab_sharing_model(true)));
    EXPECT_TRUE(shared);
}

TEST(DetectVocabSharingTest, RejectsSeparateQuantizedVocabularies) {
    bool shared = true;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(make_vocab_sharing_model(false)));
    EXPECT_FALSE(shared);
}

TEST(DetectVocabSharingTest, DetectsDistinctConstantsOverOneBuffer) {
    // A tied vocabulary read from IR: the embedding and the LM head get two different Constant
    // nodes, which view one buffer.
    ov::Tensor weight(ov::element::u8, ov::Shape{4, 2});
    ov::Tensor zero_point(ov::element::u8, ov::Shape{4, 1});
    ov::Tensor scale(ov::element::f32, ov::Shape{4, 1});
    std::fill_n(weight.data<uint8_t>(), weight.get_size(), uint8_t{200});
    std::fill_n(zero_point.data<uint8_t>(), zero_point.get_size(), uint8_t{128});
    std::fill_n(scale.data<float>(), scale.get_size(), 1.0f);
    const auto embedding_weight = std::make_shared<ov::opset10::Constant>(weight);
    const auto lm_head_weight = std::make_shared<ov::opset10::Constant>(weight);
    ASSERT_NE(embedding_weight, lm_head_weight);
    ASSERT_EQ(embedding_weight->get_data_ptr(), lm_head_weight->get_data_ptr());
    const auto embedding_vocab = make_u8_dequantization(embedding_weight,
                                                        std::make_shared<ov::opset10::Constant>(zero_point),
                                                        std::make_shared<ov::opset10::Constant>(scale));
    const auto lm_head_vocab = make_u8_dequantization(lm_head_weight,
                                                      std::make_shared<ov::opset10::Constant>(zero_point),
                                                      std::make_shared<ov::opset10::Constant>(scale));

    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1});
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto embedding = std::make_shared<ov::opset10::Gather>(embedding_vocab, ids, axis);
    auto lm_head = std::make_shared<ov::opset10::MatMul>(hidden, lm_head_vocab, false, true);
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::opset10::Result>(embedding),
                                                                    std::make_shared<ov::opset10::Result>(lm_head)},
                                                   ov::ParameterVector{ids, hidden});
    bool shared = false;

    EXPECT_FALSE(ov::npuw::DetectVocabSharing(shared).run_on_model(model));
    EXPECT_TRUE(shared);
}

TEST(DetectVocabSharingTest, DetectsSharedSymmetricI4Vocabulary) {
    bool shared = false;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(make_vocab_sharing_model(true, true)));
    EXPECT_TRUE(shared);
}

TEST(DetectVocabSharingTest, RejectsSeparateSymmetricI4Vocabularies) {
    bool shared = true;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(make_vocab_sharing_model(false, true)));
    EXPECT_FALSE(shared);
}

TEST(DetectVocabSharingTest, DetectsSharedFullPrecisionVocabulary) {
    bool shared = false;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(make_full_precision_vocab_sharing_model(true)));
    EXPECT_TRUE(shared);
}

TEST(DetectVocabSharingTest, RejectsSeparateFullPrecisionVocabularies) {
    bool shared = true;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(make_full_precision_vocab_sharing_model(false)));
    EXPECT_FALSE(shared);
}

using VocabScaleTestParams = std::tuple<bool, bool, ov::Shape>;

class DetectVocabSharingScaleTest : public ::testing::TestWithParam<VocabScaleTestParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<VocabScaleTestParams>& info) {
        const auto& [shared, symmetric, shape] = info.param;
        std::string name = std::string(shared ? "Shared" : "Separate") + (symmetric ? "I4" : "U8") + "Scale";
        if (shape.empty()) {
            return name + "Scalar";
        }
        for (const auto dimension : shape) {
            name += "_" + std::to_string(dimension);
        }
        return name;
    }
};

TEST_P(DetectVocabSharingScaleTest, ComparesStorageWithoutRestrictingScaleShape) {
    const auto& [expected_shared, symmetric, scale_shape] = GetParam();
    const auto model = make_vocab_sharing_model(expected_shared, symmetric, scale_shape);
    const auto original_nodes = model->get_ordered_ops();
    bool shared = !expected_shared;
    ov::npuw::DetectVocabSharing pass(shared);

    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_EQ(shared, expected_shared);
    EXPECT_EQ(model->get_ordered_ops(), original_nodes);
}

INSTANTIATE_TEST_SUITE_P(
    ScaleLayouts,
    DetectVocabSharingScaleTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(ov::Shape{}, ov::Shape{1}, ov::Shape{2}, ov::Shape{1, 1},
                                         ov::Shape{1, 2}, ov::Shape{4, 1}, ov::Shape{4, 2})),
    DetectVocabSharingScaleTest::getTestCaseName);

TEST(DetectVocabSharingTest, RejectsMissingVocabularyBranchAndResetsState) {
    bool shared = false;
    ov::npuw::DetectVocabSharing pass(shared);
    const auto model = make_vocab_sharing_model(true);
    EXPECT_FALSE(pass.run_on_model(model));
    ASSERT_TRUE(shared);

    model->remove_result(model->get_results().back());
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_FALSE(shared);

    const auto lm_head_only = make_vocab_sharing_model(true);
    lm_head_only->remove_result(lm_head_only->get_results().front());
    shared = true;
    EXPECT_FALSE(pass.run_on_model(lm_head_only));
    EXPECT_FALSE(shared);
}

TEST(DetectVocabSharingTest, RejectsNonTerminalMatMul) {
    const auto model = make_vocab_sharing_model(true);
    const auto result = model->get_results().back();
    const auto projection = ov::opset10::Constant::create(ov::element::f32, ov::Shape{4, 2}, {1.0f});
    const auto intermediate = std::make_shared<ov::opset10::MatMul>(result->input_value(0), projection, false, false);
    result->input(0).replace_source_output(intermediate);
    bool shared = true;

    EXPECT_FALSE(ov::npuw::DetectVocabSharing(shared).run_on_model(model));
    EXPECT_FALSE(shared);
}

TEST(DetectVocabSharingTest, RejectsManuallyAddedLmHeadOutput) {
    const auto model = make_vocab_sharing_model(true);
    model->get_results().back()->get_rt_info()["manually_added_output"] = true;
    bool shared = true;

    EXPECT_FALSE(ov::npuw::DetectVocabSharing(shared).run_on_model(model));
    EXPECT_FALSE(shared);
}

class DetectVocabSharingTerminalTest : public ::testing::TestWithParam<VocabMatMulTerminal> {};

TEST_P(DetectVocabSharingTerminalTest, DetectsSharedVocabularyThroughLmHeadTerminal) {
    const auto model = make_vocab_matmul_model(false, GetParam());
    const auto nodes = model->get_ordered_ops();
    const auto matmul = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
        return ov::is_type<ov::opset10::MatMul>(node);
    });
    ASSERT_NE(matmul, nodes.end());
    const auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1});
    const auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto embedding = std::make_shared<ov::opset10::Gather>((*matmul)->input_value(1), ids, axis);
    model->add_parameters({ids});
    model->add_results({std::make_shared<ov::opset10::Result>(embedding)});
    bool shared = false;

    EXPECT_FALSE(ov::npuw::DetectVocabSharing(shared).run_on_model(model));
    EXPECT_TRUE(shared);
}

INSTANTIATE_TEST_SUITE_P(LmHeadTerminal,
                       DetectVocabSharingTerminalTest,
                       ::testing::Values(VocabMatMulTerminal::Add,
                                         VocabMatMulTerminal::Transpose,
                                         VocabMatMulTerminal::Convert,
                                         VocabMatMulTerminal::Divide,
                                         VocabMatMulTerminal::Gated));

// InsertVocabSub128 marks every LM head that it matches. Thus detection must stop for a model with
// more than one LM head, else an extra head with its own u8 vocabulary is marked too.
TEST(DetectVocabSharingTwoLmHeadsTest, AbortsWhenF16HeadIsTiedAndU8HeadIsSeparate) {
    const auto two_heads = make_two_lm_head_model(false);

    EXPECT_FALSE(run_vocab_asym_shared_gate(two_heads.model));
    EXPECT_EQ(count_sub128_shifts(two_heads.model), 0u);
}

TEST(DetectVocabSharingTwoLmHeadsTest, AbortsWhenU8HeadIsTiedAndU8HeadIsSeparate) {
    const auto two_heads = make_two_lm_head_model(true);

    EXPECT_FALSE(run_vocab_asym_shared_gate(two_heads.model));
    EXPECT_EQ(count_sub128_shifts(two_heads.model), 0u);
}

TEST(DetectVocabSharingTwoLmHeadsTest, CountsPretransposedSeparateHead) {
    // The pretransposed head cannot share the embedding storage, but InsertVocabSub128 marks it.
    const auto two_heads = make_two_lm_head_model(true, true);

    EXPECT_FALSE(run_vocab_asym_shared_gate(two_heads.model));
    EXPECT_EQ(count_sub128_shifts(two_heads.model), 0u);
}

TEST(DetectVocabSharingTwoLmHeadsTest, IgnoresManuallyAddedSecondHead) {
    // A manually added output (for example the Eagle-3 hidden state) is not an LM head.
    const auto two_heads = make_two_lm_head_model(true);
    two_heads.model->get_results().back()->get_rt_info()["manually_added_output"] = true;

    EXPECT_TRUE(run_vocab_asym_shared_gate(two_heads.model));
    const std::set<std::shared_ptr<ov::Node>> expected{two_heads.tied_weight, two_heads.tied_zero_point};
    EXPECT_EQ(sub128_shift_sources(two_heads.model), expected);
}

TEST(DQLiftGatherAsymCWTest, RejectsNon128Subtractions) {
    const auto model = make_gather_model(127.0f);

    EXPECT_FALSE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 1);
}

TEST(DQLiftGatherAsymCWTest, RejectsUnmarked128Subtractions) {
    const auto model = make_gather_model(128.0f);
    clear_sub128_shift_markers(model);

    EXPECT_FALSE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 1);
}

TEST(HostGatherQuantAsymmTest, AcceptsPairedSub128Shifts) {
    const auto model = make_parameter_gather_model(128.0f);
    ov::npuw::patterns::opt::Context context(model, 0);
    EXPECT_TRUE(run_host_gather(model, context));
    ASSERT_TRUE(context.params_to_quant_gather_unpack.has_value());
    ASSERT_EQ(context.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1);
}

TEST(HostGatherQuantAsymmTest, MarksPairedSub128Sources) {
    const auto model = make_parameter_gather_model(128.0f);
    ov::npuw::patterns::opt::Context context(model, 0);

    EXPECT_TRUE(run_host_gather(model, context));
    ASSERT_TRUE(context.params_to_quant_gather_unpack.has_value());
    const auto& gather = context.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.begin()->second;
    ASSERT_EQ(gather.w->get_element_type(), ov::element::i8);
    ASSERT_EQ(gather.z->get_element_type(), ov::element::i8);
    ASSERT_EQ(context.closures_to_subtract_128.size(), 2);
    for (const auto& shifted_and_source : context.closures_to_subtract_128) {
        EXPECT_EQ(shifted_and_source.first->get_element_type(), ov::element::i8);
        EXPECT_TRUE(shifted_and_source.second == model->get_parameters().at(1) ||
                    shifted_and_source.second == model->get_parameters().at(2));
    }
    // Simulating partitioning behavior
    for (const auto& shifted_and_source : context.closures_to_subtract_128) {
        model->add_parameters({shifted_and_source.first});
        if (shifted_and_source.second->output(0).get_target_inputs().empty()) {
            model->remove_parameter(shifted_and_source.second);
        }
    }

    std::size_t i8_parameter_count = 0;
    std::size_t u8_parameter_count = 0;
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_element_type() == ov::element::i8) {
            ++i8_parameter_count;
        } else if (parameter->get_element_type() == ov::element::u8) {
            ++u8_parameter_count;
        }
    }
    EXPECT_EQ(i8_parameter_count, 2u);
    EXPECT_EQ(u8_parameter_count, 0u);
}

TEST(HostGatherQuantAsymmTest, KeepsRawSourcesWithoutMarkedPattern) {
    const auto model = make_parameter_gather_model(std::nullopt,
                                                   std::nullopt,
                                                   ov::element::u8,
                                                   ov::element::u8);
    ov::npuw::patterns::opt::Context context(model, 0);

    EXPECT_TRUE(run_host_gather(model, context));
    ASSERT_TRUE(context.params_to_quant_gather_unpack.has_value());
    const auto& gather = context.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.begin()->second;
    ASSERT_EQ(gather.w->get_element_type(), ov::element::u8);
    ASSERT_EQ(gather.z->get_element_type(), ov::element::u8);
    EXPECT_TRUE(context.closures_to_subtract_128.empty());
}

TEST(HostGatherQuantAsymmTest, RejectsNon128Subtractions) {
    const auto model = make_parameter_gather_model(127.0f);
    ov::npuw::patterns::opt::Context context(model, 0);
    EXPECT_FALSE(run_host_gather(model, context));
}

TEST(HostGatherQuantAsymmTest, RejectsUnmarked128Subtractions) {
    const auto model = make_parameter_gather_model(128.0f);
    clear_sub128_shift_markers(model);
    ov::npuw::patterns::opt::Context context(model, 0);

    EXPECT_FALSE(run_host_gather(model, context));
    EXPECT_FALSE(context.params_to_quant_gather_unpack.has_value());
    EXPECT_EQ(count_subtracts(model), 3);
}

namespace {

constexpr std::size_t kGroups = 16;
constexpr std::size_t kGroupSize = 128;  // kGroups * kGroupSize == hidden size of the vocabulary
constexpr std::size_t kOutFeatures = 64;

struct GQWeight {
    std::shared_ptr<ov::opset10::Parameter> weight;
    std::shared_ptr<ov::opset10::Parameter> scale;
};

GQWeight make_gq_weight(std::size_t out_features) {
    return {std::make_shared<ov::opset10::Parameter>(ov::element::i4, ov::Shape{kGroups, kGroupSize, out_features}),
            std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{kGroups, 1, out_features})};
}

// Each MatMul output is read by an op, as q/k/v in a real layer. The merge redirects readers, it does not
// move tensor names
std::shared_ptr<ov::opset10::Result> make_result(const ov::Output<ov::Node>& matmul_out) {
    return std::make_shared<ov::opset10::Result>(std::make_shared<ov::opset10::Convert>(matmul_out, ov::element::f32));
}

// W(i4) -> Convert -> Multiply(S) -> Reshape, the weight side that DQParMMGQ matches
ov::Output<ov::Node> dequantize(const GQWeight& gq) {
    const auto out_features = static_cast<int32_t>(gq.weight->get_shape()[2]);
    auto weight_convert = std::make_shared<ov::opset10::Convert>(gq.weight, ov::element::f16);
    auto scaled = std::make_shared<ov::opset10::Multiply>(weight_convert, gq.scale);
    auto shape = ov::opset10::Constant::create(ov::element::i32,
                                               ov::Shape{2},
                                               std::vector<int32_t>{kGroups * kGroupSize, out_features});
    return std::make_shared<ov::opset10::Reshape>(scaled, shape, false);
}

// A head function where the dequantized vocabulary feeds the first layer MatMuls directly, with no
// norm in between (for example EXAONE-4.0): ids -> Gather(W, Z, S) -> Convert -> Sub128 -> Subtract ->
// Multiply -> Convert -> group-quantized MatMuls. DQParMMGQ registers this Multiply as the activation
// of the MatMuls, and HostGatherQuantAsymm then replaces it.
std::shared_ptr<ov::Model> make_head_with_direct_gq_readers(std::size_t num_matmuls) {
    auto model = make_parameter_gather_model(128.0f);
    auto converted = model->get_results().front()->input_value(0);
    model->remove_result(model->get_results().front());
    for (std::size_t index = 0; index < num_matmuls; ++index) {
        const auto gq = make_gq_weight(kOutFeatures);
        auto matmul = std::make_shared<ov::opset10::MatMul>(converted, dequantize(gq), false, false);
        model->add_parameters({gq.weight, gq.scale});
        model->add_results({make_result(matmul)});
    }
    model->validate_nodes_and_infer_types();
    return model;
}

// Expected head after the host gather: the MatMuls read the host-gathered f16 embedding, nothing is merged
std::shared_ptr<ov::Model> make_host_gathered_head(std::size_t num_matmuls) {
    auto gathered = std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{1, 1, kGroups * kGroupSize});
    auto converted = std::make_shared<ov::opset10::Convert>(gathered, ov::element::f16);
    ov::ParameterVector params{gathered};
    ov::ResultVector results;
    for (std::size_t index = 0; index < num_matmuls; ++index) {
        const auto gq = make_gq_weight(kOutFeatures);
        auto matmul = std::make_shared<ov::opset10::MatMul>(converted, dequantize(gq), false, false);
        params.insert(params.end(), {gq.weight, gq.scale});
        results.push_back(make_result(matmul));
    }
    return std::make_shared<ov::Model>(results, params);
}

// Expected head after the merge over axis 2: one MatMul on the activation Multiply over the concatenated
// weights, and a Slice per original MatMul
std::shared_ptr<ov::Model> make_merged_head(std::size_t num_matmuls) {
    auto model = make_parameter_gather_model(128.0f);
    auto multiply = model->get_results().front()->input_value(0).get_node()->input_value(0);
    model->remove_result(model->get_results().front());

    const auto gq = make_gq_weight(kOutFeatures * num_matmuls);
    auto matmul = std::make_shared<ov::opset10::MatMul>(multiply, dequantize(gq), false, false);
    model->add_parameters({gq.weight, gq.scale});
    for (std::size_t index = 0; index < num_matmuls; ++index) {
        // DQParMMGQ registers the MatMuls in reverse topological order, so the last one comes first
        const auto position = num_matmuls - 1 - index;
        const auto begin = static_cast<int32_t>(position * kOutFeatures);
        const auto end = static_cast<int32_t>((position + 1) * kOutFeatures);
        auto start = ov::opset10::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 0, begin});
        auto stop = ov::opset10::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{1, 1, end});
        auto step = ov::opset10::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{1, 1, 1});
        auto slice = std::make_shared<ov::op::v8::Slice>(matmul, start, stop, step);
        model->add_results({make_result(slice)});
    }
    return model;
}

// Runs the head/tail passes in the order of Partitioner::optimize() with a context the test can inspect
class HeadTailPasses : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("npuw::test::HeadTailPasses");
    HeadTailPasses(std::shared_ptr<ov::npuw::patterns::opt::Context>& context, bool host_gather)
        : m_context(context),
          m_host_gather(host_gather) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        m_context = std::make_shared<ov::npuw::patterns::opt::Context>(model, 0);
        m_context->pmm_dims = "2";
        ov::pass::GraphRewrite rewrite;
        rewrite.add_matcher<ov::npuw::patterns::opt::DQParMMGQ>(std::ref(*m_context));
        rewrite.run_on_model(model);
        EXPECT_FALSE(m_context->par_dq_mms.empty());

        if (m_host_gather) {
            EXPECT_TRUE(run_host_gather(model, *m_context));
        }
        ov::npuw::patterns::opt::mergeParallelMatMuls(model, *m_context);

        // Partitioning adds the Parameters that the passes create (host gather output, concatenated
        // closures) to the function, do the same so the model stays valid
        ov::ParameterVector created;
        for (const auto& node : model->get_ordered_ops()) {
            auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node);
            if (param && model->get_parameter_index(param) < 0) {
                created.push_back(param);
            }
        }
        model->add_parameters(created);
        return true;
    }

private:
    std::shared_ptr<ov::npuw::patterns::opt::Context>& m_context;
    bool m_host_gather;
};

class HeadTailPassesTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        // Nodes created by the passes have no fused names
        disable_rt_info_check();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    }

    std::shared_ptr<ov::npuw::patterns::opt::Context> context;
};

class HostGatherParallelMatMulsTest : public HeadTailPassesTest, public ::testing::WithParamInterface<std::size_t> {
protected:
    void TearDown() override {
        HeadTailPassesTest::TearDown();
        ASSERT_NE(context, nullptr);
        EXPECT_TRUE(context->par_dq_mms.empty());
        EXPECT_TRUE(context->params_to_concat.empty());
        EXPECT_EQ(context->closures_to_subtract_128.size(), 2u);
        EXPECT_TRUE(vocab_weight->output(0).get_target_inputs().empty());
        EXPECT_TRUE(vocab_zero_point->output(0).get_target_inputs().empty());
    }

    std::shared_ptr<ov::op::v0::Parameter> vocab_weight;
    std::shared_ptr<ov::op::v0::Parameter> vocab_zero_point;
};

}  // namespace

// CVS-199500: the context kept the replaced vocabulary chain alive, so the u8 vocabulary Parameters
// still had readers. Partitioning then kept their closures next to the lazy Sub128 closures, and the
// weights bank detached the shared source of both. With two or more MatMuls the merge also brought
// the replaced chain back into the graph.
TEST_P(HostGatherParallelMatMulsTest, ReleasesReplacedVocabularyAndDoesNotMergeOnIt) {
    const auto num_matmuls = GetParam();
    model = make_head_with_direct_gq_readers(num_matmuls);
    vocab_weight = model->get_parameters().at(1);
    vocab_zero_point = model->get_parameters().at(2);
    manager.register_pass<HeadTailPasses>(context, true);
    model_ref = make_host_gathered_head(num_matmuls);
}

INSTANTIATE_TEST_SUITE_P(NumMatMuls, HostGatherParallelMatMulsTest, ::testing::Values(1u, 2u, 3u));

class ParallelMatMulMergeTest : public HeadTailPassesTest, public ::testing::WithParamInterface<std::size_t> {};

TEST_P(ParallelMatMulMergeTest, MergesMatMulsOnUnchangedActivation) {
    const auto num_matmuls = GetParam();
    model = make_head_with_direct_gq_readers(num_matmuls);
    manager.register_pass<HeadTailPasses>(context, false);
    model_ref = make_merged_head(num_matmuls);
}

INSTANTIATE_TEST_SUITE_P(NumMatMuls, ParallelMatMulMergeTest, ::testing::Values(2u, 3u));

namespace {

// The head of make_head_with_direct_gq_readers(1) as the plugin sees it before partitioning: the
// vocabulary and the MatMul weights are Constants that partitioning turns into closure Parameters
struct ConstantHead {
    std::shared_ptr<ov::Model> model;
    ov::Output<ov::Node> vocab_multiply;
};

ConstantHead make_constant_head() {
    constexpr std::size_t vocab_size = 64;
    constexpr std::size_t hidden_size = kGroups * kGroupSize;
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto weights = ov::opset10::Constant::create(ov::element::u8,
                                                 ov::Shape{vocab_size, hidden_size},
                                                 std::vector<uint8_t>(vocab_size * hidden_size, 200));
    auto zero_point =
        ov::opset10::Constant::create(ov::element::u8, ov::Shape{vocab_size, 1}, std::vector<uint8_t>(vocab_size, 128));
    auto scale =
        ov::opset10::Constant::create(ov::element::f16, ov::Shape{vocab_size, 1}, std::vector<float>(vocab_size, 1.0f));
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto gathered_weights = std::make_shared<ov::opset10::Gather>(weights, ids, axis);
    auto gathered_zero_point = std::make_shared<ov::opset10::Gather>(zero_point, ids, axis);
    auto gathered_scale = std::make_shared<ov::opset10::Gather>(scale, ids, axis);
    auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {128.0f});
    auto shifted_weight = std::make_shared<ov::opset10::Subtract>(
        std::make_shared<ov::opset10::Convert>(gathered_weights, ov::element::f16),
        shift);
    auto shifted_zero_point = std::make_shared<ov::opset10::Subtract>(
        std::make_shared<ov::opset10::Convert>(gathered_zero_point, ov::element::f16),
        shift);
    shifted_weight->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    shifted_zero_point->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    auto dequantized = std::make_shared<ov::opset10::Subtract>(shifted_weight, shifted_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, gathered_scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f16);

    auto gq_weight = ov::opset10::Constant::create(ov::element::i4,
                                                   ov::Shape{kGroups, kGroupSize, kOutFeatures},
                                                   std::vector<int8_t>(hidden_size * kOutFeatures, 1));
    auto gq_scale = ov::opset10::Constant::create(ov::element::f16,
                                                  ov::Shape{kGroups, 1, kOutFeatures},
                                                  std::vector<float>(kGroups * kOutFeatures, 1.0f));
    auto gq_scaled =
        std::make_shared<ov::opset10::Multiply>(std::make_shared<ov::opset10::Convert>(gq_weight, ov::element::f16),
                                                gq_scale);
    auto gq_shape =
        ov::opset10::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{hidden_size, kOutFeatures});
    auto gq_reshaped = std::make_shared<ov::opset10::Reshape>(gq_scaled, gq_shape, false);
    auto matmul = std::make_shared<ov::opset10::MatMul>(converted, gq_reshaped, false, false);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{make_result(matmul)}, ov::ParameterVector{ids});
    return {model, scaled->output(0)};
}

::intel_npu::Config make_head_partitioning_cfg() {
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*opt_desc);
    ::intel_npu::Config cfg(opt_desc);
    cfg.update({{"NPUW_ONLINE_PIPELINE", "NONE"},
                {"NPUW_FUNCALL_FOR_ALL", "YES"},
                {"NPUW_FOLD", "YES"},
                {"NPUW_HOST_GATHER", "YES"}});
    return cfg;
}

}  // namespace

// CVS-199500: partitioning decided whether to keep the u8 vocabulary Parameter next to its Sub128
// closure by get_target_inputs(). Any holder of the replaced vocabulary chain (here the test, in
// the bug the parallel MatMul context) made the disconnected Gather count as a reader, and the
// weights bank then detached the source shared by both closures. Only readers reachable from the
// model outputs may keep the source.
TEST(HostGatherSub128PartitioningTest, DropsVocabularySourceHeldByDisconnectedChain) {
    const auto head = make_constant_head();
    auto cfg = make_head_partitioning_cfg();
    ov::npuw::PartitioningContext ctx;
    ctx.use_host_gather_quant = true;

    const auto partitioning = ov::npuw::getPartitioning(head.model, cfg, ctx);

    // The vocabulary Multiply was replaced by the host gather, the test still holds it
    EXPECT_TRUE(head.vocab_multiply.get_target_inputs().empty());

    ASSERT_EQ(partitioning.functions.size(), 1u);
    const auto& function = partitioning.functions.begin()->second;
    const auto& params = function._model->get_parameters();
    EXPECT_EQ(std::count_if(params.begin(),
                            params.end(),
                            [](const std::shared_ptr<ov::op::v0::Parameter>& param) {
                                return param->get_element_type() == ov::element::u8;
                            }),
              0);
    EXPECT_EQ(std::count_if(params.begin(),
                            params.end(),
                            [](const std::shared_ptr<ov::op::v0::Parameter>& param) {
                                return param->get_element_type() == ov::element::i8;
                            }),
              2);

    const auto funcall = std::find_if(partitioning.subgraphs.begin(),
                                      partitioning.subgraphs.end(),
                                      [](const ov::npuw::Subgraph& subgraph) {
                                          return !subgraph._funcall.empty();
                                      });
    ASSERT_NE(funcall, partitioning.subgraphs.end());
    EXPECT_EQ(funcall->_lazy_closure.size(), params.size() - function._param_offset);
}

TEST(ConvertDQVocabTest, ReplacesMarkedShiftsWithLazySources) {
    const auto model = make_parameter_vocab_dq_model();
    ov::npuw::patterns::opt::Context context(model, 0);

    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::ConvertDQVocab>(std::ref(context));

    EXPECT_TRUE(rewrite.run_on_model(model));
    ASSERT_EQ(context.closures_to_subtract_128.size(), 2);
    for (const auto& shifted_and_source : context.closures_to_subtract_128) {
        EXPECT_EQ(shifted_and_source.first->get_element_type(), ov::element::i8);
        EXPECT_TRUE(shifted_and_source.second == model->get_parameters().at(0) ||
                    shifted_and_source.second == model->get_parameters().at(1));
        EXPECT_EQ(model->get_parameter_index(shifted_and_source.first), -1);
    }

    std::size_t reachable_i8_parameter_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(node);
        if (parameter && parameter->get_element_type() == ov::element::i8) {
            ++reachable_i8_parameter_count;
        }
    }
    EXPECT_EQ(reachable_i8_parameter_count, 2u);

    EXPECT_EQ(count_sub128_shifts(model), 0u);
}

class InsertVocabSub128PrePostProcessingTest : public ::testing::TestWithParam<bool> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "WithConvert" : "DirectMatMul";
    }
};

TEST_P(InsertVocabSub128PrePostProcessingTest, PreservesVocabularyConverts) {
    const auto model = make_vocab_matmul_model(GetParam());
    EXPECT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    EXPECT_EQ(count_subtracts(model), 3u);
    EXPECT_EQ(count_sub128_shifts(model), 2u);

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

INSTANTIATE_TEST_SUITE_P(Conversion,
                         InsertVocabSub128PrePostProcessingTest,
                         ::testing::Bool(),
                         InsertVocabSub128PrePostProcessingTest::getTestCaseName);

TEST(InsertVocabSub128Test, SkipsPretransposedLayout) {
    // A [H, V] LM head cannot share storage with the [V, H] embedding table.
    for (const bool convert_before_matmul : {false, true}) {
        const auto model = make_pretransposed_vocab_matmul_model(convert_before_matmul);
        EXPECT_FALSE(ov::npuw::InsertVocabSub128().run_on_model(model));
        EXPECT_EQ(count_sub128_shifts(model), 0u);
    }
}

class InsertVocabSub128LmHeadTerminalTest : public ::testing::TestWithParam<VocabMatMulTerminal> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<VocabMatMulTerminal>& info) {
        switch (info.param) {
        case VocabMatMulTerminal::Add:
            return "Add";
        case VocabMatMulTerminal::Transpose:
            return "Transpose";
        case VocabMatMulTerminal::Convert:
            return "Convert";
        case VocabMatMulTerminal::Gated:
            return "Gated";
        case VocabMatMulTerminal::Divide:
            return "Divide";
        }
        return "Unknown";
    }
};

TEST_P(InsertVocabSub128LmHeadTerminalTest, InsertsSub128BeforeLmHeadTerminal) {
    const auto model = make_vocab_matmul_model(false, GetParam());

    EXPECT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    EXPECT_EQ(count_subtracts(model), 3u);
    EXPECT_EQ(count_sub128_shifts(model), 2u);
}

INSTANTIATE_TEST_SUITE_P(LmHeadTerminal,
                         InsertVocabSub128LmHeadTerminalTest,
                         ::testing::Values(VocabMatMulTerminal::Add,
                                           VocabMatMulTerminal::Transpose,
                                           VocabMatMulTerminal::Convert,
                                           VocabMatMulTerminal::Gated,
                                           VocabMatMulTerminal::Divide),
                         InsertVocabSub128LmHeadTerminalTest::getTestCaseName);

TEST(InsertVocabSub128LmHeadTerminalTest, SkipsManuallyAddedOutput) {
    const auto model = make_vocab_matmul_model(false);
    model->get_results().front()->get_rt_info()["manually_added_output"] = true;

    EXPECT_FALSE(ov::npuw::InsertVocabSub128().run_on_model(model));
    EXPECT_EQ(count_subtracts(model), 1u);
    EXPECT_EQ(count_sub128_shifts(model), 0u);
}

TEST(LazySubtract128Test, CoversEveryU8ValueWithoutChangingSource) {
    std::vector<uint8_t> values(256);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<uint8_t>(index);
    }
    const auto constant = ov::opset10::Constant::create(ov::element::u8, ov::Shape{16, 16}, values);
    ov::npuw::weights::LazyTensor source(constant);
    const auto shifted = source.subtract_128();
    EXPECT_EQ(shifted.eval_meta().shape, constant->get_shape());
    EXPECT_EQ(shifted.eval_meta().type, ov::element::i8);
    EXPECT_EQ(shifted, source.subtract_128());
    EXPECT_NE(shifted, source);
    const auto result = shifted.eval();
    for (std::size_t index = 0; index < values.size(); ++index) {
        EXPECT_EQ(result.data<const int8_t>()[index], static_cast<int>(index) - 128);
    }
    EXPECT_EQ(constant->cast_vector<uint8_t>(), values);
}

TEST(LazySubtract128Test, HandlesVectorAndTailSizes) {
    const std::array<std::size_t, 8> sizes{1, 31, 32, 33, 63, 64, 65, 257};
    for (const auto size : sizes) {
        std::vector<uint8_t> values(size);
        for (std::size_t index = 0; index < size; ++index) {
            values[index] = static_cast<uint8_t>(index);
        }

        const auto constant = ov::opset10::Constant::create(ov::element::u8, ov::Shape{size}, values);
        const auto result = ov::npuw::weights::LazyTensor(constant).subtract_128().eval();

        ASSERT_EQ(result.get_shape(), (ov::Shape{size}));
        ASSERT_EQ(result.get_element_type(), ov::element::i8);
        for (std::size_t index = 0; index < size; ++index) {
            EXPECT_EQ(result.data<const int8_t>()[index], static_cast<int>(values[index]) - 128);
        }
    }
}

TEST(LazySubtract128Test, RejectsNonU8Input) {
    const auto constant = ov::opset10::Constant::create(ov::element::i8, ov::Shape{1}, {0});
    const auto shifted = ov::npuw::weights::LazyTensor(constant).subtract_128();
    EXPECT_THROW(shifted.eval(), ov::Exception);
    EXPECT_THROW(shifted.eval_meta(), ov::Exception);
}
