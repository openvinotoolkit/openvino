// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// CPU-backed NPUW Host Flash Attention tests for the generic (non-LLM) pipeline: the attention
// history comes from a model input (NPUW_ATTN_HISTORY_SIZE_INPUT) and the past KV parameters use
// the [B, S, H, D] layout, transposed to [B, H, S, D] right before the KV Concat.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "hfa_bshd_model.hpp"
#include "openvino/openvino.hpp"

namespace {

using ov::test::npuw::HfaBshdModelConfig;
using TensorMap = std::map<std::string, ov::Tensor>;

ov::AnyMap hfa_props(bool fused) {
    return {{"NPU_USE_NPUW", "YES"},
            {"NPUW_DEVICES", "CPU"},
            {"NPUW_FOLD", "YES"},
            {"NPUW_UNFOLD_IREQS", "NO"},
            {"NPUW_ONLINE_PIPELINE", "REP"},
            {"NPUW_ONLINE_ISOLATE", "ATTN"},
            // The test model has only 2 layers; keep both repeated attention blocks.
            {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
            {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
            {"NPUW_ATTN", "HFA"},
            {"NPUW_ATTN_HFA_FUSED", fused ? "YES" : "NO"},
            {"NPUW_ATTN_HISTORY_SIZE_INPUT", ov::test::npuw::kHfaHistoryInputName}};
}

int64_t round_up(int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// Inputs for one prefill chunk with `history` valid past positions. Past rows that HFA must not
// read (beyond the history rounded up to whole tiles) are NaN when `poison_unread_past` is set,
// so attention over the whole past buffer (i.e. HFA not engaged) produces NaNs.
TensorMap make_inputs(const std::shared_ptr<ov::Model>& model,
                      const HfaBshdModelConfig& cfg,
                      int64_t history,
                      bool poison_unread_past) {
    const auto Q = static_cast<int64_t>(cfg.query_len);
    const auto P = static_cast<int64_t>(cfg.past_len);
    const auto read_rows = round_up(history, Q);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    TensorMap inputs;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        ov::Tensor tensor(input.get_element_type(), input.get_shape());
        if (name == ov::test::npuw::kHfaHistoryInputName) {
            tensor.data<int64_t>()[0] = history;
        } else if (name == ov::test::npuw::kHfaMaskInputName) {
            // Past columns [0, P) are visible below the history; present columns [P, P+Q) are causal.
            auto* data = tensor.data<float>();
            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (int64_t row = 0; row < Q; ++row) {
                for (int64_t col = 0; col < P + Q; ++col) {
                    const bool visible = col < P ? col < history : col - P <= row;
                    data[row * (P + Q) + col] = visible ? 0.0f : neg_inf;
                }
            }
        } else if (input.get_element_type() == ov::element::f16) {
            // Past K/V in [1, P, Hkv, D] layout.
            auto* data = tensor.data<ov::float16>();
            const auto row_size = static_cast<int64_t>(tensor.get_size()) / P;
            for (int64_t row = 0; row < P; ++row) {
                const bool unread = poison_unread_past && row >= read_rows;
                for (int64_t i = 0; i < row_size; ++i) {
                    const float value = dist(rng);
                    data[row * row_size + i] = unread ? ov::float16(std::numeric_limits<float>::quiet_NaN())
                                                      : ov::float16(value);
                }
            }
        } else {
            auto* data = tensor.data<float>();
            for (std::size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = dist(rng);
            }
        }
        inputs[name] = tensor;
    }
    return inputs;
}

TensorMap infer(ov::InferRequest& request, const ov::CompiledModel& compiled, const TensorMap& inputs) {
    for (const auto& input : compiled.inputs()) {
        request.set_tensor(input, inputs.at(input.get_any_name()));
    }
    request.infer();
    TensorMap outputs;
    for (const auto& output : compiled.outputs()) {
        const auto result = request.get_tensor(output);
        ov::Tensor copy(result.get_element_type(), result.get_shape());
        result.copy_to(copy);
        outputs[output.get_any_name()] = copy;
    }
    return outputs;
}

void expect_attention_close(const TensorMap& expected, const TensorMap& actual) {
    const auto& name = ov::test::npuw::kHfaHiddenOutputName;
    const auto& ref = expected.at(name);
    const auto& out = actual.at(name);
    ASSERT_EQ(ref.get_shape(), out.get_shape());
    const auto* ref_data = ref.data<float>();
    const auto* out_data = out.data<float>();
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < ref.get_size(); ++i) {
        ASSERT_TRUE(std::isfinite(out_data[i])) << name << "[" << i << "]: HFA read masked-out past rows";
        max_diff = std::max(max_diff, std::abs(ref_data[i] - out_data[i]));
    }
    // HFA keeps its running state in the f16 KV storage precision.
    EXPECT_LE(max_diff, 2e-2f);
}

void expect_bitwise_equal(const TensorMap& expected, const TensorMap& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (const auto& [name, tensor] : expected) {
        const auto& other = actual.at(name);
        ASSERT_EQ(tensor.get_byte_size(), other.get_byte_size()) << name;
        EXPECT_EQ(std::memcmp(tensor.data(), other.data(), tensor.get_byte_size()), 0) << name;
    }
}

class HFAExplicitHistoryCPUTest : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        m_model = ov::test::npuw::build_hfa_bshd_model(m_cfg);
        m_reference = m_core.compile_model(m_model, "CPU", ov::hint::inference_precision(ov::element::f32));
        m_reference_request = m_reference.create_infer_request();
    }

    TensorMap reference(int64_t history) {
        return infer(m_reference_request, m_reference, make_inputs(m_model, m_cfg, history, false));
    }

    // Past capacity is 3 tiles, so every history below leaves at least one past tile unread.
    const HfaBshdModelConfig m_cfg{};
    ov::Core m_core;
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_reference;
    ov::InferRequest m_reference_request;
};

// One request runs consecutive prefill chunks with no past, one and two full past tiles, and a
// partially filled past tile (history not a multiple of the tile size).
TEST_P(HFAExplicitHistoryCPUTest, MatchesReferenceForFullAndPartialPastTiles) {
    auto compiled = m_core.compile_model(m_model, "NPU", hfa_props(GetParam()));
    auto request = compiled.create_infer_request();
    const auto Q = static_cast<int64_t>(m_cfg.query_len);
    for (const int64_t history : {int64_t{0}, Q, 2 * Q, Q + 3, int64_t{5}}) {
        SCOPED_TRACE("history " + std::to_string(history));
        const auto actual = infer(request, compiled, make_inputs(m_model, m_cfg, history, true));
        expect_attention_close(reference(history), actual);
    }
}

TEST_P(HFAExplicitHistoryCPUTest, ExportImportRoundTripKeepsHistoryInputAndLayout) {
    auto compiled = m_core.compile_model(m_model, "NPU", hfa_props(GetParam()));
    const auto history = static_cast<int64_t>(m_cfg.query_len) + 3;
    const auto inputs = make_inputs(m_model, m_cfg, history, true);
    auto request = compiled.create_infer_request();
    const auto expected = infer(request, compiled, inputs);

    std::stringstream blob;
    compiled.export_model(blob);
    // Only the device selection: the HFA history input name must come from the blob itself.
    auto imported = m_core.import_model(blob, "NPU", {{"NPU_USE_NPUW", "YES"}, {"NPUW_DEVICES", "CPU"}});
    auto imported_request = imported.create_infer_request();
    const auto actual = infer(imported_request, imported, inputs);

    expect_bitwise_equal(expected, actual);
    expect_attention_close(reference(history), actual);
}

INSTANTIATE_TEST_SUITE_P(FusedAndNonFused,
                         HFAExplicitHistoryCPUTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "Fused" : "NonFused";
                         });

}  // namespace
