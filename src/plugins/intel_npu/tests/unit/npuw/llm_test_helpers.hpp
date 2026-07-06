// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compiled_model.hpp"
#include "llm_compiled_model.hpp"
#include "model_builder.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "serialization.hpp"
#include "weights_bank.hpp"

namespace ov::test::npuw {

template <typename Config = LLMConfig>
inline Config make_test_model_config() {
    Config cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 4;
    cfg.vocab_size = 256;
    return cfg;
}

inline std::shared_ptr<ov::Model> build_llm_test_model() {
    ModelBuilder mb;
    return mb.build_llm(make_test_model_config());
}

/// Build an LLM test model configured so that Attention::from() can unambiguously
/// identify the past-KV dimension in the isolated attention function body.
///
/// The default test config has num_kv_heads == kPastKvLen == 4 which makes the
/// KV-cache shape [1, 4, 4, 16] ambiguous — find_context_dim sees two dims equal
/// to past_len and returns false.  By setting num_kv_heads=2 and using a past
/// length of 8, the KV shape becomes [1, 2, 8, 16] where only dim 2 equals
/// past_len=8, so Attention::from() succeeds and f._attention gets set.
inline std::shared_ptr<ov::Model> build_dynamic_attention_llm_model() {
    ov::test::npuw::LLMConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.num_kv_heads = 2;  // must differ from kPastKvLen so find_context_dim is unambiguous
    cfg.vocab_size = 256;
    cfg.force_gqa_broadcast = true;  // produces 5-input SDPA needed by the SDPA isolation pattern

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    constexpr std::size_t kSeq = 4;
    constexpr std::size_t kPast = 8;  // != num_kv_heads (2), so KV shape [1,2,8,16] is unambiguous

    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& pshape = input.get_partial_shape();
        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeq};
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeq + kPast};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeq};
        } else {
            // KV cache params: fix batch=1, past_len=kPast, leave head/head_dim from pshape
            auto static_shape = pshape;
            static_shape[0] = 1;
            static_shape[2] = kPast;
            new_shapes[name] = static_shape;
        }
    }
    model->reshape(new_shapes);
    model->validate_nodes_and_infer_types();
    return model;
}

inline LLMConfig make_test_model_config_gqa() {
    auto cfg = make_test_model_config();
    cfg.num_kv_heads = 2;  // num_heads=4 / num_kv_heads=2 -> n_rep=2
    return cfg;
}

inline std::shared_ptr<ov::Model> build_llm_gqa_test_model() {
    ModelBuilder mb;
    return mb.build_llm(make_test_model_config_gqa());
}

inline std::shared_ptr<ov::Model> build_llm_test_model_with_kv_fake_convert(const ov::element::Type fake_convert_type) {
    auto model = build_llm_test_model();
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});

    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa) {
            continue;
        }

        auto inject_fake_convert = [&](size_t input_idx, const std::string& suffix) {
            auto fake_convert_1 =
                std::make_shared<ov::op::v13::FakeConvert>(sdpa->input_value(input_idx), scale, fake_convert_type);
            auto fake_convert_2 = std::make_shared<ov::op::v13::FakeConvert>(fake_convert_1, scale, fake_convert_type);
            fake_convert_1->set_friendly_name(sdpa->get_friendly_name() + "/" + suffix + "_1");
            fake_convert_2->set_friendly_name(sdpa->get_friendly_name() + "/" + suffix + "_2");
            sdpa->input(input_idx).replace_source_output(fake_convert_2);
        };

        inject_fake_convert(1, "key_fake_convert");
        inject_fake_convert(2, "value_fake_convert");
    }

    model->validate_nodes_and_infer_types();
    return model;
}

/// Hybrid LLM: alternating linear-attention / full-attention layers.
/// 4 layers → layers 0,2 linear; layers 1,3 full attention.
/// Attention layers are Qwen3.5-style: output-gated, partial RoPE.
inline std::shared_ptr<ov::Model> build_hybrid_llm_test_model() {
    auto cfg = make_test_model_config();
    cfg.num_layers = 4;
    cfg.attn_output_gate = true;
    cfg.rotary_dim = cfg.head_dim / 4;
    cfg.is_linear_layer = make_mamba_schedule(1);
    auto mixer = std::make_shared<GatedDeltaNetMixer>();
    mixer->hidden_size = cfg.hidden_size;
    mixer->precision = cfg.precision;
    mixer->weight_fn = cfg.weight;
    mixer->num_heads = cfg.num_heads;
    mixer->key_head_dim = cfg.head_dim;
    mixer->value_head_dim = cfg.head_dim;
    cfg.linear_mixer = mixer;
    ModelBuilder mb;
    return mb.build_llm(cfg);
}

/// LFM2-style hybrid: gated short-conv mixer layers interleaved with full attention.
/// 4 layers → layers 0,2 short-conv; layers 1,3 full attention.
inline std::shared_ptr<ov::Model> build_lfm2_llm_test_model() {
    auto cfg = make_test_model_config();
    cfg.num_layers = 4;
    cfg.is_linear_layer = make_mamba_schedule(1);
    auto mixer = std::make_shared<ShortConvMixer>();
    mixer->hidden_size = cfg.hidden_size;
    mixer->precision = cfg.precision;
    mixer->weight_fn = cfg.weight;
    mixer->conv_dim = cfg.hidden_size;
    cfg.linear_mixer = mixer;
    ModelBuilder mb;
    return mb.build_llm(cfg);
}

inline std::shared_ptr<ov::Model> build_whisper_decoder_test_model() {
    ModelBuilder mb;
    return mb.build_whisper_decoder(make_test_model_config<WhisperConfig>());
}

/// Eagle3 target: regular LLM whose captured layer outputs feed the extra
/// "last_hidden_state" output (concat of 3 layers -> fc, GenAI NPU form).
inline std::shared_ptr<ov::Model> build_eagle3_target_test_model() {
    auto cfg = make_test_model_config<LLMConfig>();
    cfg.num_layers = 8;
    cfg.eagle3_capture_layers = {2, 4, 5};  // GenAI default pick: {2, N/2, N-3}
    ModelBuilder mb;
    return mb.build_llm(cfg);
}

/// Eagle3 draft: single-"midlayer" decoder taking a target "hidden_states" input
/// and emitting draft-vocab logits. Captures 3 layers (feature width 3*hidden)
/// with an "model.fc" projection, matching the real NPU export. GQA keeps it
/// close to real Eagle3 draft heads.
inline std::shared_ptr<ov::Model> build_eagle3_draft_test_model(bool use_tree_mask = false) {
    auto cfg = make_test_model_config<Eagle3DraftConfig>();
    cfg.num_kv_heads = 2;
    cfg.draft_vocab_size = 128;
    cfg.use_tree_mask = use_tree_mask;
    ModelBuilder mb;
    return mb.build_eagle3_draft(cfg);
}

/// Eagle3 draft variant with a single captured layer and a dynamic
/// "hidden_states" feature dim, so ReshapeToStatic must recover the width from
/// the "last_hidden_state" output (Eagle3Extension::get_static_input fallback).
inline std::shared_ptr<ov::Model> build_eagle3_draft_dynamic_hidden_test_model() {
    auto cfg = make_test_model_config<Eagle3DraftConfig>();
    cfg.num_kv_heads = 2;
    cfg.draft_vocab_size = 128;
    cfg.num_captured_layers = 1;
    cfg.dynamic_hidden_states = true;
    ModelBuilder mb;
    return mb.build_eagle3_draft(cfg);
}

inline std::shared_ptr<ov::Model> build_embedding_test_model() {
    ModelBuilder mb;
    return mb.build_embedding_encoder(make_test_model_config<BertConfig>());
}

inline std::shared_ptr<ov::Model> build_embedding_decoder_test_model() {
    ModelBuilder mb;
    auto cfg = make_test_model_config<LLMConfig>();
    cfg.use_kv_cache = false;
    cfg.internal_position_ids = true;
    cfg.lm_head_weight = {};
    return mb.build_llm(cfg);
}

inline std::shared_ptr<ov::Model> build_moe_llm_test_model() {
    ModelBuilder mb;
    auto cfg = make_test_model_config();
    cfg.num_experts = 8;
    cfg.num_experts_per_tok = 2;
    return mb.build_llm(cfg);
}

/// Qwen3-style MoE: separate gate/up expert MatMuls (SwiGLU) and a Softmax->TopK router
/// with ReduceSum->Divide renormalization, matching NPUW's Qwen3Expert + Qwen3Router
/// patterns (real Qwen3-30B-A3B). Contrast build_moe_llm_test_model, which emits the
/// GPT-OSS topology (fused gate_up, TopK->Softmax router). Stateful (KV cache) like the
/// real model; consumers that need static shapes run StatefulToStateless + reshape, the
/// same way production prepares an LLM.
inline std::shared_ptr<ov::Model> build_qwen3_moe_llm_test_model() {
    ModelBuilder mb;
    auto cfg = make_test_model_config();
    cfg.num_experts = 8;
    cfg.num_experts_per_tok = 2;
    cfg.moe_factory = make_qwen3_moe_ffn;
    return mb.build_llm(cfg);
}

inline std::shared_ptr<ov::Model> build_sliding_window_test_model(size_t window_size = 512,
                                                                  size_t sliding_to_full_ratio = 0,
                                                                  const SlidingMaskFn& sliding_mask_fn = {},
                                                                  size_t num_layers = 2) {
    auto cfg = make_test_model_config();
    cfg.num_layers = num_layers;
    cfg.sliding_window_size = window_size;
    cfg.sliding_to_full_ratio = sliding_to_full_ratio;
    cfg.sliding_mask_fn = sliding_mask_fn;
    ModelBuilder mb;
    return mb.build_llm(cfg);
}

inline std::shared_ptr<ov::Model> build_token_type_ids_test_model(size_t window_size = 512,
                                                                  size_t sliding_to_full_ratio = 1,
                                                                  const SlidingMaskFn& sliding_mask_fn = {}) {
    auto cfg = make_test_model_config();
    cfg.sliding_window_size = window_size;
    cfg.sliding_to_full_ratio = sliding_to_full_ratio;
    cfg.sliding_mask_fn = sliding_mask_fn;
    cfg.use_inputs_embeds = true;
    cfg.use_token_type_ids = true;
    ModelBuilder mb;
    return mb.build_llm(cfg);
}

class NullPlugin : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }
};

class MockSubCompiledModel : public ov::npuw::ICompiledModel_v0 {
public:
    MockSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         const ov::AnyMap&)
        : ov::npuw::ICompiledModel_v0(model, plugin) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return {};
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        return 0;
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}
};

struct CompileCall {
    std::string friendly_name;
    ov::AnyMap props;
    std::shared_ptr<ov::Model> model;
};

class RecordingFactory {
public:
    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [this](const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            m_calls.push_back({model->get_friendly_name(), props, model});
            return std::make_shared<MockSubCompiledModel>(model, plugin, props);
        };
    }

    const std::vector<CompileCall>& calls() const {
        return m_calls;
    }

    std::size_t count_suffix(std::string_view suffix) const {
        return std::count_if(m_calls.begin(), m_calls.end(), [suffix](const CompileCall& call) {
            return call.friendly_name.size() >= suffix.size() &&
                   call.friendly_name.compare(call.friendly_name.size() - suffix.size(), suffix.size(), suffix) == 0;
        });
    }

    std::size_t count_contains(std::string_view fragment) const {
        return std::count_if(m_calls.begin(), m_calls.end(), [fragment](const CompileCall& call) {
            return call.friendly_name.find(fragment) != std::string::npos;
        });
    }

    const CompileCall* find_suffix(std::string_view suffix) const {
        const auto it = std::find_if(m_calls.begin(), m_calls.end(), [suffix](const CompileCall& call) {
            return call.friendly_name.size() >= suffix.size() &&
                   call.friendly_name.compare(call.friendly_name.size() - suffix.size(), suffix.size(), suffix) == 0;
        });
        return it == m_calls.end() ? nullptr : &(*it);
    }

    const CompileCall* find_contains(std::string_view fragment) const {
        const auto it = std::find_if(m_calls.begin(), m_calls.end(), [fragment](const CompileCall& call) {
            return call.friendly_name.find(fragment) != std::string::npos;
        });
        return it == m_calls.end() ? nullptr : &(*it);
    }

private:
    std::vector<CompileCall> m_calls;
};

}  // namespace ov::test::npuw
