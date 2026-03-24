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
#include "openvino/runtime/iplugin.hpp"
#include "serialization.hpp"
#include "weights_bank.hpp"

namespace ov::test::npuw {

inline ModelConfig make_llm_test_model_config() {
    ModelConfig cfg;
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
    return mb.build_model(make_llm_test_model_config());
}

inline std::shared_ptr<ov::Model> build_whisper_decoder_test_model() {
    auto cfg = make_llm_test_model_config();
    cfg.use_cross_attention = true;
    ModelBuilder mb;
    return mb.build_model(cfg);
}

inline std::shared_ptr<ov::Model> build_embedding_test_model() {
    auto cfg = make_llm_test_model_config();
    cfg.use_token_type_embedding = true;
    ModelBuilder mb;
    return mb.build_model(cfg);
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
    std::string                friendly_name;
    ov::AnyMap                 props;
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
