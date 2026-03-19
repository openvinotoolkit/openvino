// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for LLMCompiledModel's CompiledModelFactory injection point.
//
// We build a tiny stateful LLM with ModelBuilder, then construct
// LLMCompiledModel with a counting factory that returns MockSubCompiledModel
// instead of a real npuw::CompiledModel.  No hardware or ICore is required:
//
//   - NullPlugin is a minimal ov::IPlugin stub (no hardware, no core).
//   - extract_npu_descriptor() returns std::nullopt when plugin->get_core()
//     is null (our null guard), so all NPU-HW-specific branches are skipped.
//
// Expected factory call counts:
//   NPUW_LLM_SHARED_HEAD=NO  (two-model pipeline)   → 2
//   NPUW_LLM_SHARED_HEAD=YES (three-model pipeline)  → 3

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>

#include "compiled_model.hpp"      // ov::npuw::ICompiledModel_v0
#include "llm_compiled_model.hpp"  // ov::npuw::LLMCompiledModel
#include "model_builder.hpp"       // ov::test::npuw::ModelBuilder/Config
#include "openvino/runtime/iplugin.hpp"
#include "serialization.hpp"       // ov::npuw::s11n::CompiledContext
#include "weights_bank.hpp"        // ov::npuw::weights::Bank

namespace {

// ---------------------------------------------------------------------------
// NullPlugin — the minimum ov::IPlugin needed to construct LLMCompiledModel.
// get_core() returns nullptr → extract_npu_descriptor() skips all HW queries.
// ---------------------------------------------------------------------------
class NullPlugin : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(
        const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> compile_model(
        const std::shared_ptr<const ov::Model>&,
        const ov::AnyMap&,
        const ov::SoPtr<ov::IRemoteContext>&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(
        std::istream&, const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(
        std::istream&, const ov::SoPtr<ov::IRemoteContext>&,
        const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(
        const ov::Tensor&, const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(
        const ov::Tensor&, const ov::SoPtr<ov::IRemoteContext>&,
        const ov::AnyMap&) const override { return {}; }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&,
                                    const ov::AnyMap&) const override { return {}; }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override { return {}; }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override { return {}; }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override { return {}; }
};

// ---------------------------------------------------------------------------
// MockSubCompiledModel — minimal ICompiledModel_v0 returned by the factory.
// LLMCompiledModel only stores the pointer during construction; no methods
// are called on it before the constructor returns.
// ---------------------------------------------------------------------------
class MockSubCompiledModel : public ov::npuw::ICompiledModel_v0 {
public:
    MockSubCompiledModel(const std::shared_ptr<ov::Model>& m,
                         const std::shared_ptr<const ov::IPlugin>& p,
                         const ov::AnyMap& /*props*/)
        : ov::npuw::ICompiledModel_v0(m, p) {}

    // ov::ICompiledModel pure virtuals
    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override { return {}; }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override { return {}; }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override { return {}; }

    // ov::npuw::ICompiledModel_v0 pure virtuals
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override { return {}; }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override { return {}; }
    std::string submodel_device(std::size_t) const override { return "CPU"; }
    std::size_t num_submodels() const override { return 0; }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override { return {}; }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class LLMCompiledModelFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        ov::test::npuw::ModelConfig cfg;
        cfg.num_layers   = 2;
        cfg.hidden_size  = 64;
        cfg.num_heads    = 4;
        cfg.head_dim     = 16;
        cfg.num_kv_heads = 4;
        cfg.vocab_size   = 256;

        ov::test::npuw::ModelBuilder mb;
        m_model  = mb.build_model(cfg);
        m_plugin = std::make_shared<NullPlugin>();
    }

    ov::npuw::LLMCompiledModel::CompiledModelFactory make_counting_factory(std::atomic<int>& n) {
        return [&n](const std::shared_ptr<ov::Model>& m,
                    const std::shared_ptr<const ov::IPlugin>& p,
                    const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            ++n;
            return std::make_shared<MockSubCompiledModel>(m, p, props);
        };
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_LLM",                  "YES"},
                {"NPUW_LLM_MAX_PROMPT_LEN",   "128"},
                {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    }

    std::shared_ptr<ov::Model>    m_model;
    std::shared_ptr<ov::IPlugin>  m_plugin;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(LLMCompiledModelFactoryTest, TwoModelPipeline) {
    std::atomic<int> calls{0};
    auto props = base_props();
    props["NPUW_LLM_SHARED_HEAD"] = std::string("NO");

    ASSERT_NO_THROW(ov::npuw::LLMCompiledModel(m_model, m_plugin, props,
                                               make_counting_factory(calls)));
    EXPECT_EQ(calls.load(), 2) << "Expected 2 factory calls: generate + prefill";
}

TEST_F(LLMCompiledModelFactoryTest, ThreeModelPipeline) {
    std::atomic<int> calls{0};
    auto props = base_props();
    props["NPUW_LLM_SHARED_HEAD"] = std::string("YES");

    ASSERT_NO_THROW(ov::npuw::LLMCompiledModel(m_model, m_plugin, props,
                                               make_counting_factory(calls)));
    EXPECT_EQ(calls.load(), 3) << "Expected 3 factory calls: generate + prefill + lm_head";
}

}  // namespace
