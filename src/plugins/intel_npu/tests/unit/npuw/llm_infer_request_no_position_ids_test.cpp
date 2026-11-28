// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct LLMNoPositionIdsTestAccess {
    using Request = ov::npuw::LLMInferRequest;

    static bool position_ids_present(const Request& req) {
        return req.m_position_ids_present;
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::build_llm_test_model_without_position_ids;
using ov::test::npuw::LLMNoPositionIdsTestAccess;
using ov::test::npuw::NullPlugin;
class FakeSubCompiledModel;

class FakeSubInferRequest final : public ov::ISyncInferRequest {
public:
    explicit FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return ov::ISyncInferRequest::get_tensor(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
};

class FakeSubCompiledModel final : public ov::npuw::ICompiledModel_v0 {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         const ov::AnyMap&)
        : ov::npuw::ICompiledModel_v0(model, plugin),
          m_model(model) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const FakeSubCompiledModel>(shared_from_this());
        return std::make_shared<FakeSubInferRequest>(std::move(self));
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        intel_npu::make_executor("no_position_ids_task", 1),
                                                        intel_npu::make_executor("no_position_ids_callback", 1));
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        // One CPU submodel, so init_pre_alloc_device() resolves to CPU staging
        // instead of requesting NPU remote tensors from the stub plugin.
        return 1;
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}

private:
    std::shared_ptr<ov::Model> m_model;
};

FakeSubInferRequest::FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model)
    : ov::ISyncInferRequest(std::move(compiled_model)) {
    for (const auto& input : get_compiled_model()->inputs()) {
        ov::ISyncInferRequest::set_tensor(input,
                                          ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        ov::ISyncInferRequest::set_tensor(
            output,
            ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void FakeSubInferRequest::infer() {
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

class NoPositionIdsFactory {
public:
    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [](const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
        };
    }
};

ov::Tensor make_i64(std::initializer_list<size_t> shape, int64_t fill_value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), fill_value);
    return tensor;
}

class LLMNoPositionIdsTest : public ::testing::Test {
protected:
    void build(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& extra_props = {}) {
        m_plugin = std::make_shared<NullPlugin>();
        NoPositionIdsFactory factory;
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", "256"},
                         {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                         {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                         {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}};
        for (const auto& kv : extra_props) {
            props[kv.first] = kv.second;
        }
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, m_plugin, props, factory.make_factory());
        ASSERT_NE(m_compiled, nullptr);
        m_request = std::make_unique<ov::npuw::LLMInferRequest>(m_compiled);
        ASSERT_NE(m_request, nullptr);
    }

    bool has_input(const std::string& name) {
        return ov::npuw::util::find_port_by_name(m_request->get_inputs(), name).has_value();
    }

    void set_input(const std::string& name, const ov::Tensor& tensor) {
        m_request->set_tensor(ov::npuw::util::find_port_by_name(m_request->get_inputs(), name).value(),
                              ov::get_tensor_impl(tensor));
    }

    ov::SoPtr<ov::ITensor> logits() {
        return m_request->get_tensor(ov::npuw::util::find_port_by_name(m_request->get_outputs(), "logits").value());
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::unique_ptr<ov::npuw::LLMInferRequest> m_request;
};

// A NoPE model has no position_ids Parameter, so the request must flag its
// absence and expose no position_ids input port.
TEST_F(LLMNoPositionIdsTest, NoPositionIdsModelSetsFlagFalse) {
    build(build_llm_test_model_without_position_ids());
    EXPECT_FALSE(LLMNoPositionIdsTestAccess::position_ids_present(*m_request));
    EXPECT_FALSE(has_input("position_ids"));
}

// Positive control: a standard RoPE model keeps the position_ids port and flag.
TEST_F(LLMNoPositionIdsTest, PositionIdsModelSetsFlagTrue) {
    build(build_llm_test_model());
    EXPECT_TRUE(LLMNoPositionIdsTestAccess::position_ids_present(*m_request));
    EXPECT_TRUE(has_input("position_ids"));
}

// A full chunked prefill followed by a generate step must run end-to-end without
// touching the absent position_ids port, synthesizing positions internally.
TEST_F(LLMNoPositionIdsTest, PrefillAndGenerateRunWithoutPositionIds) {
    build(build_llm_test_model_without_position_ids());
    ASSERT_FALSE(has_input("position_ids"));

    // Multi-chunk prompt (72 tokens over a chunk size of 32) exercises the
    // chunked-prefill loop where the position_ids tensor fetch is hoisted.
    constexpr size_t kPromptLen = 72u;
    set_input("input_ids", make_i64({1, kPromptLen}, 1));
    set_input("attention_mask", make_i64({1, kPromptLen}, 1));
    ASSERT_NO_THROW(m_request->infer());
    ASSERT_NE(logits(), nullptr);

    // Generate step: a single token with the history exposed in the mask.
    set_input("input_ids", make_i64({1, 1}, 1));
    set_input("attention_mask", make_i64({1, kPromptLen + 1}, 1));
    ASSERT_NO_THROW(m_request->infer());
    EXPECT_NE(logits(), nullptr);
}

// Static prefill (no chunking) must also run end-to-end without position_ids,
// exercising the whole-prefill guard in infer_whole_prefill.
TEST_F(LLMNoPositionIdsTest, PrefillAndGenerateRunWithoutPositionIdsStaticPrefill) {
    build(build_llm_test_model_without_position_ids(), {{"NPUW_LLM_PREFILL_HINT", "STATIC"}});
    ASSERT_FALSE(has_input("position_ids"));

    constexpr size_t kPromptLen = 72u;
    set_input("input_ids", make_i64({1, kPromptLen}, 1));
    set_input("attention_mask", make_i64({1, kPromptLen}, 1));
    ASSERT_NO_THROW(m_request->infer());
    ASSERT_NE(logits(), nullptr);

    set_input("input_ids", make_i64({1, 1}, 1));
    set_input("attention_mask", make_i64({1, kPromptLen + 1}, 1));
    ASSERT_NO_THROW(m_request->infer());
    EXPECT_NE(logits(), nullptr);
}

}  // namespace
