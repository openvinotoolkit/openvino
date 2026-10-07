// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Request-level regression tests for SwaKVCacheHelper: classifying SWA-managed
// KV names, sharing SWA tensors across generate variants, and handing SWA past
// KV off between prefill and generate. Submodels are compiled through a factory
// that returns fake sub-requests whose infer() stamps every "present" output
// with a call-local, increasing byte value ("color") and zeroes everything
// else, so every byte observed in an SWA KV tensor was moved there by the
// plugin's own SWA plumbing, not the fake compute.
//
// The low-level write algorithms are already covered by
// sliding_window_manager_update_kv_cache_test.cpp; these tests stay within the
// SWA window to exercise the wiring only, not that math.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "executor.hpp"
#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct LLMSwaCacheTestAccess {
    using Request = ov::npuw::LLMInferRequest;

    static const ov::npuw::LLMCompiledModel::KVCacheDesc& desc(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc;
    }

    static const std::vector<std::string>& swa_past_names(const Request& req) {
        return req.m_swa_past_names;
    }

    static const std::vector<std::string>& kvcache_past_names(const Request& req) {
        return req.m_kvcache_past_names;
    }

    static ov::SoPtr<ov::ITensor> prefill_past(Request& req, const std::string& name) {
        return req.m_prefill_request->get_tensor(req.m_prefill_in_ports.at(name));
    }

    static ov::SoPtr<ov::ITensor> generate_past(Request& req, const std::string& name) {
        return req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
    }

    static std::size_t generate_variant_count(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_generate_compiled_variants.size();
    }

    static ov::SoPtr<ov::ITensor> generate_variant_past(Request& req,
                                                        std::size_t variant_idx,
                                                        const std::string& name) {
        const auto& request = req.m_generate_requests.at(variant_idx);
        const auto& ports = req.m_generate_variant_in_ports.at(request);
        return request->get_tensor(ports.at(name));
    }

    static uint32_t kv_dim(const Request& req, const std::string& name, bool v_transposed) {
        const auto& desc = req.m_npuw_llm_compiled_model->m_kvcache_desc;
        return (ov::npuw::util::isPastValueParam(name) && v_transposed) ? 3u : desc.dim;
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_sliding_window_test_model;
using ov::test::npuw::LLMSwaCacheTestAccess;
using ov::test::npuw::NullPlugin;
class FakeSubCompiledModel;

// Stamps "present" outputs with a per-instance call counter ("color") on each
// infer(); everything else is zeroed.
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

private:
    uint8_t m_calls = 0u;
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
                                                        intel_npu::make_executor("swa_cache_task", 1),
                                                        intel_npu::make_executor("swa_cache_callback", 1));
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        return 1;  // keeps init_pre_alloc_device() on CPU staging, not NPU remote tensors
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
    ++m_calls;
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        const auto& name = output.get_any_name();
        const uint8_t value = (name.find("present") != std::string::npos) ? m_calls : 0u;
        std::memset(tensor->data(), value, tensor->get_byte_size());
    }
}

class SwaCacheFactory {
public:
    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [](const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
        };
    }
};

std::vector<uint8_t> materialize_bytes(const ov::SoPtr<ov::ITensor>& tensor) {
    ov::Tensor copy(tensor->get_element_type(), tensor->get_shape());
    tensor->copy_to(ov::get_tensor_impl(copy)._ptr);
    auto* data = static_cast<uint8_t*>(copy.data());
    return std::vector<uint8_t>(data, data + copy.get_byte_size());
}

ov::Tensor make_i64(std::initializer_list<size_t> shape, int64_t fill_value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), fill_value);
    return tensor;
}

constexpr size_t kWindowSize = 32;  // << kv_cache_size (192), so shrinkage is visible
constexpr size_t kNumLayers = 4;    // layers 0,2 sliding; layers 1,3 full-attention
constexpr size_t kChunkSize = 8;

class LLMSwaCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    void init(const ov::AnyMap& extra_props) {
        SwaCacheFactory factory;
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                         {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                         {"NPUW_LLM_ENABLE_SWA_KV_CACHE_SHRINK", "YES"}};
        for (const auto& [key, value] : extra_props) {
            props[key] = value;
        }
        auto model = build_sliding_window_test_model(kWindowSize, /*sliding_to_full_ratio=*/1, {}, kNumLayers);
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, m_plugin, props, factory.make_factory());
        ASSERT_NE(m_compiled, nullptr);
        m_request = std::make_unique<ov::npuw::LLMInferRequest>(m_compiled);
    }

    void set_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids) {
        const auto& inputs = m_request->get_inputs();
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "input_ids").value(),
                              ov::get_tensor_impl(input_ids));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "attention_mask").value(),
                              ov::get_tensor_impl(attention_mask));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "position_ids").value(),
                              ov::get_tensor_impl(position_ids));
    }

    void run_prefill(size_t prompt_len) {
        set_inputs(make_i64({1, prompt_len}, 1), make_i64({1, prompt_len}, 1), make_i64({1, prompt_len}, 0));
        m_request->infer();
    }

    void run_generate_step(size_t live_tokens) {
        set_inputs(make_i64({1, 1}, 1),
                   make_i64({1, live_tokens + 1}, 1),
                   make_i64({1, 1}, static_cast<int64_t>(live_tokens)));
        m_request->infer();
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::unique_ptr<ov::npuw::LLMInferRequest> m_request;
};

// Only the sliding layers (0, 2) should be classified as SWA-managed; the
// full-attention layers (1, 3) must stay on the regular contiguous-KV path.
TEST_F(LLMSwaCacheTest, OnlySlidingLayersAreClassifiedAsSwaManaged) {
    init({});
    const auto& swa_names = LLMSwaCacheTestAccess::swa_past_names(*m_request);
    const auto& kv_names = LLMSwaCacheTestAccess::kvcache_past_names(*m_request);

    ASSERT_EQ(swa_names.size(), 4u);  // 2 sliding layers * (key + value)
    for (const auto& name : swa_names) {
        EXPECT_TRUE(ov::npuw::util::is_swa_kv_cache_name(name)) << name;
    }
    for (const auto& name : kv_names) {
        EXPECT_FALSE(ov::npuw::util::is_swa_kv_cache_name(name)) << name;
    }
    for (const auto& name : swa_names) {
        EXPECT_EQ(std::find(kv_names.begin(), kv_names.end(), name), kv_names.end()) << name;
    }
}

// Two-chunk prefill (16 tokens, well inside the 32-token window) followed by
// one generate step: checks that every byte lands where the prefill<->generate
// handoff claims (update_prefill(), copy_prefill_to_generate(), update_generate()).
TEST_F(LLMSwaCacheTest, ChunkedPrefillUpdatesWindowAndHandsOffToGenerate) {
    init({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", std::to_string(kChunkSize)}});
    const auto& swa_names = LLMSwaCacheTestAccess::swa_past_names(*m_request);
    ASSERT_FALSE(swa_names.empty());

    // Chunk 1 (intermediate) is merged into the prefill past tensor (color 1);
    // chunk 2 (final) is left unmerged until the generate handoff.
    run_prefill(2 * kChunkSize);

    const auto& desc = LLMSwaCacheTestAccess::desc(*m_request);
    for (const auto& name : swa_names) {
        const auto dim = LLMSwaCacheTestAccess::kv_dim(*m_request, name, desc.v_tensors_transposed_pre);
        auto past = LLMSwaCacheTestAccess::prefill_past(*m_request, name);

        auto chunk0 = ov::npuw::util::make_tensor_slice(past, dim, 0u, static_cast<uint32_t>(kChunkSize));
        auto expected0 = std::vector<uint8_t>(chunk0->get_byte_size(), 1u);
        EXPECT_EQ(materialize_bytes(chunk0), expected0) << name << " (chunk 1, merged via update_prefill)";

        auto chunk1 = ov::npuw::util::make_tensor_slice(past,
                                                        dim,
                                                        static_cast<uint32_t>(kChunkSize),
                                                        static_cast<uint32_t>(2 * kChunkSize));
        auto expected_zero = std::vector<uint8_t>(chunk1->get_byte_size(), 0u);
        EXPECT_EQ(materialize_bytes(chunk1), expected_zero) << name << " (chunk 2, not yet merged)";
    }

    run_generate_step(2 * kChunkSize);

    for (const auto& name : swa_names) {
        const auto dim = LLMSwaCacheTestAccess::kv_dim(*m_request, name, desc.v_tensors_transposed_gen);
        auto generate_past = LLMSwaCacheTestAccess::generate_past(*m_request, name);

        auto history = ov::npuw::util::make_tensor_slice(generate_past, dim, 0u, static_cast<uint32_t>(kChunkSize));
        auto expected_chunk0 = std::vector<uint8_t>(history->get_byte_size(), 1u);
        EXPECT_EQ(materialize_bytes(history), expected_chunk0) << name << " (handed off chunk 1)";

        auto tail = ov::npuw::util::make_tensor_slice(generate_past,
                                                      dim,
                                                      static_cast<uint32_t>(kChunkSize),
                                                      static_cast<uint32_t>(2 * kChunkSize));
        auto expected_chunk1 = std::vector<uint8_t>(tail->get_byte_size(), 2u);
        EXPECT_EQ(materialize_bytes(tail), expected_chunk1) << name
                                                            << " (handed off chunk 2, color from its own "
                                                               "present output)";

        auto new_token = ov::npuw::util::make_tensor_slice(generate_past,
                                                           dim,
                                                           static_cast<uint32_t>(2 * kChunkSize),
                                                           static_cast<uint32_t>(2 * kChunkSize + 1));
        auto expected_new_token = std::vector<uint8_t>(new_token->get_byte_size(), 1u);
        EXPECT_EQ(materialize_bytes(new_token), expected_new_token)
            << name << " (appended by update_generate from the generate model's own first call)";
    }
}

// share_across_generate_variants() should alias every SWA past tensor from the
// largest generate variant onto the smaller ones, not just shape-match them.
TEST_F(LLMSwaCacheTest, SharesSwaPastTensorsAcrossGenerateVariants) {
    init({{"NPUW_LLM_GENERATE_PYRAMID", "YES"}, {"NPUW_LLM_MAX_PROMPT_LEN", "2048"}});
    ASSERT_GE(LLMSwaCacheTestAccess::generate_variant_count(m_compiled), 2u);

    const auto& swa_names = LLMSwaCacheTestAccess::swa_past_names(*m_request);
    ASSERT_FALSE(swa_names.empty());

    const auto last_variant = LLMSwaCacheTestAccess::generate_variant_count(m_compiled) - 1u;
    for (const auto& name : swa_names) {
        auto largest = LLMSwaCacheTestAccess::generate_variant_past(*m_request, last_variant, name);
        for (std::size_t i = 0; i < last_variant; ++i) {
            auto smaller = LLMSwaCacheTestAccess::generate_variant_past(*m_request, i, name);
            EXPECT_EQ(smaller->data(), largest->data()) << name << " variant " << i;
        }
    }
}

}  // namespace
