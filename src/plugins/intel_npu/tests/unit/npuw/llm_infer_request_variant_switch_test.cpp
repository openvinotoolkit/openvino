// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define private public
#define protected public
#include "llm_block_kvcache_strategy.hpp"
#include "llm_infer_request.hpp"
#undef protected
#undef private

#include "llm_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace {

using ov::test::npuw::build_llm_test_model;
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

class FakeSubAsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    explicit FakeSubAsyncInferRequest(const std::shared_ptr<ov::ISyncInferRequest>& request)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_request(request) {}

    void start_async() override {
        try {
            m_request->infer();
            if (m_callback) {
                m_callback(nullptr);
            }
        } catch (...) {
            if (m_callback) {
                m_callback(std::current_exception());
                return;
            }
            throw;
        }
    }

    void wait() override {}

    bool wait_for(const std::chrono::milliseconds&) override {
        return true;
    }

    void cancel() override {}

    void set_callback(std::function<void(std::exception_ptr)> callback) override {
        m_callback = std::move(callback);
    }

    void infer() override {
        m_request->infer();
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_request->get_profiling_info();
    }

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return m_request->get_tensor(port);
    }

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        m_request->set_tensor(port, tensor);
    }

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override {
        return m_request->get_tensors(port);
    }

    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override {
        m_request->set_tensors(port, tensors);
    }

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return m_request->query_state();
    }

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override {
        return m_request->get_compiled_model();
    }

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return m_request->get_inputs();
    }

    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return m_request->get_outputs();
    }

private:
    std::shared_ptr<ov::ISyncInferRequest> m_request;
    std::function<void(std::exception_ptr)> m_callback;
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
    ov::Any get_property(const std::string& name) const override {
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{"CPU"};
        }
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
        return std::make_shared<FakeSubAsyncInferRequest>(create_sync_infer_request());
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
        ov::ISyncInferRequest::set_tensor(output,
                                          ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void FakeSubInferRequest::infer() {
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

class VariantSwitchFactory {
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

void fill_tensor_pattern(const ov::SoPtr<ov::ITensor>& tensor, uint8_t seed) {
    ov::Tensor dense(tensor->get_element_type(), tensor->get_shape());
    auto* data = static_cast<uint8_t*>(dense.data());
    for (size_t i = 0; i < dense.get_byte_size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + (i % 251));
    }
    ov::get_tensor_impl(dense)->copy_to(tensor._ptr);
}

uint32_t kv_dim_for_name(const ov::npuw::LLMInferRequest& req, const std::string& name) {
    const auto& desc = req.m_npuw_llm_compiled_model->m_kvcache_desc;
    return (ov::npuw::util::isPastValueParam(name) && desc.v_tensors_transposed_gen) ? 3u : desc.dim;
}

void select_smallest_generate_variant(ov::npuw::LLMInferRequest& req) {
    ASSERT_GE(req.m_generate_requests.size(), 2u);
    req.m_kvcache_variant_idx = 0u;
    req.m_kvcache_request = req.m_generate_requests.front();
    req.m_kvcache_in_ports = req.m_generate_variant_in_ports.at(req.m_kvcache_request);
    req.m_kvcache_out_ports = req.m_generate_variant_out_ports.at(req.m_kvcache_request);
}

class LLMInferRequestVariantSwitchTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"},
                {"NPUW_DEVICES", "CPU"},
                {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                {"NPUW_LLM_GENERATE_PYRAMID", "YES"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    std::shared_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(const ov::AnyMap& extra_props,
                                                                      VariantSwitchFactory& factory) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_shared<ov::npuw::LLMCompiledModel>(build_llm_test_model(),
                                                            m_plugin,
                                                            props,
                                                            factory.make_factory());
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(LLMInferRequestVariantSwitchTest, ContinuousKvSwitchMigratesStoredTokensToLargerVariant) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(compiled->m_generate_compiled_variants.size(), 2u);

    ov::npuw::LLMInferRequest req(compiled);
    select_smallest_generate_variant(req);

    const uint32_t stored_tokens = req.get_current_variant_capacity();
    compiled->m_kvcache_desc.num_stored_tokens = stored_tokens;

    std::unordered_map<std::string, std::vector<uint8_t>> expected_kv_bytes;
    uint8_t seed = 17u;
    for (const auto& name : req.m_kvcache_past_names) {
        auto src = req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
        auto src_slice = ov::npuw::util::make_tensor_slice(src, kv_dim_for_name(req, name), 0u, stored_tokens);
        fill_tensor_pattern(src_slice, seed);
        expected_kv_bytes.emplace(name, materialize_bytes(src_slice));
        seed = static_cast<uint8_t>(seed + 37u);
    }

    ASSERT_TRUE(req.try_switch_to_larger_variant());
    EXPECT_EQ(req.m_kvcache_variant_idx, 1u);

    for (const auto& name : req.m_kvcache_past_names) {
        auto dst = req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
        auto dst_slice = ov::npuw::util::make_tensor_slice(dst, kv_dim_for_name(req, name), 0u, stored_tokens);
        EXPECT_EQ(materialize_bytes(dst_slice), expected_kv_bytes.at(name)) << name;
    }
}

TEST_F(LLMInferRequestVariantSwitchTest, BlockKvVariantsExposeCompatibleBindingsAcrossSwitchBoundary) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                           {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"},
                                           {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
                                           {"NPUW_LLM_GENERATE_ATTENTION_HINT", "PYRAMID"},
                                           {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"}},
                                          factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(compiled->m_generate_compiled_variants.size(), 2u);
    ASSERT_TRUE(compiled->m_is_block_kv_cache);
    auto small_variant = std::dynamic_pointer_cast<FakeSubCompiledModel>(compiled->m_generate_compiled_variants.front());
    auto large_variant = std::dynamic_pointer_cast<FakeSubCompiledModel>(compiled->m_generate_compiled_variants.back());
    ASSERT_NE(small_variant, nullptr);
    ASSERT_NE(large_variant, nullptr);

    const auto collect_block0_inputs = [](const std::shared_ptr<const ov::Model>& model, const std::string& suffix) {
        std::vector<std::string> names;
        for (const auto& input : model->inputs()) {
            const auto& name = input.get_any_name();
            if (name.find("past_key_values.") == 0 && name.find(suffix) != std::string::npos) {
                names.push_back(name);
            }
        }
        std::sort(names.begin(), names.end());
        return names;
    };

    const auto small_model = small_variant->get_runtime_model();
    const auto large_model = large_variant->get_runtime_model();
    ASSERT_NE(small_model, nullptr);
    ASSERT_NE(large_model, nullptr);

    const auto small_key_block0 = collect_block0_inputs(small_model, ".key_block_0");
    const auto small_value_block0 = collect_block0_inputs(small_model, ".value_block_0");
    const auto large_key_block0 = collect_block0_inputs(large_model, ".key_block_0");
    const auto large_value_block0 = collect_block0_inputs(large_model, ".value_block_0");

    ASSERT_FALSE(small_key_block0.empty());
    ASSERT_FALSE(small_value_block0.empty());
    EXPECT_EQ(small_key_block0, large_key_block0);
    EXPECT_EQ(small_value_block0, large_value_block0);
}

}  // namespace
