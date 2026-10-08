// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "llm_block_kvcache_strategy.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "serialization.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct PrefillInferenceRecord {
    std::vector<size_t> sequence_sizes;
    std::vector<std::string> friendly_names;
};

struct LLMVariantSwitchTestAccess {
    static std::size_t generate_variant_count(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_generate_compiled_variants.size();
    }

    static std::size_t prefill_variant_count(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_prefill_compiled_variants.size();
    }

    static std::size_t select_prefill_variant_index(const ov::npuw::LLMInferRequest& req, uint64_t tail_length) {
        return req.select_prefill_variant_index(tail_length);
    }

    static void prepare_prefill_tail_variant(ov::npuw::LLMInferRequest& req,
                                             std::size_t variant_index,
                                             uint32_t num_stored_tokens) {
        req.prepare_prefill_tail_variant(variant_index, num_stored_tokens);
    }

    static std::size_t current_prefill_variant_index(const ov::npuw::LLMInferRequest& req) {
        return req.m_prefill_variant_idx;
    }

    static void run_chunked_prefill(ov::npuw::LLMInferRequest& req,
                                    const ov::SoPtr<ov::ITensor>& input_ids,
                                    const ov::SoPtr<ov::ITensor>& attention_mask,
                                    const ov::SoPtr<ov::ITensor>& position_ids) {
        req.infer_chunked_prefill(input_ids, attention_mask, position_ids, {}, {}, {}, {});
    }

    static ov::SoPtr<ov::ITensor> prefill_input_tensor(const ov::npuw::LLMInferRequest& req, const std::string& name) {
        return req.m_prefill_request->get_tensor(req.m_prefill_in_ports.at(name));
    }

    static ov::SoPtr<ov::ITensor> prefill_output_tensor(const ov::npuw::LLMInferRequest& req, const std::string& name) {
        return req.m_prefill_request->get_tensor(req.m_prefill_out_ports.at(name));
    }

    static uint32_t prefill_kv_dim(const ov::npuw::LLMInferRequest& req, const std::string& name) {
        const auto& desc = req.m_npuw_llm_compiled_model->m_kvcache_desc;
        return (ov::npuw::util::isPastValueParam(name) && desc.v_tensors_transposed_pre) ? 3u : desc.dim;
    }

    static uint32_t stored_tokens(const ov::npuw::LLMInferRequest& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens;
    }

    static uint32_t present_chunk_tokens(const ov::npuw::LLMInferRequest& req) {
        return static_cast<uint32_t>(req.m_tokens_in_present_chunk);
    }

    static std::shared_ptr<ov::npuw::ICompiledModel_v0> generate_variant(
        const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
        std::size_t idx) {
        return compiled->m_generate_compiled_variants.at(idx);
    }

    static bool is_block_kv_cache(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_is_block_kv_cache;
    }

    static void set_num_stored_tokens(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                      uint32_t num_tokens) {
        compiled->m_kvcache_desc.num_stored_tokens = num_tokens;
    }

    static uint32_t kv_dim_for_name(const ov::npuw::LLMInferRequest& req, const std::string& name) {
        const auto& desc = req.m_npuw_llm_compiled_model->m_kvcache_desc;
        // Mirror copy_kvcache()/on_generate_variant_switch(): classify via the present name so
        // quantized aux inputs (e.g. DynamicQuantize/0/past_key_values/value/scale) resolve too.
        const auto present = ov::npuw::util::past_key_values_to_present_name(name);
        return (present.find("value") != std::string::npos && desc.v_tensors_transposed_gen) ? 3u : desc.dim;
    }

    static void select_smallest_generate_variant(ov::npuw::LLMInferRequest& req) {
        ASSERT_GE(req.m_generate_requests.size(), 2u);
        req.m_kvcache_variant_idx = 0u;
        req.m_kvcache_request = req.m_generate_requests.front();
        req.m_kvcache_in_ports = req.m_generate_variant_in_ports.at(req.m_kvcache_request);
        req.m_kvcache_out_ports = req.m_generate_variant_out_ports.at(req.m_kvcache_request);
    }

    static uint32_t current_variant_capacity(ov::npuw::LLMInferRequest& req) {
        return req.get_current_variant_capacity();
    }

    static bool try_switch_to_larger_variant(ov::npuw::LLMInferRequest& req) {
        return req.try_switch_to_larger_variant();
    }

    static std::size_t current_variant_index(const ov::npuw::LLMInferRequest& req) {
        return req.m_kvcache_variant_idx;
    }

    static const std::vector<std::string>& kvcache_past_names(const ov::npuw::LLMInferRequest& req) {
        return req.m_kvcache_past_names;
    }

    static const auto& kvcache_in_ports(const ov::npuw::LLMInferRequest& req) {
        return req.m_kvcache_in_ports;
    }

    static const auto& kvcache_request(const ov::npuw::LLMInferRequest& req) {
        return req.m_kvcache_request;
    }

    static void set_kvcache_sizes(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                  std::vector<uint32_t> sizes) {
        compiled->m_kvcache_sizes = std::move(sizes);
    }

    static std::size_t generate_request_count(const ov::npuw::LLMInferRequest& req) {
        return req.m_generate_requests.size();
    }

    static std::shared_ptr<ov::IAsyncInferRequest> select_generate_request(ov::npuw::LLMInferRequest& req,
                                                                           int64_t prompt_length) {
        return req.select_generate_request(prompt_length);
    }

    static bool is_known_generate_request(const ov::npuw::LLMInferRequest& req,
                                          const std::shared_ptr<ov::IAsyncInferRequest>& request) {
        return std::find(req.m_generate_requests.begin(), req.m_generate_requests.end(), request) !=
               req.m_generate_requests.end();
    }

    static void set_current_variant_index(ov::npuw::LLMInferRequest& req, std::size_t idx) {
        req.m_kvcache_variant_idx = idx;
    }

    // Re-serialize the model's metadata in the exact field order LLMCompiledModel::serialize() uses
    // (write_model_meta), but write a forged trailing variant count instead of the real one. All
    // fields except that count come straight from the real, valid compiled model, so the blob
    // deserializes successfully right up to the size-table/variant-count invariant check.
    static std::string serialize_meta_with_variant_count(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                                         uint32_t forged_variant_count) {
        namespace s = ov::npuw::s11n;
        std::ostringstream os;
        s::write(os, compiled->m_name);
        s::write(os, compiled->inputs());
        s::write(os, compiled->outputs());
        auto& d = compiled->m_kvcache_desc;
        s::write(os, d.max_prompt_size);
        s::write(os, d.total_size);
        s::write(os, d.num_stored_tokens);
        s::write(os, d.dim);
        s::write(os, d.max_generation_token_len);
        s::write(os, d.v_tensors_transposed_pre);
        s::write(os, d.v_tensors_transposed_gen);
        s::write(os, compiled->m_prefill_chunk_size);
        s::write(os, compiled->m_use_chunk_prefill);
        s::write(os, compiled->m_max_lora_rank);
        s::write(os, compiled->m_enable_prefix_caching);
        s::write(os, compiled->m_prefix_caching_block_size);
        s::write(os, compiled->m_prefix_caching_max_num_blocks);
        s::write(os, compiled->m_longrope_context_limit);
        s::write(os, compiled->m_is_whisper);
        s::write(os, compiled->m_eos_token_id);
        s::write(os, compiled->m_decomposed_sdpa_size);
        s::write(os, compiled->m_is_eagle);
        s::write(os, compiled->m_is_embedding);
        s::write(os, compiled->m_is_block_kv_cache);
        s::write(os, compiled->m_is_encoder_embedding);
        s::write(os, compiled->m_swa_window_size);
        s::write(os, compiled->m_longrope_tables);
        s::write(os, compiled->m_cfg);
        s::write(os, compiled->m_kvcache_sizes);
        s::write(os, forged_variant_count);
        return os.str();
    }

    static std::shared_ptr<ov::npuw::LLMCompiledModel> deserialize(std::istream& stream,
                                                                   const std::shared_ptr<const ov::IPlugin>& plugin) {
        ov::npuw::s11n::CompiledContext ctx(false, nullptr, nullptr);
        return ov::npuw::LLMCompiledModel::deserialize(stream, plugin, {}, ctx);
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::LLMVariantSwitchTestAccess;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::PrefillInferenceRecord;
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

private:
    std::shared_ptr<const FakeSubCompiledModel> m_compiled_model;
};

class FakeSubCompiledModel final : public ov::npuw::ICompiledModel_v0 {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         const ov::AnyMap&,
                         std::shared_ptr<PrefillInferenceRecord> record)
        : ov::npuw::ICompiledModel_v0(model, plugin),
          m_model(model),
          m_record(std::move(record)) {}

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

    const std::shared_ptr<PrefillInferenceRecord>& record() const {
        return m_record;
    }

    std::size_t sequence_size() const {
        for (const auto& input : m_model->inputs()) {
            if (input.get_any_name().find("input_ids") != std::string::npos) {
                return input.get_shape()[1];
            }
        }
        return 0;
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        intel_npu::make_executor("variant_switch_task", 1),
                                                        intel_npu::make_executor("variant_switch_callback", 1));
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
    std::shared_ptr<PrefillInferenceRecord> m_record;
};

FakeSubInferRequest::FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(std::move(compiled_model)) {
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
    if (const auto& record = m_compiled_model->record()) {
        record->sequence_sizes.push_back(m_compiled_model->sequence_size());
        record->friendly_names.push_back(m_compiled_model->get_runtime_model()->get_friendly_name());
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        uint8_t marker = 0;
        for (const auto character : output.get_any_name()) {
            marker = static_cast<uint8_t>(marker + static_cast<uint8_t>(character));
        }
        std::memset(tensor->data(), marker, tensor->get_byte_size());
    }
}

class VariantSwitchFactory {
public:
    explicit VariantSwitchFactory(std::shared_ptr<PrefillInferenceRecord> record = {}) : m_record(std::move(record)) {}

    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        const auto record = m_record;
        return [record](const std::shared_ptr<ov::Model>& model,
                        const std::shared_ptr<const ov::IPlugin>& plugin,
                        const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<FakeSubCompiledModel>(model, plugin, props, record);
        };
    }

private:
    std::shared_ptr<PrefillInferenceRecord> m_record;
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

ov::Tensor make_i64_iota(std::size_t sequence_size, int64_t start) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{1, sequence_size});
    std::iota(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size(), start);
    return tensor;
}

ov::Tensor make_i64_filled(std::size_t sequence_size, int64_t value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{1, sequence_size});
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), value);
    return tensor;
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

    // Fills stored KV of the smallest variant, switches to the next variant, and verifies the
    // migrated bytes match. When expect_aux is set, at least one quantized aux (scale/zp) past
    // input must participate, so the check exercises the aux name mapping too.
    void check_continuous_switch_migration(const ov::AnyMap& extra_props, bool expect_aux) {
        VariantSwitchFactory factory;
        auto compiled = create_compiled_model(extra_props, factory);
        ASSERT_NE(compiled, nullptr);
        ASSERT_EQ(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 2u);

        ov::npuw::LLMInferRequest req(compiled);
        LLMVariantSwitchTestAccess::select_smallest_generate_variant(req);

        const uint32_t stored_tokens = LLMVariantSwitchTestAccess::current_variant_capacity(req);
        LLMVariantSwitchTestAccess::set_num_stored_tokens(compiled, stored_tokens);

        bool saw_aux = false;
        std::unordered_map<std::string, std::vector<uint8_t>> expected_kv_bytes;
        uint8_t seed = 17u;
        for (const auto& name : LLMVariantSwitchTestAccess::kvcache_past_names(req)) {
            saw_aux = saw_aux || name.find("scale") != std::string::npos || name.find("zp") != std::string::npos;
            auto src = LLMVariantSwitchTestAccess::kvcache_request(req)->get_tensor(
                LLMVariantSwitchTestAccess::kvcache_in_ports(req).at(name));
            auto src_slice = ov::npuw::util::make_tensor_slice(src,
                                                               LLMVariantSwitchTestAccess::kv_dim_for_name(req, name),
                                                               0u,
                                                               stored_tokens);
            fill_tensor_pattern(src_slice, seed);
            expected_kv_bytes.emplace(name, materialize_bytes(src_slice));
            seed = static_cast<uint8_t>(seed + 37u);
        }
        if (expect_aux) {
            ASSERT_TRUE(saw_aux) << "Quantized model must expose scale/zero-point past inputs";
        }

        ASSERT_TRUE(LLMVariantSwitchTestAccess::try_switch_to_larger_variant(req));
        EXPECT_EQ(LLMVariantSwitchTestAccess::current_variant_index(req), 1u);

        for (const auto& name : LLMVariantSwitchTestAccess::kvcache_past_names(req)) {
            auto dst = LLMVariantSwitchTestAccess::kvcache_request(req)->get_tensor(
                LLMVariantSwitchTestAccess::kvcache_in_ports(req).at(name));
            auto dst_slice = ov::npuw::util::make_tensor_slice(dst,
                                                               LLMVariantSwitchTestAccess::kv_dim_for_name(req, name),
                                                               0u,
                                                               stored_tokens);
            EXPECT_EQ(materialize_bytes(dst_slice), expected_kv_bytes.at(name)) << name;
        }
    }
};

TEST_F(LLMInferRequestVariantSwitchTest, ContinuousKvSwitchMigratesStoredTokensToLargerVariant) {
    check_continuous_switch_migration({}, /*expect_aux=*/false);
}

// Same migration path, but for an i8-quantized KV cache: on_generate_variant_switch() must also
// migrate the quantized aux (scale/zero-point) past tensors to the larger variant.
TEST_F(LLMInferRequestVariantSwitchTest, ContinuousKvSwitchMigratesQuantizedAuxTensors) {
    check_continuous_switch_migration({{ov::hint::kv_cache_precision.name(), ov::element::i8}}, /*expect_aux=*/true);
}

TEST_F(LLMInferRequestVariantSwitchTest, ShorterPrefillChunkSelectsAndActivatesTailVariant) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                           {"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"},
                                           {"NPUW_LLM_PREFILL_SHORTER_CHUNK_SIZE", "32"}},
                                          factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(LLMVariantSwitchTestAccess::prefill_variant_count(compiled), 2u);

    ov::npuw::LLMInferRequest req(compiled);
    EXPECT_EQ(LLMVariantSwitchTestAccess::select_prefill_variant_index(req, 32u), 0u);
    EXPECT_EQ(LLMVariantSwitchTestAccess::select_prefill_variant_index(req, 33u), 1u);

    LLMVariantSwitchTestAccess::prepare_prefill_tail_variant(req, 0u, 0u);
    EXPECT_EQ(LLMVariantSwitchTestAccess::current_prefill_variant_index(req), 0u);
}

TEST_F(LLMInferRequestVariantSwitchTest, ShorterPrefillChunkRunsDefaultThenShortTail) {
    const auto short_record = std::make_shared<PrefillInferenceRecord>();
    VariantSwitchFactory short_factory(short_record);
    auto short_compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                 {"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"},
                                                 {"NPUW_LLM_PREFILL_SHORTER_CHUNK_SIZE", "32"}},
                                                short_factory);
    ASSERT_NE(short_compiled, nullptr);
    ov::npuw::LLMInferRequest short_request(short_compiled);

    auto input_ids = make_i64_iota(96u, 1);
    auto attention_mask = make_i64_filled(96u, 1);
    auto position_ids = make_i64_iota(96u, 0);
    LLMVariantSwitchTestAccess::run_chunked_prefill(short_request,
                                                    ov::get_tensor_impl(input_ids),
                                                    ov::get_tensor_impl(attention_mask),
                                                    ov::get_tensor_impl(position_ids));

    ASSERT_EQ(short_record->sequence_sizes, (std::vector<size_t>{64u, 32u}));
    ASSERT_EQ(short_record->friendly_names.size(), 2u);
    EXPECT_NE(short_record->friendly_names[1].find("_chunk32"), std::string::npos);
    EXPECT_EQ(LLMVariantSwitchTestAccess::stored_tokens(short_request), 96u);
    EXPECT_EQ(LLMVariantSwitchTestAccess::present_chunk_tokens(short_request), 32u);

    const auto default_record = std::make_shared<PrefillInferenceRecord>();
    VariantSwitchFactory default_factory(default_record);
    auto default_compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"}},
                              default_factory);
    ASSERT_NE(default_compiled, nullptr);
    ov::npuw::LLMInferRequest default_request(default_compiled);
    LLMVariantSwitchTestAccess::run_chunked_prefill(default_request,
                                                    ov::get_tensor_impl(input_ids),
                                                    ov::get_tensor_impl(attention_mask),
                                                    ov::get_tensor_impl(position_ids));

    ASSERT_EQ(default_record->sequence_sizes, (std::vector<size_t>{64u, 64u}));
    EXPECT_EQ(LLMVariantSwitchTestAccess::stored_tokens(default_request), 96u);
    EXPECT_EQ(LLMVariantSwitchTestAccess::present_chunk_tokens(default_request), 32u);

    for (const auto& name : LLMVariantSwitchTestAccess::kvcache_past_names(short_request)) {
        const auto short_past = LLMVariantSwitchTestAccess::prefill_input_tensor(short_request, name);
        const auto default_past = LLMVariantSwitchTestAccess::prefill_input_tensor(default_request, name);
        const auto kv_dim = LLMVariantSwitchTestAccess::prefill_kv_dim(short_request, name);
        EXPECT_EQ(materialize_bytes(ov::npuw::util::make_tensor_slice(short_past, kv_dim, 0u, 64u)),
                  materialize_bytes(ov::npuw::util::make_tensor_slice(default_past, kv_dim, 0u, 64u)))
            << name;

        std::string output_name = name;
        const auto past_prefix = output_name.find("past_key_values");
        ASSERT_NE(past_prefix, std::string::npos);
        output_name.replace(past_prefix, std::string("past_key_values").size(), "present");

        const auto short_present = LLMVariantSwitchTestAccess::prefill_output_tensor(short_request, output_name);
        const auto default_present = LLMVariantSwitchTestAccess::prefill_output_tensor(default_request, output_name);
        EXPECT_EQ(materialize_bytes(ov::npuw::util::make_tensor_slice(short_present, kv_dim, 0u, 32u)),
                  materialize_bytes(ov::npuw::util::make_tensor_slice(default_present, kv_dim, 32u, 64u)))
            << name;
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
    ASSERT_EQ(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 2u);
    ASSERT_TRUE(LLMVariantSwitchTestAccess::is_block_kv_cache(compiled));
    auto small_variant =
        std::dynamic_pointer_cast<FakeSubCompiledModel>(LLMVariantSwitchTestAccess::generate_variant(compiled, 0u));
    auto large_variant = std::dynamic_pointer_cast<FakeSubCompiledModel>(LLMVariantSwitchTestAccess::generate_variant(
        compiled,
        LLMVariantSwitchTestAccess::generate_variant_count(compiled) - 1u));
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

// Block-based KV cache relies on SplitKVCacheIntoBlocks, whose pattern no longer matches once the
// KV cache is precision-converted to i8. The two features are mutually exclusive, so enabling both
// must be rejected at compile time (guards against silently producing an unblocked / unquantized
// model). This is why the block strategy never has to handle quantized aux tensors.
TEST_F(LLMInferRequestVariantSwitchTest, BlockKvCacheRejectsQuantizedKvCache) {
    VariantSwitchFactory factory;
    try {
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                               {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"},
                               {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
                               {"NPUW_LLM_GENERATE_ATTENTION_HINT", "PYRAMID"},
                               {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"},
                               {ov::hint::kv_cache_precision.name(), ov::element::i8}},
                              factory);
        FAIL() << "Expected block-based KV cache with integer KV-cache precision to be rejected";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("cannot be combined with integer KV-cache precision"), std::string::npos)
            << ex.what();
    }
}

// The deserializer restores m_kvcache_sizes and the generate-variant count as two independent blob
// fields. A corrupted blob can therefore declare a size table that disagrees with the variant count,
// which downstream turns into an out-of-bounds read. LLMCompiledModel::deserialize() now rejects the
// mismatch at restore time. This exercises the real import path: it re-serializes a valid model's
// metadata but forges the trailing variant count so it no longer matches the 2-entry size table.
TEST_F(LLMInferRequestVariantSwitchTest, ImportRejectsSizeTableThatDisagreesWithVariantCount) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 2u);

    const std::string blob = LLMVariantSwitchTestAccess::serialize_meta_with_variant_count(compiled, /*forged=*/3u);
    std::istringstream in(blob);

    try {
        LLMVariantSwitchTestAccess::deserialize(in, m_plugin);
        FAIL() << "Expected import to reject a kvcache size table that disagrees with the variant count";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("does not match generate variant count"), std::string::npos) << ex.what();
    }
}

// The loader now enforces the invariant, and this test pins the defence-in-depth guard in the consumer:
// even if a mismatched size table reaches it, the walk is bounded to the shorter container.
TEST_F(LLMInferRequestVariantSwitchTest, SelectGenerateRequestStaysInBoundsWhenSizeTableExceedsVariants) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 2u);

    ov::npuw::LLMInferRequest req(compiled);
    ASSERT_EQ(LLMVariantSwitchTestAccess::generate_request_count(req), 2u);

    LLMVariantSwitchTestAccess::set_kvcache_sizes(
        compiled,
        {0u, 0u, std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()});

    auto selected = LLMVariantSwitchTestAccess::select_generate_request(req, /*prompt_length=*/16);

    EXPECT_TRUE(LLMVariantSwitchTestAccess::is_known_generate_request(req, selected));
}

// m_kvcache_variant_idx is derived from a std::find over m_generate_requests and then used to index
// the independently sized m_kvcache_sizes. An out-of-range index must fault loudly via .at() instead
// of reading past the size table.
TEST_F(LLMInferRequestVariantSwitchTest, CurrentVariantCapacityRejectsOutOfRangeVariantIndex) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);

    ov::npuw::LLMInferRequest req(compiled);

    LLMVariantSwitchTestAccess::set_kvcache_sizes(compiled, {128u});
    LLMVariantSwitchTestAccess::set_current_variant_index(req, 1u);

    EXPECT_THROW(LLMVariantSwitchTestAccess::current_variant_capacity(req), std::out_of_range);
}

}  // namespace
