// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "llm_block_kvcache_strategy.hpp"
#include "llm_infer_request.hpp"
#include "llm_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "serialization.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct LLMVariantSwitchTestAccess {
    static std::size_t generate_variant_count(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_generate_compiled_variants.size();
    }

    static std::shared_ptr<ov::npuw::ICompiledModel_v0> generate_variant(
        const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
        std::size_t idx) {
        return compiled->m_generate_compiled_variants.at(idx);
    }

    static void set_generate_variant(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                     std::size_t idx,
                                     std::shared_ptr<ov::npuw::ICompiledModel_v0> variant) {
        compiled->m_generate_compiled_variants.at(idx) = std::move(variant);
    }

    static bool is_block_kv_cache(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_is_block_kv_cache;
    }

    static void set_num_stored_tokens(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled, uint32_t num_tokens) {
        compiled->m_kvcache_desc.num_stored_tokens = num_tokens;
    }

    static void set_kv_dim(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled, uint32_t dim) {
        compiled->m_kvcache_desc.dim = dim;
    }

    static void validate_imported_kvcache_dim(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        compiled->validate_imported_kvcache_dim();
    }

    static void validate_imported_sequence_extents(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        compiled->validate_imported_sequence_extents();
    }

    static void set_prefill_chunk_size(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled, uint64_t size) {
        compiled->m_prefill_chunk_size = size;
    }

    static void set_max_generation_token_len(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                             uint32_t len) {
        compiled->m_kvcache_desc.max_generation_token_len = len;
    }

    static uint32_t max_prompt_size(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_kvcache_desc.max_prompt_size;
    }

    static void set_prefill_compiled(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                     std::shared_ptr<ov::npuw::ICompiledModel_v0> prefill) {
        compiled->m_prefill_compiled = std::move(prefill);
    }

    // Mirrors the production name-matching in validate_imported_kvcache_dim(): a port may carry
    // several names, so match against all of them rather than get_any_name()'s single alias.
    static uint32_t past_key_tensor_rank(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        for (const auto& port : compiled->m_generate_compiled_variants.back()->inputs()) {
            const auto& names = port.get_names();
            const bool is_past_key = std::any_of(names.begin(), names.end(), [](const std::string& name) {
                return ov::npuw::util::isPastKeyParam(name);
            });
            if (is_past_key) {
                return static_cast<uint32_t>(port.get_partial_shape().rank().get_length());
            }
        }
        return 0u;
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

    static const std::vector<uint32_t>& kvcache_sizes(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_kvcache_sizes;
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
            auto src_slice = ov::npuw::util::make_tensor_slice(
                src, LLMVariantSwitchTestAccess::kv_dim_for_name(req, name), 0u, stored_tokens);
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
            auto dst_slice = ov::npuw::util::make_tensor_slice(
                dst, LLMVariantSwitchTestAccess::kv_dim_for_name(req, name), 0u, stored_tokens);
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
    auto small_variant = std::dynamic_pointer_cast<FakeSubCompiledModel>(
        LLMVariantSwitchTestAccess::generate_variant(compiled, 0u));
    auto large_variant = std::dynamic_pointer_cast<FakeSubCompiledModel>(
        LLMVariantSwitchTestAccess::generate_variant(compiled,
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

// KVCacheDesc::dim is read verbatim from an imported blob and is later used to subscript shape
// vectors in the KV-slice helpers, so a malformed value is a heap out-of-bounds write. deserialize()
// runs validate_imported_kvcache_dim() at the trust boundary to reject such blobs.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedKvcacheDimOutOfTensorRank) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);

    const uint32_t rank = LLMVariantSwitchTestAccess::past_key_tensor_rank(compiled);
    ASSERT_GT(rank, 0u) << "test model is expected to expose past-key KV tensors";

    for (uint32_t dim = 0; dim < rank; ++dim) {
        LLMVariantSwitchTestAccess::set_kv_dim(compiled, dim);
        EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_kvcache_dim(compiled)) << "dim=" << dim;
    }

    LLMVariantSwitchTestAccess::set_kv_dim(compiled, rank);
    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_kvcache_dim(compiled), ov::Exception);

    LLMVariantSwitchTestAccess::set_kv_dim(compiled, 0xFFFFFFFFu);
    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_kvcache_dim(compiled), ov::Exception);
}

// A dim valid for the rank-4 generate KV tensors must still be rejected when a restored prefill KV
// tensor has a smaller rank, since prefill SWA updates slice along it too.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedKvcacheDimOutOfPrefillTensorRank) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(LLMVariantSwitchTestAccess::past_key_tensor_rank(compiled), 4u);

    auto past = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 8});
    past->output(0).set_names({"past_key_values.0.swa.key"});
    auto present = std::make_shared<ov::op::v0::Result>(past);
    present->output(0).set_names({"present.0.swa.key"});
    auto prefill_model = std::make_shared<ov::Model>(ov::ResultVector{present}, ov::ParameterVector{past});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(prefill_model, m_plugin, ov::AnyMap{}));

    LLMVariantSwitchTestAccess::set_kv_dim(compiled, 1u);
    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_kvcache_dim(compiled));

    LLMVariantSwitchTestAccess::set_kv_dim(compiled, 2u);
    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_kvcache_dim(compiled), ov::Exception);
}

// m_prefill_chunk_size is read verbatim from an imported blob and the chunked-prefill staging
// copies trust it to be no larger than the restored prefill model's sequence-length ports
// (ReshapeToStatic makes that true on the compile path). A blob that declares a bigger chunk than
// its own prefill ports turns `tensor->get_size() - count` into a size_t underflow at runtime -
// deserialize() runs validate_imported_sequence_extents() at the trust boundary to reject it.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedPrefillChunkSizeLargerThanPrefillPort) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));

    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, 1024u);
    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Same trust boundary, but for the non-chunked (whole) prefill path, where the staging copies
// trust m_kvcache_desc.max_prompt_size to fit the restored prefill ports instead.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedMaxPromptSizeLargerThanPrefillPort) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "STATIC"}}, factory);
    ASSERT_NE(compiled, nullptr);

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));

    auto past = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    past->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(past);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{past});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// m_kvcache_desc.max_generation_token_len is read verbatim from an imported blob and the
// infer_generate() staging copies trust it to be no larger than the restored generate variants'
// sequence-length ports. A blob declaring a bigger limit than its own generate ports turns the
// same right-aligned `get_byte_size() - count` subtraction into a size_t underflow.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedMaxGenerationTokenLenLargerThanGeneratePort) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}}, factory);
    ASSERT_NE(compiled, nullptr);

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));

    LLMVariantSwitchTestAccess::set_max_generation_token_len(compiled, 16u);
    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// VLM prefill's `inputs_embeds` is [BATCH, SEQ_LEN, EMB_SIZE]: the sequence extent is axis 1,
// not the trailing axis (the embedding width). A port restored as [1, 8, 4096] must still be
// rejected against a declared chunk size of 16, even though 16 <= 4096 (the embedding width)
// would wrongly pass a check that read the last axis instead of axis 1.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedInputsEmbedsSeqAxisSmallerThanPrefillChunkSize) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, 16u);

    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 4096});
    embeds->output(0).set_names({"inputs_embeds"});
    auto result = std::make_shared<ov::op::v0::Result>(embeds);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Negative control for the above: a port whose axis-1 (sequence) extent actually meets the
// declared chunk size must be accepted, proving axis 1 (not just "reject inputs_embeds always")
// is what gets checked.
TEST_F(LLMInferRequestVariantSwitchTest, AcceptsImportedInputsEmbedsWhenSeqAxisMeetsPrefillChunkSize) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, 16u);

    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 4096});
    embeds->output(0).set_names({"inputs_embeds"});
    auto result = std::make_shared<ov::op::v0::Result>(embeds);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));
}

// The non-sequence-dimension floor must hold even when there is no top-level port to compare
// against: a dynamic embedding width leaves the destination tensor's real storage unknown at
// import time, even though its own sequence axis (axis 1) still satisfies the declared limit.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedInputsEmbedsWithDynamicEmbSizeAndNoTopLevelPort) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, 16u);

    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, -1});
    embeds->output(0).set_names({"inputs_embeds"});
    auto result = std::make_shared<ov::op::v0::Result>(embeds);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// The sequence axis alone isn't enough: infer_chunked_prefill()'s VLM `inputs_embeds` byte-count
// scaling multiplies by the *top-level* port's embedding width (see llm_infer_request.cpp). A
// prefill child whose embedding width disagrees with the top-level model's must be rejected even
// though its sequence axis alone satisfies the declared limit.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedInputsEmbedsEmbSizeDisagreeingWithTopLevel) {
    VariantSwitchFactory factory;
    auto props = base_props();
    merge_props(props, {{"NPUW_LLM_PREFILL_HINT", "STATIC"}});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(ov::test::npuw::build_token_type_ids_test_model(),
                                                                 m_plugin,
                                                                 props,
                                                                 factory.make_factory());
    ASSERT_NE(compiled, nullptr);

    // Top-level `inputs_embeds` has EMB_SIZE=64 (see make_test_model_config()); this restored
    // prefill child disagrees (EMB_SIZE=1) while its sequence axis (2048) still fits.
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2048, 1});
    embeds->output(0).set_names({"inputs_embeds"});
    auto result = std::make_shared<ov::op::v0::Result>(embeds);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Negative control for the above: a port whose embedding width actually matches the top-level
// model's must be accepted.
TEST_F(LLMInferRequestVariantSwitchTest, AcceptsImportedInputsEmbedsWhenEmbSizeMatchesTopLevel) {
    VariantSwitchFactory factory;
    auto props = base_props();
    merge_props(props, {{"NPUW_LLM_PREFILL_HINT", "STATIC"}});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(ov::test::npuw::build_token_type_ids_test_model(),
                                                                 m_plugin,
                                                                 props,
                                                                 factory.make_factory());
    ASSERT_NE(compiled, nullptr);

    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2048, 64});
    embeds->output(0).set_names({"inputs_embeds"});
    auto result = std::make_shared<ov::op::v0::Result>(embeds);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));
}

// pad_position_ids() indexes the destination using the *source* (top-level) port's rank, so a
// rank-1 position_ids child - whose own sequence axis still satisfies the declared limit -
// silently reads/writes out of bounds against an actually rank-2 (or rank-3, VLM) top-level port.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedPositionIdsRankDisagreeingWithTopLevel) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "STATIC"}}, factory);
    ASSERT_NE(compiled, nullptr);

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2048});
    position_ids->output(0).set_names({"position_ids"});
    auto result = std::make_shared<ov::op::v0::Result>(position_ids);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{position_ids});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Same trust boundary as the rank/embedding-width checks above, but for element type: a child
// port restored with a different element type than the top-level model's disagrees with what the
// runtime copies (which size their byte counts off the top-level tensor's element type) expect.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedAttentionMaskElementTypeDisagreeingWithTopLevel) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "STATIC"}}, factory);
    ASSERT_NE(compiled, nullptr);

    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2048});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// A blob can set m_use_chunk_prefill=true with m_prefill_chunk_size=0: infer_chunked_prefill()'s
// `current_prompts_len = min(remaining_prompts, chunk_prompt_len)` is then always 0, so
// `remaining_prompts` never reaches 0 and the chunk loop never terminates. Reject it at import.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsZeroPrefillChunkSizeWhenChunkedPrefillEnabled) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));

    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, 0u);
    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// AddPositionIdsParam() synthesizes a `position_ids` child port even when the top-level model has
// none of its own (the LFM-2 case); LLMInferRequest::infer() then synthesizes its own rank-2 i64
// source tensor at runtime. A rank-1 child must still be rejected even though no top-level port
// exists to cross-check it against, since pad_position_ids() would index it using that
// synthesized source's rank.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsSynthesizedPositionIdsRankMismatchWithNoTopLevelPort) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    input_ids->output(0).set_names({"input_ids"});
    auto top_level_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_ids)},
                                    ov::ParameterVector{input_ids});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(top_level_model, m_plugin, /*serialized=*/true);

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2048});
    position_ids->output(0).set_names({"position_ids"});
    auto tiny_prefill_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(position_ids)},
                                    ov::ParameterVector{position_ids});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Same no-top-level-port trust boundary, but for element type: the synthesized source is always
// i64.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsSynthesizedPositionIdsElementTypeMismatchWithNoTopLevelPort) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    input_ids->output(0).set_names({"input_ids"});
    auto top_level_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_ids)},
                                    ov::ParameterVector{input_ids});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(top_level_model, m_plugin, /*serialized=*/true);

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2048});
    position_ids->output(0).set_names({"position_ids"});
    auto tiny_prefill_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(position_ids)},
                                    ov::ParameterVector{position_ids});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Negative control for the two tests above: a child honoring the synthesized-position contract
// (i64, rank 2, non-zero batch) must be accepted even with no top-level port to compare against.
TEST_F(LLMInferRequestVariantSwitchTest, AcceptsSynthesizedPositionIdsMatchingContractWithNoTopLevelPort) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    input_ids->output(0).set_names({"input_ids"});
    auto top_level_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_ids)},
                                    ov::ParameterVector{input_ids});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(top_level_model, m_plugin, /*serialized=*/true);

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2048});
    position_ids->output(0).set_names({"position_ids"});
    auto tiny_prefill_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(position_ids)},
                                    ov::ParameterVector{position_ids});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));
}

// Matching the top-level dimensions does not prove the destination tensor has storage: a [0,
// chunk_size] child agrees with the (dynamic) top-level batch dimension and still satisfies the
// sequence-extent check, yet get_size() == 0 for the whole tensor - infer_chunked_prefill()'s
// right-aligned staging copy then underflows subtracting chunk_prompt_len from it.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedAttentionMaskWithZeroBatchDimension) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{0, 512});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// ReshapeToStatic sizes attention_mask to max_prompt_size - the full KV window - never to the
// chunk size (see reshape_to_static.cpp), since infer_chunked_prefill() copies each completed
// chunk's mask forward to an absolute offset of num_stored_tokens, which grows up to
// max_prompt_size across chunks. A port sized to only one chunk must still be rejected, even
// though it would satisfy a (wrong) chunk-size-only check.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedChunkedPrefillAttentionMaskSizedToChunkInsteadOfMaxPromptSize) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 512});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// ReshapeToStatic sizes each generate variant's attention_mask to that variant's own KV window
// (m_kvcache_sizes[i]), not to max_generation_token_len - infer_generate() copies the full
// stored-token history into it starting at offset 0, so a port sized to only the (generally much
// smaller) per-step generation limit must still be rejected.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedGenerateAttentionMaskSizedToMaxGenerationLenInsteadOfKvSize) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_GE(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 1u);

    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 8});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_generate_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_generate_variant(
        compiled,
        0u,
        std::make_shared<FakeSubCompiledModel>(tiny_generate_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// infer_chunked_prefill() also computes attn_mask_in_tensor->get_size() - m_prefill_chunk_size
// every chunk iteration (llm_infer_request.cpp). Both scalars are independently deserialized, so
// a blob can declare a chunk size larger than max_prompt_size (the real compile path's "chunk
// covers the whole prompt" guard only runs for a fresh compile, never for an imported blob) while
// still providing a mask sized to just max_prompt_size - that alone must still be rejected.
TEST_F(LLMInferRequestVariantSwitchTest,
       RejectsImportedChunkedPrefillAttentionMaskSizedToMaxPromptSizeSmallerThanChunk) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    const uint32_t max_prompt_size = LLMVariantSwitchTestAccess::max_prompt_size(compiled);
    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, max_prompt_size + 1u);

    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, max_prompt_size});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Disabling chunked prefill (prefill_hint=STATIC) leaves m_prefill_chunk_size at its unrelated,
// compile-time config value - the chunk-size floor must not apply when chunking isn't actually
// selected at runtime, or a legitimate whole-prefill blob with a stale/larger chunk size config
// would be wrongly rejected even though ReshapeToStatic only ever sized attention_mask to
// max_prompt_size and infer_whole_prefill() never reads m_prefill_chunk_size at all.
TEST_F(LLMInferRequestVariantSwitchTest, AcceptsImportedWholePrefillAttentionMaskWhenChunkSizeExceedsMaxPromptSize) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "STATIC"}}, factory);
    ASSERT_NE(compiled, nullptr);

    const uint32_t max_prompt_size = LLMVariantSwitchTestAccess::max_prompt_size(compiled);
    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, max_prompt_size + 1u);

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));
}

// infer_generate() also computes kv_attn_mask->get_size() - kvcache_desc.max_generation_token_len
// (llm_infer_request.cpp). Both scalars are independently deserialized, so a blob can declare a
// KV window (m_kvcache_sizes[i]) smaller than the generation limit while still providing a mask
// sized to just that (smaller) window - that alone must still be rejected. Compiling with the
// generation limit already set keeps every OTHER (untouched) variant's ports correctly sized, so
// only mutating m_kvcache_sizes[0] (not max_generation_token_len globally) isolates the gap.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedGenerateAttentionMaskSizedToKvSizeSmallerThanGenerationLen) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "16"}}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_GE(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 1u);

    auto sizes = LLMVariantSwitchTestAccess::kvcache_sizes(compiled);
    ASSERT_GT(sizes.at(0), 8u);
    sizes.at(0) = 8u;
    LLMVariantSwitchTestAccess::set_kvcache_sizes(compiled, sizes);

    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 8});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_generate_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_generate_variant(
        compiled,
        0u,
        std::make_shared<FakeSubCompiledModel>(tiny_generate_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// PrefixCachingHelper::find_cached_blocks() computes max_prompt_size - m_prefill_chunk_size
// unconditionally whenever prefix caching is enabled (llm_prefix_caching.cpp). Both scalars are
// independently deserialized, and nothing but the compile path's own "chunk covers the whole
// prompt" fallback normally keeps chunk size <= max prompt size. Oversizing every restored port
// can't mask this - the scalar relationship must be validated directly.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedPrefillChunkSizeLargerThanMaxPromptSize) {
    VariantSwitchFactory factory;
    auto compiled =
        create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"}}, factory);
    ASSERT_NE(compiled, nullptr);

    const uint32_t max_prompt_size = LLMVariantSwitchTestAccess::max_prompt_size(compiled);
    const uint32_t oversized_chunk = max_prompt_size + 1u;
    LLMVariantSwitchTestAccess::set_prefill_chunk_size(compiled, oversized_chunk);

    // Sized to satisfy the attention_mask extent check (required == oversized_chunk here) on its
    // own, so only the direct chunk-size-vs-max-prompt-size scalar check can still catch this.
    auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, oversized_chunk});
    attn_mask->output(0).set_names({"attention_mask"});
    auto result = std::make_shared<ov::op::v0::Result>(attn_mask);
    auto tiny_prefill_model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{attn_mask});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// infer_chunked_prefill()/infer_whole_prefill() multiply or size their copy byte counts off
// inputs_embeds' CONCRETE runtime tensor width (input_ids->get_shape().back()), not off anything
// declared at import time. A dynamic top-level embedding-width dimension can't be cross-checked
// against the restored child port, since a future inference call could supply any width - reject
// the blob outright rather than let an unboundable future call underflow the staging copy.
TEST_F(LLMInferRequestVariantSwitchTest, RejectsImportedTopLevelInputsEmbedsWithDynamicEmbSize) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    input_ids->output(0).set_names({"input_ids"});
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1});
    embeds->output(0).set_names({"inputs_embeds"});
    auto top_level_model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_ids), std::make_shared<ov::op::v0::Result>(embeds)},
        ov::ParameterVector{input_ids, embeds});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(top_level_model, m_plugin, /*serialized=*/true);

    auto child_embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 4096});
    child_embeds->output(0).set_names({"inputs_embeds"});
    auto tiny_prefill_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(child_embeds)},
                                    ov::ParameterVector{child_embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled), ov::Exception);
}

// Negative control for the above: a dynamic top-level BATCH dimension must still be tolerated
// (the pre-existing, intentional policy) - only the embedding-width axis is rejected when
// dynamic, proving the new check is scoped to that one axis.
TEST_F(LLMInferRequestVariantSwitchTest, AcceptsImportedTopLevelInputsEmbedsWithDynamicBatchButStaticEmbSize) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    input_ids->output(0).set_names({"input_ids"});
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 4096});
    embeds->output(0).set_names({"inputs_embeds"});
    auto top_level_model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(input_ids), std::make_shared<ov::op::v0::Result>(embeds)},
        ov::ParameterVector{input_ids, embeds});
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(top_level_model, m_plugin, /*serialized=*/true);

    auto child_embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 4096});
    child_embeds->output(0).set_names({"inputs_embeds"});
    auto tiny_prefill_model =
        std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(child_embeds)},
                                    ov::ParameterVector{child_embeds});
    LLMVariantSwitchTestAccess::set_prefill_compiled(
        compiled,
        std::make_shared<FakeSubCompiledModel>(tiny_prefill_model, m_plugin, ov::AnyMap{}));

    EXPECT_NO_THROW(LLMVariantSwitchTestAccess::validate_imported_sequence_extents(compiled));
}

}  // namespace
