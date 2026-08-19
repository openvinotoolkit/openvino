// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "infer_request_utils.hpp"
#include "llm_block_kvcache_strategy.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
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

    static bool is_block_kv_cache(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_is_block_kv_cache;
    }

    static const auto& prefill_kv_seq_dims(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_kvcache_desc.kv_seq_dims;
    }

    static const auto& generate_kv_seq_dims(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_kvcache_desc.kv_seq_dims_gen;
    }

    static void set_num_stored_tokens(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                      uint32_t num_tokens) {
        compiled->m_kvcache_desc.num_stored_tokens = num_tokens;
    }

    static uint32_t kv_dim_for_name(const ov::npuw::LLMInferRequest& req, const std::string& name) {
        const auto& desc = req.m_npuw_llm_compiled_model->m_kvcache_desc;
        return desc.kv_seq_dims_gen.at(name);
    }

    static void set_kv_seq_dims(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                const std::string& name,
                                uint32_t prefill_dim,
                                uint32_t generate_dim) {
        compiled->m_kvcache_desc.kv_seq_dims.at(name) = prefill_dim;
        compiled->m_kvcache_desc.kv_seq_dims_gen.at(name) = generate_dim;
    }

    static const auto& prefill_request(const ov::npuw::LLMInferRequest& req) {
        return req.m_prefill_request;
    }

    static const auto& prefill_out_ports(const ov::npuw::LLMInferRequest& req) {
        return req.m_prefill_out_ports;
    }

    static void copy_kvcache(ov::npuw::LLMInferRequest& req) {
        req.copy_kvcache();
    }

    static void bind_past_kv(ov::npuw::LLMInferRequest& req) {
        req.bind_past_kv();
    }

    static bool past_kv_bound(const ov::npuw::LLMInferRequest& req) {
        return req.m_past_kv_bound;
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
};

TEST_F(LLMInferRequestVariantSwitchTest, NonChunkPrefillPreservesKvSequenceDimensions) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);

    const auto& prefill_dims = LLMVariantSwitchTestAccess::prefill_kv_seq_dims(compiled);
    const auto& generate_dims = LLMVariantSwitchTestAccess::generate_kv_seq_dims(compiled);
    EXPECT_FALSE(prefill_dims.empty());
    EXPECT_EQ(prefill_dims, generate_dims);
}

TEST(LLMKvCacheCopyTest, TransposesDataWhenSequenceDimensionDiffers) {
    const ov::Shape src_shape{1, 2, 3, 4};
    const ov::Shape dst_shape{1, 2, 4, 3};
    ov::Tensor src(ov::element::i32, src_shape);
    ov::Tensor dst(ov::element::i32, dst_shape);

    auto* src_data = src.data<int32_t>();
    for (size_t idx = 0; idx < src.get_size(); ++idx) {
        src_data[idx] = static_cast<int32_t>(idx);
    }

    auto src_impl = ov::get_tensor_impl(src);
    auto dst_impl = ov::get_tensor_impl(dst);
    ov::npuw::util::copy_tensor_by_dim(src_impl, dst_impl, 2u, 3u);

    const auto* dst_data = dst.data<const int32_t>();
    for (size_t head = 0; head < src_shape[1]; ++head) {
        for (size_t seq = 0; seq < src_shape[2]; ++seq) {
            for (size_t dim = 0; dim < src_shape[3]; ++dim) {
                const size_t src_idx = (head * src_shape[2] + seq) * src_shape[3] + dim;
                const size_t dst_idx = (head * dst_shape[2] + dim) * dst_shape[3] + seq;
                EXPECT_EQ(dst_data[dst_idx], src_data[src_idx]);
            }
        }
    }
}

TEST_F(LLMInferRequestVariantSwitchTest, CopyKvCacheTransposesAndAlignsShortPrefillOutputToStoredTail) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ov::npuw::LLMInferRequest req(compiled);

    const auto& past_names = LLMVariantSwitchTestAccess::kvcache_past_names(req);
    const auto value_it = std::find_if(past_names.begin(), past_names.end(), ov::npuw::util::isPastValueParam);
    ASSERT_NE(value_it, past_names.end());
    const auto& input_name = *value_it;
    const auto output_name =
        std::regex_replace(input_name, std::regex(ov::npuw::LLMInferRequest::layer_names::past_key_values), "present");

    auto dst = LLMVariantSwitchTestAccess::kvcache_request(req)->get_tensor(
        LLMVariantSwitchTestAccess::kvcache_in_ports(req).at(input_name));
    const uint32_t generate_dim = LLMVariantSwitchTestAccess::generate_kv_seq_dims(compiled).at(input_name);
    ASSERT_TRUE(generate_dim == 2u || generate_dim == 3u);
    const uint32_t prefill_dim = generate_dim == 2u ? 3u : 2u;
    LLMVariantSwitchTestAccess::set_kv_seq_dims(compiled, input_name, prefill_dim, generate_dim);

    constexpr uint32_t stored_tokens = 5u;
    constexpr uint32_t produced_tokens = 3u;
    ASSERT_GE(dst->get_shape()[generate_dim], stored_tokens);
    auto src_shape = dst->get_shape();
    std::swap(src_shape[2], src_shape[3]);
    src_shape[prefill_dim] = produced_tokens;
    auto src = ov::get_tensor_impl(ov::Tensor(dst->get_element_type(), src_shape));
    fill_tensor_pattern(src, 31u);
    LLMVariantSwitchTestAccess::prefill_request(req)->set_tensor(
        LLMVariantSwitchTestAccess::prefill_out_ports(req).at(output_name),
        src);
    ov::npuw::util::fill_tensor_bytes(dst, 0u);
    LLMVariantSwitchTestAccess::set_num_stored_tokens(compiled, stored_tokens);

    auto expected_shape = dst->get_shape();
    expected_shape[generate_dim] = produced_tokens;
    auto expected = ov::get_tensor_impl(ov::Tensor(dst->get_element_type(), expected_shape));
    ov::npuw::util::copy_tensor_by_dim(src, expected, prefill_dim, generate_dim);

    LLMVariantSwitchTestAccess::copy_kvcache(req);

    auto prefix = ov::npuw::util::make_tensor_slice(dst, generate_dim, 0u, stored_tokens - produced_tokens);
    const auto prefix_bytes = materialize_bytes(prefix);
    EXPECT_TRUE(std::all_of(prefix_bytes.begin(), prefix_bytes.end(), [](uint8_t value) {
        return value == 0u;
    }));
    auto copied_tail =
        ov::npuw::util::make_tensor_slice(dst, generate_dim, stored_tokens - produced_tokens, stored_tokens);
    EXPECT_EQ(materialize_bytes(copied_tail), materialize_bytes(expected));
}

TEST_F(LLMInferRequestVariantSwitchTest, DifferentPerParameterAxesDisablePrefillGenerateBufferSharing) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ov::npuw::LLMInferRequest req(compiled);
    const auto& name = LLMVariantSwitchTestAccess::kvcache_past_names(req).front();
    const auto generate_dim = LLMVariantSwitchTestAccess::generate_kv_seq_dims(compiled).at(name);
    LLMVariantSwitchTestAccess::set_kv_seq_dims(compiled, name, generate_dim == 2u ? 3u : 2u, generate_dim);

    LLMVariantSwitchTestAccess::bind_past_kv(req);

    EXPECT_FALSE(LLMVariantSwitchTestAccess::past_kv_bound(req));
}

TEST_F(LLMInferRequestVariantSwitchTest, BlockKvFallsBackWhenPrefillAndGenerateAxesDiffer) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                           {"NPUW_LLM_PREFILL_CHUNK_SIZE", "512"},
                                           {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
                                           {"NPUW_LLM_GENERATE_ATTENTION_HINT", "DYNAMIC"},
                                           {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"}},
                                          factory);

    ASSERT_NE(compiled, nullptr);
    ASSERT_NE(LLMVariantSwitchTestAccess::prefill_kv_seq_dims(compiled),
              LLMVariantSwitchTestAccess::generate_kv_seq_dims(compiled));
    EXPECT_FALSE(LLMVariantSwitchTestAccess::is_block_kv_cache(compiled));
}

TEST_F(LLMInferRequestVariantSwitchTest, ContinuousKvSwitchMigratesStoredTokensToLargerVariant) {
    VariantSwitchFactory factory;
    auto compiled = create_compiled_model({}, factory);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(LLMVariantSwitchTestAccess::generate_variant_count(compiled), 2u);

    ov::npuw::LLMInferRequest req(compiled);
    LLMVariantSwitchTestAccess::select_smallest_generate_variant(req);

    const uint32_t stored_tokens = LLMVariantSwitchTestAccess::current_variant_capacity(req);
    LLMVariantSwitchTestAccess::set_num_stored_tokens(compiled, stored_tokens);

    std::unordered_map<std::string, std::vector<uint8_t>> expected_kv_bytes;
    uint8_t seed = 17u;
    for (const auto& name : LLMVariantSwitchTestAccess::kvcache_past_names(req)) {
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

}  // namespace
