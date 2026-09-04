// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Request-level regression tests for continuous prefill on the contiguous KV
// strategy. The tests drive the public LLMInferRequest surface: infer() for
// routing and the npuw_stored_tokens_state variable state for the propose/grant
// negotiation. Submodels are compiled through a factory that returns fake
// sub-requests whose infer() only zeroes outputs, so every byte that the tests
// observe in a KV tensor was moved there by the plugin's own KV plumbing.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "llm_block_kvcache_strategy.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_kvcache_strategy.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct LLMContinuedPrefillTestAccess {
    using Request = ov::npuw::LLMInferRequest;

    static const ov::npuw::LLMCompiledModel::KVCacheDesc& desc(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc;
    }

    static const std::vector<std::string>& past_names(Request& req) {
        return req.m_kvcache_past_names;
    }

    static ov::SoPtr<ov::ITensor> generate_past(Request& req, const std::string& name) {
        return req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
    }

    static ov::SoPtr<ov::ITensor> prefill_past(Request& req, const std::string& name) {
        return req.m_prefill_request->get_tensor(req.m_prefill_in_ports.at(name));
    }

    static ov::SoPtr<ov::ITensor> prefill_attention_mask(Request& req) {
        return req.m_prefill_request->get_tensor(req.m_prefill_in_ports.at(Request::layer_names::attention_mask));
    }

    static bool generate_initialized(Request& req) {
        return req.m_generate_initialized;
    }

    static std::unique_ptr<ov::npuw::LLMKVCacheStrategy> take_strategy(Request& req) {
        return std::move(req.m_kvcache_strategy);
    }

    static void set_strategy(Request& req, std::unique_ptr<ov::npuw::LLMKVCacheStrategy> strategy) {
        req.m_kvcache_strategy = std::move(strategy);
    }

    static bool is_block_kv_cache(Request& req) {
        return req.m_npuw_llm_compiled_model->m_is_block_kv_cache;
    }

    static ov::npuw::LLMBlockKVCacheStrategy* block_strategy(Request& req) {
        return dynamic_cast<ov::npuw::LLMBlockKVCacheStrategy*>(req.m_kvcache_strategy.get());
    }

    static const std::unordered_map<uint32_t, ov::npuw::LayerBlockManagers>& block_managers(
        const ov::npuw::LLMBlockKVCacheStrategy& strategy) {
        return strategy.m_kv_cache_block_managers;
    }

    static uint32_t block_size(const ov::npuw::LLMBlockKVCacheStrategy& strategy) {
        return strategy.m_block_size;
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::LLMContinuedPrefillTestAccess;
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
                                                        intel_npu::make_executor("continued_prefill_task", 1),
                                                        intel_npu::make_executor("continued_prefill_callback", 1));
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

class ContinuedPrefillFactory {
public:
    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [](const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
        };
    }
};

// Delegates every strategy call to the real contiguous strategy but fails the
// continuation, modelling a failure inside the strategy.
class ThrowingContinuationStrategy final : public ov::npuw::LLMKVCacheStrategy {
public:
    ThrowingContinuationStrategy(ov::npuw::LLMInferRequest& req, std::unique_ptr<ov::npuw::LLMKVCacheStrategy> inner)
        : LLMKVCacheStrategy(req),
          m_inner(std::move(inner)) {}

    void on_initialize() override {
        m_inner->on_initialize();
    }
    void on_reset(uint32_t next_prompt_length) override {
        m_inner->on_reset(next_prompt_length);
    }
    void on_prefill_chunk_begin(uint32_t current_prompts_len) override {
        m_inner->on_prefill_chunk_begin(current_prompts_len);
    }
    void on_prefill_chunk_done(uint32_t current_prompts_len, bool is_last) override {
        m_inner->on_prefill_chunk_done(current_prompts_len, is_last);
    }
    void on_generate_kv_init() override {
        m_inner->on_generate_kv_init();
    }
    void on_generate_variant_switch(const std::shared_ptr<ov::IAsyncInferRequest>& old_req,
                                    const PortsMap& old_in_ports,
                                    const std::shared_ptr<ov::IAsyncInferRequest>& new_req,
                                    const PortsMap& new_in_ports) override {
        m_inner->on_generate_variant_switch(old_req, old_in_ports, new_req, new_in_ports);
    }
    void on_generate_step_done(uint32_t input_tokens_len) override {
        m_inner->on_generate_step_done(input_tokens_len);
    }
    void continue_prefill(uint32_t, uint32_t) override {
        OPENVINO_THROW("Injected continued-prefill failure.");
    }

private:
    std::unique_ptr<ov::npuw::LLMKVCacheStrategy> m_inner;
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

ov::Tensor make_i64(std::initializer_list<size_t> shape, int64_t fill_value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), fill_value);
    return tensor;
}

ov::Tensor make_i64_iota(std::initializer_list<size_t> shape, int64_t start) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::iota(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size(), start);
    return tensor;
}

class LLMContinuedPrefillTest : public ::testing::Test {
protected:
    void SetUp() override {
        init({});
    }

    void init(const ov::AnyMap& extra_props) {
        m_plugin = std::make_shared<NullPlugin>();
        ContinuedPrefillFactory factory;
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", "256"},
                         {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                         {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                         {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"},
                         {"NPUW_LLM_ENABLE_CONTINUOUS_PREFILL", "YES"}};
        for (const auto& [key, value] : extra_props) {
            props[key] = value;
        }
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(build_llm_test_model(),
                                                                  m_plugin,
                                                                  props,
                                                                  factory.make_factory());
        ASSERT_NE(m_compiled, nullptr);
        ASSERT_TRUE(m_compiled->get_property("NPUW_LLM_CONTINUOUS_PREFILL_SUPPORTED").as<bool>());

        m_request = std::make_unique<ov::npuw::LLMInferRequest>(m_compiled);

        // The byte-level comparisons below materialize a source and a destination
        // slice and compare them directly, which is valid only while the prefill
        // and generate KV layouts agree on the V orientation. This holds for the
        // test model; a mismatch would also disable KV buffer sharing entirely.
        const auto& desc = LLMContinuedPrefillTestAccess::desc(*m_request);
        ASSERT_EQ(desc.v_tensors_transposed_pre, desc.v_tensors_transposed_gen);
    }

    ov::npuw::LLMInferRequest& request() {
        return *m_request;
    }

    ov::SoPtr<ov::IVariableState> stored_tokens_state() {
        for (const auto& state : m_request->query_state()) {
            if (state->get_name() == "npuw_stored_tokens_state") {
                return state;
            }
        }
        ADD_FAILURE() << "npuw_stored_tokens_state is not exposed by the request";
        return {};
    }

    int64_t stored_tokens() {
        return stored_tokens_state()->get_state()->data<int64_t>()[0];
    }

    void propose(int64_t k_common) {
        auto proposal = ov::Tensor(ov::element::i64, ov::Shape{1});
        proposal.data<int64_t>()[0] = k_common;
        stored_tokens_state()->set_state(ov::get_tensor_impl(proposal));
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

    void run_full_prefill(size_t prompt_len) {
        set_inputs(make_i64({1, prompt_len}, 1), make_i64({1, prompt_len}, 1), make_i64_iota({1, prompt_len}, 0));
        m_request->infer();
    }

    void run_generate_step(size_t live_tokens) {
        set_inputs(make_i64({1, 1}, 1),
                   make_i64({1, live_tokens + 1}, 1),
                   make_i64({1, 1}, static_cast<int64_t>(live_tokens)));
        m_request->infer();
    }

    void run_delta_prefill(size_t keep, size_t delta_len) {
        set_inputs(make_i64({1, delta_len}, 1),
                   make_i64({1, keep + delta_len}, 1),
                   make_i64_iota({1, delta_len}, static_cast<int64_t>(keep)));
        m_request->infer();
    }

    uint32_t generate_seq_dim(const std::string& past_name) {
        const auto& desc = LLMContinuedPrefillTestAccess::desc(*m_request);
        const bool is_value = ov::npuw::util::isPastValueParam(past_name);
        return (is_value && desc.v_tensors_transposed_gen) ? 3u : desc.dim;
    }

    uint32_t prefill_seq_dim(const std::string& past_name) {
        const auto& desc = LLMContinuedPrefillTestAccess::desc(*m_request);
        const bool is_value = ov::npuw::util::isPastValueParam(past_name);
        return (is_value && desc.v_tensors_transposed_pre) ? 3u : desc.dim;
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::unique_ptr<ov::npuw::LLMInferRequest> m_request;
};

// A granted keep routes the next infer through the delta-only prefill: the
// preserved prefix is repacked from the generate past KV into the prefill
// layout (through the alias-safe staging path, as both views share one backing
// buffer), the restored attention mask exposes the history, and a multi-chunk
// delta lands after the preserved region without disturbing it.
TEST_F(LLMContinuedPrefillTest, GrantedContinuationRepacksGenerateKvAndRunsDeltaOnly) {
    auto& req = request();
    run_full_prefill(72);
    EXPECT_EQ(stored_tokens(), 72);
    run_generate_step(72);
    run_generate_step(73);
    EXPECT_EQ(stored_tokens(), 74);
    ASSERT_TRUE(LLMContinuedPrefillTestAccess::generate_initialized(req));

    // The live prefix sits in the generate past KV. Stamp the preserved region
    // with a per-tensor pattern and remember its logical content.
    constexpr uint32_t kKeep = 64u;
    std::unordered_map<std::string, std::vector<uint8_t>> expected_kv_bytes;
    uint8_t seed = 23u;
    for (const auto& name : LLMContinuedPrefillTestAccess::past_names(req)) {
        auto src = LLMContinuedPrefillTestAccess::generate_past(req, name);
        auto src_slice = ov::npuw::util::make_tensor_slice(src, generate_seq_dim(name), 0u, kKeep);
        fill_tensor_pattern(src_slice, seed);
        expected_kv_bytes.emplace(name, materialize_bytes(src_slice));
        seed = static_cast<uint8_t>(seed + 41u);
    }

    // Propose the whole live history. The watermark is the prefill-produced 72,
    // so the grant rounds down to the chunk-aligned 64.
    propose(74);
    EXPECT_EQ(stored_tokens(), 64);

    // Second turn: 40 delta tokens (two chunks) on top of the granted keep.
    run_delta_prefill(kKeep, 40);
    EXPECT_EQ(stored_tokens(), 104);
    EXPECT_FALSE(LLMContinuedPrefillTestAccess::generate_initialized(req));

    // The preserved prefix must now sit in the prefill past KV, repacked into
    // the prefill layout, and must have survived the delta chunk loop.
    for (const auto& name : LLMContinuedPrefillTestAccess::past_names(req)) {
        auto dst = LLMContinuedPrefillTestAccess::prefill_past(req, name);
        auto dst_slice = ov::npuw::util::make_tensor_slice(dst, prefill_seq_dim(name), 0u, kKeep);
        EXPECT_EQ(materialize_bytes(dst_slice), expected_kv_bytes.at(name)) << name;
    }

    // Without the restored history mask the delta could not attend to the
    // preserved conversation at all.
    auto mask = LLMContinuedPrefillTestAccess::prefill_attention_mask(req);
    const auto* mask_data = mask->data<int64_t>();
    EXPECT_TRUE(std::all_of(mask_data, mask_data + kKeep, [](int64_t v) {
        return v == 1;
    }));

    // The continuation hands over to an ordinary generate step.
    run_generate_step(104);
    EXPECT_EQ(stored_tokens(), 105);
}

// A conversation that ends on its prompt logits never runs a generate step, so
// the live prefix stays split between the prefill past inputs and the present
// outputs, where no continuation source exists. The coordinator grants zero
// and the caller re-sends the full history as an ordinary reset prefill.
TEST_F(LLMContinuedPrefillTest, PromptOnlyTurnGrantsZeroAndFullHistoryRecovers) {
    auto& req = request();
    run_full_prefill(96);
    EXPECT_EQ(stored_tokens(), 96);
    ASSERT_FALSE(LLMContinuedPrefillTestAccess::generate_initialized(req));

    propose(96);
    EXPECT_EQ(stored_tokens(), 0);

    // The full 112-token history at position zero satisfies the armed reset.
    run_full_prefill(112);
    EXPECT_EQ(stored_tokens(), 112);

    // Once a generate step consolidates the prefix into the generate KV, the
    // next turn negotiates a real keep again.
    run_generate_step(112);
    EXPECT_EQ(stored_tokens(), 113);
    propose(113);
    EXPECT_EQ(stored_tokens(), 96);
}

// A one-token delta must route through the granted continuation. The legacy
// length heuristic would classify it as a generate step, so the explicit
// command has to stay ahead of that heuristic.
TEST_F(LLMContinuedPrefillTest, SingleTokenDeltaRoutesToContinuedPrefill) {
    auto& req = request();
    run_full_prefill(64);
    run_generate_step(64);
    EXPECT_EQ(stored_tokens(), 65);

    propose(65);
    EXPECT_EQ(stored_tokens(), 64);

    run_delta_prefill(64, 1);
    EXPECT_EQ(stored_tokens(), 65);
    // The delta ran as a prefill, so the generate phase re-initializes next.
    EXPECT_FALSE(LLMContinuedPrefillTestAccess::generate_initialized(req));
}

// A preflight rejection mutates nothing and leaves the command pending, so a
// corrected delta still completes the continuation.
TEST_F(LLMContinuedPrefillTest, PreflightRejectionKeepsCommandRetryable) {
    auto& req = request();
    run_full_prefill(72);
    run_generate_step(72);
    EXPECT_EQ(stored_tokens(), 73);

    propose(73);
    EXPECT_EQ(stored_tokens(), 64);

    // Position ids starting at the conversation base instead of the granted
    // keep must be rejected by the sequence validation.
    set_inputs(make_i64({1, 40}, 1), make_i64({1, 104}, 1), make_i64_iota({1, 40}, 0));
    EXPECT_THROW(req.infer(), ov::Exception);

    // The grant stays armed and the corrected delta succeeds.
    EXPECT_EQ(stored_tokens(), 64);
    run_delta_prefill(64, 40);
    EXPECT_EQ(stored_tokens(), 104);
}

// A failing continuation leaves the command pending and the cache unspecified.
// The caller's recovery is reset() plus the full history, after which the
// request works again.
TEST_F(LLMContinuedPrefillTest, InjectedApplyFailureRecoversAfterReset) {
    auto& req = request();
    run_full_prefill(72);
    run_generate_step(72);
    EXPECT_EQ(stored_tokens(), 73);

    auto inner = LLMContinuedPrefillTestAccess::take_strategy(req);
    ASSERT_NE(inner, nullptr);
    LLMContinuedPrefillTestAccess::set_strategy(req,
                                                std::make_unique<ThrowingContinuationStrategy>(req, std::move(inner)));

    propose(73);
    set_inputs(make_i64({1, 40}, 1), make_i64({1, 104}, 1), make_i64_iota({1, 40}, 64));
    EXPECT_THROW(req.infer(), ov::Exception);

    // The command is still armed; the caller gives up on it with reset() and
    // re-establishes the conversation with a full prefill from position zero.
    EXPECT_EQ(stored_tokens(), 64);
    stored_tokens_state()->reset();
    run_full_prefill(80);
    EXPECT_EQ(stored_tokens(), 80);
    run_generate_step(80);
    EXPECT_EQ(stored_tokens(), 81);
}

// Block-mode variant of the fixture: the same synthetic model compiled with the
// pyramid attention hint so SplitKVCacheIntoBlocks applies, giving the block
// strategy a real block pool under the fake factory. Every sub-request comes
// without a base request, so the whole flow also proves the strategy tolerates
// absent base requests during reset and continuation.
class LLMBlockContinuedPrefillTest : public LLMContinuedPrefillTest {
protected:
    void SetUp() override {
        init({{"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
              {"NPUW_LLM_GENERATE_ATTENTION_HINT", "PYRAMID"},
              {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"}});
        ASSERT_TRUE(LLMContinuedPrefillTestAccess::is_block_kv_cache(request()));
    }

    template <typename F>
    void for_each_manager(const ov::npuw::LLMBlockKVCacheStrategy& strategy, F&& fn) {
        for (const auto& [layer_idx, layer_managers] : LLMContinuedPrefillTestAccess::block_managers(strategy)) {
            for (auto* manager : {layer_managers.key_manager.get(), layer_managers.value_manager.get()}) {
                if (manager) {
                    fn(manager);
                }
            }
        }
    }
};

// A granted continuation on the block pool truncates the metadata to the
// retained prefix without moving a byte: the retained block tensors are the KV
// storage itself, so their contents surviving the continuation and the delta
// chunk loop proves the truncate-only design. The delta then appends fresh
// blocks behind the retained ones and an ordinary generate step follows.
TEST_F(LLMBlockContinuedPrefillTest, GrantedContinuationTruncatesBlockPoolAndRunsDeltaOnly) {
    auto& req = request();
    auto* strategy = LLMContinuedPrefillTestAccess::block_strategy(req);
    ASSERT_NE(strategy, nullptr);

    run_full_prefill(72);
    EXPECT_EQ(stored_tokens(), 72);
    run_generate_step(72);
    run_generate_step(73);
    EXPECT_EQ(stored_tokens(), 74);
    ASSERT_TRUE(LLMContinuedPrefillTestAccess::generate_initialized(req));

    const uint32_t block_size = LLMContinuedPrefillTestAccess::block_size(*strategy);
    ASSERT_EQ(block_size, 32u);
    constexpr uint32_t kKeep = 64u;
    const uint32_t keep_blocks = kKeep / block_size;

    // Stamp the to-be-retained blocks with per-block patterns and remember them.
    ASSERT_FALSE(LLMContinuedPrefillTestAccess::block_managers(*strategy).empty());
    std::unordered_map<const ov::npuw::KVCacheBlockManager*, std::vector<std::vector<uint8_t>>> expected_blocks;
    uint8_t seed = 29u;
    for_each_manager(*strategy, [&](ov::npuw::KVCacheBlockManager* manager) {
        const auto allocated = manager->get_allocated_blocks();
        ASSERT_GE(allocated.size(), keep_blocks);
        std::vector<std::vector<uint8_t>> bytes;
        for (uint32_t i = 0; i < keep_blocks; ++i) {
            ASSERT_EQ(manager->get_block_tokens(allocated[i]), block_size);
            auto tensor = manager->get_block_tensor(allocated[i]);
            fill_tensor_pattern(tensor, seed);
            bytes.push_back(materialize_bytes(tensor));
            seed = static_cast<uint8_t>(seed + 41u);
        }
        expected_blocks.emplace(manager, std::move(bytes));
    });

    // Propose the whole live history; the watermark is the prefill-produced 72,
    // so the grant rounds down to the block-aligned 64.
    propose(74);
    EXPECT_EQ(stored_tokens(), 64);

    // Second turn: 40 delta tokens (two chunks) on top of the granted keep.
    run_delta_prefill(kKeep, 40);
    EXPECT_EQ(stored_tokens(), 104);
    EXPECT_FALSE(LLMContinuedPrefillTestAccess::generate_initialized(req));

    for_each_manager(*strategy, [&](ov::npuw::KVCacheBlockManager* manager) {
        // 64 retained + 40 delta = 104 tokens = three full blocks + one 8-token block.
        const auto allocated = manager->get_allocated_blocks();
        ASSERT_EQ(allocated.size(), 4u);
        uint32_t total_tokens = 0;
        for (const auto block_id : allocated) {
            total_tokens += manager->get_block_tokens(block_id);
        }
        EXPECT_EQ(total_tokens, 104u);
        // The retained prefix must not have been touched by the truncation or
        // the delta chunk loop.
        const auto& bytes = expected_blocks.at(manager);
        for (uint32_t i = 0; i < keep_blocks; ++i) {
            EXPECT_EQ(materialize_bytes(manager->get_block_tensor(allocated[i])), bytes[i]) << "block " << i;
        }
    });

    // The continuation hands over to an ordinary generate step on the block pool.
    run_generate_step(104);
    EXPECT_EQ(stored_tokens(), 105);
}

}  // namespace
