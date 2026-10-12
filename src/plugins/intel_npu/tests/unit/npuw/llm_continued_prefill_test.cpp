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
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/transpose.hpp"
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

// Inputs of the prefill sub-request captured at every infer(), keyed by port name.
using PrefillJournal = std::vector<std::unordered_map<std::string, std::vector<uint8_t>>>;

std::vector<uint8_t> materialize_bytes(const ov::SoPtr<ov::ITensor>& tensor) {
    ov::Tensor copy(tensor->get_element_type(), tensor->get_shape());
    tensor->copy_to(ov::get_tensor_impl(copy)._ptr);
    auto* data = static_cast<uint8_t*>(copy.data());
    return std::vector<uint8_t>(data, data + copy.get_byte_size());
}

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
                         std::shared_ptr<PrefillJournal> journal)
        : ov::npuw::ICompiledModel_v0(model, plugin),
          m_model(model),
          m_journal(std::move(journal)) {}

    // Only the prefill submodel is journaled.
    PrefillJournal* journal() const {
        const auto& name = m_model->get_friendly_name();
        const std::string suffix = "_prefill";
        const bool is_prefill =
            name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        return is_prefill ? m_journal.get() : nullptr;
    }

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
    std::shared_ptr<PrefillJournal> m_journal;
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
    const auto* compiled = static_cast<const FakeSubCompiledModel*>(get_compiled_model().get());
    if (auto* journal = compiled->journal()) {
        auto& entry = journal->emplace_back();
        for (const auto& input : get_compiled_model()->inputs()) {
            entry.emplace(input.get_any_name(), materialize_bytes(ov::ISyncInferRequest::get_tensor(input)));
        }
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

class ContinuedPrefillFactory {
public:
    explicit ContinuedPrefillFactory(std::shared_ptr<PrefillJournal> journal = {}) : m_journal(std::move(journal)) {}

    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [journal = m_journal](const std::shared_ptr<ov::Model>& model,
                                     const std::shared_ptr<const ov::IPlugin>& plugin,
                                     const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<FakeSubCompiledModel>(model, plugin, journal);
        };
    }

private:
    std::shared_ptr<PrefillJournal> m_journal;
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

    void init(const ov::AnyMap& extra_props,
              const std::shared_ptr<ov::Model>& model = build_llm_test_model(),
              std::shared_ptr<PrefillJournal> journal = {}) {
        m_plugin = std::make_shared<NullPlugin>();
        ContinuedPrefillFactory factory(std::move(journal));
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
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, m_plugin, props, factory.make_factory());
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
        const bool is_value = ov::npuw::util::isPastValueParam(past_name) ||
                              ov::npuw::util::isDQScaleOrZPValue(past_name);
        return (is_value && desc.v_tensors_transposed_gen) ? 3u : desc.dim;
    }

    uint32_t prefill_seq_dim(const std::string& past_name) {
        const auto& desc = LLMContinuedPrefillTestAccess::desc(*m_request);
        const bool is_value = ov::npuw::util::isPastValueParam(past_name) ||
                              ov::npuw::util::isDQScaleOrZPValue(past_name);
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

class LLMQuantizedContinuedPrefillTest : public LLMContinuedPrefillTest {
protected:
    void SetUp() override {
        init({{ov::hint::kv_cache_precision.name(), ov::element::i8}});
    }
};

TEST_F(LLMQuantizedContinuedPrefillTest, QuantizedKvCacheSupportsContinuousPrefillFlow) {
    auto& req = request();
    bool saw_quantized_aux_tensor = false;
    for (const auto& name : LLMContinuedPrefillTestAccess::past_names(req)) {
        saw_quantized_aux_tensor = saw_quantized_aux_tensor || ov::npuw::util::isDQScaleOrZPValue(name);
    }
    ASSERT_TRUE(saw_quantized_aux_tensor) << "The quantized fixture must expose scale or zero-point past tensors";

    run_full_prefill(72);
    EXPECT_EQ(stored_tokens(), 72);
    run_generate_step(72);
    EXPECT_EQ(stored_tokens(), 73);

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

    propose(73);
    EXPECT_EQ(stored_tokens(), kKeep);
    run_delta_prefill(kKeep, 40);
    EXPECT_EQ(stored_tokens(), 104);

    for (const auto& name : LLMContinuedPrefillTestAccess::past_names(req)) {
        auto dst = LLMContinuedPrefillTestAccess::prefill_past(req, name);
        auto dst_slice = ov::npuw::util::make_tensor_slice(dst, prefill_seq_dim(name), 0u, kKeep);
        EXPECT_EQ(materialize_bytes(dst_slice), expected_kv_bytes.at(name)) << name;
    }

    run_generate_step(104);
    EXPECT_EQ(stored_tokens(), 105);
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

// VLM language models feed the prefill with inputs_embeds and, depending on the
// family, token_type_ids (Gemma-3), 3-D M-RoPE position ids (Qwen2.5-VL) or
// M-RoPE plus DeepStack injections (Qwen3-VL). Every caller tensor of a
// continued prefill holds only the delta, so the tests below check that the
// chunks of a continuation stage exactly what the same chunks of a full-history
// prefill stage.
enum class VlmInputs { InputsEmbeds, TokenTypeIds, MRoPE, DeepStack };

constexpr size_t kVlmTotal = 104;      // full second-turn history
constexpr size_t kVlmFirstTurn = 72;   // history prefilled by the first turn
constexpr size_t kVlmKeep = 64;        // chunk-aligned grant, inside the image block
constexpr size_t kVlmImageBegin = 40;  // image block [40, 80), a 5 x 8 grid
constexpr size_t kVlmImageEnd = 80;
constexpr size_t kVlmImageWidth = 8;
constexpr size_t kVlmDeepstackLayers = 2;

bool is_vlm_image_token(size_t i) {
    return i >= kVlmImageBegin && i < kVlmImageEnd;
}

size_t vlm_image_tokens_before(size_t i) {
    return std::clamp(i, kVlmImageBegin, kVlmImageEnd) - kVlmImageBegin;
}

// Gemma-3 style token_type_ids input. The model builder's bidirectional image
// mask reads token_type_ids over the whole history, which does not reshape to a
// chunked prefill, so the input reaches the graph through a probe output. The
// fake sub-requests never execute the graph; the tests check the staged chunk.
void add_token_type_ids_input(const std::shared_ptr<ov::Model>& model) {
    auto token_type_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    token_type_ids->output(0).set_names({"token_type_ids"});
    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1}, {0});
    auto probe = std::make_shared<ov::op::v1::Add>(token_type_ids, zero);
    probe->output(0).set_names({"token_type_ids_probe"});
    auto result = std::make_shared<ov::op::v0::Result>(probe);
    model->add_parameters({token_type_ids});
    model->add_results({result});
    model->validate_nodes_and_infer_types();
}

// Qwen3-VL style DeepStack injection on the input embeddings, the pattern
// ReplaceDeepstackScatterWithAdd turns into a dense residual add.
void add_deepstack_injection(const std::shared_ptr<ov::Model>& model, size_t hidden_size) {
    std::shared_ptr<ov::op::v0::Parameter> embeds;
    for (const auto& param : model->get_parameters()) {
        if (param->output(0).get_names().count("inputs_embeds") > 0) {
            embeds = param;
        }
    }
    ASSERT_NE(embeds, nullptr);
    const auto consumers = embeds->output(0).get_target_inputs();

    auto masks = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    masks->output(0).set_names({"visual_pos_masks"});
    auto deepstack =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{-1, -1, static_cast<int64_t>(hidden_size)});
    deepstack->output(0).set_names({"deepstack_visual_embeds"});

    auto nonzero = std::make_shared<ov::op::v3::NonZero>(masks, ov::element::i64);
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto pos = std::make_shared<ov::op::v1::Transpose>(nonzero, perm);
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    ov::Output<ov::Node> hidden = embeds->output(0);
    for (size_t l = 0; l < kVlmDeepstackLayers; ++l) {
        auto gathered = std::make_shared<ov::op::v8::GatherND>(hidden, pos);
        auto layer = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(l)});
        auto level = std::make_shared<ov::op::v8::Gather>(deepstack, layer, axis0);
        auto add = std::make_shared<ov::op::v1::Add>(gathered, level);
        hidden = std::make_shared<ov::op::v3::ScatterNDUpdate>(hidden, pos, add);
    }
    for (auto consumer : consumers) {
        consumer.replace_source_output(hidden);
    }
    model->add_parameters({masks, deepstack});
    model->validate_nodes_and_infer_types();
}

std::shared_ptr<ov::Model> build_vlm_test_model(VlmInputs kind) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.use_inputs_embeds = true;
    if (kind == VlmInputs::MRoPE || kind == VlmInputs::DeepStack) {
        cfg.position_ids = ov::test::npuw::make_position_ids_3d();
    }
    ov::test::npuw::ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    if (kind == VlmInputs::TokenTypeIds) {
        add_token_type_ids_input(model);
    }
    if (kind == VlmInputs::DeepStack) {
        add_deepstack_injection(model, cfg.hidden_size);
    }
    return model;
}

// Copies [begin, end) of one axis into a dense tensor.
ov::Tensor slice_axis(const ov::Tensor& tensor, size_t axis, size_t begin, size_t end) {
    ov::Coordinate lo(tensor.get_shape().size(), 0u);
    ov::Coordinate hi(tensor.get_shape());
    lo[axis] = begin;
    hi[axis] = end;
    const ov::Tensor roi(tensor, lo, hi);
    ov::Tensor dense(tensor.get_element_type(), roi.get_shape());
    roi.copy_to(dense);
    return dense;
}

// The inputs of one VLM prefill. Per-token tensors are indexed by the token
// position in the full history; slice() cuts them to [begin, end) the way the
// caller does for a continued prefill.
struct VlmPrompt {
    ov::Tensor inputs_embeds;     // [1, T, hidden]
    ov::Tensor position_ids;      // [1, T] or M-RoPE [3, 1, T]
    ov::Tensor token_type_ids;    // [1, T]
    ov::Tensor visual_pos_masks;  // [1, T]
    ov::Tensor deepstack;         // [layers, image tokens, hidden]

    static VlmPrompt full_history(bool mrope, size_t hidden) {
        VlmPrompt p;
        p.inputs_embeds = ov::Tensor(ov::element::f32, ov::Shape{1, kVlmTotal, hidden});
        p.token_type_ids = ov::Tensor(ov::element::i64, ov::Shape{1, kVlmTotal});
        p.visual_pos_masks = ov::Tensor(ov::element::boolean, ov::Shape{1, kVlmTotal});
        const size_t image_tokens = kVlmImageEnd - kVlmImageBegin;
        p.deepstack = ov::Tensor(ov::element::f32, ov::Shape{kVlmDeepstackLayers, image_tokens, hidden});
        for (size_t i = 0; i < kVlmTotal; ++i) {
            for (size_t e = 0; e < hidden; ++e) {
                p.inputs_embeds.data<float>()[i * hidden + e] = static_cast<float>(i * 100 + e);
            }
            p.token_type_ids.data<int64_t>()[i] = is_vlm_image_token(i) ? 1 : 0;
            p.visual_pos_masks.data<bool>()[i] = is_vlm_image_token(i);
        }
        for (size_t i = 0; i < p.deepstack.get_size(); ++i) {
            p.deepstack.data<float>()[i] = 1e5f + static_cast<float>(i);
        }
        if (!mrope) {
            p.position_ids = make_i64_iota({1, kVlmTotal}, 0);
            return p;
        }
        // M-RoPE: text advances all three axes together, an image token takes
        // (block start, block start + row, block start + column), and the text
        // after the block resumes from the block's largest position plus one.
        p.position_ids = ov::Tensor(ov::element::i64, ov::Shape{3, 1, kVlmTotal});
        auto* pos = p.position_ids.data<int64_t>();
        const auto start = static_cast<int64_t>(kVlmImageBegin);
        const auto rows = static_cast<int64_t>((kVlmImageEnd - kVlmImageBegin) / kVlmImageWidth);
        const int64_t after_image = start + std::max<int64_t>(rows, static_cast<int64_t>(kVlmImageWidth));
        for (size_t i = 0; i < kVlmTotal; ++i) {
            int64_t t = static_cast<int64_t>(i), h = t, w = t;
            if (is_vlm_image_token(i)) {
                const auto j = static_cast<int64_t>(i - kVlmImageBegin);
                t = start;
                h = start + j / static_cast<int64_t>(kVlmImageWidth);
                w = start + j % static_cast<int64_t>(kVlmImageWidth);
            } else if (i >= kVlmImageEnd) {
                t = h = w = after_image + static_cast<int64_t>(i - kVlmImageEnd);
            }
            pos[i] = t;
            pos[kVlmTotal + i] = h;
            pos[2 * kVlmTotal + i] = w;
        }
        return p;
    }

    VlmPrompt slice(size_t begin, size_t end) const {
        VlmPrompt p;
        p.inputs_embeds = slice_axis(inputs_embeds, 1, begin, end);
        p.position_ids = slice_axis(position_ids, position_ids.get_shape().size() - 1, begin, end);
        p.token_type_ids = slice_axis(token_type_ids, 1, begin, end);
        p.visual_pos_masks = slice_axis(visual_pos_masks, 1, begin, end);
        p.deepstack = slice_axis(deepstack, 1, vlm_image_tokens_before(begin), vlm_image_tokens_before(end));
        return p;
    }
};

class LLMVlmContinuedPrefillTest : public LLMContinuedPrefillTest, public ::testing::WithParamInterface<VlmInputs> {
protected:
    void SetUp() override {}

    bool mrope() const {
        return GetParam() == VlmInputs::MRoPE || GetParam() == VlmInputs::DeepStack;
    }

    // Compiles a fresh model and request whose prefill sub-request is journaled.
    // The base fixture asserts that the capability reports support.
    void start_session() {
        m_journal = std::make_shared<PrefillJournal>();
        init({}, build_vlm_test_model(GetParam()), m_journal);
    }

    void set_input(const std::string& name, const ov::Tensor& tensor) {
        const auto port = ov::npuw::util::find_port_by_name(request().get_inputs(), name);
        if (port.has_value()) {
            request().set_tensor(port.value(), ov::get_tensor_impl(tensor));
        }
    }

    void run_prefill(const VlmPrompt& prompt, size_t history_len) {
        set_input("inputs_embeds", prompt.inputs_embeds);
        set_input("attention_mask", make_i64({1, history_len}, 1));
        set_input("position_ids", prompt.position_ids);
        set_input("token_type_ids", prompt.token_type_ids);
        set_input("visual_pos_masks", prompt.visual_pos_masks);
        set_input("deepstack_visual_embeds", prompt.deepstack);
        request().infer();
    }

    void run_vlm_generate_step(size_t live_tokens, size_t hidden) {
        VlmPrompt step;
        step.inputs_embeds = ov::Tensor(ov::element::f32, ov::Shape{1, 1, hidden});
        std::fill_n(step.inputs_embeds.data<float>(), hidden, 0.5f);
        step.position_ids = mrope() ? make_i64({3, 1, 1}, static_cast<int64_t>(live_tokens))
                                    : make_i64({1, 1}, static_cast<int64_t>(live_tokens));
        step.token_type_ids = make_i64({1, 1}, 0);
        step.visual_pos_masks = ov::Tensor(ov::element::boolean, ov::Shape{1, 1});
        step.visual_pos_masks.data<bool>()[0] = false;
        step.deepstack = ov::Tensor(ov::element::f32, ov::Shape{kVlmDeepstackLayers, 1, hidden});
        std::fill_n(step.deepstack.data<float>(), step.deepstack.get_size(), 0.f);
        run_prefill(step, live_tokens + 1);
    }

    // First turn, one generate step, then a negotiated grant of kVlmKeep.
    void prepare_continuation(const VlmPrompt& full, size_t hidden) {
        run_prefill(full.slice(0, kVlmFirstTurn), kVlmFirstTurn);
        run_vlm_generate_step(kVlmFirstTurn, hidden);
        propose(kVlmFirstTurn + 1);
        ASSERT_EQ(stored_tokens(), static_cast<int64_t>(kVlmKeep));
    }

    std::vector<std::string> staged_inputs() const {
        std::vector<std::string> names{"inputs_embeds", "attention_mask", "position_ids"};
        if (GetParam() == VlmInputs::TokenTypeIds) {
            names.push_back("token_type_ids");
        }
        if (GetParam() == VlmInputs::DeepStack) {
            names.push_back("deepstack_visual_embeds");
        }
        return names;
    }

    std::shared_ptr<PrefillJournal> m_journal;
};

// Turn two continued at a grant inside the image block stages exactly the
// chunks a full-history prefill of the same conversation stages for the same
// absolute range. The grant is chunk-aligned, so the chunk boundaries match.
TEST_P(LLMVlmContinuedPrefillTest, ContinuedPrefillStagesTheSameChunksAsFullHistory) {
    constexpr size_t kHidden = 64;
    const auto full = VlmPrompt::full_history(mrope(), kHidden);

    // Reference: the whole second-turn history prefilled on a fresh request.
    start_session();
    run_prefill(full, kVlmTotal);
    const PrefillJournal reference = *m_journal;
    ASSERT_EQ(reference.size(), 4u);  // chunks of 32 over 104 tokens

    // Continuation: first turn, generate, grant, then only the delta.
    start_session();
    prepare_continuation(full, kHidden);
    m_journal->clear();
    run_prefill(full.slice(kVlmKeep, kVlmTotal), kVlmTotal);
    EXPECT_EQ(stored_tokens(), static_cast<int64_t>(kVlmTotal));
    ASSERT_EQ(m_journal->size(), 2u);  // chunks [64, 96) and [96, 104)

    for (size_t chunk = 0; chunk < m_journal->size(); ++chunk) {
        const auto& expected = reference.at(chunk + kVlmKeep / 32u);
        const auto& actual = m_journal->at(chunk);
        for (const auto& name : staged_inputs()) {
            ASSERT_EQ(actual.count(name), 1u) << name;
            EXPECT_EQ(actual.at(name), expected.at(name)) << name << " differs in delta chunk " << chunk;
        }
    }
}

// A per-token input left at full-history length is a preflight rejection: the
// grant stays armed and the correctly sliced delta still completes.
TEST_P(LLMVlmContinuedPrefillTest, FullHistoryInputIsRejectedBeforeTheRepack) {
    constexpr size_t kHidden = 64;
    const auto full = VlmPrompt::full_history(mrope(), kHidden);
    start_session();
    prepare_continuation(full, kHidden);

    auto wrong = full.slice(kVlmKeep, kVlmTotal);
    switch (GetParam()) {
    case VlmInputs::InputsEmbeds:
    case VlmInputs::MRoPE:
        wrong.position_ids = full.position_ids;
        break;
    case VlmInputs::TokenTypeIds:
        wrong.token_type_ids = full.token_type_ids;
        break;
    case VlmInputs::DeepStack:
        wrong.deepstack = full.deepstack;
        break;
    }
    EXPECT_THROW(run_prefill(wrong, kVlmTotal), ov::Exception);
    EXPECT_EQ(stored_tokens(), static_cast<int64_t>(kVlmKeep));

    run_prefill(full.slice(kVlmKeep, kVlmTotal), kVlmTotal);
    EXPECT_EQ(stored_tokens(), static_cast<int64_t>(kVlmTotal));
}

INSTANTIATE_TEST_SUITE_P(
    VlmInputs,
    LLMVlmContinuedPrefillTest,
    ::testing::Values(VlmInputs::InputsEmbeds, VlmInputs::TokenTypeIds, VlmInputs::MRoPE, VlmInputs::DeepStack),
    [](const ::testing::TestParamInfo<VlmInputs>& info) {
        switch (info.param) {
        case VlmInputs::InputsEmbeds:
            return std::string("InputsEmbeds");
        case VlmInputs::TokenTypeIds:
            return std::string("TokenTypeIds");
        case VlmInputs::MRoPE:
            return std::string("MRoPE");
        case VlmInputs::DeepStack:
            return std::string("DeepStack");
        }
        return std::string();
    });

}  // namespace
