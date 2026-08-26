// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Request-level tests for the LongRoPE short/long regime state machine.
//
// llm_longrope_kv_test.cpp proves the coefficient math against an independent
// reference; what is left, and what these tests cover, is everything around it: when a
// crossing is detected, that the rewrite runs after the KV of the previous stage has
// been migrated into the request being rewritten, which rows it touches, and that a
// transition which cannot be completed leaves the recorded regime describing the cache
// as it really is.
//
// The observation trick is a second, "control" model that is identical except for a
// context limit no position in the test can reach. Both models are driven with the same
// inputs through fake sub-requests whose outputs depend only on the port name, so their
// KV caches agree byte for byte up to the re-rotation - which is then the only thing the
// comparison can be measuring.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_longrope_kv.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct LLMLongRopeRegimeTestAccess {
    using Request = ov::npuw::LLMInferRequest;

    static bool long_regime(Request& req) {
        return req.m_longrope_long_regime;
    }

    static const std::vector<std::string>& past_names(Request& req) {
        return req.m_kvcache_past_names;
    }

    static ov::SoPtr<ov::ITensor> generate_past(Request& req, const std::string& name) {
        return req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
    }

    static uint32_t stored_tokens(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens;
    }

    static uint32_t seq_dim(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc.dim;
    }

    static ov::npuw::patterns::pre_compute::LongRopeCosSin& tables(Request& req) {
        return req.m_npuw_llm_compiled_model->m_longrope_tables;
    }

    static bool has_long(const ov::npuw::LLMCompiledModel& compiled) {
        return compiled.m_longrope_tables.has_long;
    }

    // Drives the transition directly, which is the only way to reach the rejection
    // paths - a real request always hands over its own, well-formed port map.
    static void sync_regime(Request& req,
                            const ov::npuw::LLMInferRequest::PortsMap& in_ports,
                            uint32_t num_cached_tokens,
                            bool is_long) {
        req.sync_longrope_kv_regime(req.m_kvcache_request, in_ports, num_cached_tokens, is_long);
    }

    static const ov::npuw::LLMInferRequest::PortsMap& generate_in_ports(Request& req) {
        return req.m_kvcache_in_ports;
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_longrope_llm_test_model;
using ov::test::npuw::LLMLongRopeRegimeTestAccess;
using ov::test::npuw::NullPlugin;
class FakeSubCompiledModel;

// A position no prompt in this file reaches, so the model never leaves the short
// regime and its KV cache is never re-rotated.
constexpr int64_t kUnreachableLimit = 4096;
// Crossed by generated token 16 of a 16-token prompt.
constexpr int64_t kCrossingLimit = 16;

// Deterministic, finite, port-dependent contents so two models with the same port names
// produce the same KV bytes.
void fill_pattern(const ov::SoPtr<ov::ITensor>& tensor, size_t seed) {
    const size_t count = tensor->get_size();
    const auto type = tensor->get_element_type();
    if (type == ov::element::f16) {
        auto* data = tensor->data<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = ov::float16(std::sin(static_cast<float>(i + seed) * 0.137f) * 1.25f);
        }
    } else if (type == ov::element::f32) {
        auto* data = tensor->data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = std::sin(static_cast<float>(i + seed) * 0.137f) * 1.25f;
        }
    } else {
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
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
                                                        intel_npu::make_executor("longrope_regime_task", 1),
                                                        intel_npu::make_executor("longrope_regime_callback", 1));
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
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
        fill_pattern(ov::ISyncInferRequest::get_tensor(output), std::hash<std::string>{}(output.get_any_name()));
    }
}

ov::npuw::LLMCompiledModel::CompiledModelFactory make_fake_factory() {
    return [](const std::shared_ptr<ov::Model>& model,
              const std::shared_ptr<const ov::IPlugin>& plugin,
              const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
        return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
    };
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

std::vector<float> to_floats(const ov::SoPtr<ov::ITensor>& tensor) {
    ov::Tensor dense(tensor->get_element_type(), tensor->get_shape());
    tensor->copy_to(ov::get_tensor_impl(dense)._ptr);
    std::vector<float> out(dense.get_size());
    if (dense.get_element_type() == ov::element::f16) {
        const auto* data = dense.data<ov::float16>();
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<float>(data[i]);
        }
    } else {
        const auto* data = dense.data<float>();
        std::copy_n(data, out.size(), out.begin());
    }
    return out;
}

// One conversation driver over a freshly compiled model with the given context limit.
class LongRopePipeline {
public:
    explicit LongRopePipeline(int64_t context_limit) {
        m_plugin = std::make_shared<NullPlugin>();
        const ov::AnyMap props{{"NPUW_LLM", "YES"},
                               {"NPUW_DEVICES", "CPU"},
                               {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                               {"NPUW_LLM_MIN_RESPONSE_LEN", "32"},
                               {"NPUW_LLM_CACHE_ROPE", "YES"}};
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(build_longrope_llm_test_model(context_limit),
                                                                  m_plugin,
                                                                  props,
                                                                  make_fake_factory());
        m_request = std::make_unique<ov::npuw::LLMInferRequest>(m_compiled);
    }

    ov::npuw::LLMInferRequest& request() {
        return *m_request;
    }

    const ov::npuw::LLMCompiledModel& compiled() const {
        return *m_compiled;
    }

    void prefill(size_t prompt_len, int64_t first_position) {
        set_inputs(make_i64({1, prompt_len}, 1),
                   make_i64({1, prompt_len}, 1),
                   make_i64_iota({1, prompt_len}, first_position));
        m_request->infer();
    }

    void generate_step(size_t live_tokens, int64_t position) {
        set_inputs(make_i64({1, 1}, 1), make_i64({1, live_tokens + 1}, 1), make_i64({1, 1}, position));
        m_request->infer();
    }

    // The past keys of every layer, flattened, in m_kvcache_past_names order.
    std::vector<std::vector<float>> generate_keys() {
        std::vector<std::vector<float>> out;
        for (const auto& name : LLMLongRopeRegimeTestAccess::past_names(*m_request)) {
            if (ov::npuw::util::isPastKeyParam(name)) {
                out.push_back(to_floats(LLMLongRopeRegimeTestAccess::generate_past(*m_request, name)));
            }
        }
        return out;
    }

    ov::Shape generate_key_shape() {
        for (const auto& name : LLMLongRopeRegimeTestAccess::past_names(*m_request)) {
            if (ov::npuw::util::isPastKeyParam(name)) {
                return LLMLongRopeRegimeTestAccess::generate_past(*m_request, name)->get_shape();
            }
        }
        return {};
    }

private:
    void set_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids) {
        const auto& inputs = m_request->get_inputs();
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "input_ids").value(),
                              ov::get_tensor_impl(input_ids));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "attention_mask").value(),
                              ov::get_tensor_impl(attention_mask));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "position_ids").value(),
                              ov::get_tensor_impl(position_ids));
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::unique_ptr<ov::npuw::LLMInferRequest> m_request;
};

// Applies the same turn the request is expected to have applied, to a copy of the
// control run's keys. The coefficient math itself is validated independently in
// llm_longrope_kv_test.cpp; here it is the reference for placement and row range.
std::vector<std::vector<float>> expected_turn(std::vector<std::vector<float>> keys,
                                              const ov::Shape& shape,
                                              uint32_t seq_dim,
                                              ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                                              uint32_t num_tokens,
                                              bool to_long) {
    const auto delta = ov::npuw::longrope::make_regime_delta(tables, 0, num_tokens, to_long);
    for (auto& layer : keys) {
        auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape, layer.data()));
        ov::npuw::longrope::rerotate_keys(tensor, seq_dim, num_tokens, delta);
    }
    return keys;
}

void expect_keys_near(const std::vector<std::vector<float>>& actual,
                      const std::vector<std::vector<float>>& expected,
                      float tol) {
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_FALSE(actual.empty());
    for (size_t layer = 0; layer < actual.size(); ++layer) {
        ASSERT_EQ(actual[layer].size(), expected[layer].size());
        for (size_t i = 0; i < actual[layer].size(); ++i) {
            ASSERT_NEAR(actual[layer][i], expected[layer][i], tol) << "layer " << layer << " element " << i;
        }
    }
}

// The models this file relies on must actually reach the LongRoPE path, otherwise every
// comparison below would pass on two untransformed graphs.
TEST(LLMLongRopeRegime, ModelsReachTheLongRopePath) {
    LongRopePipeline crossing(kCrossingLimit);
    LongRopePipeline control(kUnreachableLimit);

    EXPECT_TRUE(LLMLongRopeRegimeTestAccess::has_long(crossing.compiled()));
    // Its limit sits past the whole context, so it can only ever run short.
    EXPECT_FALSE(LLMLongRopeRegimeTestAccess::has_long(control.compiled()));
    EXPECT_TRUE(LLMLongRopeRegimeTestAccess::tables(crossing.request()).is_valid());
}

// The first generate step past the limit turns exactly the cached rows, and it does so
// after the prefill KV has been migrated into the generate request - had it run before,
// the migration would have overwritten the turned keys with the untouched ones and the
// two runs would agree.
TEST(LLMLongRopeRegime, CrossingTurnsTheMigratedCache) {
    constexpr size_t kPrompt = 16;

    LongRopePipeline control(kUnreachableLimit);
    control.prefill(kPrompt, 0);
    control.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    EXPECT_FALSE(LLMLongRopeRegimeTestAccess::long_regime(control.request()));
    const auto control_keys = control.generate_keys();

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    EXPECT_FALSE(LLMLongRopeRegimeTestAccess::long_regime(crossing.request()));
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    EXPECT_TRUE(LLMLongRopeRegimeTestAccess::long_regime(crossing.request()));

    const auto expected = expected_turn(control_keys,
                                        crossing.generate_key_shape(),
                                        LLMLongRopeRegimeTestAccess::seq_dim(crossing.request()),
                                        LLMLongRopeRegimeTestAccess::tables(crossing.request()),
                                        kPrompt,
                                        true);
    expect_keys_near(crossing.generate_keys(), expected, 2e-3f);
    // A turn that did nothing would trivially satisfy the comparison above.
    EXPECT_NE(control_keys, crossing.generate_keys());
}

// A prompt that is already past the limit finds an empty cache: the regime is recorded,
// nothing is rewritten, and the following steps do not cross again.
TEST(LLMLongRopeRegime, FreshLongPromptRecordsTheRegimeWithoutTouchingTheCache) {
    constexpr size_t kPrompt = 20;  // positions 0..19, so the prompt itself is long

    LongRopePipeline control(kUnreachableLimit);
    control.prefill(kPrompt, 0);
    control.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    const auto control_keys = control.generate_keys();

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    EXPECT_TRUE(LLMLongRopeRegimeTestAccess::long_regime(crossing.request()));
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    EXPECT_TRUE(LLMLongRopeRegimeTestAccess::long_regime(crossing.request()));

    expect_keys_near(crossing.generate_keys(), control_keys, 0.0f);
}

// Trimming the cache back below the limit - what speculative decoding does when its
// proposal is rejected - flips the regime back and returns the keys to where they were.
TEST(LLMLongRopeRegime, TrimmingBackBelowTheLimitTurnsTheCacheBack) {
    constexpr size_t kPrompt = 16;
    constexpr uint32_t kTrimTo = 10;

    LongRopePipeline control(kUnreachableLimit);
    control.prefill(kPrompt, 0);
    control.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    const auto control_keys = control.generate_keys();

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    ASSERT_TRUE(LLMLongRopeRegimeTestAccess::long_regime(crossing.request()));

    crossing.generate_step(kTrimTo, kTrimTo);
    EXPECT_FALSE(LLMLongRopeRegimeTestAccess::long_regime(crossing.request()));
    EXPECT_EQ(LLMLongRopeRegimeTestAccess::stored_tokens(crossing.request()), kTrimTo + 1u);

    // Only the rows that survived the trim were turned twice; compare just those.
    const auto keys = crossing.generate_keys();
    const auto shape = crossing.generate_key_shape();
    const size_t head_dim = shape.back();
    const size_t row_stride = head_dim;  // [batch, kv_heads, seq, head_dim]
    const size_t seq_len = shape[LLMLongRopeRegimeTestAccess::seq_dim(crossing.request())];
    ASSERT_EQ(keys.size(), control_keys.size());
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        for (size_t plane = 0; plane * seq_len * row_stride < keys[layer].size(); ++plane) {
            for (size_t token = 0; token < kTrimTo; ++token) {
                for (size_t ch = 0; ch < head_dim; ++ch) {
                    const size_t idx = (plane * seq_len + token) * row_stride + ch;
                    ASSERT_NEAR(keys[layer][idx], control_keys[layer][idx], 4e-3f)
                        << "layer " << layer << " token " << token << " channel " << ch;
                }
            }
        }
    }
}

// A transition that cannot be carried out must fail before it writes anything and must
// leave the recorded regime describing the cache as it really is - otherwise the next
// call would see no flip and run against keys from the other rotation frame.
TEST(LLMLongRopeRegime, RejectedTransitionLeavesTheCacheAndTheRegimeAlone) {
    constexpr size_t kPrompt = 8;
    constexpr uint32_t kCached = kPrompt + 1;  // the prompt plus the token just generated

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));  // position 8, still short
    auto& req = crossing.request();
    ASSERT_FALSE(LLMLongRopeRegimeTestAccess::long_regime(req));
    ASSERT_EQ(LLMLongRopeRegimeTestAccess::stored_tokens(req), kCached);
    const auto before = crossing.generate_keys();

    // A port map without the past-key inputs: the same shape of failure as a KV layout
    // the rewrite does not support.
    ov::npuw::LLMInferRequest::PortsMap broken;
    for (const auto& [name, port] : LLMLongRopeRegimeTestAccess::generate_in_ports(req)) {
        if (!ov::npuw::util::isPastKeyParam(name)) {
            broken.emplace(name, port);
        }
    }

    EXPECT_THROW(LLMLongRopeRegimeTestAccess::sync_regime(req, broken, kCached, true), ov::Exception);
    EXPECT_FALSE(LLMLongRopeRegimeTestAccess::long_regime(req));
    expect_keys_near(crossing.generate_keys(), before, 0.0f);

    // The flip is still pending, so a well-formed retry performs it in full.
    LLMLongRopeRegimeTestAccess::sync_regime(req,
                                             LLMLongRopeRegimeTestAccess::generate_in_ports(req),
                                             kCached,
                                             true);
    EXPECT_TRUE(LLMLongRopeRegimeTestAccess::long_regime(req));
    EXPECT_NE(crossing.generate_keys(), before);
}

// A KV layout the rewrite cannot turn is refused when the model is compiled, not at the
// crossing token: by then there is no correct thing left to do.
class LongRopeCompileRejection : public ::testing::Test {
protected:
    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"},
                {"NPUW_DEVICES", "CPU"},
                {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                {"NPUW_LLM_MIN_RESPONSE_LEN", "32"},
                {"NPUW_LLM_CACHE_ROPE", "YES"}};
    }

    void compile(int64_t context_limit, const ov::AnyMap& extra) {
        auto props = base_props();
        for (const auto& [key, value] : extra) {
            props[key] = value;
        }
        ov::npuw::LLMCompiledModel(build_longrope_llm_test_model(context_limit),
                                   std::make_shared<NullPlugin>(),
                                   props,
                                   make_fake_factory());
    }

    void expect_longrope_rejection(int64_t context_limit, const ov::AnyMap& extra) {
        try {
            compile(context_limit, extra);
            FAIL() << "compilation was expected to be rejected";
        } catch (const ov::Exception& e) {
            EXPECT_NE(std::string(e.what()).find("LongRoPE regime changes"), std::string::npos) << e.what();
        }
    }
};

TEST_F(LongRopeCompileRejection, BlockBasedKvCache) {
    const ov::AnyMap block_props{{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                 {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"},
                                 {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
                                 {"NPUW_LLM_GENERATE_ATTENTION_HINT", "PYRAMID"},
                                 {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"}};
    expect_longrope_rejection(kCrossingLimit, block_props);
    // The same configuration is fine for a model whose long regime is out of reach.
    EXPECT_NO_THROW(compile(kUnreachableLimit, block_props));
}

TEST_F(LongRopeCompileRejection, QuantizedKvCache) {
    expect_longrope_rejection(kCrossingLimit, {{ov::hint::kv_cache_precision.name(), ov::element::i8}});
}

}  // anonymous namespace
