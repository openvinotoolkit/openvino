// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"

// Tests for ov::npuw::LLMInferRequest::trim_kvcache_for_speculative_decoding().
//
// Speculative decoding may re-run the generate step from an earlier position, so the KV cache has to
// be rewound to that position: `num_stored_tokens` is reduced to the incoming `position_ids[0]`.
//
// `num_stored_tokens` is a uint32_t, so a position id *greater* than the number of tokens processed
// so far would make `num_stored_tokens - position_id` wrap around to a huge value instead of going
// negative, leaving the descriptor claiming a cache far larger than the one that exists. The trim is
// therefore skipped (with a warning) whenever the position id runs ahead of the stored token count.

namespace ov::test::npuw {

// `m_kvcache_desc` is private to LLMCompiledModel and the trim entry point is protected in
// LLMInferRequest, so the test reaches both through a friend struct - same approach as
// LLMVariantSwitchTestAccess.
struct LLMTrimKVCacheTestAccess {
    static void set_num_stored_tokens(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled,
                                      uint32_t num_tokens) {
        compiled->m_kvcache_desc.num_stored_tokens = num_tokens;
    }

    static uint32_t num_stored_tokens(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_kvcache_desc.num_stored_tokens;
    }

    static uint32_t total_size(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        return compiled->m_kvcache_desc.total_size;
    }

    static void trim(ov::npuw::LLMInferRequest& req, const ov::SoPtr<ov::ITensor>& position_ids) {
        req.trim_kvcache_for_speculative_decoding(position_ids);
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::LLMTrimKVCacheTestAccess;
using ov::test::npuw::NullPlugin;

class FakeSubCompiledModel;

// The trim only touches the KV-cache descriptor, never a sub-request. These fakes exist purely so
// that constructing an LLMInferRequest succeeds: its constructor wraps every compiled variant into
// an async request and reads the ports back off it, which the shared MockSubCompiledModel cannot do
// (it returns a null request).
class FakeSubInferRequest final : public ov::ISyncInferRequest {
public:
    explicit FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model);

    void infer() override {}
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
                                                        intel_npu::make_executor("trim_kvcache_task", 1),
                                                        intel_npu::make_executor("trim_kvcache_callback", 1));
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

ov::npuw::LLMCompiledModel::CompiledModelFactory make_fake_factory() {
    return [](const std::shared_ptr<ov::Model>& model,
              const std::shared_ptr<const ov::IPlugin>& plugin,
              const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
        return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
    };
}

// position_ids as the generate step supplies it: [batch, seq_len] i64, value in element 0.
ov::SoPtr<ov::ITensor> make_position_ids(int64_t position_id) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{1, 1});
    tensor.data<int64_t>()[0] = position_id;
    return ov::get_tensor_impl(tensor);
}

class LLMTrimKVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    std::shared_ptr<ov::npuw::LLMCompiledModel> create_compiled_model() const {
        const ov::AnyMap props{{"NPUW_LLM", "YES"},
                               {"NPUW_DEVICES", "CPU"},
                               {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                               {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
        return std::make_shared<ov::npuw::LLMCompiledModel>(build_llm_test_model(),
                                                            m_plugin,
                                                            props,
                                                            make_fake_factory());
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

// ---------------------------------------------------------------------------
// The safety check: a position id ahead of the stored tokens must be ignored
// ---------------------------------------------------------------------------

TEST_F(LLMTrimKVCacheTest, PositionIdBeyondStoredTokensLeavesCacheUntouched) {
    auto compiled = create_compiled_model();
    ASSERT_NE(compiled, nullptr);
    ov::npuw::LLMInferRequest req(compiled);

    LLMTrimKVCacheTestAccess::set_num_stored_tokens(compiled, 10u);
    LLMTrimKVCacheTestAccess::trim(req, make_position_ids(20));

    const auto stored = LLMTrimKVCacheTestAccess::num_stored_tokens(compiled);
    EXPECT_EQ(stored, 10u) << "trimming to a position id past the processed tokens must be skipped";
    // Without the guard, 10u - 20u wraps to 4294967286 - assert the cache never claims more tokens
    // than it can physically hold.
    EXPECT_LE(stored, LLMTrimKVCacheTestAccess::total_size(compiled)) << "num_stored_tokens wrapped around";
}

// A negative position id is the worst case for the unsigned cast: static_cast<uint32_t>(-1) is
// 4294967295, so without the guard the subtraction would leave an enormous count behind.
TEST_F(LLMTrimKVCacheTest, NegativePositionIdLeavesCacheUntouched) {
    auto compiled = create_compiled_model();
    ASSERT_NE(compiled, nullptr);
    ov::npuw::LLMInferRequest req(compiled);

    LLMTrimKVCacheTestAccess::set_num_stored_tokens(compiled, 10u);
    LLMTrimKVCacheTestAccess::trim(req, make_position_ids(-1));

    const auto stored = LLMTrimKVCacheTestAccess::num_stored_tokens(compiled);
    EXPECT_EQ(stored, 10u);
    EXPECT_LE(stored, LLMTrimKVCacheTestAccess::total_size(compiled)) << "num_stored_tokens wrapped around";
}

// ---------------------------------------------------------------------------
// The guard must not disturb the cases the trim exists for
// ---------------------------------------------------------------------------

TEST_F(LLMTrimKVCacheTest, TrimsDownToPositionId) {
    auto compiled = create_compiled_model();
    ASSERT_NE(compiled, nullptr);
    ov::npuw::LLMInferRequest req(compiled);

    LLMTrimKVCacheTestAccess::set_num_stored_tokens(compiled, 10u);
    LLMTrimKVCacheTestAccess::trim(req, make_position_ids(4));

    EXPECT_EQ(LLMTrimKVCacheTestAccess::num_stored_tokens(compiled), 4u)
        << "the rejected speculative tokens must be dropped from the cache";
}

TEST_F(LLMTrimKVCacheTest, PositionIdEqualToStoredTokensIsANoOp) {
    auto compiled = create_compiled_model();
    ASSERT_NE(compiled, nullptr);
    ov::npuw::LLMInferRequest req(compiled);

    LLMTrimKVCacheTestAccess::set_num_stored_tokens(compiled, 10u);
    LLMTrimKVCacheTestAccess::trim(req, make_position_ids(10));

    EXPECT_EQ(LLMTrimKVCacheTestAccess::num_stored_tokens(compiled), 10u)
        << "nothing was speculated past the stored tokens, so nothing may be dropped";
}

TEST_F(LLMTrimKVCacheTest, TrimToZeroClearsWholeCache) {
    auto compiled = create_compiled_model();
    ASSERT_NE(compiled, nullptr);
    ov::npuw::LLMInferRequest req(compiled);

    LLMTrimKVCacheTestAccess::set_num_stored_tokens(compiled, 10u);
    LLMTrimKVCacheTestAccess::trim(req, make_position_ids(0));

    EXPECT_EQ(LLMTrimKVCacheTestAccess::num_stored_tokens(compiled), 0u);
}

}  // namespace
