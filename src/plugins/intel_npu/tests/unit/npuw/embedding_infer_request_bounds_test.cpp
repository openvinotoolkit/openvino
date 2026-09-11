// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for EmbeddingInferRequest::infer_whole_prefill(). The plugin pads
// input_ids/attention_mask right-aligned into static tensors sized for
// NPUW_LLM_MAX_PROMPT_LEN. The only guard before that copy inspects
// shape()[INPUT_IDS_SEQ_LEN_DIM] (dim 1); it does not check tensor rank or total
// element count against the destination capacity, and both input_ids/attention_mask
// ports stay dynamic, so ISyncInferRequest::check_tensor never validates their shape
// either. A caller can therefore submit a tensor whose total size exceeds the static
// destination while still passing the dim-1 check.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

#include "executor.hpp"
#include "embedding/embedding_infer_request.hpp"
#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace {

using ov::test::npuw::build_embedding_decoder_test_model;
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
                                                        intel_npu::make_executor("embedding_bounds_task", 1),
                                                        intel_npu::make_executor("embedding_bounds_callback", 1));
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
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

class EmbeddingBoundsFactory {
public:
    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [](const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
        };
    }
};

class EmbeddingInferRequestBoundsTest : public ::testing::Test {
protected:
    static constexpr size_t kMaxPromptLen = 64;

    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
        EmbeddingBoundsFactory factory;
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_TEXT_EMBED", "YES"},
                         {"NPUW_LLM_SHARED_HEAD", "NO"},
                         {"NPUW_LLM_MAX_PROMPT_LEN", std::to_string(kMaxPromptLen)}};
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(build_embedding_decoder_test_model(),
                                                                  m_plugin,
                                                                  props,
                                                                  factory.make_factory());
        ASSERT_NE(m_compiled, nullptr);
        m_request = std::make_shared<ov::npuw::EmbeddingInferRequest>(m_compiled);
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::shared_ptr<ov::npuw::EmbeddingInferRequest> m_request;
};

// A rank-3 input_ids tensor whose dim-1 value stays within the prompt limit, but whose
// total element count exceeds the static [1, kMaxPromptLen] destination the plugin pads
// into. Must be rejected instead of silently accepted and copied out of bounds.
TEST_F(EmbeddingInferRequestBoundsTest, RejectsInputIdsExceedingDestinationCapacity) {
    const auto& inputs = m_request->get_inputs();
    auto input_ids_port = ov::npuw::util::find_port_by_name(inputs, "input_ids");
    auto attn_mask_port = ov::npuw::util::find_port_by_name(inputs, "attention_mask");
    ASSERT_TRUE(input_ids_port.has_value());
    ASSERT_TRUE(attn_mask_port.has_value());

    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, 1, kMaxPromptLen + 1});
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, kMaxPromptLen});
    std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), 1);
    std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

    m_request->set_tensor(*input_ids_port, ov::get_tensor_impl(input_ids));
    m_request->set_tensor(*attn_mask_port, ov::get_tensor_impl(attention_mask));

    EXPECT_THROW(m_request->infer(), ov::Exception);
}

}  // namespace
