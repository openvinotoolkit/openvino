// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>

#include "common_test_utils/test_assertions.hpp"
#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"
#include "whisper/whisper_infer_request.hpp"

namespace {

using ov::test::npuw::build_whisper_decoder_test_model;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::WhisperConfig;

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
                                                        intel_npu::make_executor("whisper_security_task", 1),
                                                        intel_npu::make_executor("whisper_security_callback", 1));
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

ov::npuw::LLMCompiledModel::CompiledModelFactory make_fake_factory() {
    return [](const std::shared_ptr<ov::Model>& model,
              const std::shared_ptr<const ov::IPlugin>& plugin,
              const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
        return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
    };
}

ov::Tensor make_input_ids(size_t seq_len) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{1, seq_len});
    std::iota(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size(), int64_t{1});
    return tensor;
}

ov::Tensor make_encoder_hidden_states(size_t encoder_seq_len, size_t hidden_size) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{1, encoder_seq_len, hidden_size});
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.f);
    return tensor;
}

class WhisperInferRequestSecurityTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
        ov::AnyMap props{{"NPUW_LLM", "YES"},
                         {"NPUW_DEVICES", "CPU"},
                         {"NPUW_WHISPER", "YES"},
                         {"NPUW_WHISPER_EOS_TOKEN", "42"}};
        m_compiled = std::make_shared<ov::npuw::LLMCompiledModel>(build_whisper_decoder_test_model(),
                                                                  m_plugin,
                                                                  props,
                                                                  make_fake_factory());
        ASSERT_NE(m_compiled, nullptr);
        m_request = std::make_shared<ov::npuw::WhisperInferRequest>(m_compiled);
    }

    void set_inputs(const ov::Tensor& input_ids, const ov::Tensor& encoder_hidden_states) {
        const auto& inputs = m_request->get_inputs();
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "input_ids").value(),
                              ov::get_tensor_impl(input_ids));
        m_request->set_tensor(ov::npuw::util::find_port_by_name(inputs, "encoder_hidden_states").value(),
                              ov::get_tensor_impl(encoder_hidden_states));
    }

    static constexpr size_t kHiddenSize = 64;

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::npuw::LLMCompiledModel> m_compiled;
    std::shared_ptr<ov::npuw::WhisperInferRequest> m_request;
};

TEST_F(WhisperInferRequestSecurityTest, OversizedPromptIsRejectedNotOverflowed) {
    constexpr size_t kOversizedPrompt = ov::npuw::LLMCompiledModel::whisper_max_prompt_size + 60;
    set_inputs(make_input_ids(kOversizedPrompt),
               make_encoder_hidden_states(WhisperConfig{}.max_source_positions, kHiddenSize));

    OV_EXPECT_THROW_HAS_SUBSTRING(m_request->infer(), ov::Exception, "longer than the compiled prefill input");
}

TEST_F(WhisperInferRequestSecurityTest, PromptWithinCapacitySucceeds) {
    set_inputs(make_input_ids(ov::npuw::LLMCompiledModel::whisper_max_prompt_size),
               make_encoder_hidden_states(WhisperConfig{}.max_source_positions, kHiddenSize));

    EXPECT_NO_THROW(m_request->infer());
}

TEST_F(WhisperInferRequestSecurityTest, MismatchedEncoderHiddenStatesIsRejected) {
    set_inputs(make_input_ids(ov::npuw::LLMCompiledModel::whisper_max_prompt_size),
               make_encoder_hidden_states(WhisperConfig{}.max_source_positions + 1, kHiddenSize));

    OV_EXPECT_THROW_HAS_SUBSTRING(m_request->infer(), ov::Exception, "does not match the compiled prefill input");
}

}  // namespace
