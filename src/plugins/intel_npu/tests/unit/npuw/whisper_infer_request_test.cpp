// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_stored_tokens_state.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "whisper/whisper_infer_request.hpp"

namespace {

using ov::test::npuw::build_whisper_decoder_test_model;
using ov::test::npuw::make_test_model_config;
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
                                                        intel_npu::make_executor("whisper_infer_task", 1),
                                                        intel_npu::make_executor("whisper_infer_callback", 1));
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

ov::npuw::LLMCompiledModel::CompiledModelFactory make_fake_factory() {
    return [](const std::shared_ptr<ov::Model>& model,
              const std::shared_ptr<const ov::IPlugin>& plugin,
              const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
        return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
    };
}

void set_input(ov::npuw::WhisperInferRequest& request, const std::string& name, const ov::Tensor& tensor) {
    const auto port = ov::npuw::util::find_port_by_name(request.get_inputs(), name);
    ASSERT_TRUE(port.has_value()) << "Missing input " << name;
    request.set_tensor(*port, ov::get_tensor_impl(tensor));
}

}  // namespace

TEST(WhisperInferRequest, ResetStoredTokensStartsFreshPrefillForSingleToken) {
    auto plugin = std::make_shared<NullPlugin>();
    auto model = build_whisper_decoder_test_model();
    ov::AnyMap properties{{"NPUW_LLM", "YES"},
                          {"NPUW_WHISPER", "YES"},
                          {"NPUW_WHISPER_EOS_TOKEN", "42"},
                          {"NPUW_DEVICES", "CPU"}};
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, plugin, properties, make_fake_factory());
    ov::npuw::WhisperInferRequest request(compiled);

    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, 2});
    std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), 1);
    const auto config = make_test_model_config<ov::test::npuw::WhisperConfig>();
    ov::Tensor encoder_hidden_states(
        ov::element::f32,
        ov::Shape{1, config.get_encoder_seq_len(), config.hidden_size});
    std::fill_n(encoder_hidden_states.data<float>(), encoder_hidden_states.get_size(), 0.0f);
    set_input(request, "input_ids", input_ids);
    set_input(request, "encoder_hidden_states", encoder_hidden_states);

    ASSERT_NO_THROW(request.infer());
    ASSERT_EQ(request.query_state().size(), 1u);
    auto stored_tokens_state = request.query_state().front();
    ASSERT_EQ(stored_tokens_state->get_name(), ov::npuw::StoredTokensState::state_name);
    ASSERT_EQ(stored_tokens_state->get_state()->data<int64_t>()[0], 2);

    stored_tokens_state->reset();
    ov::Tensor one_token(ov::element::i64, ov::Shape{1, 1});
    one_token.data<int64_t>()[0] = 1;
    set_input(request, "input_ids", one_token);

    ASSERT_NO_THROW(request.infer());
    EXPECT_EQ(stored_tokens_state->get_state()->data<int64_t>()[0], 1);
}