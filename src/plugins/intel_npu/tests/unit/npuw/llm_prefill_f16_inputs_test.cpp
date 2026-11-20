// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "infer_request_utils.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"

namespace {

class CpuSubInferRequest final : public ov::ISyncInferRequest {
public:
    CpuSubInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled, ov::InferRequest request)
        : ov::ISyncInferRequest(compiled),
          m_request(std::move(request)) {
        const auto& inputs = compiled->inputs();
        const auto& outputs = compiled->outputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            set_tensor(inputs[i], ov::get_tensor_impl(m_request.get_input_tensor(i)));
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            set_tensor(outputs[i], ov::get_tensor_impl(m_request.get_output_tensor(i)));
        }
    }

    void infer() override {
        const auto& inputs = get_compiled_model()->inputs();
        const auto& outputs = get_compiled_model()->outputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            m_request.set_input_tensor(i, ov::make_tensor(get_tensor(inputs[i])));
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            m_request.set_output_tensor(i, ov::make_tensor(get_tensor(outputs[i])));
        }
        m_request.infer();
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    ov::InferRequest m_request;
};

class CpuSubCompiledModel final : public ov::test::npuw::MockSubCompiledModel {
public:
    CpuSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                        const std::shared_ptr<const ov::IPlugin>& plugin,
                        const ov::AnyMap& props,
                        ov::Core& core)
        : MockSubCompiledModel(model, plugin, props),
          m_cpu(core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32))) {}

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<CpuSubInferRequest>(shared_from_this(), m_cpu.create_infer_request());
    }
    std::size_t num_submodels() const override {
        return 1;
    }

private:
    mutable ov::CompiledModel m_cpu;
};

class TestableLLMInferRequest final : public ov::npuw::LLMInferRequest {
public:
    using ov::npuw::LLMInferRequest::infer_prefill;
    using ov::npuw::LLMInferRequest::LLMInferRequest;
};

std::vector<std::vector<float>> run_prefill(ov::Core& core, bool f16ic, size_t chunk_size) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.use_inputs_embeds = true;
    ov::AnyMap props{{"NPUW_LLM", "YES"},
                     {"NPUW_LLM_SHARED_HEAD", "NO"},
                     {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                     {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                     {"NPUW_LLM_PREFILL_CHUNK_SIZE", std::to_string(chunk_size)},
                     {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                     {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"},
                     {"++NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"NPUW_F16IC", f16ic}}}};
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(
        ov::test::npuw::build_per_layer_inputs_probe_model(cfg),
        std::make_shared<ov::test::npuw::NullPlugin>(),
        props,
        [&core](const auto& model, const auto& plugin, const auto& config) {
            return std::make_shared<CpuSubCompiledModel>(model, plugin, config, core);
        });
    TestableLLMInferRequest request(compiled);
    const auto set = [&](const std::string& name, const ov::Tensor& tensor) {
        request.set_tensor(ov::npuw::util::find_port_by_name(compiled->inputs(), name).value(),
                           ov::get_tensor_impl(tensor));
    };
    std::vector<std::vector<float>> outputs;
    const auto step = [&](size_t input_len, size_t past_len, bool prefill) {
        SCOPED_TRACE(input_len);
        SCOPED_TRACE(past_len);
        ov::Tensor embeds(ov::element::f32, {1, input_len, cfg.hidden_size});
        for (size_t i = 0; i < embeds.get_size(); ++i) {
            embeds.data<float>()[i] = std::sin(static_cast<float>(past_len * cfg.hidden_size + i) * 0.13f);
        }
        ov::Tensor ple(ov::element::f32, {1, input_len, 1, 1});
        for (size_t i = 0; i < ple.get_size(); ++i) {
            ple.data<float>()[i] = 0.1234f + static_cast<float>(past_len + i) * 0.01f;
        }
        ov::Tensor mask(ov::element::i64, {1, past_len + input_len});
        std::fill_n(mask.data<int64_t>(), mask.get_size(), int64_t{1});
        ov::Tensor positions(ov::element::i64, {1, input_len});
        std::iota(positions.data<int64_t>(), positions.data<int64_t>() + input_len, static_cast<int64_t>(past_len));
        set("inputs_embeds", embeds);
        set("per_layer_inputs", ple);
        set("attention_mask", mask);
        set("position_ids", positions);
        if (prefill && input_len == 1) {
            request.infer_prefill(ov::get_tensor_impl(embeds),
                                  ov::get_tensor_impl(mask),
                                  ov::get_tensor_impl(positions),
                                  {},
                                  ov::get_tensor_impl(ple),
                                  {},
                                  {});
        } else {
            request.infer();
        }

        const auto ple_output = request.get_tensor(
            ov::npuw::util::find_port_by_name(compiled->outputs(), "per_layer_inputs_probe_output").value());
        const auto* ple_data = ple_output->data<float>() + ple_output->get_size() - input_len;
        for (size_t i = 0; i < input_len; ++i) {
            const float expected =
                prefill && f16ic ? static_cast<float>(ov::float16(ple.data<float>()[i])) : ple.data<float>()[i];
            EXPECT_FLOAT_EQ(ple_data[i], expected) << i;
        }
        const auto logits =
            request.get_tensor(ov::npuw::util::find_port_by_name(compiled->outputs(), "logits").value());
        outputs.emplace_back(logits->data<float>(), logits->data<float>() + logits->get_size());
    };
    for (const size_t prompt_len : {size_t{67}, size_t{64}, size_t{1}}) {
        step(prompt_len, 0, true);
        step(1, prompt_len, false);
    }
    return outputs;
}

TEST(LLMPrefillF16InputsTest, CpuLogitsMatchWithWholeAndChunkedPrefill) {
    ov::Core core;
    for (const size_t chunk_size : {size_t{0}, size_t{64}}) {
        SCOPED_TRACE(chunk_size);
        const auto expected = run_prefill(core, false, chunk_size);
        const auto actual = run_prefill(core, true, chunk_size);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t step = 0; step < actual.size(); ++step) {
            SCOPED_TRACE(step);
            ASSERT_EQ(actual[step].size(), expected[step].size());
            for (size_t i = 0; i < actual[step].size(); ++i) {
                ASSERT_NEAR(actual[step][i], expected[step][i], 1e-2f * (1.f + std::abs(expected[step][i]))) << i;
            }
        }
    }
}

}  // namespace
