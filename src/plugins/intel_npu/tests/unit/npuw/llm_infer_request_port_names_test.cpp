// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for the port-name registration logic in ov::npuw::LLMInferRequest.
//
// A single compiled-model port may expose several tensor names (aliases).
// LLMInferRequest must register every one of those names in its port maps and
// LoRA state list, because later lookups derive keys from *other* requests'
// port names (e.g. kv-cache copy maps "past_key_values.*" -> "present.*") and
// those keys may not coincide with the arbitrary alias that get_any_name()
// happens to return. Relying on get_any_name() therefore silently drops a port
// or registers a non-matching name.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_lora_states.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace ov::test::npuw {

struct LLMPortNameRegistrationTestAccess {
    using PortsMap = ov::npuw::LLMInferRequest::PortsMap;

    static const PortsMap& prefill_in_ports(const ov::npuw::LLMInferRequest& req) {
        return req.m_prefill_in_ports;
    }
    static const PortsMap& prefill_out_ports(const ov::npuw::LLMInferRequest& req) {
        return req.m_prefill_out_ports;
    }
    static const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request(const ov::npuw::LLMInferRequest& req) {
        return req.m_prefill_request;
    }
    static const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests(
        const ov::npuw::LLMInferRequest& req) {
        return req.m_generate_requests;
    }
    static const PortsMap& generate_variant_in_ports(const ov::npuw::LLMInferRequest& req,
                                                     const std::shared_ptr<ov::IAsyncInferRequest>& gen_req) {
        return req.m_generate_variant_in_ports.at(gen_req);
    }
    static const PortsMap& generate_variant_out_ports(const ov::npuw::LLMInferRequest& req,
                                                      const std::shared_ptr<ov::IAsyncInferRequest>& gen_req) {
        return req.m_generate_variant_out_ports.at(gen_req);
    }
    static const std::vector<ov::SoPtr<ov::IVariableState>>& variable_states(const ov::npuw::LLMInferRequest& req) {
        return req.m_variableStates;
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::LLMPortNameRegistrationTestAccess;
using ov::test::npuw::NullPlugin;
class FakeSubCompiledModel;

// LoRA-matching alias injected onto a prefill input port whose canonical name is
// NOT a LoRA name (input_ids). Matches ov::npuw::util::matchLoRAMatMulAString,
// i.e. the regex "^lora_state.*MatMul\.A$".
constexpr const char* kInjectedLoraName = "lora_state.injected.MatMul.A";
// Generic alias suffix appended to every port so that get_names() returns >1 name.
constexpr const char* kAliasSuffix = "__npuw_test_alias";
constexpr const char* kHiddenStatesName = "hidden_states";

std::shared_ptr<ov::Model> build_llm_test_model_with_hidden_states() {
    auto model = build_llm_test_model();
    const auto& results = model->get_results();
    const auto logits_result = std::find_if(results.begin(), results.end(), [](const auto& result) {
        return result->output(0).get_names().count("logits") != 0u;
    });
    OPENVINO_ASSERT(logits_result != results.end(), "Synthetic LLM must expose logits");

    const auto lm_head = ov::as_type_ptr<ov::op::v0::MatMul>((*logits_result)->input_value(0).get_node_shared_ptr());
    OPENVINO_ASSERT(lm_head, "Synthetic LLM logits must be produced by the LM-head MatMul");

    auto hidden_states = std::make_shared<ov::op::v0::Result>(lm_head->input_value(0));
    hidden_states->set_friendly_name(kHiddenStatesName);
    hidden_states->output(0).set_names({kHiddenStatesName});
    model->add_results({hidden_states});
    model->validate_nodes_and_infer_types();
    return model;
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
                                                        intel_npu::make_executor("port_names_task", 1),
                                                        intel_npu::make_executor("port_names_callback", 1));
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

// Add a second unique tensor name to every sub-model port so that get_names()
// returns more than one alias.
class AliasInjectingFactory {
public:
    explicit AliasInjectingFactory(bool inject_lora_alias) : m_inject_lora_alias(inject_lora_alias) {}

    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        const bool inject_lora_alias = m_inject_lora_alias;
        return [inject_lora_alias](const std::shared_ptr<ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            // Give every input/output port a second, unique alias name.
            for (auto&& port : model->inputs()) {
                port.add_names({port.get_any_name() + kAliasSuffix});
            }
            for (auto&& port : model->outputs()) {
                port.add_names({port.get_any_name() + kAliasSuffix});
            }
            // Tag input_ids with a LoRA-matching alias. This is not a name that
            // get_any_name() would report, so only a full get_names() scan finds it.
            if (inject_lora_alias) {
                for (auto&& port : model->inputs()) {
                    if (port.get_names().count("input_ids") != 0) {
                        port.add_names({kInjectedLoraName});
                        break;
                    }
                }
            }
            return std::make_shared<FakeSubCompiledModel>(model, plugin, props);
        };
    }

private:
    bool m_inject_lora_alias;
};

class LLMInferRequestPortNamesTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"},
                {"NPUW_DEVICES", "CPU"},
                {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    }

    std::shared_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(AliasInjectingFactory& factory) const {
        return std::make_shared<ov::npuw::LLMCompiledModel>(build_llm_test_model(),
                                                            m_plugin,
                                                            base_props(),
                                                            factory.make_factory());
    }

    std::shared_ptr<ov::npuw::LLMCompiledModel> create_shared_head_compiled_model(AliasInjectingFactory& factory) const {
        auto props = base_props();
        props["NPUW_LLM_SHARED_HEAD"] = "YES";
        return std::make_shared<ov::npuw::LLMCompiledModel>(build_llm_test_model_with_hidden_states(),
                                                            m_plugin,
                                                            std::move(props),
                                                            factory.make_factory());
    }

    // Asserts that every tensor name of every port is registered as a key in `map`
    // and that it maps back to that very port. Returns the maximum number of names
    // seen on any single port so the caller can confirm the multi-alias scenario
    // was actually exercised.
    static std::size_t expect_all_names_registered(const std::vector<ov::Output<const ov::Node>>& ports,
                                                   const LLMPortNameRegistrationTestAccess::PortsMap& map) {
        std::size_t max_names = 0;
        for (const auto& port : ports) {
            const auto& names = port.get_names();
            max_names = std::max(max_names, names.size());
            for (const auto& name : names) {
                const auto it = map.find(name);
                EXPECT_NE(it, map.end()) << "Tensor name '" << name << "' was not registered in the port map";
                if (it == map.end()) {
                    continue;
                }
                EXPECT_EQ(it->second.get_node(), port.get_node())
                    << "Tensor name '" << name << "' is registered to a different port";
            }
        }
        return max_names;
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(LLMInferRequestPortNamesTest, PrefillPortsRegisterEveryTensorName) {
    AliasInjectingFactory factory(/*inject_lora_alias=*/false);
    auto compiled = create_compiled_model(factory);
    ASSERT_NE(compiled, nullptr);

    ov::npuw::LLMInferRequest req(compiled);

    const auto& prefill = LLMPortNameRegistrationTestAccess::prefill_request(req);
    ASSERT_NE(prefill, nullptr);

    const std::size_t max_in_names =
        expect_all_names_registered(prefill->get_compiled_model()->inputs(),
                                    LLMPortNameRegistrationTestAccess::prefill_in_ports(req));
    const std::size_t max_out_names =
        expect_all_names_registered(prefill->get_compiled_model()->outputs(),
                                    LLMPortNameRegistrationTestAccess::prefill_out_ports(req));

    EXPECT_GT(max_in_names, 1u);
    EXPECT_GT(max_out_names, 1u);
}

TEST_F(LLMInferRequestPortNamesTest, GenerateVariantPortsRegisterEveryTensorName) {
    AliasInjectingFactory factory(/*inject_lora_alias=*/false);
    auto compiled = create_compiled_model(factory);
    ASSERT_NE(compiled, nullptr);

    ov::npuw::LLMInferRequest req(compiled);

    const auto& generate_requests = LLMPortNameRegistrationTestAccess::generate_requests(req);
    ASSERT_FALSE(generate_requests.empty());

    std::size_t max_in_names = 0;
    std::size_t max_out_names = 0;
    for (const auto& gen_req : generate_requests) {
        ASSERT_NE(gen_req, nullptr);
        max_in_names = std::max(
            max_in_names,
            expect_all_names_registered(gen_req->get_compiled_model()->inputs(),
                                        LLMPortNameRegistrationTestAccess::generate_variant_in_ports(req, gen_req)));
        max_out_names = std::max(
            max_out_names,
            expect_all_names_registered(gen_req->get_compiled_model()->outputs(),
                                        LLMPortNameRegistrationTestAccess::generate_variant_out_ports(req, gen_req)));
    }

    EXPECT_GT(max_in_names, 1u);
    EXPECT_GT(max_out_names, 1u);
}

TEST_F(LLMInferRequestPortNamesTest, LoraStateUsesMatchingTensorNameNotAnyName) {
    AliasInjectingFactory factory(/*inject_lora_alias=*/true);
    auto compiled = create_compiled_model(factory);
    ASSERT_NE(compiled, nullptr);

    ov::npuw::LLMInferRequest req(compiled);

    ASSERT_TRUE(ov::npuw::util::matchLoRAMatMulAString(kInjectedLoraName));

    const auto& states = LLMPortNameRegistrationTestAccess::variable_states(req);
    bool found = false;
    for (const auto& state : states) {
        if (state->get_name() == kInjectedLoraName) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "No VariableState was registered under the LoRA-matching alias '" << kInjectedLoraName << "'";
}

TEST_F(LLMInferRequestPortNamesTest, SharedHeadBindsAdditionalHiddenStateOutput) {
    AliasInjectingFactory factory(/*inject_lora_alias=*/false);
    auto compiled = create_shared_head_compiled_model(factory);
    ASSERT_NE(compiled, nullptr);

    ov::npuw::LLMInferRequest req(compiled);

    const auto& prefill = LLMPortNameRegistrationTestAccess::prefill_request(req);
    ASSERT_NE(prefill, nullptr);
    const auto passthrough_output =
        std::find_if(prefill->get_compiled_model()->outputs().begin(),
                     prefill->get_compiled_model()->outputs().end(),
                     [](const auto& output) {
                         return output.get_names().count(ov::npuw::LLMCompiledModel::shared_head_passthrough_output) !=
                                0u;
                     });
    ASSERT_NE(passthrough_output, prefill->get_compiled_model()->outputs().end());

    const auto hidden_states_output =
        std::find_if(compiled->outputs().begin(), compiled->outputs().end(), [](const auto& output) {
            return output.get_names().count(kHiddenStatesName) != 0u;
        });
    ASSERT_NE(hidden_states_output, compiled->outputs().end());

    // This tensor is bound in LLMInferRequest's shared-head setup. Without the passthrough
    // alias scan and binding, the public hidden-states output remains an empty tensor.
    const auto hidden_states = req.get_tensor(*hidden_states_output);
    ASSERT_NE(hidden_states, nullptr);
    EXPECT_EQ(hidden_states->get_shape(), (ov::Shape{1u, 1u, 64u}));
}

}  // namespace
