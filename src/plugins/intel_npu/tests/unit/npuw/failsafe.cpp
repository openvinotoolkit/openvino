// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "v1/elements/failsafe.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

constexpr const char* kCandidateProperty = "TEST_CANDIDATE_NAME";

std::shared_ptr<ov::Model> make_test_model(const std::vector<std::string>& input_names,
                                           const std::vector<std::string>& output_names) {
    OPENVINO_ASSERT(input_names.size() == output_names.size(),
                    "Failsafe test model requires matching input and output counts");

    ov::ParameterVector inputs;
    ov::ResultVector outputs;
    for (std::size_t idx = 0; idx < input_names.size(); ++idx) {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1});
        input->set_friendly_name(input_names[idx]);
        auto zero = ov::opset10::Constant::create(ov::element::f32, ov::Shape{1}, {0.f});
        auto add = std::make_shared<ov::opset10::Add>(input, zero);
        add->set_friendly_name(output_names[idx] + "_add");
        auto result = std::make_shared<ov::opset10::Result>(add);
        result->set_friendly_name(output_names[idx]);
        inputs.push_back(input);
        outputs.push_back(result);
    }

    return std::make_shared<ov::Model>(outputs, inputs, "FailsafeTestModel");
}

std::shared_ptr<ov::Model> make_test_model() {
    return make_test_model({"input"}, {"output"});
}

class NullPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }
};

struct CandidateState {
    std::string name;
    std::vector<std::string>* events = nullptr;
    int create_request_failures = 0;
    int infer_failures = 0;
    float output_bias = 0.f;
    std::vector<float> last_input_values;
    std::vector<float> last_output_values;
    std::vector<const void*> bound_input_ptrs;
};

class TestCompiledModel;

class TestInferRequest final : public ov::ISyncInferRequest {
public:
    TestInferRequest(std::shared_ptr<const TestCompiledModel> compiled_model, std::shared_ptr<CandidateState> state);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    bool is_input_port(const ov::Output<const ov::Node>& port) const {
        auto found = ov::ISyncInferRequest::find_port(port);
        return found.found() && !found.is_output();
    }

    std::shared_ptr<CandidateState> m_state;
};

class TestCompiledModel final : public ov::ICompiledModel {
public:
    TestCompiledModel(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      std::shared_ptr<CandidateState> state)
        : ov::ICompiledModel(model, plugin),
          m_model(model),
          m_state(std::move(state)) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        if (name == kCandidateProperty) {
            return m_state->name;
        }
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{m_state->name};
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        if (m_state->events != nullptr) {
            m_state->events->push_back("create-request:" + m_state->name);
        }
        if (m_state->create_request_failures-- > 0) {
            OPENVINO_THROW("create request failed for ", m_state->name);
        }

        auto self = std::static_pointer_cast<const TestCompiledModel>(shared_from_this());
        return std::make_shared<TestInferRequest>(std::move(self), m_state);
    }

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        get_task_executor(),
                                                        get_callback_executor());
    }

private:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<CandidateState> m_state;
};

TestInferRequest::TestInferRequest(std::shared_ptr<const TestCompiledModel> compiled_model, std::shared_ptr<CandidateState> state)
    : ov::ISyncInferRequest(std::move(compiled_model)),
      m_state(std::move(state)) {
    const auto& inputs = get_compiled_model()->inputs();
    for (const auto& input : inputs) {
        ov::ISyncInferRequest::set_tensor(input,
                                          ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    const auto& outputs = get_compiled_model()->outputs();
    for (const auto& output : outputs) {
        ov::ISyncInferRequest::set_tensor(output,
                                          ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void TestInferRequest::infer() {
    if (m_state->events != nullptr) {
        m_state->events->push_back("infer:" + m_state->name);
    }
    if (m_state->infer_failures-- > 0) {
        OPENVINO_THROW("infer failed for ", m_state->name);
    }

    const auto& inputs = get_compiled_model()->inputs();
    const auto& outputs = get_compiled_model()->outputs();
    OPENVINO_ASSERT(inputs.size() == outputs.size(), "Test infer request expects matching input and output counts");
    OPENVINO_ASSERT(!inputs.empty(), "Test infer request expects at least one input");

    m_state->last_input_values.clear();
    m_state->last_output_values.clear();
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
        const auto& input = ov::ISyncInferRequest::get_tensor(inputs[idx]);
        const auto& output = ov::ISyncInferRequest::get_tensor(outputs[idx]);
        const auto input_value = input->data<const float>()[0];
        const auto output_value = input_value + m_state->output_bias + static_cast<float>(idx);
        m_state->last_input_values.push_back(input_value);
        m_state->last_output_values.push_back(output_value);
        output->data<float>()[0] = output_value;
    }
}

ov::SoPtr<ov::ITensor> TestInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    return ov::ISyncInferRequest::get_tensor(port);
}

void TestInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    if (is_input_port(port)) {
        m_state->bound_input_ptrs.push_back(tensor->data());
    }
    ov::ISyncInferRequest::set_tensor(port, tensor);
}

ov::npuw::failsafe::CompiledModel::Factory make_factory(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    std::map<std::string, std::shared_ptr<CandidateState>> states,
    std::set<std::string> compile_failures = {}) {
    return [model, plugin, states = std::move(states), compile_failures = std::move(compile_failures)](
               const std::string& device) -> ov::SoPtr<ov::ICompiledModel> {
        const auto it = states.find(device);
        OPENVINO_ASSERT(it != states.end(), "Missing test state for device ", device);

        const auto& state = it->second;
        if (state->events != nullptr) {
            state->events->push_back("compile:" + state->name);
        }
        if (compile_failures.count(device) > 0) {
            OPENVINO_THROW("compile failed for ", state->name);
        }
        return {std::make_shared<TestCompiledModel>(model, plugin, state), {}};
    };
}

std::string error_message(const ov::Exception& ex) {
    return ex.what() == nullptr ? std::string{} : std::string(ex.what());
}

}  // namespace

TEST(FailsafeCompiledModelTest, CompileFailureFallsBackToLaterCandidate) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU", &events});
    auto second = std::make_shared<CandidateState>(CandidateState{"CPU", &events});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", first}, {"CPU", second}}, {"NPU"}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->active_device_name(), "CPU");
    EXPECT_EQ(compiled->get_property(kCandidateProperty).as<std::string>(), "CPU");
    EXPECT_EQ(events, (std::vector<std::string>{"compile:NPU", "compile:CPU"}));
}

TEST(FailsafeCompiledModelTest, CreateInferRequestFailureFallsBackToLaterCandidate) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU", &events, 1});
    auto second = std::make_shared<CandidateState>(CandidateState{"CPU", &events});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", first}, {"CPU", second}}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    auto request = compiled->create_sync_infer_request();

    ASSERT_NE(request, nullptr);
    EXPECT_EQ(compiled->active_device_name(), "CPU");
    EXPECT_EQ(events,
              (std::vector<std::string>{"compile:NPU", "create-request:NPU", "compile:CPU", "create-request:CPU"}));
}

TEST(FailsafeCompiledModelTest, InferenceFailureFallsBackAndRebindsTensors) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU", &events, 0, 1, 10.f});
    auto second = std::make_shared<CandidateState>(CandidateState{"CPU", &events, 0, 0, 20.f});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", first}, {"CPU", second}}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    auto request = compiled->create_sync_infer_request();
    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 3.f;
    request->set_tensor(model->inputs().front(), input);

    ASSERT_NO_THROW(request->infer());
    EXPECT_EQ(compiled->active_device_name(), "CPU");
    EXPECT_EQ(first->bound_input_ptrs, (std::vector<const void*>{input->data()}));
    EXPECT_EQ(second->bound_input_ptrs, (std::vector<const void*>{input->data()}));
    ASSERT_EQ(second->last_input_values.size(), 1u);
    ASSERT_EQ(second->last_output_values.size(), 1u);
    EXPECT_FLOAT_EQ(second->last_input_values.front(), 3.f);
    EXPECT_FLOAT_EQ(second->last_output_values.front(), 23.f);
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 23.f);
    EXPECT_EQ(events,
              (std::vector<std::string>{"compile:NPU",
                                        "create-request:NPU",
                                        "infer:NPU",
                                        "compile:CPU",
                                        "create-request:CPU",
                                        "infer:CPU"}));
}

TEST(FailsafeCompiledModelTest, UserProvidedOutputBufferReceivesFailoverResult) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU", &events, 0, 1, 10.f});
    auto second = std::make_shared<CandidateState>(CandidateState{"CPU", &events, 0, 0, 20.f});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", first}, {"CPU", second}}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    auto request = compiled->create_sync_infer_request();
    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    auto output = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 3.f;
    output->data<float>()[0] = -1.f;
    request->set_tensor(model->inputs().front(), input);
    request->set_tensor(model->outputs().front(), output);

    ASSERT_NO_THROW(request->infer());
    EXPECT_EQ(compiled->active_device_name(), "CPU");
    EXPECT_EQ(request->get_tensor(model->outputs().front())._ptr, output._ptr);
    EXPECT_FLOAT_EQ(output->data<const float>()[0], 23.f);
    EXPECT_EQ(events,
              (std::vector<std::string>{"compile:NPU",
                                        "create-request:NPU",
                                        "infer:NPU",
                                        "compile:CPU",
                                        "create-request:CPU",
                                        "infer:CPU"}));
}

TEST(FailsafeCompiledModelTest, ImportedInnerModelPortsAreRemappedByIndex) {
    auto model = make_test_model({"wrapper_tokens", "wrapper_style"}, {"wrapper_audio", "wrapper_duration"});
    auto imported_model = make_test_model({"cached_tokens", "cached_style"}, {"cached_audio", "cached_duration"});
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;
    constexpr float npu_output_bias = 10.f;
    constexpr float cpu_output_bias = 20.f;
    auto npu = std::make_shared<CandidateState>(CandidateState{"NPU", &events, 0, 0, npu_output_bias});
    auto cpu = std::make_shared<CandidateState>(CandidateState{"CPU", &events, 0, 0, cpu_output_bias});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(imported_model, plugin, {{"NPU", npu}, {"CPU", cpu}}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    auto request = compiled->create_sync_infer_request();

    auto first_input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    auto second_input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    auto second_output = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    constexpr float first_input_value = 3.f;
    constexpr float second_input_value = 7.f;
    constexpr float initial_output_value = -1.f;
    first_input->data<float>()[0] = first_input_value;
    second_input->data<float>()[0] = second_input_value;
    second_output->data<float>()[0] = initial_output_value;

    constexpr std::size_t first_port_index = 0u;
    constexpr std::size_t second_port_index = 1u;
    ASSERT_NO_THROW(request->set_tensor(model->inputs()[first_port_index], first_input));
    ASSERT_NO_THROW(request->set_tensor(model->inputs()[second_port_index], second_input));
    ASSERT_NO_THROW(request->set_tensor(model->outputs()[second_port_index], second_output));

    ASSERT_NO_THROW(request->infer());
    ASSERT_EQ(npu->bound_input_ptrs, (std::vector<const void*>{first_input->data(), second_input->data()}));
    ASSERT_EQ(npu->last_input_values.size(), 2u);
    ASSERT_EQ(npu->last_output_values.size(), 2u);
    EXPECT_FLOAT_EQ(npu->last_input_values[first_port_index], first_input_value);
    EXPECT_FLOAT_EQ(npu->last_input_values[second_port_index], second_input_value);

    const float first_expected_output = first_input_value + npu_output_bias + static_cast<float>(first_port_index);
    const float second_expected_output = second_input_value + npu_output_bias + static_cast<float>(second_port_index);
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs()[first_port_index])->data<const float>()[0], first_expected_output);
    EXPECT_EQ(request->get_tensor(model->outputs()[second_port_index])._ptr, second_output._ptr);
    EXPECT_FLOAT_EQ(second_output->data<const float>()[0], second_expected_output);
    EXPECT_EQ(events, (std::vector<std::string>{"compile:NPU", "create-request:NPU", "infer:NPU"}));
}

TEST(FailsafeCompiledModelTest, ChainedInferenceSurvivesUpstreamAndDownstreamFailover) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();

    std::vector<std::string> first_events;
    auto first_npu = std::make_shared<CandidateState>(CandidateState{"NPU", &first_events, 0, 1, 10.f});
    auto first_cpu = std::make_shared<CandidateState>(CandidateState{"CPU", &first_events, 0, 0, 20.f});
    auto first_so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", first_npu}, {"CPU", first_cpu}}));
    auto first_compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(first_so._ptr);

    std::vector<std::string> second_events;
    auto second_npu = std::make_shared<CandidateState>(CandidateState{"NPU", &second_events, 0, 1, 1.f});
    auto second_cpu = std::make_shared<CandidateState>(CandidateState{"CPU", &second_events, 0, 0, 2.f});
    auto second_so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", second_npu}, {"CPU", second_cpu}}));
    auto second_compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(second_so._ptr);

    ASSERT_NE(first_compiled, nullptr);
    ASSERT_NE(second_compiled, nullptr);
    auto first_request = first_compiled->create_sync_infer_request();
    auto second_request = second_compiled->create_sync_infer_request();

    auto source = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    source->data<float>()[0] = 3.f;
    first_request->set_tensor(model->inputs().front(), source);

    auto chained_tensor = first_request->get_tensor(model->outputs().front());
    second_request->set_tensor(model->inputs().front(), chained_tensor);

    ASSERT_NO_THROW(first_request->infer());
    EXPECT_EQ(first_compiled->active_device_name(), "CPU");
    ASSERT_EQ(first_cpu->last_input_values.size(), 1u);
    ASSERT_EQ(first_cpu->last_output_values.size(), 1u);
    EXPECT_FLOAT_EQ(first_cpu->last_input_values.front(), 3.f);
    EXPECT_FLOAT_EQ(first_cpu->last_output_values.front(), 23.f);
    EXPECT_EQ(first_request->get_tensor(model->outputs().front())._ptr, chained_tensor._ptr);
    EXPECT_FLOAT_EQ(chained_tensor->data<const float>()[0], 23.f);

    ASSERT_NO_THROW(second_request->infer());
    EXPECT_EQ(second_compiled->active_device_name(), "CPU");
    EXPECT_EQ(second_npu->bound_input_ptrs, (std::vector<const void*>{chained_tensor->data()}));
    EXPECT_EQ(second_cpu->bound_input_ptrs, (std::vector<const void*>{chained_tensor->data()}));
    ASSERT_EQ(second_cpu->last_input_values.size(), 1u);
    ASSERT_EQ(second_cpu->last_output_values.size(), 1u);
    EXPECT_FLOAT_EQ(second_cpu->last_input_values.front(), 23.f);
    EXPECT_FLOAT_EQ(second_cpu->last_output_values.front(), 25.f);
    EXPECT_FLOAT_EQ(second_request->get_tensor(model->outputs().front())->data<const float>()[0], 25.f);
}

TEST(FailsafeCompiledModelTest, CompileFailureWithoutFallbackPropagatesError) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU"});

    try {
        (void)ov::npuw::failsafe::CompiledModel::create(
            model,
            plugin,
            {"NPU"},
            make_factory(model, plugin, {{"NPU", first}}, {"NPU"}));
        FAIL() << "Expected compile failure";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(error_message(ex).find("compile failed for NPU"), std::string::npos);
    }
}

TEST(FailsafeCompiledModelTest, CreateInferRequestFailureWithoutFallbackPropagatesError) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU", nullptr, 1});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU"},
        make_factory(model, plugin, {{"NPU", first}}));
    auto compiled = so._ptr;
    ASSERT_NE(compiled, nullptr);

    try {
        (void)compiled->create_infer_request();
        FAIL() << "Expected infer-request creation failure";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(error_message(ex).find("create request failed for NPU"), std::string::npos);
    }
}

TEST(FailsafeCompiledModelTest, InferFailureWithoutFallbackPropagatesError) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto first = std::make_shared<CandidateState>(CandidateState{"NPU", nullptr, 0, 1});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU"},
        make_factory(model, plugin, {{"NPU", first}}));
    auto compiled = so._ptr;
    ASSERT_NE(compiled, nullptr);
    auto request = compiled->create_infer_request();

    try {
        request->infer();
        FAIL() << "Expected inference failure";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(error_message(ex).find("infer failed for NPU"), std::string::npos);
    }
}

TEST(FailsafeCompiledModelTest, SingleDeviceReturnsRawCompiledModel) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto first = std::make_shared<CandidateState>(CandidateState{"CPU"});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"CPU"},
        make_factory(model, plugin, {{"CPU", first}}));

    EXPECT_EQ(std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr), nullptr);
    ASSERT_NE(so._ptr, nullptr);
    EXPECT_EQ(so->get_property(kCandidateProperty).as<std::string>(), "CPU");
    auto devices = so->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
    EXPECT_EQ(devices, (std::vector<std::string>{"CPU"}));
}

TEST(FailsafeCompiledModelTest, ExecutionDevicesReturnsActiveDevice) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto npu = std::make_shared<CandidateState>(CandidateState{"NPU"});
    auto cpu = std::make_shared<CandidateState>(CandidateState{"CPU"});

    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", npu}, {"CPU", cpu}}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->active_device_name(), "NPU");
    auto devices = compiled->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
    EXPECT_EQ(devices, (std::vector<std::string>{"NPU"}));
}

TEST(FailsafeCompiledModelTest, ExecutionDevicesUpdatesAfterFailover) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto npu = std::make_shared<CandidateState>(CandidateState{"NPU"});
    auto cpu = std::make_shared<CandidateState>(CandidateState{"CPU"});

    // NPU compile fails → failsafe falls back to CPU at creation time.
    auto so = ov::npuw::failsafe::CompiledModel::create(
        model, plugin, {"NPU", "CPU"},
        make_factory(model, plugin, {{"NPU", npu}, {"CPU", cpu}}, {"NPU"}));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(so._ptr);

    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->active_device_name(), "CPU");
    auto devices = compiled->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
    EXPECT_EQ(devices, (std::vector<std::string>{"CPU"}));
}
