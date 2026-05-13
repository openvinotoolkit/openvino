// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "v1/elements/accuracy_checked.hpp"
#include "v1/elements/failsafe.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

// ---------------------------------------------------------------------------
// Shared test infrastructure (mirrored from failsafe.cpp)
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> make_test_model() {
    auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1});
    input->set_friendly_name("input");
    auto zero = ov::opset10::Constant::create(ov::element::f32, ov::Shape{1}, {0.f});
    auto add = std::make_shared<ov::opset10::Add>(input, zero);
    add->set_friendly_name("output_add");
    auto result = std::make_shared<ov::opset10::Result>(add);
    result->set_friendly_name("output");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "AccTestModel");
}

class NullPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override { return {}; }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override { return {}; }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override { return {}; }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override { return {}; }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override { return {}; }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override { return {}; }
};

// Compiled model that adds a fixed bias to its single input and writes the
// result to its single output.  Optionally tracks which events occurred.
struct ModelState {
    std::string name;
    std::vector<std::string>* events = nullptr;
    float output_bias = 0.f;
    float last_input_value = 0.f;
    float last_output_value = 0.f;
    // Optional glitch: from call number glitch_at_call onwards, output_bias is
    // replaced with glitch_bias.  Set glitch_at_call <= 0 to disable.
    int infer_count = 0;
    int glitch_at_call = 0;
    float glitch_bias = 0.f;
};

class TestCompiledModel;

class TestInferRequest final : public ov::ISyncInferRequest {
public:
    TestInferRequest(std::shared_ptr<const TestCompiledModel> cm, std::shared_ptr<ModelState> state);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return ov::ISyncInferRequest::get_tensor(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override { return {}; }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override { return {}; }

private:
    std::shared_ptr<ModelState> m_state;
};

class TestCompiledModel final : public ov::ICompiledModel {
public:
    TestCompiledModel(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      std::shared_ptr<ModelState> state)
        : ov::ICompiledModel(model, plugin), m_model(model), m_state(std::move(state)) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override { return m_model; }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{m_state->name};
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        if (m_state->events) {
            m_state->events->push_back("create-request:" + m_state->name);
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
    std::shared_ptr<ModelState> m_state;
};

TestInferRequest::TestInferRequest(std::shared_ptr<const TestCompiledModel> cm, std::shared_ptr<ModelState> state)
    : ov::ISyncInferRequest(cm), m_state(std::move(state)) {
    for (const auto& input : get_compiled_model()->inputs()) {
        ov::ISyncInferRequest::set_tensor(
            input, ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        ov::ISyncInferRequest::set_tensor(
            output, ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void TestInferRequest::infer() {
    if (m_state->events) {
        m_state->events->push_back("infer:" + m_state->name);
    }
    ++m_state->infer_count;
    const float bias = (m_state->glitch_at_call > 0 && m_state->infer_count >= m_state->glitch_at_call)
                           ? m_state->glitch_bias
                           : m_state->output_bias;
    const auto& input_port = get_compiled_model()->inputs().front();
    const auto& output_port = get_compiled_model()->outputs().front();
    const auto& in_tensor = ov::ISyncInferRequest::get_tensor(input_port);
    const auto& out_tensor = ov::ISyncInferRequest::get_tensor(output_port);
    m_state->last_input_value = in_tensor->data<const float>()[0];
    m_state->last_output_value = m_state->last_input_value + bias;
    out_tensor->data<float>()[0] = m_state->last_output_value;
}

// Helper to build an ov::SoPtr<ov::ICompiledModel> backed by a TestCompiledModel.
ov::SoPtr<ov::ICompiledModel> make_test_compiled_model(const std::shared_ptr<ov::Model>& model,
                                                        const std::shared_ptr<const ov::IPlugin>& plugin,
                                                        std::shared_ptr<ModelState> state) {
    return {std::make_shared<TestCompiledModel>(model, plugin, std::move(state)), {}};
}

// A checker that passes when |actual - reference| <= threshold for a scalar f32 tensor.
ov::npuw::accuracy_checked::CompiledModel::Checker make_threshold_checker(float threshold) {
    return [threshold](const ov::SoPtr<ov::ITensor>& actual,
                       const ov::SoPtr<ov::ITensor>& reference) -> bool {
        const float a = actual->data<const float>()[0];
        const float r = reference->data<const float>()[0];
        return std::abs(a - r) <= threshold;
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(AccuracyCheckedCompiledModelTest, NullRefReturnsMainUnwrapped) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    auto main_state = std::make_shared<ModelState>(ModelState{"main", nullptr, 0.f});
    auto main_cm = make_test_compiled_model(model, plugin, main_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(model, plugin, main_cm, {}, make_threshold_checker(0.f));

    // When ref is null create() must return the unwrapped main model.
    EXPECT_EQ(std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr), nullptr);
    ASSERT_NE(so._ptr, nullptr);
}

TEST(AccuracyCheckedCompiledModelTest, AccurateInferencePassesThrough) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;

    auto main_state = std::make_shared<ModelState>(ModelState{"main", &events, 10.f});
    auto ref_state  = std::make_shared<ModelState>(ModelState{"ref",  &events, 10.f});  // same bias → same output

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.1f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    ASSERT_NE(compiled, nullptr);

    auto request = compiled->create_sync_infer_request();
    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 5.f;
    request->set_tensor(model->inputs().front(), input);

    ASSERT_NO_THROW(request->infer());

    // Both main and ref should have been inferred.
    EXPECT_EQ(main_state->last_input_value, 5.f);
    EXPECT_EQ(ref_state->last_input_value, 5.f);
    // Output = input + bias = 5 + 10 = 15.
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 15.f);

    // Model should NOT have switched.
    EXPECT_FALSE(compiled->has_switched_to_reference());
}

TEST(AccuracyCheckedCompiledModelTest, InaccurateInferenceSwitchesToReference) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;

    // main bias=10, ref bias=11 → difference=1 > threshold=0.5 → fail
    auto main_state = std::make_shared<ModelState>(ModelState{"main", &events, 10.f});
    auto ref_state  = std::make_shared<ModelState>(ModelState{"ref",  &events, 11.f});

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.5f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    ASSERT_NE(compiled, nullptr);

    auto request = compiled->create_sync_infer_request();
    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 3.f;
    request->set_tensor(model->inputs().front(), input);

    ASSERT_NO_THROW(request->infer());

    // Output should be the reference value (3 + 11 = 14), copied into the main output buffer.
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 14.f);
    EXPECT_TRUE(compiled->has_switched_to_reference());
}

TEST(AccuracyCheckedCompiledModelTest, PermanentSwitchSkipsMainOnSubsequentInfers) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;

    auto main_state = std::make_shared<ModelState>(ModelState{"main", &events, 10.f});
    auto ref_state  = std::make_shared<ModelState>(ModelState{"ref",  &events, 11.f});

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.5f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    auto request = compiled->create_sync_infer_request();

    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 1.f;
    request->set_tensor(model->inputs().front(), input);

    // First infer triggers the switch.
    ASSERT_NO_THROW(request->infer());
    ASSERT_TRUE(compiled->has_switched_to_reference());

    events.clear();  // Reset event log.

    input->data<float>()[0] = 2.f;
    ASSERT_NO_THROW(request->infer());

    // After permanent switch the main model must NOT be invoked.
    for (const auto& ev : events) {
        EXPECT_NE(ev, "infer:main") << "main should not be inferred after switch";
    }
    // Reference should have been inferred with the new input.
    EXPECT_FLOAT_EQ(ref_state->last_input_value, 2.f);
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 2.f + 11.f);
}

TEST(AccuracyCheckedCompiledModelTest, UserOutputBufferReceivesReferenceValueOnSwitch) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();

    auto main_state = std::make_shared<ModelState>(ModelState{"main", nullptr, 10.f});
    auto ref_state  = std::make_shared<ModelState>(ModelState{"ref",  nullptr, 20.f});

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.5f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    auto request = compiled->create_sync_infer_request();

    auto input  = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    auto output = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0]  = 4.f;
    output->data<float>()[0] = -1.f;  // sentinel
    request->set_tensor(model->inputs().front(), input);
    request->set_tensor(model->outputs().front(), output);

    ASSERT_NO_THROW(request->infer());

    // The user-provided output tensor must hold the reference result (4 + 20 = 24).
    EXPECT_FLOAT_EQ(output->data<const float>()[0], 24.f);
    // And get_tensor() must return the same object.
    EXPECT_EQ(request->get_tensor(model->outputs().front())._ptr, output._ptr);
}

TEST(AccuracyCheckedCompiledModelTest, NewRequestFromSwitchedModelStartsOnReference) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;

    auto main_state = std::make_shared<ModelState>(ModelState{"main", &events, 10.f});
    auto ref_state  = std::make_shared<ModelState>(ModelState{"ref",  &events, 20.f});

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.5f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);

    // Trigger switch via first request.
    {
        auto req1 = compiled->create_sync_infer_request();
        auto in1 = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
        in1->data<float>()[0] = 1.f;
        req1->set_tensor(model->inputs().front(), in1);
        req1->infer();
        ASSERT_TRUE(compiled->has_switched_to_reference());
    }

    events.clear();

    // A second request created after the switch should use reference directly.
    auto req2 = compiled->create_sync_infer_request();
    auto in2 = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    in2->data<float>()[0] = 5.f;
    req2->set_tensor(model->inputs().front(), in2);
    ASSERT_NO_THROW(req2->infer());

    for (const auto& ev : events) {
        EXPECT_NE(ev, "infer:main") << "new request after switch must not use main";
    }
    EXPECT_FLOAT_EQ(ref_state->last_input_value, 5.f);
    EXPECT_FLOAT_EQ(req2->get_tensor(model->outputs().front())->data<const float>()[0], 5.f + 20.f);
}

TEST(AccuracyCheckedCompiledModelTest, ChainedWithFailsafeModel) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;

    // Failsafe inner layer: NPU fails, falls back to CPU.
    auto npu_state = std::make_shared<ModelState>(ModelState{"NPU", &events, 10.f});
    auto cpu_state = std::make_shared<ModelState>(ModelState{"CPU", &events, 10.f});

    auto failsafe_factory = [&](const std::string& device) -> ov::SoPtr<ov::ICompiledModel> {
        if (device == "NPU") {
            OPENVINO_THROW("NPU not available");
        }
        if (device == "CPU") {
            return make_test_compiled_model(model, plugin, cpu_state);
        }
        OPENVINO_THROW("Unknown device: ", device);
    };

    auto failsafe_cm = ov::npuw::failsafe::CompiledModel::create(model, plugin, {"NPU", "CPU"}, failsafe_factory);

    // Reference is a separate CPU model with a different bias (simulate inaccuracy).
    auto ref_state = std::make_shared<ModelState>(ModelState{"ref_cpu", &events, 20.f});
    auto ref_cm = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, failsafe_cm, ref_cm, make_threshold_checker(0.5f));
    auto acc_compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    ASSERT_NE(acc_compiled, nullptr);

    auto request = acc_compiled->create_sync_infer_request();
    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 2.f;
    request->set_tensor(model->inputs().front(), input);

    // Failsafe uses CPU (bias=10), ref uses ref_cpu (bias=20). diff=10 > 0.5 → switch.
    ASSERT_NO_THROW(request->infer());

    EXPECT_TRUE(acc_compiled->has_switched_to_reference());
    // Output should be ref result: 2 + 20 = 22.
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 22.f);
}

TEST(AccuracyCheckedCompiledModelTest, ExecutionDevicesReflectsActiveModel) {
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();

    auto main_state = std::make_shared<ModelState>(ModelState{"NPU", nullptr, 10.f});
    auto ref_state  = std::make_shared<ModelState>(ModelState{"CPU", nullptr, 11.f});

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.5f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    ASSERT_NE(compiled, nullptr);

    // Before switch: reports main device.
    auto devs_before = compiled->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
    EXPECT_EQ(devs_before, (std::vector<std::string>{"NPU"}));

    // Trigger switch.
    auto request = compiled->create_sync_infer_request();
    auto input = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    input->data<float>()[0] = 1.f;
    request->set_tensor(model->inputs().front(), input);
    request->infer();
    ASSERT_TRUE(compiled->has_switched_to_reference());

    // After switch: reports reference device.
    auto devs_after = compiled->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
    EXPECT_EQ(devs_after, (std::vector<std::string>{"CPU"}));
}

TEST(AccuracyCheckedCompiledModelTest, RepeatingBlockAccuracyFailsAtThirdCall) {
    // In NPUW a function body (repeating block) is compiled once and its infer
    // request pair (main + ref) is *reused* for every call-site invocation
    // within a single forward pass.  The AccuracyChecked::InferRequest wraps
    // that pair and is called once per instance.  The shared
    // m_switched_to_reference flag ensures that once any instance detects an
    // accuracy failure, all subsequent invocations automatically use reference.
    //
    // Scenario: main is accurate on calls 1 and 2, but "glitches" from call 3
    // onwards (output bias shifts by 1).  Reference is always stable.
    // Threshold = 0.5, so |glitch| = 1 triggers the permanent switch on call 3.
    auto model = make_test_model();
    auto plugin = std::make_shared<NullPlugin>();
    std::vector<std::string> events;

    auto main_state = std::make_shared<ModelState>(ModelState{"main", &events, 10.f});
    main_state->glitch_at_call = 3;
    main_state->glitch_bias    = 11.f;  // diverges from ref by 1 on call 3+

    auto ref_state = std::make_shared<ModelState>(ModelState{"ref", &events, 10.f});

    auto main_cm = make_test_compiled_model(model, plugin, main_state);
    auto ref_cm  = make_test_compiled_model(model, plugin, ref_state);

    auto so = ov::npuw::accuracy_checked::CompiledModel::create(
        model, plugin, main_cm, ref_cm, make_threshold_checker(0.5f));
    auto compiled = std::dynamic_pointer_cast<ov::npuw::accuracy_checked::CompiledModel>(so._ptr);
    ASSERT_NE(compiled, nullptr);

    // One AccuracyChecked::InferRequest reused for all N instances of the block.
    auto request = compiled->create_sync_infer_request();
    auto input   = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1}));
    request->set_tensor(model->inputs().front(), input);

    // -- Forward pass 1, instance 1: main call #1, bias=10. Accurate. ---------
    input->data<float>()[0] = 1.f;
    ASSERT_NO_THROW(request->infer());
    EXPECT_FALSE(compiled->has_switched_to_reference());
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 11.f);  // 1+10

    // -- Forward pass 1, instance 2: main call #2, bias=10. Accurate. ---------
    input->data<float>()[0] = 2.f;
    ASSERT_NO_THROW(request->infer());
    EXPECT_FALSE(compiled->has_switched_to_reference());
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 12.f);  // 2+10

    // -- Forward pass 1, instance 3: main call #3, bias glitches to 11.
    //    main=3+11=14, ref=3+10=13, |14-13|=1 > 0.5 → permanent switch. ------
    input->data<float>()[0] = 3.f;
    ASSERT_NO_THROW(request->infer());
    EXPECT_TRUE(compiled->has_switched_to_reference());
    // Output must be the reference result (3+10=13), not the glitched main (14).
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 13.f);

    // -- Forward pass 2: all instances use reference directly. -----------------
    events.clear();
    input->data<float>()[0] = 10.f;
    ASSERT_NO_THROW(request->infer());  // instance 1
    input->data<float>()[0] = 20.f;
    ASSERT_NO_THROW(request->infer());  // instance 2
    input->data<float>()[0] = 30.f;
    ASSERT_NO_THROW(request->infer());  // instance 3

    for (const auto& ev : events) {
        EXPECT_NE(ev, "infer:main") << "main must not be invoked after permanent reference switch";
    }
    // Last output via reference: 30+10=40.
    EXPECT_FLOAT_EQ(request->get_tensor(model->outputs().front())->data<const float>()[0], 40.f);
}
