// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Verifies that ZeroInferRequest refuses to alias a state output onto a state
// input whose shared Level Zero buffer is too small for the output.
//
// A state input and its state output are two views of the same variable and
// share one buffer, sized from the input. If the output needs more bytes than
// that buffer holds, the device would write past it. This test builds a request
// whose metadata pairs a small state input with a large state output (via a stub
// graph) and checks that construction throws instead of aliasing the undersized
// buffer.
//
// The test needs a real NPU (the mocked init structs still use the driver for
// host tensor allocation) and self-skips when no NPU device is present.

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "common/zero_init_mock.hpp"
#include "compiled_model.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/openvino.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_infer_request.hpp"

namespace {

// A graph that returns caller-provided metadata and no real device handle.
class StubGraph : public ::intel_npu::IGraph {
public:
    explicit StubGraph(::intel_npu::NetworkMetadata metadata) : _metadata(std::move(metadata)) {}

    const ::intel_npu::NetworkMetadata& get_metadata() const override {
        return _metadata;
    }
    const std::optional<std::size_t> get_batch_size() const override {
        return std::nullopt;
    }
    std::optional<bool> is_profiling_blob() const override {
        return std::nullopt;
    }
    void* get_handle() const override {
        return nullptr;
    }
    std::optional<std::string_view> get_compatibility_descriptor() const override {
        return std::nullopt;
    }

private:
    ::intel_npu::NetworkMetadata _metadata;
};

// inputs[0] is a state input of shape {small}; outputs[0] is a state output of
// shape {large}; both name-matched and cross-linked. The shared buffer is sized
// from the input, so it is undersized for the output.
::intel_npu::NetworkMetadata make_state_metadata(size_t input_elems, size_t output_elems) {
    ::intel_npu::NetworkMetadata metadata;
    metadata.name = "state_tensor_mismatch_model";

    ::intel_npu::IODescriptor state_input;
    state_input.nameFromCompiler = "state";
    state_input.precision = ov::element::f32;
    state_input.shapeFromCompiler = ov::PartialShape{static_cast<int64_t>(input_elems)};
    state_input.isStateInput = true;
    state_input.relatedDescriptorIndex = 0;  // -> outputs[0]
    state_input.indexUsedByDriver = 0;

    ::intel_npu::IODescriptor state_output;
    state_output.nameFromCompiler = "state";
    state_output.precision = ov::element::f32;
    state_output.shapeFromCompiler = ov::PartialShape{static_cast<int64_t>(output_elems)};
    state_output.isStateOutput = true;
    state_output.relatedDescriptorIndex = 0;  // -> inputs[0]
    state_output.indexUsedByDriver = 1;

    metadata.inputs = {state_input};
    metadata.outputs = {state_output};
    return metadata;
}

// A trivial model providing one input port and one output port for the request.
std::shared_ptr<ov::Model> make_host_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    param->set_friendly_name("input");
    param->get_output_tensor(0).set_names({"input"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    result->set_friendly_name("output");
    result->get_output_tensor(0).set_names({"output"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "state_tensor_mismatch");
}

::intel_npu::FilteredConfig make_config() {
    auto options = std::make_shared<::intel_npu::OptionsDesc>();
    options->add<::intel_npu::LOG_LEVEL>();
    ::intel_npu::FilteredConfig config(options);
    config.enable(::intel_npu::LOG_LEVEL::key().data(), true);
    config.update({{::intel_npu::LOG_LEVEL::key().data(), "LOG_NONE"}});
    return config;
}

}  // namespace

// A state output must not be aliased onto a state input whose buffer is smaller
// than the output requires; ZeroInferRequest construction must throw instead.
TEST(ZeroInferRequestStateTensorTests, ThrowsWhenStateOutputAliasesUndersizedBuffer) {
    constexpr size_t INPUT_ELEMS = 1;         // small state input
    constexpr size_t OUTPUT_ELEMS = 1048576;  // large state output

    ov::Core core;
    const auto devices = core.get_available_devices();
    const bool has_npu = std::any_of(devices.begin(), devices.end(), [](const std::string& d) {
        return d.rfind("NPU", 0) == 0;
    });
    if (!has_npu) {
        GTEST_SKIP() << "No NPU device available";
    }

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    try {
        auto init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();
        init_struct = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(init_mock);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not initialize NPU driver structures: " << e.what();
    }

    auto config = make_config();
    auto model = make_host_model();
    auto graph = std::make_shared<StubGraph>(make_state_metadata(INPUT_ELEMS, OUTPUT_ELEMS));

    // The device is only dereferenced by CompiledModel::create_infer_request(),
    // which this test does not call (the request is constructed directly), so a
    // null device is sufficient here.
    auto compiled_model = std::make_shared<::intel_npu::CompiledModel>(
        model,
        std::make_shared<ov::test::utils::MockPlugin>(),  // only to satisfy the non-null plugin requirement
        nullptr,
        std::static_pointer_cast<::intel_npu::IGraph>(graph),
        config,
        std::nullopt);

    // The undersized-alias check runs in the ZeroInferRequest constructor.
    std::shared_ptr<::intel_npu::ZeroInferRequest> request;
    EXPECT_THROW(request = std::make_shared<::intel_npu::ZeroInferRequest>(init_struct, compiled_model, config),
                 ov::Exception);
}
