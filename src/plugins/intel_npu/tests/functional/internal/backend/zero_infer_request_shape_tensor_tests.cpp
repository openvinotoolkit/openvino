// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Verifies that ZeroInferRequest rejects a shape tensor whose element count does
// not match the rank of its related tensor.
//
// A shape tensor carries one dimension value per dimension of its related
// dynamic tensor, so its element count (N) must equal that tensor's rank (R).
// This test builds an inference request whose metadata violates that invariant
// (N != R) using a stub graph, skips real Level Zero pipeline creation, and
// checks that infer() throws instead of indexing the related shape out of range.
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

// A ZeroInferRequest that skips real Level Zero pipeline creation, so infer()
// reaches the host-side shape-tensor handling without a device pipeline.
class NoPipelineZeroInferRequest : public ::intel_npu::ZeroInferRequest {
public:
    NoPipelineZeroInferRequest(const std::shared_ptr<::intel_npu::ZeroInitStructsHolder>& initStructs,
                               const std::shared_ptr<const ::intel_npu::ICompiledModel>& compiledModel,
                               const ::intel_npu::Config& config)
        : ::intel_npu::ZeroInferRequest(initStructs, compiledModel, config) {}

protected:
    void create_pipeline_impl() override {}
};

// inputs[0] is the shape tensor (N elements) bound to inputs[1] (rank R), with
// N != R. The shape tensor comes first so it is the first input processed.
::intel_npu::NetworkMetadata make_mismatched_metadata(size_t shape_tensor_element_count, size_t related_rank) {
    ::intel_npu::NetworkMetadata metadata;
    metadata.name = "shape_tensor_mismatch_model";

    ::intel_npu::IODescriptor shape_tensor;
    shape_tensor.nameFromCompiler = "foo";
    shape_tensor.precision = ov::element::u32;
    shape_tensor.shapeFromCompiler = ov::PartialShape{static_cast<int64_t>(shape_tensor_element_count)};  // 1-D {N}
    shape_tensor.isShapeTensor = true;
    shape_tensor.relatedDescriptorIndex = 1;
    shape_tensor.indexUsedByDriver = 0;

    ::intel_npu::IODescriptor related;
    related.nameFromCompiler = "foo";
    related.precision = ov::element::f32;
    related.shapeFromCompiler = ov::PartialShape(ov::Shape(related_rank, 3));  // static rank-R shape
    related.isShapeTensor = false;
    related.relatedDescriptorIndex = 0;
    related.indexUsedByDriver = 1;

    metadata.inputs = {shape_tensor, related};

    ::intel_npu::IODescriptor output;
    output.nameFromCompiler = "output";
    output.precision = ov::element::f32;
    output.shapeFromCompiler = ov::PartialShape(ov::Shape(related_rank, 3));
    output.isShapeTensor = false;
    output.indexUsedByDriver = 0;

    metadata.outputs = {output};
    return metadata;
}

// A trivial model providing one input port and one output port for the request.
std::shared_ptr<ov::Model> make_host_model(size_t rank) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(rank, 3));
    param->set_friendly_name("foo");
    param->get_output_tensor(0).set_names({"foo"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    result->set_friendly_name("output");
    result->get_output_tensor(0).set_names({"output"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "shape_tensor_mismatch");
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

// A shape tensor must hold exactly one value per dimension of its related tensor.
// When its element count does not match that tensor's rank, prepare_inputs() must
// reject it and throw rather than index the related shape out of range.
TEST(ZeroInferRequestShapeTensorTests, ThrowsWhenShapeTensorCountDoesNotMatchRank) {
    constexpr size_t N = 1000;  // shape tensor element count
    constexpr size_t R = 2;     // related tensor rank  (N != R)

    ov::Core core;
    const auto devices = core.get_available_devices();
    const bool has_npu = std::any_of(devices.begin(), devices.end(), [](const std::string& d) {
        return d.rfind("NPU", 0) == 0;
    });
    if (!has_npu) {
        GTEST_SKIP() << "No NPU device available";
    }

    // The mocked init structs still talk to the real driver for host tensor
    // allocation; if that fails there is no usable NPU, so skip.
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    try {
        auto init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();
        init_struct = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(init_mock);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not initialize NPU driver structures: " << e.what();
    }

    auto config = make_config();
    auto model = make_host_model(R);
    auto graph = std::make_shared<StubGraph>(make_mismatched_metadata(N, R));

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

    auto request = std::make_shared<NoPipelineZeroInferRequest>(init_struct, compiled_model, config);

    EXPECT_THROW(request->infer(), ov::Exception);
}
