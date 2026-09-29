// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/state_conversion_executor.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/usm_host_tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

class GpuOnlyHostTensor : public USMHostTensor {
public:
    using USMHostTensor::USMHostTensor;

    const void* data() const override {
        OPENVINO_THROW("CPU fallback was used for the GPU conversion test");
    }
};

class PaddedHostTensor : public USMHostTensor {
public:
    // Expose a 2x2 view over a 2x3 allocation to exercise row padding.
    explicit PaddedHostTensor(const std::shared_ptr<RemoteContextImpl>& context)
        : USMHostTensor(context, ov::element::f32, {2, 3}) {}

    const ov::Shape& get_shape() const override { return m_shape; }
    const ov::Strides& get_strides() const override { return m_strides; }

private:
    ov::Shape m_shape{2, 2};
    ov::Strides m_strides{3 * sizeof(float), sizeof(float)};
};

class GpuOnlyHostView : public ov::ITensor {
public:
    explicit GpuOnlyHostView(std::shared_ptr<ov::ITensor> tensor) : m_tensor(std::move(tensor)) {}

    void set_shape(ov::Shape shape) override { m_tensor->set_shape(std::move(shape)); }
    const ov::element::Type& get_element_type() const override { return m_tensor->get_element_type(); }
    const ov::Shape& get_shape() const override { return m_tensor->get_shape(); }
    const ov::Strides& get_strides() const override { return m_tensor->get_strides(); }
    void* data() override { return m_tensor->data(); }
    void* data_rw() override { return m_tensor->data_rw(); }
    void* data(const ov::element::Type& type) override { return m_tensor->data(type); }
    void* data_rw(const ov::element::Type& type) override { return m_tensor->data_rw(type); }

    const void* data() const override {
        OPENVINO_ASSERT(!m_const_data_accessed, "CPU fallback was used for the host conversion test");
        m_const_data_accessed = true;
        return static_cast<const ov::ITensor&>(*m_tensor).data();
    }

    const void* data(const ov::element::Type& type) const override {
        return static_cast<const ov::ITensor&>(*m_tensor).data(type);
    }

private:
    std::shared_ptr<ov::ITensor> m_tensor;
    mutable bool m_const_data_accessed = false;
};

cldnn::network::ptr make_state_network(const ov::Shape& shape,
                                       const ov::element::Type& src_type,
                                       const ov::element::Type& dst_type) {
    auto& engine = get_test_engine();
    const layout state_layout{shape, dst_type, format::bfyx};
    topology state_topology;
    state_topology.add(input_layout("input", state_layout));
    state_topology.add(read_value{"read_value", {input_info("input")}, "state", {state_layout}, src_type});
    state_topology.add(assign{"assign", {input_info("read_value")}, "state", state_layout});
    // Prepare conversion kernels before export; the import test performs its own round trip.
    return get_network(engine, state_topology, get_test_default_config(engine), get_test_stream_ptr(), false);
}

template <typename Src, typename Dst>
void check_state_values(VariableState& variable,
                        const std::shared_ptr<ov::ITensor>& source,
                        const std::vector<Src>& values,
                        cldnn::stream& stream,
                        const cldnn::memory::ptr& source_memory = nullptr) {
    if (source_memory)
        source_memory->copy_from(stream, values.data(), true);
    else
        std::copy(values.begin(), values.end(), static_cast<Src*>(source->data()));

    std::vector<Dst> expected(values.size());
    ov::reference::convert(values.data(), expected.data(), values.size());
    ASSERT_NO_THROW(variable.set_state(ov::SoPtr<ov::ITensor>(source)));
    ASSERT_NE(variable.get_memory(), nullptr);
    ASSERT_EQ(variable.get_layout().get_shape(), source->get_shape());
    cldnn::mem_lock<Dst, mem_lock_type::read> actual(variable.get_memory(), stream);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(static_cast<float>(expected[i])))
            ASSERT_TRUE(std::isnan(static_cast<float>(actual[i]))) << "element " << i;
        else
            ASSERT_EQ(actual[i], expected[i]) << "element " << i;
    }
}

}  // namespace

// IRemoteTensor::data() always throws, so if the fast-path `return` is ever dropped again and
// execution falls through to the fallback path, this call throws instead of silently passing.
TEST(convert_and_copy_test, remote_tensor_fast_path_does_not_fall_through) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    const ov::Shape shape{1, 2, 2, 2};
    const ov::element::Type et = ov::element::f32;

    auto src_remote = std::make_shared<RemoteTensorImpl>(context, shape, et);
    auto src_mem = src_remote->get_original_memory();
    std::vector<float> src_values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    set_values(src_mem, src_values);

    layout dst_layout{shape, et, format::bfyx};
    auto dst_mem = engine.allocate_memory(dst_layout);

    OV_ASSERT_NO_THROW(convert_and_copy(src_remote.get(), dst_mem, stream, dst_layout, false));

    cldnn::mem_lock<float, mem_lock_type::read> dst_ptr(dst_mem, stream);
    for (size_t i = 0; i < src_values.size(); ++i) {
        ASSERT_EQ(dst_ptr[i], src_values[i]);
    }
}

TEST(convert_and_copy_test, string_host_tensor_has_live_elements) {
    auto& engine = get_test_engine();
    if (!engine.use_unified_shared_memory()) {
        GTEST_SKIP() << "GPU host tensor does not use USM on this device";
    }
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto tensor = context->create_host_tensor(ov::element::string, ov::Shape{2});
    ASSERT_TRUE(tensor);
    ASSERT_FALSE(std::dynamic_pointer_cast<USMHostTensor>(tensor._ptr));
    EXPECT_EQ(tensor->get_shape(), (ov::Shape{2}));
    tensor->data<std::string>()[0] = std::string(64, 'A');
    tensor->data<std::string>()[1] = "short";
    EXPECT_EQ(tensor->data<std::string>()[0], std::string(64, 'A'));
    EXPECT_EQ(tensor->data<std::string>()[1], "short");
    tensor = {};
}

TEST(convert_and_copy_gpu_test, variable_state_reuses_program_kernels_for_sources_shapes_and_types) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    auto network = make_state_network({2, 6}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    const state_conversion_key key{data_types::f32, data_types::f16};
    std::vector<state_conversion_key> keys{key, key, {data_types::bf16, data_types::f16}};
    program->prepare_state_conversions(keys);
    auto executor = program->get_state_conversion_executor();
    ASSERT_NE(executor, nullptr);
    ASSERT_EQ(executor->get_keys().size(), keys.size() - 1);
    for (const auto& required_key : keys)
        ASSERT_TRUE(executor->has_kernel(required_key));
    program->prepare_state_conversions(keys);
    EXPECT_EQ(program->get_state_conversion_executor(), executor);
    EXPECT_THROW(program->prepare_state_conversions({{data_types::bf16, data_types::f16}}), ov::Exception);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    VariableState variable(network->get_variable_info("state"), context, network->get_shape_predictor(), program);

    auto host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{2, 6});
    std::vector<float> first_values{0.f, -0.f, 1.f, -1.f, 0.1f, -0.1f,
                                    65504.f, 1e-8f, 3.5f, -3.5f,
                                    std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    check_state_values<float, ov::float16>(variable, host, first_values, stream,
                                           host->get_impl()->get_original_memory());
    EXPECT_TRUE(variable.is_set());

    auto predictor = network->get_shape_predictor();
    std::weak_ptr<cldnn::program> program_weak = program;
    network.reset();
    program.reset();
    ASSERT_FALSE(program_weak.expired());

    auto larger_host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{3, 7});
    std::vector<float> larger_values(21);
    for (size_t i = 0; i < larger_values.size(); ++i)
        larger_values[i] = (static_cast<int>(i) - 10) / 4.f;
    check_state_values<float, ov::float16>(variable, larger_host, larger_values, stream,
                                           larger_host->get_impl()->get_original_memory());

    auto device = std::make_shared<RemoteTensorImpl>(context, ov::Shape{3, 7}, ov::element::f32,
                                                     TensorType::BT_USM_DEVICE_INTERNAL);
    auto device_memory = device->get_original_memory();
    ASSERT_EQ(device_memory->get_allocation_type(), allocation_type::usm_device);
    check_state_values<float, ov::float16>(variable, device, larger_values, stream, device_memory);
    EXPECT_EQ(program_weak.lock()->get_state_conversion_executor(), executor);

    auto retained_program = program_weak.lock();
    VariableStateInfo bf16_info{"bf16", layout{ov::Shape{2, 6}, ov::element::f16, format::bfyx},
                                static_cast<ov::element::Type_t>(ov::element::bf16)};
    VariableState bf16_state(bf16_info, context, predictor, retained_program);
    auto bf16_host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::bf16, ov::Shape{2, 6});
    const std::vector<ov::bfloat16> bf16_values{0.f, -0.f, 1.f, -1.f, 0.1f, -0.1f,
                                              65504.f, 1e-8f, 3.5f, -3.5f,
                                              std::numeric_limits<float>::infinity(),
                                              std::numeric_limits<float>::quiet_NaN()};
    check_state_values<ov::bfloat16, ov::float16>(bf16_state, bf16_host, bf16_values, stream,
                                                 bf16_host->get_impl()->get_original_memory());
}

TEST(convert_and_copy_gpu_test, variable_state_stages_aligned_and_unaligned_host_inputs) {
    auto& engine = get_test_engine();
    const auto alignment = static_cast<size_t>(engine.get_device_info().cacheline_size.value_or(0));
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_device) ||
        !engine.supports_allocation(allocation_type::usm_host) || alignment <= sizeof(float))
        GTEST_SKIP() << "OpenCL FP16, USM host/device allocations, and cache-line alignment are required";

    ASSERT_EQ(alignment % sizeof(float), 0);
    const ov::Shape aligned_shape{alignment / sizeof(float)};
    auto network = make_state_network(aligned_shape, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_state_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    VariableState variable(network->get_variable_info("state"), context, network->get_shape_predictor(), program);

    std::unique_ptr<void, decltype(&ov::util::aligned_free)> storage(
        ov::util::aligned_alloc(alignment, 2 * alignment), ov::util::aligned_free);
    ASSERT_NE(storage, nullptr);
    auto aligned_view = ov::make_tensor(ov::element::f32, aligned_shape, storage.get());
    auto aligned_source = std::make_shared<GpuOnlyHostView>(aligned_view);
    std::vector<float> aligned_values(ov::shape_size(aligned_shape));
    for (size_t i = 0; i < aligned_values.size(); ++i)
        aligned_values[i] = static_cast<float>(i) - 4.f;
    check_state_values<float, ov::float16>(variable, aligned_source, aligned_values, stream);

    auto* unaligned_ptr = static_cast<float*>(storage.get()) + 1;
    auto unaligned_view = ov::make_tensor(ov::element::f32, aligned_shape, unaligned_ptr);
    auto unaligned_source = std::make_shared<GpuOnlyHostView>(unaligned_view);
    std::vector<float> staged_values(ov::shape_size(aligned_shape), -0.5f);
    check_state_values<float, ov::float16>(variable, unaligned_source, staged_values, stream);

    const ov::Shape unaligned_size_shape{alignment / sizeof(float) + 1};
    auto unaligned_size_view = ov::make_tensor(ov::element::f32, unaligned_size_shape, storage.get());
    auto unaligned_size_source = std::make_shared<GpuOnlyHostView>(unaligned_size_view);
    std::vector<float> unaligned_size_values(ov::shape_size(unaligned_size_shape), 0.25f);
    check_state_values<float, ov::float16>(variable, unaligned_size_source, unaligned_size_values, stream);
}

TEST(convert_and_copy_gpu_test, variable_state_owns_staging_until_consumed_or_destroyed) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    const ov::Shape shape{2, 3};
    auto network = make_state_network(shape, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_state_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    const auto& info = network->get_variable_info("state");
    VariableState variable(info, context, network->get_shape_predictor(), program);

    auto source = ov::make_tensor(ov::element::f32, shape);
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    std::copy(values.begin(), values.end(), static_cast<float*>(source->data()));
    std::weak_ptr<ov::ITensor> pending_source = source;
    ASSERT_NO_THROW(variable.set_state(ov::SoPtr<ov::ITensor>(source)));
    // State must keep the original values when the caller reuses the input buffer.
    std::fill_n(static_cast<float*>(source->data()), values.size(), 0.f);
    source.reset();
    EXPECT_TRUE(pending_source.expired());

    auto result = variable.get_memory();
    EXPECT_TRUE(pending_source.expired());
    {
        cldnn::mem_lock<ov::float16, mem_lock_type::read> actual(result, stream);
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_EQ(actual[i], ov::float16(values[i]));
    }

    source = ov::make_tensor(ov::element::f32, shape);
    std::fill_n(static_cast<float*>(source->data()), values.size(), 0.25f);
    pending_source = source;
    ASSERT_NO_THROW(variable.set_state(ov::SoPtr<ov::ITensor>(source)));
    source.reset();
    EXPECT_TRUE(pending_source.expired());
    ASSERT_NO_THROW(variable.reset());
    EXPECT_TRUE(pending_source.expired());

    {
        VariableState temporary(info, context, network->get_shape_predictor(), program);
        source = ov::make_tensor(ov::element::f32, shape);
        std::fill_n(static_cast<float*>(source->data()), values.size(), -0.5f);
        pending_source = source;
        ASSERT_NO_THROW(temporary.set_state(ov::SoPtr<ov::ITensor>(source)));
        source.reset();
        EXPECT_TRUE(pending_source.expired());
    }
    EXPECT_TRUE(pending_source.expired());
}

TEST(convert_and_copy_gpu_test, variable_state_padded_roi_uses_gpu_without_packing) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM allocations are required";

    auto network = make_state_network({2, 2}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_state_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    for (bool transpose : {false, true}) {
        auto info = network->get_variable_info("state");
        info.transpose_required = transpose;
        VariableState variable(info, context, network->get_shape_predictor(), program);
        ov::Tensor parent(ov::element::f32, {2, 4});
        const std::vector<float> physical_values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        std::copy(physical_values.begin(), physical_values.end(), parent.data<float>());
        ov::Tensor view(parent, {0, 2}, {2, 4});
        ASSERT_FALSE(view.is_continuous());
        auto source = std::make_shared<GpuOnlyHostView>(get_tensor_impl(view)._ptr);
        // The wrapper rejects a second data() access, detecting a CPU fallback.
        ASSERT_NO_THROW(variable.set_state(ov::SoPtr<ov::ITensor>(source)));
        source.reset();
        view = {};
        parent = {};
        cldnn::mem_lock<ov::float16, mem_lock_type::read> result(variable.get_memory(), stream);
        const std::vector<float> expected = transpose ? std::vector<float>{3.f, 7.f, 4.f, 8.f}
                                                       : std::vector<float>{3.f, 4.f, 7.f, 8.f};
        for (size_t i = 0; i < expected.size(); ++i)
            ASSERT_EQ(result[i], ov::float16(expected[i]));
    }

    auto source = engine.allocate_memory(layout{ov::Shape{2, 2}, data_types::f32, format::bfyx},
                                         allocation_type::usm_host);
    auto destination = engine.allocate_memory(layout{ov::Shape{2, 2}, data_types::f16, format::bfyx},
                                              allocation_type::usm_device);
    const std::vector<float> values{1.f, 2.f, 3.f, 4.f};
    source->copy_from(stream, values.data(), true);
    state_conversion_input input;
    ASSERT_TRUE(state_conversion_executor::prepare_input(source->get_layout(), {8, 4}, false, input));
    auto executor = program->get_state_conversion_executor();
    const std::vector<std::vector<float>> expected_values{
        {-7.f, -7.f, -7.f, -7.f}, {1.f, 2.f, -7.f, -7.f}, {3.f, 4.f, -7.f, -7.f},
        {1.f, 3.f, -7.f, 2.f}};
    // Out-of-bounds reads and writes must leave the affected output untouched.
    for (int scenario = 0; scenario < 4; ++scenario) {
        auto invalid = input;
        if (scenario == 0) {
            invalid.source_offset = 4;
        } else if (scenario == 1) {
            invalid.padded = true;
            invalid.strides[4] = 4;
        } else if (scenario == 2) {
            invalid.source_offset = 2;
        } else if (scenario == 3) {
            invalid.transpose = true;
            invalid.dimensions[4] = 3;
        }
        const std::vector<ov::float16> sentinel(4, ov::float16(-7.f));
        destination->copy_from(stream, sentinel.data(), true);
        auto completion = executor->execute({data_types::f32, data_types::f16}, source, destination, stream, invalid);
        ASSERT_NE(completion, nullptr);
        completion->wait();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> result(destination, stream);
        for (size_t i = 0; i < sentinel.size(); ++i) {
            EXPECT_EQ(result[i], ov::float16(expected_values[scenario][i]))
                << "scenario " << scenario << ", element " << i;
        }
    }
    auto small_destination = engine.allocate_memory(layout{ov::Shape{2}, data_types::f16, format::bfyx},
                                                    allocation_type::usm_device);
    EXPECT_THROW(executor->execute({data_types::f32, data_types::f16}, source, small_destination, stream, input),
                 ov::Exception);
}

TEST(convert_and_copy_gpu_test, cpu_padded_roi_preserves_values_and_transpose) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    ov::Tensor parent(ov::element::f32, {2, 4});
    const std::vector<float> values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::copy(values.begin(), values.end(), parent.data<float>());
    ov::Tensor view(parent, {0, 2}, {2, 4});
    const layout source_layout(ov::Shape{2, 2}, data_types::f32, format::bfyx, padding({0, 0}, {0, 2}));
    for (bool transpose : {false, true}) {
        auto destination = engine.allocate_memory(layout{ov::Shape{2, 2}, data_types::f32, format::bfyx});
        ASSERT_NO_THROW(convert_and_copy(get_tensor_impl(view)._ptr.get(), destination, stream, source_layout, transpose));
        cldnn::mem_lock<float, mem_lock_type::read> result(destination, stream);
        const std::vector<float> expected = transpose ? std::vector<float>{3.f, 7.f, 4.f, 8.f}
                                                       : std::vector<float>{3.f, 4.f, 7.f, 8.f};
        for (size_t i = 0; i < expected.size(); ++i)
            ASSERT_EQ(result[i], expected[i]);
    }
}

TEST(convert_and_copy_gpu_test, variable_state_fallback_conditions) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    auto network = make_state_network({2, 3}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_state_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    const auto& info = network->get_variable_info("state");
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};

    VariableState without_program(info, context, network->get_shape_predictor());
    auto host = std::make_shared<USMHostTensor>(context, ov::element::f32, ov::Shape{2, 3});
    check_state_values<float, ov::float16>(without_program, host, values, stream,
                                           host->get_impl()->get_original_memory());

    VariableState wrong_type(info, context, network->get_shape_predictor(), program);
    auto bf16_host = std::make_shared<USMHostTensor>(context, ov::element::bf16, ov::Shape{2, 3});
    const std::vector<ov::bfloat16> bf16_values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    check_state_values<ov::bfloat16, ov::float16>(wrong_type, bf16_host, bf16_values, stream,
                                                  bf16_host->get_impl()->get_original_memory());

    auto same_type_info = info;
    same_type_info.m_user_specified_type = ov::element::f16;
    VariableState same_type(same_type_info, context, network->get_shape_predictor(), program);
    auto f16_host = std::make_shared<USMHostTensor>(context, ov::element::f16, ov::Shape{2, 3});
    const std::vector<ov::float16> f16_values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    check_state_values<ov::float16, ov::float16>(same_type, f16_host, f16_values, stream,
                                                 f16_host->get_impl()->get_original_memory());

    VariableStateInfo unsupported_info{"unsupported", layout{ov::Shape{2, 3}, ov::element::f32, format::bfyx},
                                       static_cast<ov::element::Type_t>(ov::element::f16)};
    VariableState unsupported(unsupported_info, context, network->get_shape_predictor(), program);
    check_state_values<ov::float16, float>(unsupported, f16_host, f16_values, stream,
                                           f16_host->get_impl()->get_original_memory());

    VariableState padded(info, context, network->get_shape_predictor(), program);
    auto padded_host = std::make_shared<PaddedHostTensor>(context);
    const std::vector<float> physical_values{1.f, 2.f, 99.f, 3.f, 4.f, 99.f};
    padded_host->get_impl()->get_original_memory()->copy_from(stream, physical_values.data(), true);
    ASSERT_NO_THROW(padded.set_state(ov::SoPtr<ov::ITensor>(padded_host)));
    ASSERT_NE(padded.get_memory(), nullptr);
    {
        cldnn::mem_lock<ov::float16, mem_lock_type::read> padded_result(padded.get_memory(), stream);
        const std::vector<float> logical_values{1.f, 2.f, 3.f, 4.f};
        for (size_t i = 0; i < logical_values.size(); ++i)
            ASSERT_EQ(padded_result[i], ov::float16(logical_values[i]));
    }

    auto transpose_info = info;
    transpose_info.transpose_required = true;
    VariableState transposed(transpose_info, context, network->get_shape_predictor(), program);
    auto transpose_host = std::make_shared<USMHostTensor>(context, ov::element::f32, ov::Shape{2, 3});
    transpose_host->get_impl()->get_original_memory()->copy_from(stream, values.data(), true);
    ASSERT_NO_THROW(transposed.set_state(ov::SoPtr<ov::ITensor>(transpose_host)));
    ASSERT_NE(transposed.get_memory(), nullptr);
    {
        cldnn::mem_lock<ov::float16, mem_lock_type::read> transpose_result(transposed.get_memory(), stream);
        const std::vector<float> transposed_values{1.f, -4.f, -2.f, 5.f, 3.f, -6.f};
        for (size_t i = 0; i < transposed_values.size(); ++i)
            ASSERT_EQ(transpose_result[i], ov::float16(transposed_values[i]));
    }

    VariableState empty(info, context, network->get_shape_predictor(), program);
    auto empty_tensor = ov::make_tensor(ov::element::f32, ov::Shape{0, 3});
    ASSERT_NO_THROW(empty.set_state(ov::SoPtr<ov::ITensor>(empty_tensor)));
    EXPECT_TRUE(empty.is_set());
    EXPECT_EQ(empty.get_memory(), nullptr);
}

TEST(convert_and_copy_gpu_test, variable_state_missing_prepared_kernel_falls_back_to_cpu) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    auto network = make_state_network({2, 3}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_state_conversions({});
    ASSERT_EQ(program->get_state_conversion_executor(), nullptr);
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    VariableState variable(network->get_variable_info("state"), context, network->get_shape_predictor(), program);
    auto source = ov::make_tensor(ov::element::f32, ov::Shape{2, 3});
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    auto& stream = context->get_engine().get_service_stream();
    check_state_values<float, ov::float16>(variable, source, values, stream);

    if (engine.supports_allocation(allocation_type::cl_mem)) {
        auto remote = std::make_shared<RemoteTensorImpl>(context, ov::Shape{2, 3}, ov::element::f32);
        ASSERT_EQ(remote->get_original_memory()->get_allocation_type(), allocation_type::cl_mem);
        try {
            variable.set_state(ov::SoPtr<ov::ITensor>(remote));
            FAIL() << "Expected an inaccessible remote source error";
        } catch (const ov::Exception& e) {
            EXPECT_NE(std::string(e.what()).find("CPU conversion cannot access a remote source tensor"),
                      std::string::npos);
        }
    }
}

TEST(convert_and_copy_gpu_test, imported_state_conversion_kernel_runs_without_recompilation) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    const state_conversion_key key{data_types::f32, data_types::f16};
    auto network = make_state_network({2, 3}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_state_conversions({key});
    std::stringstream blob;
    auto stream_ptr = get_test_stream_ptr();
    {
        BinaryOutputBuffer output(blob);
        output.set_stream(stream_ptr.get());
        program->save(output);
    }
    network.reset();
    program.reset();

    blob.seekg(0);
    BinaryInputBuffer input(blob, engine);
    auto restored = std::make_shared<cldnn::program>(engine, get_test_default_config(engine));
    restored->load(input);
    ASSERT_NE(restored->get_state_conversion_executor(), nullptr);
    ASSERT_TRUE(restored->get_state_conversion_executor()->has_kernel(key));
    auto imported_executor = restored->get_state_conversion_executor();
    ASSERT_NO_THROW(restored->prepare_state_conversions({key}));
    EXPECT_EQ(restored->get_state_conversion_executor(), imported_executor);
    EXPECT_THROW(restored->prepare_state_conversions({{data_types::bf16, data_types::f16}}), ov::Exception);
    auto restored_network = std::make_shared<cldnn::network>(restored, 0);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    VariableState variable(restored_network->get_variable_info("state"), context,
                           restored_network->get_shape_predictor(), restored);
    auto source = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{2, 3});
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    check_state_values<float, ov::float16>(variable, source, values, stream,
                                           source->get_impl()->get_original_memory());
}
