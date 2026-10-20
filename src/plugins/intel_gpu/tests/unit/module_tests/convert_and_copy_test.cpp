// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/tensor_conversion.hpp"
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
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

class ControlledConversionEvent : public cldnn::event {
public:
    bool fail_wait = false;
    size_t wait_calls = 0;

protected:
    void wait_impl() override {
        ++wait_calls;
        if (fail_wait)
            OPENVINO_THROW("Injected conversion wait failure");
    }
    void set_impl() override {}
    bool is_set_impl() override { return _set; }
};

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
void check_converted_values(const cldnn::memory::ptr& memory,
                            const std::vector<Src>& values,
                            cldnn::stream& stream) {
    std::vector<Dst> expected(values.size());
    ov::reference::convert(values.data(), expected.data(), values.size());
    cldnn::mem_lock<Dst, mem_lock_type::read> actual(memory, stream);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(static_cast<float>(expected[i])))
            ASSERT_TRUE(std::isnan(static_cast<float>(actual[i]))) << "element " << i;
        else
            ASSERT_EQ(actual[i], expected[i]) << "element " << i;
    }
}

// Isolate this overload call to avoid an MSVC 19.42 C1001 in the fallback test.
void convert_host_tensor(const ov::ITensor* source,
                         const cldnn::memory::ptr& destination,
                         cldnn::stream& stream,
                         const cldnn::layout& source_layout) {
    ov::intel_gpu::convert_and_copy(source, destination, stream, source_layout, false);
}

template <typename Src, typename Dst>
void check_async_values(const std::shared_ptr<tensor_conversion>& conversion,
                        const std::shared_ptr<ov::ITensor>& source,
                        const std::vector<Src>& values,
                        const cldnn::memory::ptr& destination,
                        cldnn::stream& stream,
                        const cldnn::memory::ptr& source_memory = nullptr) {
    if (source_memory)
        source_memory->copy_from(stream, values.data(), true);
    else
        std::copy(values.begin(), values.end(), static_cast<Src*>(source->data()));

    const auto source_layout = make_layout(source->get_element_type(), source->get_shape());
    PendingConversion pending;
    ASSERT_TRUE(convert_and_copy_async(source.get(), destination, stream, source_layout, conversion, pending));
    ASSERT_NO_THROW(pending.wait());
    check_converted_values<Src, Dst>(destination, values, stream);
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

    ASSERT_NO_THROW(variable.set_state(ov::SoPtr<ov::ITensor>(source)));
    ASSERT_NE(variable.get_memory(), nullptr);
    ASSERT_EQ(variable.get_layout().get_shape(), source->get_shape());
    check_converted_values<Src, Dst>(variable.get_memory(), values, stream);
}

}  // namespace

TEST(convert_and_copy_test, pending_conversion_wait_releases_resources) {
    static_assert(!std::is_copy_constructible_v<PendingConversion>);
    static_assert(!std::is_copy_assignable_v<PendingConversion>);
    static_assert(!std::is_move_constructible_v<PendingConversion>);
    static_assert(!std::is_move_assignable_v<PendingConversion>);

    PendingConversion pending;
    pending.staging = get_test_engine().allocate_memory(layout{ov::Shape{1}, data_types::f32, format::bfyx});
    std::weak_ptr<cldnn::memory> staging = pending.staging;
    auto done = std::make_shared<ControlledConversionEvent>();
    pending.done = done;

    ASSERT_NO_THROW(pending.wait());
    EXPECT_EQ(done->wait_calls, 1);
    EXPECT_EQ(pending.done, nullptr);
    EXPECT_EQ(pending.staging, nullptr);
    EXPECT_TRUE(staging.expired());
    EXPECT_NO_THROW(pending.wait());
    EXPECT_EQ(done->wait_calls, 1);
}

TEST(convert_and_copy_test, pending_conversion_failed_wait_keeps_resources_until_retry) {
    PendingConversion pending;
    pending.staging = get_test_engine().allocate_memory(layout{ov::Shape{1}, data_types::f32, format::bfyx});
    std::weak_ptr<cldnn::memory> staging = pending.staging;
    auto done = std::make_shared<ControlledConversionEvent>();
    done->fail_wait = true;
    pending.done = done;

    EXPECT_THROW(pending.wait(), ov::Exception);
    EXPECT_EQ(done->wait_calls, 1);
    EXPECT_EQ(pending.done, done);
    EXPECT_FALSE(staging.expired());

    done->fail_wait = false;
    ASSERT_NO_THROW(pending.wait());
    EXPECT_EQ(done->wait_calls, 2);
    EXPECT_EQ(pending.done, nullptr);
    EXPECT_TRUE(staging.expired());
}

TEST(convert_and_copy_test, pending_conversion_explicit_cleanup_does_not_wait) {
    PendingConversion pending;
    pending.staging = get_test_engine().allocate_memory(layout{ov::Shape{1}, data_types::f32, format::bfyx});
    std::weak_ptr<cldnn::memory> staging = pending.staging;
    auto done = std::make_shared<ControlledConversionEvent>();
    done->fail_wait = true;
    pending.done = done;

    ASSERT_NO_THROW(get_test_stream().finish());
    pending.clear_after_completion();
    EXPECT_EQ(done->wait_calls, 0);
    EXPECT_EQ(pending.done, nullptr);
    EXPECT_EQ(pending.staging, nullptr);
    EXPECT_TRUE(staging.expired());
}

TEST(convert_and_copy_test, pending_conversion_without_event_keeps_staging_until_explicit_cleanup) {
    PendingConversion pending;
    pending.staging = get_test_engine().allocate_memory(layout{ov::Shape{1}, data_types::f32, format::bfyx});
    std::weak_ptr<cldnn::memory> staging = pending.staging;

    ASSERT_NO_THROW(pending.wait());
    EXPECT_FALSE(staging.expired());
    ASSERT_NO_THROW(get_test_stream().finish());
    pending.clear_after_completion();
    EXPECT_TRUE(staging.expired());
}

TEST(convert_and_copy_test, pending_conversion_destruction_does_not_wait) {
    auto done = std::make_shared<ControlledConversionEvent>();
    done->fail_wait = true;
    {
        PendingConversion pending;
        pending.done = done;
    }
    EXPECT_EQ(done->wait_calls, 0);
}

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

TEST(convert_and_copy_gpu_test, async_util_snapshots_host_inputs_without_variable_state_or_program) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    const ov::Shape shape{2, 3};
    const layout src_layout{shape, data_types::f32, format::bfyx};
    const layout dst_layout{shape, data_types::f16, format::bfyx};
    auto network = make_state_network(shape, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    auto conversion = program->get_tensor_conversion();
    std::weak_ptr<cldnn::program> program_weak = program;
    network.reset();
    program.reset();
    ASSERT_TRUE(program_weak.expired());

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    auto dst = context->get_engine().allocate_memory(dst_layout, allocation_type::usm_device);
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    PendingConversion pending;
    for (const bool usm_host : {false, true}) {
        std::shared_ptr<ov::ITensor> source;
        if (usm_host)
            source = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, shape);
        else
            source = ov::make_tensor(ov::element::f32, shape);
        std::copy(values.begin(), values.end(), static_cast<float*>(source->data()));
        std::weak_ptr<ov::ITensor> source_weak = source;

        ASSERT_TRUE(convert_and_copy_async(source.get(), dst, stream, src_layout, conversion, pending));
        std::weak_ptr<cldnn::memory> staging_weak = pending.staging;
        EXPECT_NE(pending.done, nullptr);
        EXPECT_FALSE(staging_weak.expired());
        std::fill_n(static_cast<float*>(source->data()), values.size(), 0.f);
        source.reset();
        EXPECT_TRUE(source_weak.expired());

        ASSERT_NO_THROW(pending.wait());
        EXPECT_TRUE(staging_weak.expired());
        EXPECT_EQ(pending.done, nullptr);
        cldnn::mem_lock<ov::float16, mem_lock_type::read> actual(dst, stream);
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_EQ(actual[i], ov::float16(values[i]));
    }
}

TEST(convert_and_copy_gpu_test, async_util_converts_remote_sources_without_variable_state) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    const ov::Shape shape{2, 3};
    const layout src_layout{shape, data_types::f32, format::bfyx};
    const layout dst_layout{shape, data_types::f16, format::bfyx};
    auto network = make_state_network(shape, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    auto conversion = program->get_tensor_conversion();
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    const std::vector<float> zeros(values.size(), 0.f);

    for (const auto tensor_type : {TensorType::BT_USM_DEVICE_INTERNAL, TensorType::BT_BUF_INTERNAL}) {
        auto source = std::make_shared<RemoteTensorImpl>(context, shape, ov::element::f32, tensor_type);
        auto source_memory = source->get_original_memory();
        const auto allocation = tensor_type == TensorType::BT_USM_DEVICE_INTERNAL ? allocation_type::usm_device
                                                                                 : allocation_type::cl_mem;
        ASSERT_EQ(source_memory->get_allocation_type(), allocation);
        source_memory->copy_from(stream, values.data(), true);
        auto dst = context->get_engine().allocate_memory(dst_layout, allocation);
        PendingConversion pending;

        ASSERT_TRUE(convert_and_copy_async(source.get(), dst, stream, src_layout, conversion, pending));
        EXPECT_NE(pending.done, nullptr);
        source_memory->copy_from(stream, zeros.data(), true);
        source_memory.reset();
        source.reset();
        ASSERT_NO_THROW(pending.wait());
        EXPECT_EQ(pending.staging, nullptr);
        cldnn::mem_lock<ov::float16, mem_lock_type::read> actual(dst, stream);
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_EQ(actual[i], ov::float16(values[i]));
    }
}

TEST(convert_and_copy_gpu_test, async_util_unsupported_input_leaves_pending_empty_for_cpu_fallback) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    const ov::Shape shape{2, 3};
    const layout src_layout{shape, data_types::f32, format::bfyx};
    const layout dst_layout{shape, data_types::f16, format::bfyx};
    auto source = ov::make_tensor(ov::element::f32, shape);
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    std::copy(values.begin(), values.end(), static_cast<float*>(source->data()));
    auto dst = engine.allocate_memory(dst_layout);
    auto& stream = get_test_stream();
    PendingConversion pending;

    EXPECT_FALSE(convert_and_copy_async(source.get(), dst, stream, src_layout, nullptr, pending));
    EXPECT_EQ(pending.staging, nullptr);
    EXPECT_EQ(pending.done, nullptr);
    auto network = make_state_network(shape, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    EXPECT_EQ(program->get_tensor_conversion(false), nullptr);
    auto unprepared = std::make_shared<tensor_conversion>();
    EXPECT_FALSE(convert_and_copy_async(source.get(), dst, stream, src_layout, unprepared, pending));
    program->prepare_tensor_conversions({});
    ASSERT_EQ(program->get_tensor_conversion(false), nullptr);
    program->prepare_tensor_conversions({{data_types::f16, data_types::f16}, {data_types::f16, data_types::f32}});
    unprepared = program->get_tensor_conversion();
    ASSERT_NE(unprepared, nullptr);
    EXPECT_TRUE(unprepared->get_keys().empty());
    EXPECT_TRUE(unprepared->get_kernels().empty());
    ASSERT_NO_THROW(program->prepare_tensor_conversions({}));
    EXPECT_THROW(program->prepare_tensor_conversions({{data_types::f32, data_types::f16}}), ov::Exception);
    EXPECT_FALSE(convert_and_copy_async(source.get(), dst, stream, src_layout, unprepared, pending));
    EXPECT_EQ(pending.staging, nullptr);
    EXPECT_EQ(pending.done, nullptr);

    ASSERT_NO_THROW(convert_host_tensor(source.get(), dst, stream, src_layout));
    check_converted_values<float, ov::float16>(dst, values, stream);

    auto prepared_network = make_state_network(shape, ov::element::f32, ov::element::f16);
    auto prepared_program = prepared_network->get_program();
    prepared_program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    const auto conversion = prepared_program->get_tensor_conversion();
    const std::vector<tensor_conversion_key> missing_keys{
        {data_types::bf16, data_types::f16}, {data_types::f16, data_types::f16}, {data_types::f16, data_types::f32}};
    for (const auto& key : missing_keys) {
        auto unsupported_source = ov::make_tensor(ov::element::Type(key.first), shape);
        if (key.first == data_types::bf16)
            ov::reference::convert(values.data(), static_cast<ov::bfloat16*>(unsupported_source->data()), values.size());
        else
            ov::reference::convert(values.data(), static_cast<ov::float16*>(unsupported_source->data()), values.size());
        auto destination = engine.allocate_memory(layout{shape, key.second, format::bfyx});
        const auto unsupported_layout = make_layout(unsupported_source->get_element_type(), shape);
        EXPECT_FALSE(convert_and_copy_async(unsupported_source.get(), destination, stream, unsupported_layout,
                                            conversion, pending));
        EXPECT_EQ(pending.staging, nullptr);
        EXPECT_EQ(pending.done, nullptr);
        ASSERT_NO_THROW(convert_host_tensor(unsupported_source.get(), destination, stream, unsupported_layout));
        if (key.second == data_types::f32)
            check_converted_values<float, float>(destination, values, stream);
        else
            check_converted_values<float, ov::float16>(destination, values, stream);
    }

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto remote = std::make_shared<RemoteTensorImpl>(context, shape, ov::element::f32);
    EXPECT_FALSE(convert_and_copy_async(remote.get(), dst, stream, src_layout, unprepared, pending));
    EXPECT_EQ(pending.staging, nullptr);
    EXPECT_EQ(pending.done, nullptr);
    try {
        convert_host_tensor(remote.get(), dst, stream, src_layout);
        FAIL() << "Expected an inaccessible remote source error";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("CPU conversion cannot access a remote source tensor"), std::string::npos);
    }
}

TEST(convert_and_copy_gpu_test, async_util_reuses_kernels_for_sources_shapes_and_types) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    auto network = make_state_network({2, 6}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    const tensor_conversion_key key{data_types::f32, data_types::f16};
    std::vector<tensor_conversion_key> keys{key, key, {data_types::bf16, data_types::f16}};
    program->prepare_tensor_conversions(keys);
    auto executor = program->get_tensor_conversion();
    ASSERT_NE(executor, nullptr);
    ASSERT_EQ(executor->get_keys().size(), keys.size() - 1);
    for (const auto& required_key : keys)
        ASSERT_TRUE(executor->has_kernel(required_key));
    program->prepare_tensor_conversions(keys);
    EXPECT_EQ(program->get_tensor_conversion(), executor);
    EXPECT_THROW(program->prepare_tensor_conversions({{data_types::bf16, data_types::f16}}), ov::Exception);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    auto destination = context->get_engine().allocate_memory(
        layout{ov::Shape{2, 6}, data_types::f16, format::bfyx}, allocation_type::usm_device);

    auto host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{2, 6});
    std::vector<float> first_values{0.f, -0.f, 1.f, -1.f, 0.1f, -0.1f,
                                    65504.f, 1e-8f, 3.5f, -3.5f,
                                    std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    check_async_values<float, ov::float16>(executor, host, first_values, destination, stream,
                                          host->get_impl()->get_original_memory());

    std::weak_ptr<cldnn::program> program_weak = program;
    network.reset();
    program.reset();
    ASSERT_TRUE(program_weak.expired());

    auto larger_host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{3, 7});
    destination = context->get_engine().allocate_memory(
        layout{ov::Shape{3, 7}, data_types::f16, format::bfyx}, allocation_type::usm_device);
    std::vector<float> larger_values(21);
    for (size_t i = 0; i < larger_values.size(); ++i)
        larger_values[i] = (static_cast<int>(i) - 10) / 4.f;
    check_async_values<float, ov::float16>(executor, larger_host, larger_values, destination, stream,
                                          larger_host->get_impl()->get_original_memory());

    auto device = std::make_shared<RemoteTensorImpl>(context, ov::Shape{3, 7}, ov::element::f32,
                                                     TensorType::BT_USM_DEVICE_INTERNAL);
    auto device_memory = device->get_original_memory();
    ASSERT_EQ(device_memory->get_allocation_type(), allocation_type::usm_device);
    check_async_values<float, ov::float16>(executor, device, larger_values, destination, stream, device_memory);
    EXPECT_TRUE(executor->has_kernel(key));

    destination = context->get_engine().allocate_memory(
        layout{ov::Shape{2, 6}, data_types::f16, format::bfyx}, allocation_type::usm_device);
    auto bf16_host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::bf16, ov::Shape{2, 6});
    const std::vector<ov::bfloat16> bf16_values{0.f, -0.f, 1.f, -1.f, 0.1f, -0.1f,
                                              65504.f, 1e-8f, 3.5f, -3.5f,
                                              std::numeric_limits<float>::infinity(),
                                              std::numeric_limits<float>::quiet_NaN()};
    check_async_values<ov::bfloat16, ov::float16>(executor, bf16_host, bf16_values, destination, stream,
                                                bf16_host->get_impl()->get_original_memory());
}

TEST(convert_and_copy_gpu_test, concurrent_preparation_reuses_tensor_conversion_kernels) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16)
        GTEST_SKIP() << "OpenCL FP16 is required";

    auto network = make_state_network({2, 3}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    EXPECT_EQ(program->get_tensor_conversion(false), nullptr);
    const std::vector<tensor_conversion_key> keys{{data_types::f32, data_types::f16}};
    std::vector<std::future<std::shared_ptr<tensor_conversion>>> preparations;
    for (size_t i = 0; i < 4; ++i) {
        preparations.push_back(std::async(std::launch::async, [program, keys] {
            auto conversion = program->get_tensor_conversion();
            program->prepare_tensor_conversions(keys);
            return conversion;
        }));
    }
    std::shared_ptr<tensor_conversion> conversion;
    for (auto& preparation : preparations) {
        std::shared_ptr<tensor_conversion> prepared;
        ASSERT_NO_THROW(prepared = preparation.get());
        ASSERT_NE(prepared, nullptr);
        if (!conversion)
            conversion = prepared;
        EXPECT_EQ(prepared, conversion);
    }

    EXPECT_EQ(program->get_tensor_conversion(false), conversion);
    EXPECT_EQ(conversion->get_keys(), keys);
    const auto kernels = conversion->get_kernels();
    ASSERT_EQ(kernels.size(), 1);
    ASSERT_NO_THROW(program->prepare_tensor_conversions(keys));
    EXPECT_EQ(conversion->get_kernels(), kernels);
}

TEST(convert_and_copy_gpu_test, variable_state_keeps_tensor_conversion_after_program_destruction) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    const ov::Shape shape{2, 3};
    auto network = make_state_network(shape, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    std::weak_ptr<cldnn::network> network_weak = network;
    std::weak_ptr<cldnn::program> program_weak = program;
    std::weak_ptr<tensor_conversion> conversion_weak = program->get_tensor_conversion();
    std::weak_ptr<RemoteContextImpl> context_weak = context;
    auto variable = std::make_unique<VariableState>(network->get_variable_info("state"), context,
                                                   network->get_shape_predictor(),
                                                   program->get_tensor_conversion());
    ASSERT_TRUE(program->get_tensor_conversion()->compatible_with(context->get_engine()));

    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    {
        VariableState without_conversion(network->get_variable_info("state"), context, network->get_shape_predictor());
        auto host = ov::make_tensor(ov::element::f32, shape);
        check_state_values<float, ov::float16>(without_conversion, host, values, stream);

        auto wrong_type = ov::make_tensor(ov::element::bf16, shape);
        const std::vector<ov::bfloat16> bf16_values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
        check_state_values<ov::bfloat16, ov::float16>(*variable, wrong_type, bf16_values, stream);

        auto same_type_info = network->get_variable_info("state");
        same_type_info.m_user_specified_type = ov::element::f16;
        VariableState same_type(same_type_info, context, network->get_shape_predictor(),
                               program->get_tensor_conversion());
        auto f16_host = ov::make_tensor(ov::element::f16, shape);
        const std::vector<ov::float16> f16_values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
        check_state_values<ov::float16, ov::float16>(same_type, f16_host, f16_values, stream);
    }
    auto source = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, shape);
    source->get_impl()->get_original_memory()->copy_from(stream, values.data(), true);
    ASSERT_NO_THROW(variable->set_state(ov::SoPtr<ov::ITensor>(source)));
    source.reset();
    network.reset();
    program.reset();
    context.reset();
    ASSERT_TRUE(network_weak.expired());
    ASSERT_TRUE(program_weak.expired());
    ASSERT_FALSE(conversion_weak.expired());
    ASSERT_FALSE(context_weak.expired());

    {
        cldnn::mem_lock<ov::float16, mem_lock_type::read> result(variable->get_memory(), stream);
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_EQ(result[i], ov::float16(values[i]));
    }

    source = std::make_shared<GpuOnlyHostTensor>(context_weak.lock(), ov::element::f32, shape);
    const std::vector<float> next_values{0.5f, 1.5f, -2.5f, 3.5f, -4.5f, 5.5f};
    check_state_values<float, ov::float16>(*variable, source, next_values, stream,
                                          source->get_impl()->get_original_memory());
    ASSERT_NO_THROW(variable->set_state(ov::SoPtr<ov::ITensor>(source)));
    ASSERT_NO_THROW(variable->reset());
    EXPECT_FALSE(variable->is_set());
    check_state_values<float, ov::float16>(*variable, source, values, stream,
                                          source->get_impl()->get_original_memory());

    auto larger_source = std::make_shared<GpuOnlyHostTensor>(context_weak.lock(), ov::element::f32, ov::Shape{3, 7});
    const std::vector<float> larger_values(21, 0.25f);
    check_state_values<float, ov::float16>(*variable, larger_source, larger_values, stream,
                                          larger_source->get_impl()->get_original_memory());
    larger_source.reset();
    auto empty = ov::make_tensor(ov::element::f32, ov::Shape{0, 3});
    ASSERT_NO_THROW(variable->set_state(ov::SoPtr<ov::ITensor>(empty)));
    EXPECT_TRUE(variable->is_set());
    EXPECT_EQ(variable->get_memory(), nullptr);
    ASSERT_NO_THROW(variable->reset());

    // Destroy the state with a submitted conversion and no external conversion/context owners.
    ASSERT_NO_THROW(variable->set_state(ov::SoPtr<ov::ITensor>(source)));
    source.reset();
    variable.reset();
    EXPECT_TRUE(conversion_weak.expired());
    EXPECT_TRUE(context_weak.expired());
}

TEST(convert_and_copy_gpu_test, async_util_stages_aligned_and_unaligned_host_inputs) {
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
    program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    auto conversion = program->get_tensor_conversion();
    auto destination = context->get_engine().allocate_memory(
        layout{aligned_shape, data_types::f16, format::bfyx}, allocation_type::usm_device);

    std::unique_ptr<void, decltype(&ov::util::aligned_free)> storage(
        ov::util::aligned_alloc(alignment, 2 * alignment), ov::util::aligned_free);
    ASSERT_NE(storage, nullptr);
    auto aligned_view = ov::make_tensor(ov::element::f32, aligned_shape, storage.get());
    auto aligned_source = std::make_shared<GpuOnlyHostView>(aligned_view);
    std::vector<float> aligned_values(ov::shape_size(aligned_shape));
    for (size_t i = 0; i < aligned_values.size(); ++i)
        aligned_values[i] = static_cast<float>(i) - 4.f;
    check_async_values<float, ov::float16>(conversion, aligned_source, aligned_values, destination, stream);

    auto* unaligned_ptr = static_cast<float*>(storage.get()) + 1;
    auto unaligned_view = ov::make_tensor(ov::element::f32, aligned_shape, unaligned_ptr);
    auto unaligned_source = std::make_shared<GpuOnlyHostView>(unaligned_view);
    std::vector<float> staged_values(ov::shape_size(aligned_shape), -0.5f);
    check_async_values<float, ov::float16>(conversion, unaligned_source, staged_values, destination, stream);

    const ov::Shape unaligned_size_shape{alignment / sizeof(float) + 1};
    auto unaligned_size_view = ov::make_tensor(ov::element::f32, unaligned_size_shape, storage.get());
    auto unaligned_size_source = std::make_shared<GpuOnlyHostView>(unaligned_size_view);
    std::vector<float> unaligned_size_values(ov::shape_size(unaligned_size_shape), 0.25f);
    destination = context->get_engine().allocate_memory(
        layout{unaligned_size_shape, data_types::f16, format::bfyx}, allocation_type::usm_device);
    check_async_values<float, ov::float16>(conversion, unaligned_size_source, unaligned_size_values, destination, stream);
}

TEST(convert_and_copy_gpu_test, async_util_preserves_padding_and_transpose_without_packing) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM allocations are required";

    auto network = make_state_network({2, 2}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    auto conversion = program->get_tensor_conversion();
    auto destination = context->get_engine().allocate_memory(
        layout{ov::Shape{2, 2}, data_types::f16, format::bfyx}, allocation_type::usm_device);
    for (bool transpose : {false, true}) {
        ov::Tensor parent(ov::element::f32, {2, 4});
        const std::vector<float> physical_values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        std::copy(physical_values.begin(), physical_values.end(), parent.data<float>());
        ov::Tensor view(parent, {0, 2}, {2, 4});
        ASSERT_FALSE(view.is_continuous());
        auto source = std::make_shared<GpuOnlyHostView>(get_tensor_impl(view)._ptr);
        const layout source_layout(ov::Shape{2, 2}, data_types::f32, format::bfyx, padding({0, 0}, {0, 2}));
        PendingConversion pending;
        ASSERT_TRUE(convert_and_copy_async(source.get(), destination, stream, source_layout, conversion,
                                           pending, transpose));
        source.reset();
        view = {};
        parent = {};
        ASSERT_NO_THROW(pending.wait());
        const std::vector<float> expected = transpose ? std::vector<float>{3.f, 7.f, 4.f, 8.f}
                                                       : std::vector<float>{3.f, 4.f, 7.f, 8.f};
        check_converted_values<float, ov::float16>(destination, expected, stream);

        auto padded_host = std::make_shared<PaddedHostTensor>(context);
        const std::vector<float> padded_values{1.f, 2.f, 99.f, 3.f, 4.f, 99.f};
        padded_host->get_impl()->get_original_memory()->copy_from(stream, padded_values.data(), true);
        const layout padded_layout(ov::Shape{2, 2}, data_types::f32, format::bfyx, padding({0, 0}, {0, 1}));
        ASSERT_TRUE(convert_and_copy_async(padded_host.get(), destination, stream, padded_layout, conversion,
                                           pending, transpose));
        ASSERT_NO_THROW(pending.wait());
        const std::vector<float> padded_expected = transpose ? std::vector<float>{1.f, 3.f, 2.f, 4.f}
                                                             : std::vector<float>{1.f, 2.f, 3.f, 4.f};
        check_converted_values<float, ov::float16>(destination, padded_expected, stream);
    }

    auto dense_host = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{2, 3});
    const std::vector<float> dense_values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    dense_host->get_impl()->get_original_memory()->copy_from(stream, dense_values.data(), true);
    const layout dense_layout{ov::Shape{2, 3}, data_types::f32, format::bfyx};
    destination = context->get_engine().allocate_memory(
        layout{ov::Shape{2, 3}, data_types::f16, format::bfyx}, allocation_type::usm_device);
    PendingConversion pending;
    ASSERT_TRUE(convert_and_copy_async(dense_host.get(), destination, stream, dense_layout, conversion, pending, true));
    ASSERT_NO_THROW(pending.wait());
    check_converted_values<float, ov::float16>(destination, {1.f, -4.f, -2.f, 5.f, 3.f, -6.f}, stream);
}

TEST(convert_and_copy_gpu_test, tensor_conversion_guards_source_and_destination_bounds) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    auto network = make_state_network({2, 2}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_tensor_conversions({{data_types::f32, data_types::f16}});
    auto& stream = get_test_stream();
    auto source = engine.allocate_memory(layout{ov::Shape{2, 2}, data_types::f32, format::bfyx},
                                         allocation_type::usm_host);
    auto destination = engine.allocate_memory(layout{ov::Shape{2, 2}, data_types::f16, format::bfyx},
                                              allocation_type::usm_device);
    const std::vector<float> values{1.f, 2.f, 3.f, 4.f};
    source->copy_from(stream, values.data(), true);
    tensor_conversion_input input;
    ASSERT_TRUE(tensor_conversion::prepare_input(source->get_layout(), {8, 4}, false, input));
    auto executor = program->get_tensor_conversion();
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

TEST(convert_and_copy_gpu_test, imported_tensor_conversion_kernel_runs_without_recompilation) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl || !engine.get_device_info().supports_fp16 ||
        !engine.supports_allocation(allocation_type::usm_host) ||
        !engine.supports_allocation(allocation_type::usm_device))
        GTEST_SKIP() << "OpenCL FP16 and USM host/device allocations are required";

    auto stream_ptr = get_test_stream_ptr();
    for (const bool unsupported_request : {false, true}) {
        auto empty_network = make_state_network({2, 3}, ov::element::f32, ov::element::f16);
        auto empty_program = empty_network->get_program();
        ASSERT_EQ(empty_program->get_tensor_conversion(false), nullptr);
        if (unsupported_request) {
            empty_program->prepare_tensor_conversions({{data_types::f16, data_types::f32}});
            ASSERT_NE(empty_program->get_tensor_conversion(), nullptr);
            EXPECT_TRUE(empty_program->get_tensor_conversion()->get_keys().empty());
        }
        const auto conversion_before_save = empty_program->get_tensor_conversion(false);
        std::stringstream empty_blob;
        {
            BinaryOutputBuffer output(empty_blob);
            output.set_stream(stream_ptr.get());
            empty_program->save(output);
        }
        EXPECT_EQ(empty_program->get_tensor_conversion(false), conversion_before_save);
        empty_blob.seekg(0);
        BinaryInputBuffer input(empty_blob, engine);
        auto restored = std::make_shared<cldnn::program>(engine, get_test_default_config(engine));
        restored->load(input);
        EXPECT_EQ(restored->get_tensor_conversion(false), nullptr);
        ASSERT_NO_THROW(restored->prepare_tensor_conversions({}));
        ASSERT_EQ(restored->get_tensor_conversion(false), nullptr);
    }

    const tensor_conversion_key key{data_types::f32, data_types::f16};
    auto network = make_state_network({2, 3}, ov::element::f32, ov::element::f16);
    auto program = network->get_program();
    program->prepare_tensor_conversions({key});
    std::stringstream blob;
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
    EXPECT_EQ(restored->get_tensor_conversion(false), nullptr);
    restored->load(input);
    ASSERT_NE(restored->get_tensor_conversion(false), nullptr);
    ASSERT_TRUE(restored->get_tensor_conversion()->has_kernel(key));
    auto imported_executor = restored->get_tensor_conversion();
    ASSERT_NO_THROW(restored->prepare_tensor_conversions({}));
    ASSERT_NO_THROW(restored->prepare_tensor_conversions({key}));
    EXPECT_EQ(restored->get_tensor_conversion(), imported_executor);
    EXPECT_THROW(restored->prepare_tensor_conversions({{data_types::bf16, data_types::f16}}), ov::Exception);
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto& stream = context->get_engine().get_service_stream();
    restored.reset();
    auto source = std::make_shared<GpuOnlyHostTensor>(context, ov::element::f32, ov::Shape{2, 3});
    const std::vector<float> values{1.f, -2.f, 3.f, -4.f, 5.f, -6.f};
    auto destination = context->get_engine().allocate_memory(
        layout{ov::Shape{2, 3}, data_types::f16, format::bfyx}, allocation_type::usm_device);
    check_async_values<float, ov::float16>(imported_executor, source, values, destination, stream,
                                          source->get_impl()->get_original_memory());
}
