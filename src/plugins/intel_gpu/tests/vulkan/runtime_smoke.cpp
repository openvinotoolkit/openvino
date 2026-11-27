// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {

bool is_gpu_device(const std::string& device) {
    return device == "GPU" || device.rfind("GPU.", 0) == 0;
}

template <typename Operation>
void expect_ov_exception(Operation operation, std::string_view message) {
    try {
        operation();
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string_view(error.what()).find(message), std::string_view::npos) << error.what();
        return;
    }
    ADD_FAILURE() << "Expected an OpenVINO exception containing " << message;
}

class VulkanRuntimeSmoke : public testing::Test {
protected:
    void SetUp() override {
        const auto devices = m_core.get_available_devices();
        const auto gpu = std::find_if(devices.begin(), devices.end(), is_gpu_device);
        ASSERT_NE(gpu, devices.end()) << "No Vulkan GPU detected";
        m_device_name = *gpu;
        ASSERT_EQ(m_core.get_property(m_device_name, "GPU_RUNTIME_TYPE").as<std::string>(), "VULKAN");
    }

    ov::Core m_core;
    std::string m_device_name;
};

TEST_F(VulkanRuntimeSmoke, AdvertisesOnlyAvailableProperties) {
    EXPECT_TRUE(m_core.get_property(m_device_name, ov::device::capabilities).empty());
    const auto properties = m_core.get_property(m_device_name, ov::supported_properties);
    for (const auto* name : {ov::optimal_batch_size.name(),
                             ov::max_batch_size.name(),
                             ov::range_for_async_infer_requests.name(),
                             ov::range_for_streams.name(),
                             ov::device::gops.name(),
                             ov::device::luid.name(),
                             ov::intel_gpu::cacheline_size.name(),
                             ov::intel_gpu::execution_units_count.name(),
                             ov::intel_gpu::uarch_version.name(),
                             ov::enable_profiling.name(),
                             ov::num_streams.name(),
                             "CONFIG_FILE"}) {
        SCOPED_TRACE(name);
        EXPECT_EQ(std::find(properties.begin(), properties.end(), name), properties.end()) << name;
        EXPECT_THROW(m_core.get_property(m_device_name, name), ov::Exception);
    }
    EXPECT_EQ(std::find(properties.begin(), properties.end(), ov::cache_dir.name()), properties.end());
    EXPECT_THROW(m_core.set_property(m_device_name, ov::enable_profiling(true)), ov::Exception);
    EXPECT_GT(m_core.get_property(m_device_name, ov::intel_gpu::device_max_alloc_mem_size), 0U);
    EXPECT_EQ(m_core.get_property(m_device_name, ov::compatibility_check), ov::CompatibilityCheck::UNSUPPORTED);
}

TEST_F(VulkanRuntimeSmoke, RejectsUnavailablePciInfo) {
    const auto properties = m_core.get_property(m_device_name, ov::supported_properties);
    EXPECT_EQ(std::find(properties.begin(), properties.end(), ov::device::pci_info.name()), properties.end());
    expect_ov_exception(
        [&] {
            m_core.get_property(m_device_name, ov::device::pci_info);
        },
        "[GPU] Property DEVICE_PCI_INFO is not in a list of supported properties");
}

TEST_F(VulkanRuntimeSmoke, RejectsLargeAllocationProperty) {
    const auto& property = ov::intel_gpu::hint::enable_large_allocations;
    const auto properties = m_core.get_property(m_device_name, ov::supported_properties);
    EXPECT_EQ(std::find(properties.begin(), properties.end(), property.name()), properties.end());

    constexpr std::string_view message = "[GPU] Property GPU_ENABLE_LARGE_ALLOCATIONS is not in a list of supported properties";
    expect_ov_exception(
        [&] {
            m_core.get_property(m_device_name, property);
        },
        message);
    expect_ov_exception(
        [&] {
            m_core.set_property(m_device_name, property(true));
        },
        message);
}

TEST_F(VulkanRuntimeSmoke, CopiesRemoteTensorsAndReportsMemoryUsage) {
    const auto tracked_buffer_bytes = [&] {
        return m_core.get_property(m_device_name, ov::intel_gpu::memory_statistics).at("vk_buffer");
    };
    EXPECT_EQ(tracked_buffer_bytes(), 0U);

    auto context = m_core.get_default_context(m_device_name);
    EXPECT_TRUE(context.get_params().empty());
    EXPECT_EQ(tracked_buffer_bytes(), 0U);
    for (const auto& shape : {ov::Shape{}, ov::Shape{256}, ov::Shape{257}}) {
        const auto count = ov::shape_size(shape);
        ov::Tensor input(ov::element::u8, shape);
        ov::Tensor output(ov::element::u8, shape);
        auto* bytes = input.data<uint8_t>();
        for (size_t i = 0; i < count; ++i) {
            bytes[i] = static_cast<uint8_t>(i ^ 0xA5);
        }
        {
            auto source = context.create_tensor(ov::element::u8, shape);
            auto destination = context.create_tensor(ov::element::u8, shape);
            EXPECT_TRUE(source.get_params().empty());
            EXPECT_EQ(tracked_buffer_bytes(), 2 * count);
            source.copy_from(input);
            source.copy_to(destination);
            destination.copy_to(output);
        }
        EXPECT_EQ(tracked_buffer_bytes(), 0U);
        EXPECT_TRUE(std::equal(bytes, bytes + count, output.data<const uint8_t>())) << shape;
    }
}

TEST_F(VulkanRuntimeSmoke, RejectsStringTensors) {
    auto context = m_core.get_default_context(m_device_name);
    expect_ov_exception(
        [&] {
            context.create_tensor(ov::element::string, {1});
        },
        "String tensors are unsupported");
}

TEST_F(VulkanRuntimeSmoke, RejectsDynamicElementType) {
    auto context = m_core.get_default_context(m_device_name);
    expect_ov_exception(
        [&] {
            context.create_tensor(ov::element::dynamic, {1});
        },
        "Dynamic element types are unsupported");
}

TEST_F(VulkanRuntimeSmoke, RejectsUnsupportedTensorRankBeforeChangingShape) {
    auto context = m_core.get_default_context(m_device_name);
    const ov::Shape supported_shape(8, 1);
    const ov::Shape unsupported_shape(9, 1);
    auto remote = context.create_tensor(ov::element::u8, supported_shape);
    const auto strides = remote.get_strides();
    ov::Tensor input(ov::element::u8, supported_shape);
    *input.data<uint8_t>() = 42;
    remote.copy_from(input);

    EXPECT_THROW(context.create_tensor(ov::element::u8, unsupported_shape), ov::Exception);
    EXPECT_THROW(remote.set_shape(unsupported_shape), ov::Exception);
    ASSERT_EQ(remote.get_shape(), supported_shape);
    EXPECT_EQ(remote.get_strides(), strides);
    ov::Tensor output(ov::element::u8, supported_shape);
    remote.copy_to(output);
    EXPECT_EQ(*output.data<uint8_t>(), 42);
}

TEST_F(VulkanRuntimeSmoke, PreservesTensorWhenGrowthFails) {
    auto context = m_core.get_default_context(m_device_name);
    const ov::Shape original_shape{3};
    auto remote = context.create_tensor(ov::element::u8, original_shape);
    ov::Tensor input(ov::element::u8, original_shape);
    std::fill_n(input.data<uint8_t>(), input.get_size(), 42);
    remote.copy_from(input);

    const auto limit = m_core.get_property(m_device_name, ov::intel_gpu::device_max_alloc_mem_size);
    expect_ov_exception(
        [&] {
            remote.set_shape({static_cast<size_t>(limit + 1)});
        },
        "Reduce the requested allocation size.");
    ASSERT_EQ(remote.get_shape(), original_shape);
    ov::Tensor output(ov::element::u8, original_shape);
    remote.copy_to(output);
    EXPECT_TRUE(std::equal(input.data<uint8_t>(), input.data<uint8_t>() + input.get_size(), output.data<uint8_t>()));
    remote.set_shape({5});
    EXPECT_EQ(remote.get_shape(), ov::Shape{5});
}

TEST_F(VulkanRuntimeSmoke, RejectsModelExecution) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{257});
    auto add = std::make_shared<ov::op::v1::Add>(parameter, parameter);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{parameter});
    EXPECT_TRUE(m_core.query_model(model, m_device_name).empty());
    expect_ov_exception(
        [&] {
            m_core.compile_model(model, m_device_name);
        },
        "Model compilation is unavailable");
}

TEST(VulkanNoDriverSmoke, ReportsNoCompatibleDevices) {
    ov::Core core;
    EXPECT_TRUE(core.get_property("GPU", ov::available_devices).empty());
    const auto devices = core.get_available_devices();
    EXPECT_TRUE(std::none_of(devices.begin(), devices.end(), is_gpu_device));
    expect_ov_exception(
        [&] {
            core.get_property("GPU", ov::device::full_name);
        },
        "no supported devices found");
}

}  // namespace

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    if (argc == 2 && std::string_view(argv[1]) == "--expect-no-driver") {
        testing::GTEST_FLAG(filter) = "VulkanNoDriverSmoke.*";
    } else if (testing::GTEST_FLAG(filter) == "*") {
        testing::GTEST_FLAG(filter) = "VulkanRuntimeSmoke.*";
    }
    return RUN_ALL_TESTS();
}
