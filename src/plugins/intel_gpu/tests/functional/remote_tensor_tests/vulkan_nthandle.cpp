// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(OV_GPU_WITH_OCL_RT) && defined(__linux__)
#include <array>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

#include <unistd.h>

#include <vulkan/vulkan.h>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

// On Linux use UUID (16 bytes) for Vulkan<->OpenCL device matching
using DeviceId = std::array<unsigned char, CL_UUID_SIZE_KHR>;

bool get_context_device_luid(cl_context cl_ctx, DeviceId& cl_luid) {
    size_t devices_size = 0;
    if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size) != CL_SUCCESS ||
        devices_size < sizeof(cl_device_id)) {
        return false;
    }

    std::vector<cl_device_id> cl_devices(devices_size / sizeof(cl_device_id));
    if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, cl_devices.data(), nullptr) != CL_SUCCESS ||
        cl_devices.empty()) {
        return false;
    }

    // On Linux: UUID is always present when cl_khr_device_uuid is supported; no validity flag
    return clGetDeviceInfo(cl_devices[0], CL_DEVICE_UUID_KHR, cl_luid.size(), cl_luid.data(), nullptr) == CL_SUCCESS;
}

bool get_context_first_device(cl_context cl_ctx, cl_device_id& cl_device) {
    size_t devices_size = 0;
    if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size) != CL_SUCCESS ||
        devices_size < sizeof(cl_device_id)) {
        return false;
    }

    std::vector<cl_device_id> cl_devices(devices_size / sizeof(cl_device_id));
    if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, cl_devices.data(), nullptr) != CL_SUCCESS ||
        cl_devices.empty()) {
        return false;
    }

    cl_device = cl_devices[0];
    return true;
}

bool supports_external_import_handle_type(cl_device_id cl_device, cl_uint handle_type) {
    size_t import_types_size = 0;
    cl_int status = clGetDeviceInfo(cl_device,
                                    CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR,
                                    0,
                                    nullptr,
                                    &import_types_size);
    if (status != CL_SUCCESS || import_types_size < sizeof(cl_uint)) {
        return false;
    }

    std::vector<cl_uint> import_types(import_types_size / sizeof(cl_uint));
    status = clGetDeviceInfo(cl_device,
                             CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR,
                             import_types_size,
                             import_types.data(),
                             nullptr);
    if (status != CL_SUCCESS) {
        return false;
    }

    return std::find(import_types.begin(), import_types.end(), handle_type) != import_types.end();
}

bool has_device_extension(VkPhysicalDevice physical_device, const char* extension_name) {
    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) != VK_SUCCESS) {
        return false;
    }

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    if (vkEnumerateDeviceExtensionProperties(physical_device,
                                             nullptr,
                                             &extension_count,
                                             available_extensions.data()) != VK_SUCCESS) {
        return false;
    }

    return std::any_of(available_extensions.begin(),
                       available_extensions.end(),
                       [extension_name](const VkExtensionProperties& extension) {
                           return std::strcmp(extension.extensionName, extension_name) == 0;
                       });
}

std::shared_ptr<ov::Model> make_copy_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(param, zero);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

using ExternalMemoryHandle = int;

constexpr ExternalMemoryHandle invalid_external_memory_handle() {
    return -1;
}

// Use DMA_BUF on Linux: Intel GPU OpenCL supports cl_khr_external_memory_dma_buf
// but not cl_khr_external_memory_opaque_fd. vkGetMemoryFdKHR (VK_KHR_external_memory_fd)
// exports both OPAQUE_FD and DMA_BUF fds; VK_EXT_external_memory_dma_buf enables the latter.
constexpr VkExternalMemoryHandleTypeFlagBits k_external_memory_handle_type =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
constexpr cl_uint k_cl_external_memory_handle_type = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR;
constexpr const char* k_vulkan_external_memory_extension = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
constexpr const char* k_vulkan_dma_buf_extension = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
constexpr const char* k_get_memory_handle_proc_name = "vkGetMemoryFdKHR";

void close_external_memory_handle(ExternalMemoryHandle& handle) {
    if (handle != invalid_external_memory_handle()) {
        close(handle);
        handle = invalid_external_memory_handle();
    }
}

bool export_vulkan_memory_handle(VkDevice device, VkDeviceMemory memory, ExternalMemoryHandle& handle) {
    auto get_memory_handle =
        reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, k_get_memory_handle_proc_name));
    if (!get_memory_handle) {
        ADD_FAILURE() << "Failed to get " << k_get_memory_handle_proc_name;
        return false;
    }

    VkMemoryGetFdInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handle_info.memory = memory;
    handle_info.handleType = k_external_memory_handle_type;

    const VkResult res = get_memory_handle(device, &handle_info, &handle);
    EXPECT_EQ(res, VK_SUCCESS);
    EXPECT_NE(handle, invalid_external_memory_handle());
    return res == VK_SUCCESS && handle != invalid_external_memory_handle();
}

struct VulkanTestContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VulkanTestContext() = default;
    VulkanTestContext(const VulkanTestContext&) = delete;
    VulkanTestContext& operator=(const VulkanTestContext&) = delete;

    VulkanTestContext(VulkanTestContext&& other) noexcept {
        instance = other.instance;
        physical_device = other.physical_device;
        device = other.device;
        other.instance = VK_NULL_HANDLE;
        other.physical_device = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
    }

    VulkanTestContext& operator=(VulkanTestContext&& other) noexcept {
        if (this != &other) {
            this->~VulkanTestContext();
            instance = other.instance;
            physical_device = other.physical_device;
            device = other.device;
            other.instance = VK_NULL_HANDLE;
            other.physical_device = VK_NULL_HANDLE;
            other.device = VK_NULL_HANDLE;
        }
        return *this;
    }

    ~VulkanTestContext() {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }
};

struct VulkanSharedBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ExternalMemoryHandle shared_handle = invalid_external_memory_handle();

    VulkanSharedBuffer() = default;
    VulkanSharedBuffer(const VulkanSharedBuffer&) = delete;
    VulkanSharedBuffer& operator=(const VulkanSharedBuffer&) = delete;

    VulkanSharedBuffer(VulkanSharedBuffer&& other) noexcept {
        device = other.device;
        buffer = other.buffer;
        memory = other.memory;
        shared_handle = other.shared_handle;
        other.device = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        other.memory = VK_NULL_HANDLE;
        other.shared_handle = invalid_external_memory_handle();
    }

    VulkanSharedBuffer& operator=(VulkanSharedBuffer&& other) noexcept {
        if (this != &other) {
            this->~VulkanSharedBuffer();
            device = other.device;
            buffer = other.buffer;
            memory = other.memory;
            shared_handle = other.shared_handle;
            other.device = VK_NULL_HANDLE;
            other.buffer = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
            other.shared_handle = invalid_external_memory_handle();
        }
        return *this;
    }

    ~VulkanSharedBuffer() {
        close_external_memory_handle(shared_handle);
        if (buffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
};

uint32_t find_memory_type(uint32_t memory_type_bits,
                          VkMemoryPropertyFlags required_properties,
                          const VkPhysicalDeviceMemoryProperties& memory_properties) {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const bool type_supported = (memory_type_bits & (1u << i)) != 0;
        const bool has_properties =
            (memory_properties.memoryTypes[i].propertyFlags & required_properties) == required_properties;
        if (type_supported && has_properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool get_vk_device_luid(VkPhysicalDevice physical_device, DeviceId& vk_luid) {
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &id_properties;

    vkGetPhysicalDeviceProperties2(physical_device, &properties2);

    // On Linux: use 16-byte UUID
    std::memcpy(vk_luid.data(), id_properties.deviceUUID, vk_luid.size());
    return true;
}

VulkanTestContext create_vulkan_test_context(const DeviceId& target_luid) {
    VulkanTestContext context;

    const char* instance_extensions[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = 1;
    instance_info.ppEnabledExtensionNames = instance_extensions;

    VkResult res = vkCreateInstance(&instance_info, nullptr, &context.instance);
    EXPECT_EQ(res, VK_SUCCESS);
    if (res != VK_SUCCESS) {
        return {};
    }

    uint32_t device_count = 0;
    res = vkEnumeratePhysicalDevices(context.instance, &device_count, nullptr);
    EXPECT_EQ(res, VK_SUCCESS);
    if (res != VK_SUCCESS || device_count == 0) {
        return {};
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    res = vkEnumeratePhysicalDevices(context.instance, &device_count, physical_devices.data());
    EXPECT_EQ(res, VK_SUCCESS);
    if (res != VK_SUCCESS) {
        return {};
    }

    for (auto physical_device : physical_devices) {
        DeviceId vk_luid{};
        if (!get_vk_device_luid(physical_device, vk_luid)) {
            continue;
        }

        if (std::memcmp(vk_luid.data(), target_luid.data(), target_luid.size()) != 0) {
            continue;
        }

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        if (queue_family_count == 0) {
            continue;
        }

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

        uint32_t selected_queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 ||
                (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0) {
                selected_queue_family = i;
                break;
            }
        }
        if (selected_queue_family == UINT32_MAX) {
            continue;
        }

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = selected_queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        std::vector<const char*> device_extensions = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                                                       k_vulkan_external_memory_extension};

        device_extensions.push_back(k_vulkan_dma_buf_extension);
    #ifdef VK_EXT_memory_budget
        if (has_device_extension(physical_device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
    #endif

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        device_info.ppEnabledExtensionNames = device_extensions.data();

        context.physical_device = physical_device;
        res = vkCreateDevice(physical_device, &device_info, nullptr, &context.device);
        EXPECT_EQ(res, VK_SUCCESS);
        if (res != VK_SUCCESS) {
            return {};
        }

        return context;
    }

    return {};
}

VulkanSharedBuffer create_vulkan_shared_buffer(VulkanTestContext& context, size_t byte_size) {
    VulkanSharedBuffer shared_buffer;
    shared_buffer.device = context.device;

    VkPhysicalDeviceMemoryProperties mem_properties{};
    vkGetPhysicalDeviceMemoryProperties(context.physical_device, &mem_properties);

    VkExternalMemoryBufferCreateInfo external_buffer_info{};
    external_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_buffer_info.handleTypes = k_external_memory_handle_type;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = &external_buffer_info;
    buffer_info.size = byte_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(context.device, &buffer_info, nullptr, &shared_buffer.buffer);
    EXPECT_EQ(res, VK_SUCCESS);
    if (res != VK_SUCCESS) {
        return {};
    }

    VkMemoryRequirements mem_requirements{};
    vkGetBufferMemoryRequirements(context.device, shared_buffer.buffer, &mem_requirements);

    uint32_t memory_type_index =
        find_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_properties);
    if (memory_type_index == UINT32_MAX) {
        ADD_FAILURE() << "Failed to find DEVICE_LOCAL Vulkan memory type for shared buffer";
        return {};
    }

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = k_external_memory_handle_type;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &export_info;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;

    res = vkAllocateMemory(context.device, &alloc_info, nullptr, &shared_buffer.memory);
    EXPECT_EQ(res, VK_SUCCESS);
    if (res != VK_SUCCESS) {
        return {};
    }

    res = vkBindBufferMemory(context.device, shared_buffer.buffer, shared_buffer.memory, 0);
    EXPECT_EQ(res, VK_SUCCESS);
    if (res != VK_SUCCESS) {
        return {};
    }

    if (export_vulkan_memory_handle(context.device, shared_buffer.memory, shared_buffer.shared_handle)) {
        std::cout << "[INFO] Vulkan shared buffer config: usage=STORAGE|XFER_SRC|XFER_DST, memory=DEVICE_LOCAL\n";
    }

    return shared_buffer;
}

TEST(GpuSharedBufferRemoteTensor, smoke_VulkanRemoteInputToRemoteOutputCopyAndCompare) {
    std::cout << "skip because driver on ubuntu 22 too old" << std::endl;
    GTEST_SKIP();
    ov::Core core;
    const ov::Shape shape{16'000};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    const std::string selected_gpu_id = "0";
    const std::string selected_gpu_device = "GPU." + selected_gpu_id;

    auto candidate_ctx = core.get_default_context(selected_gpu_device).as<ov::intel_gpu::ocl::ClContext>();
    auto params = candidate_ctx.get_params();
    auto it = params.find(ov::intel_gpu::ocl_context.name());
    if (it == params.end()) {
        FAIL() << "Failed to get OpenCL context for " << selected_gpu_device;
    }

    auto cl_ctx = static_cast<cl_context>(it->second.as<ov::intel_gpu::ocl::gpu_handle_param>());
    cl_device_id cl_device = nullptr;
    ASSERT_TRUE(get_context_first_device(cl_ctx, cl_device));
    if (!supports_external_import_handle_type(cl_device, k_cl_external_memory_handle_type)) {
        GTEST_SKIP() << "Device does not support required external-memory handle import type";
    }

    DeviceId cl_luid{};
    if (!get_context_device_luid(cl_ctx, cl_luid)) {
        FAIL() << "Failed to get LUID for " << selected_gpu_device;
    }

    VulkanTestContext vk_ctx = create_vulkan_test_context(cl_luid);
    if (vk_ctx.device == VK_NULL_HANDLE) {
        GTEST_SKIP() << "Failed to create Vulkan context for selected GPU LUID";
    }

    auto vk_input_shared = create_vulkan_shared_buffer(vk_ctx, byte_size);
    auto vk_output_shared = create_vulkan_shared_buffer(vk_ctx, byte_size);
    ASSERT_NE(vk_input_shared.shared_handle, invalid_external_memory_handle());
    ASSERT_NE(vk_output_shared.shared_handle, invalid_external_memory_handle());

    auto ov_ctx = core.get_default_context(selected_gpu_device).as<ov::intel_gpu::ocl::ClContext>();

    ov::RemoteTensor remote_input_tensor;
    ov::RemoteTensor remote_output_tensor;

    try {
        remote_input_tensor = ov_ctx.create_tensor(ov::element::f32,
                                                   shape,
                                                   reinterpret_cast<void*>(vk_input_shared.shared_handle),
                                                   ov::intel_gpu::MemType::SHARED_BUF);
        remote_output_tensor = ov_ctx.create_tensor(ov::element::f32,
                                                    shape,
                                                    reinterpret_cast<void*>(vk_output_shared.shared_handle),
                                                    ov::intel_gpu::MemType::SHARED_BUF);
    } catch (const ov::Exception& ex) {
        std::cout << "[INFO] Vulkan NT handle import not supported on this device: " << ex.what() << "\n";
        GTEST_SKIP() << "Vulkan NT handle import not supported on this configuration";
    }

    std::vector<float> input_init(element_count, 2.0f);
    ov::Tensor host_input_init(ov::element::f32, shape);
    std::memcpy(host_input_init.data(), input_init.data(), byte_size);
    remote_input_tensor.copy_from(host_input_init);

    std::vector<float> output_init(element_count, 0.0f);
    ov::Tensor host_output_init(ov::element::f32, shape);
    std::memcpy(host_output_init.data(), output_init.data(), byte_size);
    remote_output_tensor.copy_from(host_output_init);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, ov_ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_tensor(compiled.input(), remote_input_tensor);
    infer_req.set_tensor(compiled.output(), remote_output_tensor);

    ov::Tensor host_input(ov::element::f32, shape);
    remote_input_tensor.copy_to(host_input);
    const auto* input_values = host_input.data<const float>();
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(input_values[i], 2.0f) << "Input mismatch at index " << i;
    }

    infer_req.infer();

    ov::Tensor host_output(ov::element::f32, shape);
    remote_output_tensor.copy_to(host_output);
    const auto* output_values = host_output.data<const float>();

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_values[i], 2.0f) << "Mismatch at index " << i;
    }
}

}

#endif
