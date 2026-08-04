// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_engine.hpp"
#include "vk_kernel_builder.hpp"
#include "vk_memory.hpp"
#include "vk_stream.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cldnn {
namespace vk {

namespace {

vk_engine_init_data init_vk_engine_data() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "OpenVINO";
    app_info.applicationVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkInstance raw_instance = VK_NULL_HANDLE;
    VK_CALL(vkCreateInstance(&instance_info, nullptr, &raw_instance), "vkCreateInstance");
    auto instance = vk::make_instance(raw_instance);

    uint32_t device_count = 0;
    VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, nullptr), "vkEnumeratePhysicalDevices");
    OPENVINO_ASSERT(device_count > 0, "[GPU] No Vulkan physical devices found");

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, physical_devices.data()), "vkEnumeratePhysicalDevices");

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    for (const auto& candidate : physical_devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device = candidate;
            break;
        }
    }
    if (physical_device == VK_NULL_HANDLE)
        physical_device = physical_devices.front();

    auto device = std::make_shared<vk_device>(instance, physical_device);

    vk_engine_init_data init;
    init.instance = instance;
    init.physical_device = physical_device;
    init.device = device;
    return init;
}

}  // namespace

vk_engine::vk_engine()
    : vk_engine(init_vk_engine_data()) {}

vk_engine::vk_engine(vk_engine_init_data init)
    : engine(std::move(init.device)) {
    auto device = downcast<vk_device>(_device.get());
    OPENVINO_ASSERT(device, "[GPU] Invalid device type stored in vk_engine");

    _device_handle.instance = device->get_instance();
    _device_handle.physical = device->get_physical_device();
    _device_handle.device = device->get_device();
    _device_handle.queue_family = device->get_queue_family_index();
    _device_handle.queue = device->get_queue();

    _service_stream = std::make_unique<vk_stream>(_device_handle.device,
                                                  _device_handle.queue,
                                                  _device_handle.queue_family,
                                                  ExecutionConfig());
}

engine_types vk_engine::type() const {
    return engine_types::vulkan;
}

runtime_types vk_engine::runtime_type() const {
    return runtime_types::vulkan;
}

backend_types vk_engine::backend_type() const {
    return backend_types::vulkan;
}

memory::ptr vk_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    auto mem = std::make_shared<vk_memory>(this,
                                           layout,
                                           type,
                                           _device_handle.device,
                                           _device_handle.physical,
                                           _device_handle.queue);
    if (reset) {
        mem->fill(get_service_stream(), 0, {}, true);
    }
    return mem;
}

memory::ptr vk_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    OPENVINO_THROW("[GPU] Vulkan reinterpret_handle: not implemented yet");
}

memory::ptr vk_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_THROW("[GPU] Vulkan create_subbuffer: not implemented yet");
}

memory::ptr vk_engine::create_hostbuffer(void* cpu_address, size_t data_size, allocation_type _allocation_type, const layout output_layout) {
    OPENVINO_THROW("[GPU] Vulkan create_hostbuffer: not implemented yet");
}

memory::ptr vk_engine::create_hostbuffer(const void* cpu_address, size_t data_size, allocation_type _allocation_type, const layout output_layout) {
    OPENVINO_THROW("[GPU] Vulkan create_hostbuffer: not implemented yet");
}

memory::ptr vk_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_THROW("[GPU] Vulkan reinterpret_buffer: not implemented yet");
}

memory::ptr vk_engine::import_buffer(const layout& layout, ov::intel_gpu::os_handle_param external_handle) {
    OPENVINO_THROW("[GPU] Vulkan import_buffer: not implemented yet");
}

bool vk_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    auto params1 = mem1.get_internal_params(runtime_types::vulkan);
    auto params2 = mem2.get_internal_params(runtime_types::vulkan);
    return params1.mem != nullptr && params1.mem == params2.mem;
}

void* vk_engine::get_user_context(runtime_types rt_type) const {
    return nullptr;
}

allocation_type vk_engine::detect_usm_allocation_type(const void* memory) const {
    if (use_unified_shared_memory()) {
        return allocation_type::usm_host;
    }
    return allocation_type::unknown;
}

stream::ptr vk_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<vk_stream>(_device_handle.device,
                                       _device_handle.queue,
                                       _device_handle.queue_family,
                                       config);
}

stream::ptr vk_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    OPENVINO_THROW("[GPU] Vulkan create_stream with external handle: not implemented yet");
}

std::shared_ptr<kernel_builder> vk_engine::create_kernel_builder() const {
    return std::make_shared<vk_kernel_builder>(_device_handle.device);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void vk_engine::create_onednn_engine(const ExecutionConfig& config) {
    OPENVINO_THROW("[GPU] Vulkan runtime does not support oneDNN");
}
#endif

std::shared_ptr<cldnn::engine> vk_engine::create() {
    return std::make_shared<vk_engine>();
}

std::shared_ptr<cldnn::engine> create_vk_engine(const device::ptr device, runtime_types runtime_type) {
    OPENVINO_ASSERT(runtime_type == runtime_types::vulkan, "[GPU] Invalid runtime type specified for Vulkan engine");
    return vk_engine::create();
}

}  // namespace vk
}  // namespace cldnn
