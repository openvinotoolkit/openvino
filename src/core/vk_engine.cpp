// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_engine.hpp"

#include "runtime/except.hpp"
#include "vk_kernel_builder.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

namespace {

// Platform defaults come from the environment (platform_config_from_env in
// vk_platform.hpp) unless a programmatic vk_platform_config was supplied.

uint32_t requested_api_version(vk_platform platform) {
    // Vulkan SC is based on Vulkan 1.2; MoltenVK is conservative (many
    // portability devices report 1.1/1.2), so cap both at 1.2. Desktop asks
    // for 1.3.
    return platform == vk_platform::desktop ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;
}

// True when VK_KHR_portability_enumeration is provided by the driver. Only
// MoltenVK (or forced testing) enables it.
bool portability_enumeration_available() {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data());
    return std::any_of(props.begin(), props.end(), [](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
    });
}

vk_engine_init_data init_vk_engine_data(const vk_platform_config& config) {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "OpenVINO";
    app_info.applicationVersion = 1;
    app_info.apiVersion = requested_api_version(config.platform);

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    // MoltenVK (and, when forced, any driver) enumerates portability devices
    // via VK_KHR_portability_enumeration + the portability bit.
    const bool portability = config.platform == vk_platform::moltenvk || config.force_portability_enumeration;
    std::vector<const char*> instance_extensions;
    if (portability && portability_enumeration_available()) {
        instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    if (!instance_extensions.empty()) {
        instance_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
        instance_info.ppEnabledExtensionNames = instance_extensions.data();
    }

    VkInstance raw_instance = VK_NULL_HANDLE;
    VK_CALL(vkCreateInstance(&instance_info, nullptr, &raw_instance), "vkCreateInstance");
    auto instance = make_instance(raw_instance);

    uint32_t device_count = 0;
    VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, nullptr), "vkEnumeratePhysicalDevices");
    OPENVINO_ASSERT(device_count > 0, "[GPU] No Vulkan physical devices found");

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, physical_devices.data()), "vkEnumeratePhysicalDevices");

    // Pick the best compute-capable device for the requested device name.
    // The cross-platform core serves every device name: "GPU" → discrete/
    // integrated GPUs, "CPU" → CPU-type drivers (SwiftShader, Lavapipe, ...),
    // "NPU" → Vulkan SC-class devices (none on desktop today). CPU devices are
    // never picked for "GPU" and vice versa.
    const std::string device_name = [&] {
        const std::string raw = config.device_name.empty() ? std::string("GPU") : config.device_name;
        return raw.substr(0, raw.find('.'));
    }();
    const bool want_cpu = device_name == "CPU";
    const bool want_npu = device_name == "NPU";

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    int best_score = -1;
    for (const auto& candidate : physical_devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(candidate, &properties);

        // Skip devices without compute queue support (e.g. pure presentation devices)
        try {
            find_compute_queue_family(candidate);
        } catch (...) {
            continue;
        }

        const bool is_cpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        if (is_cpu != want_cpu)
            continue;  // NPU has no Vulkan device class yet — nothing matches

        int score = 0;
        switch (properties.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 100; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 50;  break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            score = 100; break;
            default:                                     score = 1;   break;
        }
        if (score > best_score) {
            best_score = score;
            physical_device = candidate;
        }
    }
    if (physical_device == VK_NULL_HANDLE) {
        OPENVINO_THROW("[GPU] No Vulkan device matching the requested device name '",
                       device_name,
                       "' was found (GPU = discrete/integrated, CPU = a CPU-type driver like "
                       "SwiftShader/Lavapipe, NPU = a Vulkan SC device; none present)");
    }

    vk_engine_init_data init;
    init.instance = instance;
    init.physical_device = physical_device;
    init.device = std::make_shared<vk_device>(instance, physical_device);
    init.config = config;
    return init;
}

}  // namespace

vk_engine::vk_engine()
    : vk_engine(platform_config_from_env()) {}

vk_engine::vk_engine(const vk_platform_config& config)
    : vk_engine(init_vk_engine_data(config)) {}

vk_engine::vk_engine(vk_engine_init_data init)
    : _config(std::move(init.config))
    , _device(std::move(init.device)) {
    _device_handle.instance = _device->get_instance();
    _device_handle.physical = _device->get_physical_device();
    _device_handle.device = _device->get_device();
    _device_handle.queue_family = _device->get_queue_family_index();
    _device_handle.queue = _device->get_queue();

    _service_stream = std::make_unique<vk_stream>(_device_handle.device,
                                                  _device_handle.queue,
                                                  _device_handle.queue_family);
}

vk_engine::~vk_engine() = default;

vk_memory_ptr vk_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    auto mem = std::make_shared<vk_memory>(_device_handle.device,
                                           _device_handle.physical,
                                           _device_handle.queue,
                                           layout,
                                           type);
    if (reset) {
        mem->fill(0);
    }
    return mem;
}

vk_memory_ptr vk_engine::reinterpret_buffer(const vk_memory_ptr& mem, const layout& new_layout) {
    OPENVINO_ASSERT(mem, "[GPU] Vulkan reinterpret_buffer: memory is null");
    return std::make_shared<vk_memory>(mem, new_layout);
}

vk_stream_ptr vk_engine::create_stream(const ExecutionConfig&) const {
    return std::make_shared<vk_stream>(_device_handle.device,
                                       _device_handle.queue,
                                       _device_handle.queue_family);
}

std::shared_ptr<vk_kernel_builder> vk_engine::create_kernel_builder() const {
    return std::make_shared<vk_kernel_builder>(_device_handle.device, _config);
}

std::shared_ptr<vk_engine> vk_engine::create(const vk_platform_config& config) {
    return std::make_shared<vk_engine>(config);
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
