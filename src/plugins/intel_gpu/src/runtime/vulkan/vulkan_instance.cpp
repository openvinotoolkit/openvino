// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_instance.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "vulkan_api.hpp"

namespace cldnn {
namespace vulkan {

vulkan_instance::ptr vulkan_instance::create() {
    const auto extensions =
        enumerate_vulkan<VkExtensionProperties>("vkEnumerateInstanceExtensionProperties", [](uint32_t* count, VkExtensionProperties* entries) {
            return vkEnumerateInstanceExtensionProperties(nullptr, count, entries);
        });

    const auto has_portability_enumeration = std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
        return std::string(extension.extensionName) == VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    });

    std::vector<const char*> enabled_extensions;
    VkInstanceCreateFlags flags = 0;
    if (has_portability_enumeration) {
        enabled_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkApplicationInfo application_info{};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "OpenVINO Intel GPU plugin";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application_info.pEngineName = "OpenVINO Intel GPU Vulkan runtime";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application_info.apiVersion = required_api_version;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.flags = flags;
    create_info.pApplicationInfo = &application_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames = enabled_extensions.empty() ? nullptr : enabled_extensions.data();

    auto instance = ptr(new vulkan_instance(VK_NULL_HANDLE, has_portability_enumeration));
    VkInstance handle = VK_NULL_HANDLE;
    const auto result = vkCreateInstance(&create_info, nullptr, &handle);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER || result == VK_ERROR_INITIALIZATION_FAILED) {
        return {};
    }
    check_vk_result(result, "vkCreateInstance");
    instance->_instance = handle;
    return instance;
}

vulkan_instance::~vulkan_instance() {
    if (_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(_instance, nullptr);
    }
}

}  // namespace vulkan
}  // namespace cldnn
