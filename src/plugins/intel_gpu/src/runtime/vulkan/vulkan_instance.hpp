// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <memory>

namespace cldnn {
namespace vulkan {

class vulkan_instance {
public:
    using ptr = std::shared_ptr<vulkan_instance>;

    static ptr create();

    vulkan_instance(const vulkan_instance&) = delete;
    vulkan_instance& operator=(const vulkan_instance&) = delete;

    ~vulkan_instance();

    VkInstance get() const {
        return _instance;
    }

    bool portability_enumeration_enabled() const {
        return _portability_enumeration_enabled;
    }

private:
    vulkan_instance(VkInstance instance, bool portability_enumeration_enabled)
        : _instance(instance),
          _portability_enumeration_enabled(portability_enumeration_enabled) {}

    VkInstance _instance = VK_NULL_HANDLE;
    bool _portability_enumeration_enabled = false;
};

}  // namespace vulkan
}  // namespace cldnn
