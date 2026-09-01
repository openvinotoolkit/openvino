// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel.hpp"

#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "vulkan_device.hpp"

namespace cldnn {
namespace vulkan {
struct vulkan_kernel::shared_state {
    shared_state(std::shared_ptr<vulkan_device> device_owner, std::vector<uint8_t> binary, std::string name)
        : device_owner(std::move(device_owner)),
          binary(std::move(binary)),
          entry_point(std::move(name)) {
        shader = this->device_owner->get_pipeline_cache().get_or_create_shader(this->binary, entry_point);
    }

    std::shared_ptr<vulkan_device> device_owner;
    std::vector<uint8_t> binary;
    std::string entry_point;
    std::shared_ptr<const vulkan_shader_state> shader;
};

vulkan_kernel::vulkan_kernel(std::shared_ptr<vulkan_device> device, std::vector<uint8_t> spirv, std::string entry_point)
    : _state(std::make_shared<shared_state>(std::move(device), std::move(spirv), std::move(entry_point))) {}

vulkan_kernel::vulkan_kernel(std::shared_ptr<shared_state> state) : _state(std::move(state)) {}

std::shared_ptr<kernel> vulkan_kernel::clone(bool) const {
    return std::shared_ptr<kernel>(new vulkan_kernel(_state));
}

bool vulkan_kernel::is_same(const kernel& other) const {
    const auto* other_kernel = dynamic_cast<const vulkan_kernel*>(&other);
    return other_kernel != nullptr && _state == other_kernel->_state;
}

std::string vulkan_kernel::get_id() const {
    return _state->entry_point;
}

std::vector<uint8_t> vulkan_kernel::get_binary() const {
    return _state->binary;
}

std::string vulkan_kernel::get_build_log() const {
    return {};
}

std::shared_ptr<const vulkan_pipeline_state> vulkan_kernel::get_or_create_pipeline(uint32_t descriptor_count,
                                                                                   uint32_t push_constants_size,
                                                                                   const vulkan_specialization_constants& specialization_constants) {
    OPENVINO_ASSERT(push_constants_size <= _state->device_owner->get_max_push_constants_size(),
                    "[GPU][Vulkan] Requested ",
                    push_constants_size,
                    " push-constant bytes, but the device limit is ",
                    _state->device_owner->get_max_push_constants_size());
    return _state->device_owner->get_pipeline_cache().get_or_create_pipeline(_state->shader, descriptor_count, push_constants_size, specialization_constants);
}

}  // namespace vulkan
}  // namespace cldnn
