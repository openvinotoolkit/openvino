// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel.hpp"

#include <algorithm>
#include <limits>
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
                                                                                   const std::array<size_t, 3>& local_size,
                                                                                   const vulkan_specialization_constants& specialization_constants) {
    OPENVINO_ASSERT(push_constants_size <= _state->device_owner->get_max_push_constants_size(),
                    "[GPU][Vulkan] Requested ",
                    push_constants_size,
                    " push-constant bytes, but the device limit is ",
                    _state->device_owner->get_max_push_constants_size());
    const auto& interface = _state->shader->interface;
    OPENVINO_ASSERT(descriptor_count == interface.descriptor_bindings.size(),
                    "[GPU][Vulkan] Kernel '",
                    _state->entry_point,
                    "' expects ",
                    interface.descriptor_bindings.size(),
                    " storage buffers, but dispatch provides ",
                    descriptor_count);
    OPENVINO_ASSERT(push_constants_size == interface.push_constant_size,
                    "[GPU][Vulkan] Kernel '",
                    _state->entry_point,
                    "' expects ",
                    interface.push_constant_size,
                    " push-constant bytes, but dispatch provides ",
                    push_constants_size);

    auto resolved_specialization_constants = specialization_constants;
    for (size_t axis = 0; axis < interface.local_size_specialization_ids.size(); ++axis) {
        const auto id = interface.local_size_specialization_ids[axis];
        if (!id.has_value()) {
            OPENVINO_ASSERT(local_size[axis] == interface.local_size_defaults[axis],
                            "[GPU][Vulkan] Local work-group size axis ",
                            axis,
                            " must use the shader's fixed value ",
                            interface.local_size_defaults[axis],
                            ", but dispatch requests ",
                            local_size[axis]);
            continue;
        }
        const auto existing = std::find_if(resolved_specialization_constants.begin(), resolved_specialization_constants.end(), [&](const auto& constant) {
            return constant.id == *id;
        });
        OPENVINO_ASSERT(local_size[axis] <= std::numeric_limits<uint32_t>::max(),
                        "[GPU][Vulkan] Local work-group size exceeds the 32-bit specialization range");
        const auto value = static_cast<uint32_t>(local_size[axis]);
        if (existing == resolved_specialization_constants.end()) {
            resolved_specialization_constants.push_back({*id, value});
        } else {
            OPENVINO_ASSERT(existing->value == value, "[GPU][Vulkan] Local-size specialization constant ", *id, " does not match dispatch axis ", axis);
        }
    }
    return _state->device_owner->get_pipeline_cache().get_or_create_pipeline(_state->shader, resolved_specialization_constants);
}

}  // namespace vulkan
}  // namespace cldnn
