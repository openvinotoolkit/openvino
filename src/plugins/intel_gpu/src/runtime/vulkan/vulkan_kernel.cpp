// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel.hpp"

#include <cstring>
#include <mutex>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_device.hpp"

namespace cldnn {
namespace vulkan {
namespace {

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

constexpr uint32_t spirv_magic = 0x07230203;

}  // namespace

struct vulkan_kernel::shared_state {
    shared_state(std::shared_ptr<vulkan_device> device_owner, std::vector<uint8_t> binary, std::string name)
        : device_owner(std::move(device_owner)),
          device(this->device_owner->get_device()),
          binary(std::move(binary)),
          entry_point(std::move(name)) {
        OPENVINO_ASSERT(this->binary.size() >= sizeof(uint32_t) && this->binary.size() % sizeof(uint32_t) == 0,
                        "[GPU][Vulkan] SPIR-V binary size must be a non-zero multiple of four bytes");

        uint32_t magic = 0;
        std::memcpy(&magic, this->binary.data(), sizeof(magic));
        OPENVINO_ASSERT(magic == spirv_magic, "[GPU][Vulkan] Invalid SPIR-V magic number");

        std::vector<uint32_t> words(this->binary.size() / sizeof(uint32_t));
        std::memcpy(words.data(), this->binary.data(), this->binary.size());

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = this->binary.size();
        shader_info.pCode = words.data();
        check_vk_result(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");
    }

    ~shared_state() {
        if (pipeline.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        }
        if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipeline.pipeline_layout, nullptr);
        }
        if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, pipeline.descriptor_set_layout, nullptr);
        }
        if (shader_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shader_module, nullptr);
        }
    }

    std::shared_ptr<vulkan_device> device_owner;
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    std::vector<uint8_t> binary;
    std::string entry_point;
    std::mutex mutex;
    vulkan_pipeline_state pipeline;
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

const vulkan_pipeline_state& vulkan_kernel::get_or_create_pipeline(uint32_t descriptor_count, uint32_t push_constants_size) {
    std::lock_guard<std::mutex> lock(_state->mutex);
    if (_state->pipeline.pipeline != VK_NULL_HANDLE) {
        OPENVINO_ASSERT(_state->pipeline.descriptor_count == descriptor_count && _state->pipeline.push_constants_size == push_constants_size,
                        "[GPU][Vulkan] A kernel cannot be reused with a different descriptor or push-constant layout");
        return _state->pipeline;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(descriptor_count);
    for (uint32_t index = 0; index < descriptor_count; ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
    descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = descriptor_count;
    descriptor_layout_info.pBindings = bindings.data();
    check_vk_result(vkCreateDescriptorSetLayout(_state->device, &descriptor_layout_info, nullptr, &_state->pipeline.descriptor_set_layout),
                    "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = push_constants_size;

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &_state->pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = push_constants_size == 0 ? 0 : 1;
    pipeline_layout_info.pPushConstantRanges = push_constants_size == 0 ? nullptr : &push_constant_range;
    check_vk_result(vkCreatePipelineLayout(_state->device, &pipeline_layout_info, nullptr, &_state->pipeline.pipeline_layout), "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = _state->shader_module;
    stage_info.pName = _state->entry_point.c_str();

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = _state->pipeline.pipeline_layout;
    check_vk_result(vkCreateComputePipelines(_state->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_state->pipeline.pipeline),
                    "vkCreateComputePipelines");

    _state->pipeline.descriptor_count = descriptor_count;
    _state->pipeline.push_constants_size = push_constants_size;
    return _state->pipeline;
}

}  // namespace vulkan
}  // namespace cldnn
