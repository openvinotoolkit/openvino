// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime/except.hpp"
#include "vk_common.hpp"
#include "vk_spirv_reflection.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

class vk_kernel {
public:
    vk_kernel(VkDevice device,
              VkPipeline pipeline,
              VkPipelineLayout layout,
              VkDescriptorSetLayout ds_layout,
              std::vector<uint32_t> spirv,
              vk_kernel_reflection reflection)
        : _device(device)
        , _pipeline(pipeline)
        , _layout(layout)
        , _ds_layout(ds_layout)
        , _kernel_id(reflection.name)
        , _spirv(std::move(spirv))
        , _reflection(std::move(reflection)) {}

    ~vk_kernel() {
        if (_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(_device, _pipeline, nullptr);
        }
        if (_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(_device, _layout, nullptr);
        }
        if (_ds_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(_device, _ds_layout, nullptr);
        }
    }

    vk_kernel(const vk_kernel&) = delete;
    vk_kernel& operator=(const vk_kernel&) = delete;

    static std::shared_ptr<vk_kernel> create_kernel(VkDevice device,
                                                    const std::vector<uint32_t>& spirv,
                                                    const vk_kernel_reflection& reflection,
                                                    VkPipelineCache pipeline_cache = VK_NULL_HANDLE);

    const std::string& get_id() const { return _kernel_id; }
    VkPipeline get_handle() const { return _pipeline; }
    VkPipelineLayout get_layout() const { return _layout; }

    const vk_kernel_reflection& get_reflection() const { return _reflection; }

    std::vector<uint8_t> get_binary() const {
        std::vector<uint8_t> binary;
        binary.resize(_spirv.size() * sizeof(uint32_t));
        std::memcpy(binary.data(), _spirv.data(), binary.size());
        return binary;
    }

private:
    VkDevice _device;
    VkPipeline _pipeline;
    VkPipelineLayout _layout;
    VkDescriptorSetLayout _ds_layout;
    std::string _kernel_id;
    std::vector<uint32_t> _spirv;
    vk_kernel_reflection _reflection;
};

using vk_kernel_ptr = std::shared_ptr<vk_kernel>;

inline vk_kernel_ptr vk_kernel::create_kernel(VkDevice device,
                                              const std::vector<uint32_t>& spirv,
                                              const vk_kernel_reflection& reflection,
                                              VkPipelineCache pipeline_cache) {
    constexpr uint32_t max_buffer_bindings = 16;

    std::vector<VkDescriptorSetLayoutBinding> bindings(max_buffer_bindings);
    for (uint32_t i = 0; i < max_buffer_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // Descriptor types per binding from the clspv reflection: sampled images
    // are SAMPLED_IMAGE, write-only images STORAGE_IMAGE, everything else
    // stays STORAGE_BUFFER.
    for (const auto& arg : reflection.args) {
        switch (arg.kind) {
        case vk_arg_kind::sampled_image:
            bindings[arg.binding].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            break;
        case vk_arg_kind::storage_image:
            bindings[arg.binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            break;
        default:
            break;
        }
    }

    VkDescriptorSetLayoutCreateInfo ds_layout_info{};
    ds_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    ds_layout_info.pBindings = bindings.data();

    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VK_CALL(vkCreateDescriptorSetLayout(device, &ds_layout_info, nullptr, &ds_layout), "vkCreateDescriptorSetLayout");

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size() * sizeof(uint32_t);
    module_info.pCode = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CALL(vkCreateShaderModule(device, &module_info, nullptr, &module), "vkCreateShaderModule");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = 128;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &ds_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CALL(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), "vkCreatePipelineLayout");

    // When the module uses spec-constant workgroup sizes, specialize
    // SpecId 0/1/2 (gl_WorkGroupSize x/y/z) with the kernel's required size.
    VkSpecializationMapEntry spec_entries[3]{};
    uint32_t spec_values[3] = {1, 1, 1};
    VkSpecializationInfo spec_info{};
    if (reflection.uses_spec_wgsize) {
        spec_values[0] = reflection.local_size[0];
        spec_values[1] = reflection.local_size[1];
        spec_values[2] = reflection.local_size[2];
        for (uint32_t i = 0; i < 3; ++i) {
            spec_entries[i].constantID = i;
            spec_entries[i].offset = i * sizeof(uint32_t);
            spec_entries[i].size = sizeof(uint32_t);
        }
        spec_info.mapEntryCount = 3;
        spec_info.pMapEntries = spec_entries;
        spec_info.dataSize = sizeof(spec_values);
        spec_info.pData = spec_values;
    }

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = reflection.name.c_str();
    if (reflection.uses_spec_wgsize)
        pipeline_info.stage.pSpecializationInfo = &spec_info;
    pipeline_info.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CALL(vkCreateComputePipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");

    vkDestroyShaderModule(device, module, nullptr);

    return std::make_shared<vk_kernel>(device, pipeline, layout, ds_layout, spirv, reflection);
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
