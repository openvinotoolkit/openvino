// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_pipeline_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <tuple>

#include "openvino/core/except.hpp"

namespace cldnn {
namespace vulkan {
namespace {

constexpr uint32_t spirv_magic = 0x07230203;
constexpr uint32_t local_size_specialization_id = 0;

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

bool diagnostics_enabled() {
    const auto* value = std::getenv("OV_GPU_VULKAN_CACHE_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

uint64_t elapsed_nanoseconds(const std::chrono::steady_clock::time_point& start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace

bool vulkan_pipeline_cache::pipeline_key::operator<(const pipeline_key& other) const {
    return std::tie(shader_identity, descriptor_count, push_constants_size, specialization_constants) <
           std::tie(other.shader_identity, other.descriptor_count, other.push_constants_size, other.specialization_constants);
}

vulkan_pipeline_cache::vulkan_pipeline_cache(VkDevice device) : _device(device), _diagnostics_enabled(diagnostics_enabled()) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE, "[GPU][Vulkan] Cannot create a pipeline cache for a null device");

    VkPipelineCacheCreateInfo cache_info{};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    check_vk_result(vkCreatePipelineCache(_device, &cache_info, nullptr, &_driver_cache), "vkCreatePipelineCache");
}

vulkan_pipeline_cache::~vulkan_pipeline_cache() {
    for (auto& entry : _pipelines) {
        auto& pipeline = *entry.second;
        if (pipeline.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(_device, pipeline.pipeline, nullptr);
        }
        if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(_device, pipeline.pipeline_layout, nullptr);
        }
        if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(_device, pipeline.descriptor_set_layout, nullptr);
        }
    }
    for (auto& entry : _shaders) {
        if (entry.second->module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(_device, entry.second->module, nullptr);
        }
    }
    if (_driver_cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(_device, _driver_cache, nullptr);
    }

    if (_diagnostics_enabled) {
        std::clog << "[GPU][Vulkan][Cache] shader_hits=" << _shader_hits << " shader_misses=" << _shader_misses << " pipeline_hits=" << _pipeline_hits
                  << " pipeline_misses=" << _pipeline_misses << " shader_create_ms=" << static_cast<double>(_shader_creation_nanoseconds) / 1'000'000.0
                  << " pipeline_create_ms=" << static_cast<double>(_pipeline_creation_nanoseconds) / 1'000'000.0 << std::endl;
    }
}

std::shared_ptr<const vulkan_shader_state> vulkan_pipeline_cache::get_or_create_shader(const std::vector<uint8_t>& spirv, const std::string& entry_point) {
    OPENVINO_ASSERT(spirv.size() >= sizeof(uint32_t) && spirv.size() % sizeof(uint32_t) == 0,
                    "[GPU][Vulkan] SPIR-V binary size must be a non-zero multiple of four bytes");
    uint32_t magic = 0;
    std::memcpy(&magic, spirv.data(), sizeof(magic));
    OPENVINO_ASSERT(magic == spirv_magic, "[GPU][Vulkan] Invalid SPIR-V magic number");

    std::lock_guard<std::mutex> lock(_mutex);
    const shader_key key{entry_point, spirv};
    const auto existing = _shaders.find(key);
    if (existing != _shaders.end()) {
        ++_shader_hits;
        return existing->second;
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size();
    shader_info.pCode = words.data();

    auto shader = std::make_shared<vulkan_shader_state>();
    shader->identity = _next_shader_identity++;
    shader->entry_point = entry_point;
    check_vk_result(vkCreateShaderModule(_device, &shader_info, nullptr, &shader->module), "vkCreateShaderModule");
    _shaders.emplace(key, shader);
    ++_shader_misses;
    _shader_creation_nanoseconds += elapsed_nanoseconds(start);
    return shader;
}

std::shared_ptr<const vulkan_pipeline_state> vulkan_pipeline_cache::get_or_create_pipeline(const std::shared_ptr<const vulkan_shader_state>& shader,
                                                                                           uint32_t descriptor_count,
                                                                                           uint32_t push_constants_size,
                                                                                           uint32_t specialized_local_size_x,
                                                                                           const specialization_constants_desc& specialization_constants) {
    OPENVINO_ASSERT(shader != nullptr && shader->module != VK_NULL_HANDLE, "[GPU][Vulkan] Cannot create a pipeline for a null shader");

    specialization_key constants;
    constants.reserve(specialization_constants.size() + (specialized_local_size_x == 0 ? 0 : 1));
    if (specialized_local_size_x != 0) {
        constants.emplace_back(local_size_specialization_id, specialized_local_size_x);
    }
    for (const auto& constant : specialization_constants) {
        OPENVINO_ASSERT(constant.id != local_size_specialization_id || specialized_local_size_x == 0,
                        "[GPU][Vulkan] Specialization constant id 0 is reserved for the local work-group size");
        constants.emplace_back(constant.id, constant.value);
    }
    std::sort(constants.begin(), constants.end());
    for (size_t index = 1; index < constants.size(); ++index) {
        OPENVINO_ASSERT(constants[index - 1].first != constants[index].first, "[GPU][Vulkan] Duplicate specialization constant id ", constants[index].first);
    }

    std::lock_guard<std::mutex> lock(_mutex);
    pipeline_key key{shader->identity, descriptor_count, push_constants_size, constants};
    const auto existing = _pipelines.find(key);
    if (existing != _pipelines.end()) {
        ++_pipeline_hits;
        return existing->second;
    }

    const auto start = std::chrono::steady_clock::now();
    auto pipeline = std::make_shared<vulkan_pipeline_state>();
    pipeline->descriptor_count = descriptor_count;
    pipeline->push_constants_size = push_constants_size;
    pipeline->shader = shader;

    try {
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
        descriptor_layout_info.pBindings = bindings.empty() ? nullptr : bindings.data();
        check_vk_result(vkCreateDescriptorSetLayout(_device, &descriptor_layout_info, nullptr, &pipeline->descriptor_set_layout),
                        "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = push_constants_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &pipeline->descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = push_constants_size == 0 ? 0 : 1;
        pipeline_layout_info.pPushConstantRanges = push_constants_size == 0 ? nullptr : &push_constant_range;
        check_vk_result(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &pipeline->pipeline_layout), "vkCreatePipelineLayout");

        std::vector<VkSpecializationMapEntry> specialization_entries;
        std::vector<uint32_t> specialization_values;
        specialization_entries.reserve(constants.size());
        specialization_values.reserve(constants.size());
        for (const auto& constant : constants) {
            VkSpecializationMapEntry entry{};
            entry.constantID = constant.first;
            entry.offset = static_cast<uint32_t>(specialization_values.size() * sizeof(uint32_t));
            entry.size = sizeof(uint32_t);
            specialization_entries.push_back(entry);
            specialization_values.push_back(constant.second);
        }

        VkSpecializationInfo specialization_info{};
        specialization_info.mapEntryCount = static_cast<uint32_t>(specialization_entries.size());
        specialization_info.pMapEntries = specialization_entries.data();
        specialization_info.dataSize = specialization_values.size() * sizeof(uint32_t);
        specialization_info.pData = specialization_values.data();

        VkPipelineShaderStageCreateInfo stage_info{};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = shader->module;
        stage_info.pName = shader->entry_point.c_str();
        stage_info.pSpecializationInfo = specialization_entries.empty() ? nullptr : &specialization_info;

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = pipeline->pipeline_layout;
        check_vk_result(vkCreateComputePipelines(_device, _driver_cache, 1, &pipeline_info, nullptr, &pipeline->pipeline), "vkCreateComputePipelines");
    } catch (...) {
        if (pipeline->pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(_device, pipeline->pipeline, nullptr);
        }
        if (pipeline->pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(_device, pipeline->pipeline_layout, nullptr);
        }
        if (pipeline->descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(_device, pipeline->descriptor_set_layout, nullptr);
        }
        throw;
    }

    _pipelines.emplace(std::move(key), pipeline);
    ++_pipeline_misses;
    _pipeline_creation_nanoseconds += elapsed_nanoseconds(start);
    return pipeline;
}

}  // namespace vulkan
}  // namespace cldnn
