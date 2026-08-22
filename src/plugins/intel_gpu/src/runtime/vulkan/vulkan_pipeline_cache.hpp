// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace vulkan {

struct vulkan_specialization_constant {
    uint32_t id = 0;
    uint32_t value = 0;
};

using vulkan_specialization_constants = std::vector<vulkan_specialization_constant>;

struct vulkan_shader_state {
    uint64_t identity = 0;
    VkShaderModule module = VK_NULL_HANDLE;
    std::string entry_point;
};

struct vulkan_pipeline_state {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t descriptor_count = 0;
    uint32_t push_constants_size = 0;
    std::shared_ptr<const vulkan_shader_state> shader;
};

class vulkan_pipeline_cache final {
public:
    vulkan_pipeline_cache(VkDevice device, VkPhysicalDevice physical_device);
    ~vulkan_pipeline_cache();

    vulkan_pipeline_cache(const vulkan_pipeline_cache&) = delete;
    vulkan_pipeline_cache& operator=(const vulkan_pipeline_cache&) = delete;

    std::shared_ptr<const vulkan_shader_state> get_or_create_shader(const std::vector<uint8_t>& spirv, const std::string& entry_point);
    std::shared_ptr<const vulkan_pipeline_state> get_or_create_pipeline(const std::shared_ptr<const vulkan_shader_state>& shader,
                                                                        uint32_t descriptor_count,
                                                                        uint32_t push_constants_size,
                                                                        const vulkan_specialization_constants& specialization_constants);

private:
    using shader_key = std::pair<std::string, std::vector<uint8_t>>;
    using specialization_key = std::vector<std::pair<uint32_t, uint32_t>>;

    struct pipeline_key {
        uint64_t shader_identity = 0;
        uint32_t descriptor_count = 0;
        uint32_t push_constants_size = 0;
        specialization_key specialization_constants;

        bool operator<(const pipeline_key& other) const;
    };

    void save_persistent_cache() noexcept;

    VkDevice _device = VK_NULL_HANDLE;
    VkPipelineCache _driver_cache = VK_NULL_HANDLE;
    std::filesystem::path _persistent_cache_path;
    std::vector<uint8_t> _persistent_cache_header;
    std::mutex _mutex;
    std::map<shader_key, std::shared_ptr<vulkan_shader_state>> _shaders;
    std::map<pipeline_key, std::shared_ptr<vulkan_pipeline_state>> _pipelines;
    uint64_t _next_shader_identity = 0;
    bool _diagnostics_enabled = false;
    uint64_t _shader_hits = 0;
    uint64_t _shader_misses = 0;
    uint64_t _pipeline_hits = 0;
    uint64_t _pipeline_misses = 0;
    uint64_t _shader_creation_nanoseconds = 0;
    uint64_t _pipeline_creation_nanoseconds = 0;
    uint64_t _persistent_cache_loaded_bytes = 0;
    uint64_t _persistent_cache_saved_bytes = 0;
    bool _persistent_cache_enabled = false;
    bool _persistent_cache_rejected = false;
};

}  // namespace vulkan
}  // namespace cldnn
