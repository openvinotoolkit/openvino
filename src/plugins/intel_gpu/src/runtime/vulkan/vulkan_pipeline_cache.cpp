// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_pipeline_cache.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>

#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"

namespace cldnn {
namespace vulkan {
namespace {

constexpr uint32_t spirv_magic = 0x07230203;
constexpr uint32_t persistent_cache_format_version = 1;
constexpr uint64_t max_persistent_cache_payload_bytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::array<uint8_t, 8> persistent_cache_magic{'O', 'V', 'V', 'K', 'P', 'C', '0', '1'};

struct persistent_cache_identity {
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t driver_version = 0;
    uint32_t api_version = 0;
    uint32_t driver_id = 0;
    uint32_t portability_subset = 0;
    uint64_t build_identity_hash = 0;
    std::string build_identity;
    std::array<uint8_t, 4> conformance_version{};
    std::array<uint8_t, VK_UUID_SIZE> device_uuid{};
    std::array<uint8_t, VK_UUID_SIZE> driver_uuid{};
    std::array<uint8_t, VK_UUID_SIZE> pipeline_cache_uuid{};
};

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

uint64_t stable_string_hash(const char* value) {
    constexpr uint64_t offset_basis = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset_basis;
    if (value == nullptr) {
        return hash;
    }
    for (; *value != '\0'; ++value) {
        hash ^= static_cast<uint8_t>(*value);
        hash *= prime;
    }
    return hash;
}

template <typename T>
void append_value(std::vector<uint8_t>& bytes, const T& value) {
    const auto* data = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(T));
}

template <size_t Size>
void append_bytes(std::vector<uint8_t>& bytes, const std::array<uint8_t, Size>& value) {
    bytes.insert(bytes.end(), value.begin(), value.end());
}

template <typename T>
bool read_value(const std::vector<uint8_t>& bytes, size_t& offset, T& value) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        return false;
    }
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

persistent_cache_identity query_persistent_cache_identity(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceDriverProperties driver_properties{};
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    id_properties.pNext = &driver_properties;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties);

    persistent_cache_identity identity;
    identity.vendor_id = properties.properties.vendorID;
    identity.device_id = properties.properties.deviceID;
    identity.driver_version = properties.properties.driverVersion;
    identity.api_version = properties.properties.apiVersion;
    identity.driver_id = static_cast<uint32_t>(driver_properties.driverID);
    const auto* build_number = ov::get_openvino_version().buildNumber;
    identity.build_identity = build_number == nullptr ? std::string{} : build_number;
    identity.build_identity_hash = stable_string_hash(build_number);
    identity.conformance_version = {driver_properties.conformanceVersion.major,
                                    driver_properties.conformanceVersion.minor,
                                    driver_properties.conformanceVersion.subminor,
                                    driver_properties.conformanceVersion.patch};
    std::copy_n(id_properties.deviceUUID, VK_UUID_SIZE, identity.device_uuid.begin());
    std::copy_n(id_properties.driverUUID, VK_UUID_SIZE, identity.driver_uuid.begin());
    std::copy_n(properties.properties.pipelineCacheUUID, VK_UUID_SIZE, identity.pipeline_cache_uuid.begin());

    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) == VK_SUCCESS) {
        std::vector<VkExtensionProperties> extensions(extension_count);
        const auto result =
            extension_count == 0 ? VK_SUCCESS : vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());
        if (result == VK_SUCCESS) {
            identity.portability_subset = static_cast<uint32_t>(std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
                return std::strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0;
            }));
        }
    }
    return identity;
}

std::vector<uint8_t> serialize_identity(const persistent_cache_identity& identity) {
    std::vector<uint8_t> bytes;
    bytes.reserve(persistent_cache_magic.size() + sizeof(uint32_t) * 8 + sizeof(uint64_t) + identity.build_identity.size() +
                  identity.conformance_version.size() + VK_UUID_SIZE * 3);
    append_bytes(bytes, persistent_cache_magic);
    append_value(bytes, persistent_cache_format_version);
    append_value(bytes, identity.vendor_id);
    append_value(bytes, identity.device_id);
    append_value(bytes, identity.driver_version);
    append_value(bytes, identity.api_version);
    append_value(bytes, identity.driver_id);
    append_value(bytes, identity.portability_subset);
    append_value(bytes, identity.build_identity_hash);
    append_value(bytes, static_cast<uint32_t>(identity.build_identity.size()));
    bytes.insert(bytes.end(), identity.build_identity.begin(), identity.build_identity.end());
    append_bytes(bytes, identity.conformance_version);
    append_bytes(bytes, identity.device_uuid);
    append_bytes(bytes, identity.driver_uuid);
    append_bytes(bytes, identity.pipeline_cache_uuid);
    return bytes;
}

std::string hex_bytes(const std::array<uint8_t, VK_UUID_SIZE>& bytes) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : bytes) {
        stream << std::setw(2) << static_cast<uint32_t>(byte);
    }
    return stream.str();
}

std::filesystem::path make_persistent_cache_path(const persistent_cache_identity& identity) {
    const auto* directory = std::getenv("OV_GPU_VULKAN_PIPELINE_CACHE_DIR");
    if (directory == nullptr || directory[0] == '\0') {
        return {};
    }

    std::ostringstream filename;
    filename << "openvino-vulkan-" << std::hex << identity.vendor_id << '-' << identity.device_id << '-' << identity.driver_version << '-'
             << identity.build_identity_hash << '-' << hex_bytes(identity.device_uuid) << '-' << hex_bytes(identity.driver_uuid) << '-'
             << hex_bytes(identity.pipeline_cache_uuid) << ".bin";
    return std::filesystem::path(directory) / filename.str();
}

std::vector<uint8_t> read_cache_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const auto end = stream.tellg();
    if (end <= 0 || static_cast<uint64_t>(end) > max_persistent_cache_payload_bytes + 1024) {
        return {};
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return stream ? bytes : std::vector<uint8_t>{};
}

bool extract_cache_payload(const std::vector<uint8_t>& file, const std::vector<uint8_t>& expected_header, std::vector<uint8_t>& payload) {
    if (file.size() < expected_header.size() + sizeof(uint64_t) || !std::equal(expected_header.begin(), expected_header.end(), file.begin())) {
        return false;
    }
    size_t offset = expected_header.size();
    uint64_t payload_size = 0;
    if (!read_value(file, offset, payload_size) || payload_size > max_persistent_cache_payload_bytes || payload_size != file.size() - offset) {
        return false;
    }
    payload.assign(file.begin() + static_cast<std::ptrdiff_t>(offset), file.end());
    return true;
}

}  // namespace

bool vulkan_pipeline_cache::pipeline_key::operator<(const pipeline_key& other) const {
    return std::tie(shader_identity, specialization_constants) < std::tie(other.shader_identity, other.specialization_constants);
}

vulkan_pipeline_cache::vulkan_pipeline_cache(VkDevice device, VkPhysicalDevice physical_device) : _device(device), _diagnostics_enabled(diagnostics_enabled()) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE && physical_device != VK_NULL_HANDLE,
                    "[GPU][Vulkan] Cannot create a pipeline cache for a null logical or physical device");

    const auto identity = query_persistent_cache_identity(physical_device);
    _persistent_cache_header = serialize_identity(identity);
    try {
        _persistent_cache_path = make_persistent_cache_path(identity);
        _persistent_cache_enabled = !_persistent_cache_path.empty();
        if (_persistent_cache_enabled) {
            std::filesystem::create_directories(_persistent_cache_path.parent_path());
        }
    } catch (...) {
        _persistent_cache_path.clear();
        _persistent_cache_enabled = false;
    }

    std::vector<uint8_t> initial_data;
    if (_persistent_cache_enabled) {
        try {
            const auto file = read_cache_file(_persistent_cache_path);
            if (!file.empty() && !extract_cache_payload(file, _persistent_cache_header, initial_data)) {
                _persistent_cache_rejected = true;
            }
        } catch (...) {
            _persistent_cache_rejected = true;
            initial_data.clear();
        }
    }

    VkPipelineCacheCreateInfo cache_info{};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_info.initialDataSize = initial_data.size();
    cache_info.pInitialData = initial_data.empty() ? nullptr : initial_data.data();
    auto result = vkCreatePipelineCache(_device, &cache_info, nullptr, &_driver_cache);
    if (result != VK_SUCCESS && !initial_data.empty()) {
        _persistent_cache_rejected = true;
        initial_data.clear();
        cache_info.initialDataSize = 0;
        cache_info.pInitialData = nullptr;
        result = vkCreatePipelineCache(_device, &cache_info, nullptr, &_driver_cache);
    }
    check_vk_result(result, "vkCreatePipelineCache");
    _persistent_cache_loaded_bytes = initial_data.size();
}

vulkan_pipeline_cache::~vulkan_pipeline_cache() {
    save_persistent_cache();
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
                  << " pipeline_create_ms=" << static_cast<double>(_pipeline_creation_nanoseconds) / 1'000'000.0
                  << " persistent_enabled=" << _persistent_cache_enabled << " persistent_loaded_bytes=" << _persistent_cache_loaded_bytes
                  << " persistent_saved_bytes=" << _persistent_cache_saved_bytes << " persistent_rejected=" << _persistent_cache_rejected
                  << " persistent_path=\"" << _persistent_cache_path.string() << "\"" << std::endl;
    }
}

void vulkan_pipeline_cache::save_persistent_cache() noexcept {
    if (!_persistent_cache_enabled || _driver_cache == VK_NULL_HANDLE) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(_mutex);
        size_t payload_size = 0;
        if (vkGetPipelineCacheData(_device, _driver_cache, &payload_size, nullptr) != VK_SUCCESS || payload_size == 0 ||
            payload_size > max_persistent_cache_payload_bytes) {
            return;
        }
        std::vector<uint8_t> payload(payload_size);
        if (vkGetPipelineCacheData(_device, _driver_cache, &payload_size, payload.data()) != VK_SUCCESS || payload_size > payload.size()) {
            return;
        }
        payload.resize(payload_size);

        auto file = _persistent_cache_header;
        append_value(file, static_cast<uint64_t>(payload.size()));
        file.insert(file.end(), payload.begin(), payload.end());

        const auto unique_suffix =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + '-' + std::to_string(reinterpret_cast<uintptr_t>(this));
        auto temporary_path = _persistent_cache_path;
        temporary_path += ".tmp." + unique_suffix;
        {
            std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
            if (!stream) {
                return;
            }
            stream.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
            stream.flush();
            if (!stream) {
                std::error_code ignored;
                std::filesystem::remove(temporary_path, ignored);
                return;
            }
        }

        std::error_code error;
        std::filesystem::rename(temporary_path, _persistent_cache_path, error);
        if (error) {
            std::filesystem::remove(_persistent_cache_path, error);
            error.clear();
            std::filesystem::rename(temporary_path, _persistent_cache_path, error);
        }
        if (error) {
            std::filesystem::remove(temporary_path, error);
            return;
        }
        _persistent_cache_saved_bytes = payload.size();
    } catch (...) {
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
    shader->interface = vulkan_kernel_interface::reflect(spirv, entry_point);
    check_vk_result(vkCreateShaderModule(_device, &shader_info, nullptr, &shader->module), "vkCreateShaderModule");
    _shaders.emplace(key, shader);
    ++_shader_misses;
    _shader_creation_nanoseconds += elapsed_nanoseconds(start);
    return shader;
}

std::shared_ptr<const vulkan_pipeline_state> vulkan_pipeline_cache::get_or_create_pipeline(const std::shared_ptr<const vulkan_shader_state>& shader,
                                                                                           const vulkan_specialization_constants& specialization_constants) {
    OPENVINO_ASSERT(shader != nullptr && shader->module != VK_NULL_HANDLE, "[GPU][Vulkan] Cannot create a pipeline for a null shader");

    specialization_key constants;
    constants.reserve(specialization_constants.size());
    for (const auto& constant : specialization_constants) {
        constants.emplace_back(constant.id, constant.value);
    }
    std::sort(constants.begin(), constants.end());
    for (size_t index = 1; index < constants.size(); ++index) {
        OPENVINO_ASSERT(constants[index - 1].first != constants[index].first, "[GPU][Vulkan] Duplicate specialization constant id ", constants[index].first);
    }
    for (const auto& constant : constants) {
        OPENVINO_ASSERT(std::binary_search(shader->interface.specialization_ids.begin(), shader->interface.specialization_ids.end(), constant.first),
                        "[GPU][Vulkan] Shader does not declare specialization constant id ",
                        constant.first);
    }

    std::lock_guard<std::mutex> lock(_mutex);
    pipeline_key key{shader->identity, constants};
    const auto existing = _pipelines.find(key);
    if (existing != _pipelines.end()) {
        ++_pipeline_hits;
        return existing->second;
    }

    const auto start = std::chrono::steady_clock::now();
    auto pipeline = std::make_shared<vulkan_pipeline_state>();
    pipeline->descriptor_count = static_cast<uint32_t>(shader->interface.descriptor_bindings.size());
    pipeline->push_constants_size = shader->interface.push_constant_size;
    pipeline->shader = shader;

    try {
        std::vector<VkDescriptorSetLayoutBinding> bindings(shader->interface.descriptor_bindings.size());
        for (size_t index = 0; index < bindings.size(); ++index) {
            bindings[index].binding = shader->interface.descriptor_bindings[index].binding;
            bindings[index].descriptorType = shader->interface.descriptor_bindings[index].type;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
        descriptor_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptor_layout_info.pBindings = bindings.empty() ? nullptr : bindings.data();
        check_vk_result(vkCreateDescriptorSetLayout(_device, &descriptor_layout_info, nullptr, &pipeline->descriptor_set_layout),
                        "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_constant_range{};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = shader->interface.push_constant_size;

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &pipeline->descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = shader->interface.push_constant_size == 0 ? 0 : 1;
        pipeline_layout_info.pPushConstantRanges = shader->interface.push_constant_size == 0 ? nullptr : &push_constant_range;
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
