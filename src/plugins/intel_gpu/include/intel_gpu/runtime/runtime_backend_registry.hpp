// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine_configuration.hpp"

namespace cldnn {

class engine;
struct device;

struct gpu_operation_lowering_capabilities {
    bool direct_divide = false;
    bool direct_binary_power = false;
};

enum class gpu_cached_kernel_artifact : uint8_t {
    native_device_binary,
    spirv,
};

struct gpu_kernel_cache_policy {
    static constexpr uint32_t current_artifact_schema_version = 1;

    gpu_cached_kernel_artifact artifact = gpu_cached_kernel_artifact::native_device_binary;
    uint32_t artifact_schema_version = current_artifact_schema_version;
};

struct runtime_backend_descriptor {
    engine_types engine_type;
    runtime_types runtime_type;
    const char* name;
    gpu_operation_lowering_capabilities operation_lowering;
    gpu_kernel_cache_policy kernel_cache;
};

/// Describes the runtimes compiled into this Intel GPU plugin binary. Runtime
/// selection is performed during device/context construction, never per dispatch.
class runtime_backend_registry {
public:
    static const std::vector<runtime_backend_descriptor>& compiled_backends();
    static const runtime_backend_descriptor& default_backend();
    static const runtime_backend_descriptor& get(runtime_types runtime_type);

    static std::map<std::string, std::shared_ptr<device>> query_devices(engine_types engine_type,
                                                                        runtime_types runtime_type,
                                                                        void* user_context,
                                                                        void* user_device,
                                                                        int context_device_id,
                                                                        int target_tile_id,
                                                                        bool initialize_devices);
    static std::shared_ptr<engine> create_engine(engine_types engine_type, runtime_types runtime_type, const std::shared_ptr<device>& device);

    static std::string make_device_id(runtime_types runtime_type, const std::string& backend_device_id);
    static std::string make_public_device_id(runtime_types runtime_type, const std::string& backend_device_id);
    static std::string select_default_device_id(const std::vector<std::pair<runtime_types, std::string>>& available_devices);
    static bool parse_device_id(const std::string& device_id, runtime_types& runtime_type, std::string& backend_device_id);
};

}  // namespace cldnn
