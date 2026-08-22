// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "engine_configuration.hpp"

namespace cldnn {

class engine;
struct device;

enum class runtime_interop_kind {
    opencl,
    level_zero,
    native,
};

struct runtime_backend_descriptor {
    engine_types engine_type;
    runtime_types runtime_type;
    const char* name;
    runtime_interop_kind interop_kind;
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
    static std::shared_ptr<engine> create_engine(engine_types engine_type,
                                                 runtime_types runtime_type,
                                                 const std::shared_ptr<device>& device);

    static std::string make_device_id(runtime_types runtime_type, const std::string& backend_device_id);
    static std::string make_public_device_id(runtime_types runtime_type, const std::string& backend_device_id);
    static bool parse_device_id(const std::string& device_id, runtime_types& runtime_type, std::string& backend_device_id);
};

}  // namespace cldnn
