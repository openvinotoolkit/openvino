// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/stream.hpp"
#include "cldnn/runtime/debug_configuration.hpp"

#include "ocl/ocl_memory.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

memory::memory(engine* engine, const layout& layout, allocation_type type, bool reused)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), _type(type), _reused(reused) {
    if (!_reused && _engine) {
        _engine->add_memory_used(_bytes_count);
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 1) {
        GPU_DEBUG_COUT << "Allocate " << _bytes_count << " bytes of " << type << " allocation type"
                       << " (current=" << _engine->get_used_device_memory() << ";"
                       << " max=" << _engine->get_max_used_device_memory() << ")" << std::endl;
    }
}

memory::~memory() {
    if (!_reused && _engine) {
        _engine->subtract_memory_used(_bytes_count);
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 1) {
        GPU_DEBUG_COUT << "Free " << _bytes_count << " bytes"
                       << " (current=" << _engine->get_used_device_memory() << ";"
                       << " max=" << _engine->get_max_used_device_memory() << ")" << std::endl;
    }
}

std::unique_ptr<surfaces_lock> surfaces_lock::create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream) {
    switch (engine_type) {
    case engine_types::ocl: return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, stream));
    default: throw std::runtime_error("Unsupported engine type in surfaces_lock::create");
    }
}

}  // namespace cldnn
