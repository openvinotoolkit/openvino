// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

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
        _engine->add_memory_used(_bytes_count, type);
        GPU_DEBUG_LOG << "Allocate " << _bytes_count << " bytes of " << type << " allocation type"
                      << " (current=" << _engine->get_used_device_memory(type) << ";"
                      << " max=" << _engine->get_max_used_device_memory(type) << ")" << std::endl;
    }
}

memory::~memory() {
    if (!_reused && _engine) {
        try {
            _engine->subtract_memory_used(_bytes_count, _type);
        } catch (...) {}
        GPU_DEBUG_LOG << "Free " << _bytes_count << " bytes of " << _type << " allocation type"
                      << " (current=" << _engine->get_used_device_memory(_type) << ";"
                      << " max=" << _engine->get_max_used_device_memory(_type) << ")" << std::endl;
    }
}

std::unique_ptr<surfaces_lock> surfaces_lock::create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream) {
    switch (engine_type) {
    case engine_types::ocl: return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, stream));
    default: throw std::runtime_error("Unsupported engine type in surfaces_lock::create");
    }
}

}  // namespace cldnn
