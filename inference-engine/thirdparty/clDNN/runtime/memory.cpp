// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/stream.hpp"

#include "ocl/ocl_memory.hpp"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {

memory::memory(engine* engine, const layout& layout, allocation_type type, bool reused)
    : _engine(engine), _layout(layout), _bytes_count(_layout.bytes_count()), _type(type), _reused(reused) {}

memory::~memory() {
    if (!_reused && _engine) {
        // TODO: Make memory usage tracker static in memory class
        _engine->get_memory_pool().subtract_memory_used(_bytes_count);
    }
}

std::unique_ptr<surfaces_lock> surfaces_lock::create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream) {
    switch (engine_type) {
    case engine_types::ocl: return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, stream));
    default: throw std::runtime_error("Unsupported engine type in surfaces_lock::create");
    }
}

}  // namespace cldnn
