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

ov::Shape MemoryUsageTracker::shapes_math(const ov::Shape& shape1, const ov::Shape& shape2, math_op op) {
    std::vector<size_t> result;

    OPENVINO_ASSERT(shape1.size() == shape2.size());

    for (size_t i = 0; i < shape1.size(); i++) {
        if (op == math_op::SUB && shape1[i] < shape2[i])
            return std::vector<size_t>();

        if (op == math_op::SUB)
            result.push_back(shape1[i] - shape2[i]);
        else if (op == math_op::SUM)
            result.push_back(shape1[i] + shape2[i]);
        else if (op == math_op::MUL)
            result.push_back(shape1[i] * shape2[i]);
    }

    return result;
}

void MemoryUsageTracker::add_shape(const std::string& id, const ov::Shape& shape) {
    auto& shapes = _shapes_info[id];
    if (shapes.size() >= _max_deque_size)
        shapes.pop_front();

    shapes.push_back(shape);
}

bool MemoryUsageTracker::can_preallocate(size_t current_buffer_size, size_t desired_buffer_size) {
    auto device_mem_usage = _engine->get_used_device_memory(cldnn::allocation_type::usm_device);

    if (desired_buffer_size <= current_buffer_size)
        return true;

    const auto memory_threshold = 0.95;
    float ration = static_cast<float>(desired_buffer_size) / static_cast<float>(current_buffer_size);

    // TOOD: need to exclude constants from global memory size
    return device_mem_usage * ration < _engine->get_device_info().max_global_mem_size * memory_threshold;
}

std::pair<bool, ov::Shape>
    MemoryUsageTracker::predict_preallocated_shape_size(const std::string& id,
                                                        const ov::Shape& current_shape,
                                                        bool can_reuse_buffer) {
    add_shape(id, current_shape);

    // Save shape information and exit without pre-allocation suggestion if current
    // buffer can be reused
    if (can_reuse_buffer)
        return {false, {}};

    // Check if there is enough data for prediction
    auto& shapes = _shapes_info[id];
    if (shapes.size() == _max_deque_size) {
        std::vector<ov::Shape> diffs;
        for (size_t i = _max_deque_size - 1; i >= 1; --i) {
            auto result = shapes_math(shapes[i], shapes[i - 1], math_op::SUB);
            if (result.empty())
                break;
            diffs.push_back(result);
        }

        bool is_same_diff = diffs.size() > 1;
        for (size_t i = 1; i < diffs.size(); ++i) {
            if (diffs[0] != diffs[i]) {
                is_same_diff = false;
                break;
            }
        }

        if (is_same_diff) {
            // Apply preallocation for the next N iterations
            ov::Shape mul_shape(diffs[0].size(), _next_iters_preallocation_count);
            auto diff = shapes_math(diffs[0], mul_shape, math_op::MUL);
            auto new_shape = shapes_math(current_shape, diff, math_op::SUM);
            return {true, new_shape};
        } else {
            // Apply percentage buffer preallocation
            auto current_shape_size = ov::shape_size(current_shape);
            ov::Shape new_shape_size(current_shape.size(), 1);
            new_shape_size[0] = static_cast<size_t>(current_shape_size * _buffers_preallocation_ratio);
            return {true, new_shape_size};
        }
    }

    return {false, {}};
}

}  // namespace cldnn
