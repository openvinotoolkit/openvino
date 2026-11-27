// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/strides.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cldnn {

using tensor_conversion_key = std::pair<data_types, data_types>;

struct tensor_conversion_input {
    size_t count = 1;
    size_t source_span_bytes = 0;
    uint64_t source_offset = 0;
    std::array<uint64_t, 6> dimensions{1, 1, 1, 1, 1, 1};
    std::array<uint64_t, 6> strides{0, 0, 0, 0, 0, 0};
    bool padded = false;
    bool transpose = false;
};

class kernels_cache;

// Prepares and holds conversion kernels shared by all states of one program.
// Serializes preparation, lookup, and kernel submission across InferRequests.
class tensor_conversion {
public:
    static bool supports(tensor_conversion_key key) {
        switch (key.first) {
        case data_types::bf16:
            return key.second == data_types::f16;
        case data_types::f32:
            return key.second == data_types::f16 || key.second == data_types::f64;
        case data_types::f64:
            return key.second == data_types::f32;
        case data_types::i32:
            return key.second == data_types::i64 || key.second == data_types::u64 || key.second == data_types::u32;
        default:
            return false;
        }
    }

    // Compiles supported keys once; later calls must request the same keys.
    void prepare(const engine& engine, kernels_cache& cache, const std::vector<tensor_conversion_key>& requested_keys);

    void restore(const engine& engine, const std::vector<tensor_conversion_key>& keys,
                 const std::vector<kernel::ptr>& kernels);

    bool compatible_with(const engine& engine) const;

    bool has_kernel(tensor_conversion_key key) const;

    std::vector<tensor_conversion_key> get_keys() const;

    std::vector<kernel::ptr> get_kernels() const;

    // Validate before copying input; execute() reuses the computed span and kernel indices.
    static bool prepare_input(const layout& source_layout, const ov::Strides& strides, bool transpose,
                              tensor_conversion_input& input);

    event::ptr execute(tensor_conversion_key key, memory::cptr src, memory::cptr dst, stream& stream,
                       const tensor_conversion_input& input, const std::vector<event::ptr>& dependencies = {});

private:
    void set_kernels(const std::vector<tensor_conversion_key>& keys, const std::vector<kernel::ptr>& kernels,
                     const engine& engine);

    struct kernel_info {
        kernel::ptr compiled_kernel;
        size_t max_work_group_size;
    };
    std::map<tensor_conversion_key, kernel_info> _kernels;
    void* _context = nullptr;
    mutable std::mutex _mutex;
    bool _prepared = false;
};

}  // namespace cldnn
