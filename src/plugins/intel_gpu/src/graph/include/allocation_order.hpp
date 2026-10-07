// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "intel_gpu/runtime/layout.hpp"

namespace cldnn {

// Snapshot the expensive layout-dependent part of allocation ordering once per node.
// Unbounded layouts have no allocation size yet and retain their processing order.
class allocation_order_key {
public:
    allocation_order_key(layout output_layout, size_t unique_id, size_t processing_number) : _unique_id(unique_id), _processing_number(processing_number) {
        if (output_layout.is_dynamic() && output_layout.has_upper_bound()) {
            // Normalize this local copy to upper-bound dimensions, preserving the tensor/format conversion for allocation ordering.
            const auto upper_bound_tensor = output_layout.get_tensor();
            output_layout.set_tensor(upper_bound_tensor);
        }
        _is_dynamic = output_layout.is_dynamic();
        if (!_is_dynamic) {
            _bytes_count = output_layout.bytes_count();
        }
    }

    bool operator<(const allocation_order_key& rhs) const {
        if (_is_dynamic != rhs._is_dynamic) {
            return !_is_dynamic;
        }
        if (_is_dynamic) {
            return _processing_number < rhs._processing_number;
        }
        if (_bytes_count != rhs._bytes_count) {
            return _bytes_count > rhs._bytes_count;
        }
        return _unique_id < rhs._unique_id;
    }

private:
    size_t _unique_id;
    size_t _processing_number;
    size_t _bytes_count = 0;
    bool _is_dynamic = false;
};

}  // namespace cldnn
