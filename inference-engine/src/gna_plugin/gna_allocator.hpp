// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include "gna_device.hpp"
#include "polymorh_allocator.hpp"

/**
 * wrap GNA interface into c++ allocator friendly one
 */
class GNAAllocator {
    std::reference_wrapper<GNADeviceHelper> _device;

 public:
    typedef uint8_t value_type;

    explicit GNAAllocator(GNADeviceHelper &device) : _device(device) {
    }
    uint8_t *allocate(std::size_t n) {
        uint32_t granted = 0;
        auto result = _device.get().alloc(n, &granted);
        if (result == nullptr || granted == 0) {
            throw std::bad_alloc();
        }
        return result;
    }
    void deallocate(uint8_t *p, std::size_t n) {
        _device.get().free();
    }
};
