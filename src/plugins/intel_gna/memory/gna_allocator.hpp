// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <functional>

#include "gna_device.hpp"
#include "memory/gna_mem_requests.hpp"

namespace GNAPluginNS {
namespace memory {
/**
 * wrap GNA interface into c++ allocator friendly one
 */
class GNAAllocator {
    std::shared_ptr<GNADeviceHelper> _device;

 public:
    typedef uint8_t value_type;

    explicit GNAAllocator(std::shared_ptr<GNADeviceHelper> device) : _device(std::move(device)) {
    }
    uint8_t *allocate(std::size_t n) {
        uint32_t granted = 0;
        auto result = _device->alloc(n, &granted);
        if (result == nullptr || granted == 0) {
            throw std::bad_alloc();
        }
        return result;
    }
    void deallocate(uint8_t *p, std::size_t n) {
        _device->free(p);
    }
    void setTag(void* memPtr, GNAPluginNS::memory::rRegion tagValue) {
        _device->tagMemoryRegion(memPtr, tagValue);
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
