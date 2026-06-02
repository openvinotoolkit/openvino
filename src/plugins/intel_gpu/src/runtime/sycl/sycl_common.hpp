// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_wrapper.hpp"
#include "sycl_device_detector.hpp"
#include "openvino/core/except.hpp"

#include <vector>

namespace cldnn {
namespace sycl {

using sycl_queue_type = ::sycl::queue;
using sycl_kernel_type = ::sycl::kernel;

class sycl_error : public ov::Exception {
public:
    explicit sycl_error(::sycl::exception const& err);
};

#define SYCL_ERR_MSG_FMT(error) ("[GPU] " + std::string(error.what()) +  std::string(", SYCL error code: ") + std::to_string(error.code().value()))

inline bool is_device_available(const device_info& info) {
    sycl_device_detector detector;
    auto devices = detector.get_available_devices(nullptr, nullptr);
    for (auto& device : devices) {
        if (device.second->get_info().uuid.uuid == info.uuid.uuid) {
            return true;
        }
    }

    return false;
}

}  // namespace sycl
}  // namespace cldnn

namespace sycl {

/*
    UsmPointer requires associated context to free it.
    Simple wrapper class for usm allocated pointer.
*/
class UsmHolder {
public:
    UsmHolder(::sycl::context context, void* ptr, size_t size, bool shared_memory = false)
    : _context(context)
    , _ptr(ptr)
    , _size(size)
    , _shared_memory(shared_memory) {
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Can not create UsmHolder with nullptr");
    }

    void* ptr() { return _ptr; }
    size_t size() { return _size; }
    void memFree() {
        try {
            if (!_shared_memory)
                ::sycl::free(_ptr, _context);
            } catch (...) {
            // Exception may happen only when clMemFreeINTEL function is unavailable, thus can't free memory properly
        }
        _ptr = nullptr;
    }
    ~UsmHolder() {
        memFree();
    }

private:
    ::sycl::context _context;
    void* _ptr;
    size_t _size;  // hold size of allocation because SYCL doesn't provide API to get it from pointer
    bool _shared_memory = false;
};
/*
    USM base class. Different usm types should derive from this class.
*/
class UsmMemory {
public:
    explicit UsmMemory(::sycl::context context, ::sycl::device device)
    : _context(context)
    , _device(device) {}

    UsmMemory(::sycl::context context, ::sycl::device device, void* usm_ptr, size_t size, size_t offset)
    : _context(context)
    , _device(device)
    , _usm_pointer(std::make_shared<UsmHolder>(_context, reinterpret_cast<uint8_t*>(usm_ptr) + offset, size, true)) {}

    size_t size() const {
        if (is_empty()) {
            return 0;
        }
        return _usm_pointer->size();
    }

    // Get methods returns original pointer allocated by SYCL.
    void* get() const {
        if (is_empty()) {
            return nullptr;
        }
        return _usm_pointer->ptr();
    }

    bool is_empty() const { return _usm_pointer.get() == nullptr; }

    void allocateHost(size_t size, size_t alignment = 0) {
        auto ptr = ::sycl::aligned_alloc_host(alignment, size, _context);
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate host USM memory");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr, size);
    }

    void allocateShared(size_t size, size_t alignment = 0) {
        auto ptr = ::sycl::aligned_alloc_shared(alignment, size, _device, _context);
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate shared USM memory");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr, size);
    }

    void allocateDevice(size_t size, size_t alignment = 0) {
        auto ptr = ::sycl::aligned_alloc_device(alignment, size, _device, _context);
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate device USM memory");
        _usm_pointer = std::make_shared<UsmHolder>(_context, ptr, size);
    }

    void freeMem() {
        _usm_pointer.reset();
    }

    virtual ~UsmMemory() = default;

protected:
    ::sycl::context _context;
    ::sycl::device _device;
    std::shared_ptr<UsmHolder> _usm_pointer = nullptr;
};

inline bool operator==(const UsmMemory &lhs, const UsmMemory &rhs) {
    return lhs.get() == rhs.get();
}

inline bool operator!=(const UsmMemory &lhs, const UsmMemory &rhs) {
    return !operator==(lhs, rhs);
}
}
