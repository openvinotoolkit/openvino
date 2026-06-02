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

class UsmHelper {
public:
    explicit UsmHelper(const ::sycl::context& ctx, const ::sycl::device& device, bool use_usm) : _ctx(ctx), _device(device), _use_usm(use_usm) {
    }

    void* allocate_host(size_t size, size_t alignment) const {
        if (!_use_usm)
            throw std::runtime_error("[CLDNN] USM is not enabled");
        return ::sycl::aligned_alloc_host(alignment, size, _ctx);
    }

    void* allocate_shared(size_t size, size_t alignment) const {
        if (!_use_usm)
            throw std::runtime_error("[CLDNN] USM is not enabled");
        return ::sycl::aligned_alloc_shared(alignment, size, _device, _ctx);
    }

    void* allocate_device(size_t size, size_t alignment) const {
        if (!_use_usm)
            throw std::runtime_error("[CLDNN] USM is not enabled");
        return ::sycl::aligned_alloc_device(alignment, size, _device, _ctx);
    }

    void free_mem(void* ptr) const {
        if (!_use_usm)
            throw std::runtime_error("[CLDNN] USM is not enabled");
        ::sycl::free(ptr, _ctx);
    }

    ::sycl::event enqueue_memcpy(::sycl::queue& queue, void* dst_ptr, const void* src_ptr,
                                 size_t bytes_count) const {
        if (!_use_usm)
            throw std::runtime_error("[CLDNN] USM is not enabled");
        return queue.memcpy(dst_ptr, src_ptr, bytes_count);
    }

    template<typename T>
    ::sycl::event enqueue_fill_mem(::sycl::queue& queue, void* dst_ptr, const T& pattern,
                                   size_t count, const std::vector<::sycl::event>& depEvents = {}) const {
        if (!_use_usm)
            throw std::runtime_error("[CLDNN] USM is not enabled");
        return queue.fill(dst_ptr, pattern, count, depEvents);
    }

    ::sycl::usm::alloc get_usm_allocation_type(const void* usm_ptr) const {
        return ::sycl::get_pointer_type(usm_ptr, _ctx);
    }


//    size_t get_usm_allocation_size(const void* usm_ptr) const {
//        if (!_get_mem_alloc_info_fn) {
//            throw std::runtime_error("[GPU] clGetMemAllocInfoINTEL is nullptr");
//        }
//
//        size_t ret_val;
//        size_t ret_val_size;
//        _get_mem_alloc_info_fn(_ctx.get(), usm_ptr, CL_MEM_ALLOC_SIZE_INTEL, sizeof(size_t), &ret_val, &ret_val_size);
//        return ret_val;
//    }

private:
    ::sycl::context _ctx;
    ::sycl::device _device;
    bool _use_usm;
};

/*
    UsmPointer requires associated context to free it.
    Simple wrapper class for usm allocated pointer.
*/
class UsmHolder {
public:
    UsmHolder(const UsmHelper& usm_helper, void* ptr, size_t size, bool shared_memory = false)
    : _usmHelper(usm_helper)
    , _ptr(ptr)
    , _size(size)
    , _shared_memory(shared_memory) { }

    void* ptr() { return _ptr; }
    size_t size() { return _size; }
    void memFree() {
        try {
            if (!_shared_memory)
                _usmHelper.free_mem(_ptr);
            } catch (...) {
            // Exception may happen only when clMemFreeINTEL function is unavailable, thus can't free memory properly
        }
        _ptr = nullptr;
    }
    ~UsmHolder() {
        memFree();
    }

private:
    const UsmHelper& _usmHelper;
    void* _ptr;
    size_t _size;  // hold size of allocation because SYCL doesn't provide API to get it from pointer
    bool _shared_memory = false;
};
/*
    USM base class. Different usm types should derive from this class.
*/
class UsmMemory {
public:
    explicit UsmMemory(const UsmHelper& usmHelper) : _usmHelper(usmHelper) { }
    UsmMemory(const UsmHelper& usmHelper, void* usm_ptr, size_t size, size_t offset) // Memo: better to use UsmMemory& instead of void*
    : _usmHelper(usmHelper)
    , _usm_pointer(std::make_shared<UsmHolder>(_usmHelper, reinterpret_cast<uint8_t*>(usm_ptr) + offset, size, true)) {
        if (!usm_ptr) {
            throw std::runtime_error("[GPU] Can't share null usm pointer");
        }
    }

    size_t size() { return _usm_pointer->size(); }

    // Get methods returns original pointer allocated by SYCL.
    void* get() const { return _usm_pointer->ptr(); }

    void allocateHost(size_t size) {
        auto ptr = _usmHelper.allocate_host(size, 0);
        _check_error(size, ptr, "Host");
        _allocate(ptr, size);
    }

    void allocateShared(size_t size) {
        auto ptr = _usmHelper.allocate_shared(size, 0);
        _check_error(size, ptr, "Shared");
        _allocate(ptr, size);
    }

    void allocateDevice(size_t size) {
        auto ptr = _usmHelper.allocate_device(size, 0);
        _check_error(size, ptr, "Device");
        _allocate(ptr, size);
    }

    void freeMem() {
        if (!_usm_pointer)
            throw std::runtime_error("[CL ext] Can not free memory of empty UsmHolder");
        _usm_pointer->memFree();
    }

    virtual ~UsmMemory() = default;

protected:
    const UsmHelper& _usmHelper;
    std::shared_ptr<UsmHolder> _usm_pointer = nullptr;

private:
    void _allocate(void* ptr, size_t size) {
        _usm_pointer = std::make_shared<UsmHolder>(_usmHelper, ptr, size);
    }

    void _check_error(size_t size, void* ptr, const char* usm_type) {
        if (ptr == nullptr) {
            std::stringstream sout;
            sout << "[SYCL ext] Can not allocate " << size << " bytes for USM " << usm_type << ". ptr: " << ptr << std::endl;
            throw std::runtime_error(sout.str());
        }
    }
};

inline bool operator==(const UsmMemory &lhs, const UsmMemory &rhs) {
    return lhs.get() == rhs.get();
}

inline bool operator!=(const UsmMemory &lhs, const UsmMemory &rhs) {
    return !operator==(lhs, rhs);
}
}
