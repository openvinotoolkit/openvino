/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include "cldnn.hpp"
#include "compounds.h"
#include "layout.hpp"
#include "engine.hpp"
#include <memory>
#include <iterator>
#include <string>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_memory Memory description and management
/// @{

template <typename T>
struct pointer;

struct memory_impl;

/// @brief Shared memory descriptor type.
enum class shared_mem_type {
    /// @brief Structure unitialized or contains no information.
    shared_mem_empty,

    /// @brief Structure describes shared CL buffer.
    shared_mem_buffer,

    /// @brief Structure describes shared CL image.
    shared_mem_image,

    /// @brief Structure describes shared VA/DXVA surface
    shared_mem_vasurface,

    /// @brief Structure describes shared D3D11 buffer
    shared_mem_dxbuffer
};

using shared_handle = void*;
using shared_surface = uint32_t;

/// @brief Low-level API handles required for using cldnn memory objects in external API calls.
struct shared_mem_params {
    shared_mem_type mem_type;     ///< shared buffer type
    shared_handle context;        ///< OpenCL context for external operations
    shared_handle user_device;    ///< DX/VA device for external operations
    shared_handle mem;            ///< memory object handle
#ifdef WIN32
    shared_handle surface;        ///< VA/DXVA surface handle
#else
    shared_surface surface;
#endif
    uint32_t plane;               ///< shared surface plane
};
/// @brief Represents buffer with particular @ref layout.
/// @details Usually allocated by @ref engine except cases when attached to user-allocated buffer.
struct memory {
    friend struct data;
    friend struct mutable_data;
    friend struct network;
    friend struct network_output;

    /// Allocate memory on @p engine using specified @p layout
    static memory allocate(const engine& engine, const layout& layout, uint32_t net_id = 0);

    /// Create shared memory object on @p engine using user-supplied memory buffer @p buf using specified @p layout
    static memory share_buffer(const engine& engine, const layout& layout, shared_handle buf, uint32_t net_id = 0);

    /// Create shared memory object on @p engine using user-supplied 2D image @p img using specified @p layout
    static memory share_image(const engine& engine, const layout& layout, shared_handle img, uint32_t net_id = 0);

    /// Create shared memory object on @p engine over specified @p plane of video decoder surface @p surf using specified @p layout
#ifdef WIN32
    static memory share_surface(const engine& engine, const layout& layout, shared_handle surf, uint32_t plane,
        uint32_t net_id = 0);
    static memory share_dx_buffer(const engine& engine, const layout& layout, shared_handle res, uint32_t net_id = 0);
#else
    static memory share_surface(const engine& engine, const layout& layout, shared_surface surf, uint32_t plane,
        uint32_t net_id = 0);
#endif

    /// Create memory object attached to the buffer allocated by user.
    /// @param ptr  The pointer to user allocated buffer.
    /// @param size Size (in bytes) of the buffer. Should be equal to @p layout.data_size()
    /// @note User is responsible for buffer deallocation. Buffer lifetime should be bigger than lifetime of the memory object.
    template <typename T>
    static memory attach(const cldnn::layout& layout, T* ptr, size_t size, uint32_t net_id = 0) {
        if (!ptr)
            throw std::invalid_argument("pointer should not be null");
        size_t data_size = size * sizeof(T);
        if (data_size != layout.bytes_count()) {
            std::string err_str("buffer size mismatch - input size " + std::to_string(data_size) + " layout size " +
                                std::to_string(layout.bytes_count()));
            throw std::invalid_argument(err_str);
        }

        return attach_impl(layout, static_cast<void*>(ptr), net_id);
    }

    explicit memory(memory_impl* data)
        : _impl(data) {
        if (_impl == nullptr)
            throw std::invalid_argument("implementation pointer should not be null");
    }

    memory(const memory& other) : _impl(other._impl) {
        retain();
    }

    memory& operator=(const memory& other) {
        if (_impl == other._impl)
            return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~memory() { release(); }

    friend bool operator==(const memory& lhs, const memory& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const memory& lhs, const memory& rhs) { return !(lhs == rhs); }

    /// number of elements of _layout.data_type stored in memory
    size_t count() const;

    /// number of bytes used by memory
    size_t size() const;

    /// Associated @ref layout
    const layout& get_layout() const;
    int get_net_id() const;

    /// Test if memory is allocated by @p engine
    bool is_allocated_by(const engine& engine) const;

    bool is_the_same_buffer(const memory& other) const;

    shared_mem_params get_internal_params() const;

    /// Creates the @ref pointer object to get an access memory data
    template <typename T>
    friend struct cldnn::pointer;
    template <typename T>
    cldnn::pointer<T> pointer() const;

    /// C API memory handle
    memory_impl* get() const { return _impl; }

private:
    friend struct engine;
    memory_impl* _impl;

    template <typename T>
    T* lock() const {
        if (data_type_traits::align_of(get_layout().data_type) % alignof(T) != 0) {
            throw std::logic_error("memory data type alignment do not match");
        }
        return static_cast<T*>(lock_impl());
    }

    void unlock() const;

    void* lock_impl() const;
    static memory attach_impl(const cldnn::layout& layout, void* ptr, uint32_t net_id);

    void retain();
    void release();
};

/// @brief Helper class to get an access @ref memory data
/// @details
/// This class provides an access to @ref memory data following RAII idiom and exposes basic C++ collection members.
/// @ref memory object is locked on construction of pointer and "unlocked" on descruction.
/// Objects of this class could be used in many STL utility functions like copy(), transform(), etc.
/// As well as in range-for loops.
template <typename T>
struct pointer {
    /// @brief Constructs pointer from @ref memory and locks @c (pin) ref@ memory object.
    explicit pointer(const memory& mem) : _mem(mem), _size(_mem.size() / sizeof(T)), _ptr(_mem.lock<T>()) {}

    /// @brief Unlocks @ref memory
    ~pointer() { _mem.unlock(); }

    /// @brief Copy construction.
    pointer(const pointer& other) : pointer(other._mem) {}

    /// @brief Copy assignment.
    pointer& operator=(const pointer& other) {
        if (this->_mem != other._mem)
            do_copy(other._mem);
        return *this;
    }

    /// @brief Returns the number of elements (of type T) stored in memory
    size_t size() const { return _size; }

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
    typedef stdext::checked_array_iterator<T*> iterator;
    typedef stdext::checked_array_iterator<const T*> const_iterator;

    iterator begin() & { return stdext::make_checked_array_iterator(_ptr, size()); }
    iterator end() & { return stdext::make_checked_array_iterator(_ptr, size(), size()); }

    const_iterator begin() const& { return stdext::make_checked_array_iterator(_ptr, size()); }
    const_iterator end() const& { return stdext::make_checked_array_iterator(_ptr, size(), size()); }
#else
    typedef T* iterator;
    typedef const T* const_iterator;
    iterator begin() & { return _ptr; }
    iterator end() & { return _ptr + size(); }
    const_iterator begin() const& { return _ptr; }
    const_iterator end() const& { return _ptr + size(); }
#endif

    /// @brief Provides indexed access to pointed memory.
    T& operator[](size_t idx) const& {
        assert(idx < _size);
        return _ptr[idx];
    }

    /// @brief Returns the raw pointer to pointed memory.
    T* data() & { return _ptr; }
    /// @brief Returns the constant raw pointer to pointed memory
    const T* data() const& { return _ptr; }

    friend bool operator==(const pointer& lhs, const pointer& rhs) { return lhs._mem == rhs._mem; }
    friend bool operator!=(const pointer& lhs, const pointer& rhs) { return !(lhs == rhs); }

    // do not use this class as temporary object
    // ReSharper disable CppMemberFunctionMayBeStatic, CppMemberFunctionMayBeConst
    /// Prevents to use pointer as temporary object
    void data() && {}
    /// Prevents to use pointer as temporary object
    void begin() && {}
    /// Prevents to use pointer as temporary object
    void end() && {}
    /// Prevents to use pointer as temporary object
    void operator[](size_t idx) && {}
    // ReSharper restore CppMemberFunctionMayBeConst, CppMemberFunctionMayBeStatic

private:
    memory _mem;
    size_t _size;
    T* _ptr;

    // TODO implement exception safe code.
    void do_copy(const memory& mem) {
        auto ptr = mem.lock<T>();
        _mem.unlock();
        _mem = mem;
        _size = _mem.size() / sizeof(T);
        _ptr = ptr;
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <typename T>
pointer<T> memory::pointer() const {
    return cldnn::pointer<T>(*this);
}
#endif

/// @}

/// @}

}  // namespace cldnn
