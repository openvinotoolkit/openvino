// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"
#include "memory_caps.hpp"
#include "event.hpp"
#include "engine_configuration.hpp"

namespace cldnn {

struct engine;
class stream;

struct memory {
    using ptr = std::shared_ptr<memory>;
    using cptr = std::shared_ptr<const memory>;
    memory(engine* engine, const layout& layout,  allocation_type type, bool reused = false);

    virtual ~memory();
    virtual void* lock(const stream& stream) = 0;
    virtual void unlock(const stream& stream) = 0;
    virtual event::ptr fill(stream& stream, unsigned char pattern) = 0;
    virtual event::ptr fill(stream& stream) = 0;

    size_t size() const { return _bytes_count; }
    size_t count() const { return _layout.count(); }
    virtual shared_mem_params get_internal_params() const = 0;
    virtual bool is_allocated_by(const engine& engine) const { return &engine == _engine; }
    engine* get_engine() const { return _engine; }
    const layout& get_layout() const { return _layout; }
    allocation_type get_allocation_type() const { return _type; }
    // TODO: must be moved outside memory class
    virtual bool is_memory_reset_needed(layout l) {
        // To avoid memory reset, output memory must meet the following requirements:
        // - To be Weights format (Data memory can be reused by memory_pool, which can lead to errors)
        // - To have zero paddings
        // - To be completely filled with data
        if ((!format::is_weights_format(l.format) && !format::is_simple_data_format(l.format)) ||
             format::is_winograd(l.format) || format::is_image_2d(l.format)) {
            return true;
        }

        if (l.data_padding.lower_size() != tensor(0) || l.data_padding.upper_size() != tensor(0)) {
            return true;
        }

        if (_bytes_count == (l.data_type == data_types::bin ? ceil_div(l.count(), 32) : l.count()) * data_type_traits::size_of(l.data_type)) {
            return false;
        }

        return true;
    }

    virtual void copy_from_other(const stream& /* stream */, const memory& /* other */) {}

protected:
    engine* _engine;
    const layout _layout;
    // layout bytes count, needed because of traits static map destruction
    // before run of memory destructor, when engine is static
    size_t _bytes_count;

private:
    allocation_type _type;
    bool _reused;
};

struct simple_attached_memory : memory {
    simple_attached_memory(const layout& layout, void* pointer)
        : memory(nullptr, layout, allocation_type::unknown), _pointer(pointer) {}

    void* lock(const stream& /* stream */) override { return _pointer; }
    void unlock(const stream& /* stream */) override {}
    event::ptr fill(stream& /* stream */, unsigned char) override { return event::ptr{nullptr}; }
    event::ptr fill(stream& /* stream */) override { return event::ptr{nullptr}; }
    shared_mem_params get_internal_params() const override { return { shared_mem_type::shared_mem_empty, nullptr, nullptr, nullptr,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0}; };

private:
    void* _pointer;
};

template <class T>
struct mem_lock {
    explicit mem_lock(memory::ptr mem, const stream& stream) : _mem(mem), _stream(stream), _ptr(reinterpret_cast<T*>(_mem->lock(_stream))) {}

    ~mem_lock() {
        _ptr = nullptr;
        _mem->unlock(_stream);
    }

    size_t size() const { return _mem->size() / sizeof(T); }

    mem_lock(const mem_lock& other) = delete;
    mem_lock& operator=(const mem_lock& other) = delete;

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
    auto begin() & { return stdext::make_checked_array_iterator(_ptr, size()); }
    auto end() & { return stdext::make_checked_array_iterator(_ptr, size(), size()); }
#else
    T* begin() & { return _ptr; }
    T* end() & { return _ptr + size(); }
#endif

    /// @brief Provides indexed access to pointed memory.
    T& operator[](size_t idx) const& {
        assert(idx < size());
        return _ptr[idx];
    }

    T* data() const & { return _ptr; }

    /// Prevents to use mem_lock as temporary object
    void data() && {}
    /// Prevents to use mem_lock as temporary object
    void begin() && {}
    /// Prevents to use mem_lock as temporary object
    void end() && {}
    /// Prevents to use mem_lock as temporary object
    void operator[](size_t idx) && {}

private:
    memory::ptr _mem;
    const stream& _stream;
    T* _ptr;
};

struct surfaces_lock {
    surfaces_lock() = default;
    virtual ~surfaces_lock() = default;

    surfaces_lock(const surfaces_lock& other) = delete;
    surfaces_lock& operator=(const surfaces_lock& other) = delete;

    static std::unique_ptr<surfaces_lock> create(engine_types engine_type, std::vector<memory::ptr> mem, const stream& stream);
};

}  // namespace cldnn
