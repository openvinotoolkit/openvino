// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"
#include "memory_caps.hpp"
#include "event.hpp"
#include "engine_configuration.hpp"

#include <type_traits>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace cldnn {

class engine;
class stream;

enum class mem_lock_type : int32_t {
    read,
    write,
    read_write
};

class MemoryTracker {
public:
    explicit MemoryTracker(engine* engine, void* buffer_ptr, size_t buffer_size, allocation_type alloc_type);
    ~MemoryTracker();

    size_t size() const { return m_buffer_size; }

private:
    engine* m_engine;
    void* m_buffer_ptr;
    size_t m_buffer_size;
    allocation_type m_alloc_type;
};

struct memory {
    using ptr = std::shared_ptr<memory>;
    using cptr = std::shared_ptr<const memory>;
    memory(engine* engine, const layout& layout, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);

    virtual ~memory() = default;
    virtual void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) = 0;
    virtual void unlock(const stream& stream) = 0;
    virtual event::ptr fill(stream& stream, unsigned char pattern, bool blocking = true) = 0;
    virtual event::ptr fill(stream& stream, bool blocking = true) = 0;
    // only supports gpu_usm
    virtual void* buffer_ptr() const { return nullptr; }

    size_t size() const { return _bytes_count; }
    size_t count() const { return _layout.count(); }
    virtual shared_mem_params get_internal_params() const = 0;
    virtual bool is_allocated_by(const engine& engine) const { return &engine == _engine && _type != allocation_type::unknown; }
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

        if (l.data_padding) {
            return true;
        }

        if (_bytes_count == l.bytes_count()) {
            return false;
        }

        return true;
    }

    // Device <== Host
    virtual event::ptr copy_from(stream& stream, const void* src_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) = 0;

    // Device <== Device
    virtual event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) = 0;

    // Device ==> Host
    virtual event::ptr copy_to(stream& stream, void* dst_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const = 0;

    // Device ==> Device
    virtual event::ptr copy_to(stream& stream, memory& dst_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
        return dst_mem.copy_from(stream, *this, src_offset, dst_offset, size, blocking);
    }

    virtual event::ptr copy_from(stream& stream, const memory& src_mem, bool blocking = true) {
        const auto zero_offset = 0;
        const auto data_size = src_mem._bytes_count;
        return copy_from(stream, src_mem, zero_offset, zero_offset, data_size, blocking);
    }

    virtual event::ptr copy_from(stream& stream, const void* src_ptr, bool blocking = true) {
        const auto zero_offset = 0;
        const auto data_size = _bytes_count;
        return copy_from(stream, src_ptr, zero_offset, zero_offset, data_size, blocking);
    }

    virtual event::ptr copy_to(stream& stream, memory& other, bool blocking = true) const {
        const auto zero_offset = 0;
        const auto data_size = other._bytes_count;
        return copy_to(stream, other, zero_offset, zero_offset, data_size, blocking);
    }

    virtual event::ptr copy_to(stream& stream, void* dst_ptr, bool blocking = true) const {
        const auto zero_offset = 0;
        const auto data_size = _bytes_count;
        return copy_to(stream, dst_ptr, zero_offset, zero_offset, data_size, blocking);
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    virtual dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const {
        throw std::runtime_error("[CLDNN] Can't convert memory object to onednn");
    }
#endif

    std::shared_ptr<MemoryTracker> get_mem_tracker() const { return m_mem_tracker; }
    GPU_DEBUG_CODE(bool from_memory_pool = false);

protected:
    engine* _engine;
    const layout _layout;
    // layout bytes count, needed because of traits static map destruction
    // before run of memory destructor, when engine is static
    size_t _bytes_count;
    std::shared_ptr<MemoryTracker> m_mem_tracker = nullptr;

private:
    allocation_type _type;
};

struct simple_attached_memory : memory {
    simple_attached_memory(const layout& layout, void* pointer)
        : memory(nullptr, layout, allocation_type::unknown, nullptr), _pointer(pointer) {}

    void* lock(const stream& /* stream */, mem_lock_type /* type */) override { return _pointer; }
    void unlock(const stream& /* stream */) override {}
    event::ptr fill(stream& /* stream */, unsigned char, bool) override { return nullptr; }
    event::ptr fill(stream& /* stream */, bool) override { return nullptr; }
    shared_mem_params get_internal_params() const override { return { shared_mem_type::shared_mem_empty, nullptr, nullptr, nullptr,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0}; };

    event::ptr copy_from(stream& stream, const void* src_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    event::ptr copy_to(stream& stream, void* dst_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    void* _pointer;
};

template <class T, mem_lock_type lock_type = mem_lock_type::read_write>
struct mem_lock {
    explicit mem_lock(memory::ptr mem, const stream& stream) : _mem(std::move(mem)), _stream(stream),
                      _ptr(reinterpret_cast<T*>(_mem->lock(_stream, lock_type))) {}

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
    T* data() && = delete;
    /// Prevents to use mem_lock as temporary object
    T* begin() && = delete;
    /// Prevents to use mem_lock as temporary object
    T* end() && = delete;
    /// Prevents to use mem_lock as temporary object
    T& operator[](size_t idx) && = delete;

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

template<typename T>
inline std::vector<T> read_vector(cldnn::memory::ptr mem, const cldnn::stream& stream) {
    cldnn::data_types mem_dtype = mem->get_layout().data_type;
    if (mem_dtype == data_types::f16 || mem_dtype == data_types::f32) {
        if (!std::is_floating_point<T>::value && !std::is_same<T, ov::float16>::value) {
            OPENVINO_ASSERT(false, "[GPU] read_vector: attempt to convert floating point memory to non-floating point memory");
        }
    }

    std::vector<T> out_vecs;
    if (mem->get_allocation_type() == allocation_type::usm_host || mem->get_allocation_type() == allocation_type::usm_shared) {
        switch (mem_dtype) {
            case data_types::i32: {
                auto p_mem = reinterpret_cast<int32_t*>(mem->buffer_ptr());
                for (size_t i = 0; i < mem->count(); ++i) {
                    out_vecs.push_back(static_cast<T>(p_mem[i]));
                }
                break;
            }
            case data_types::i64: {
                auto p_mem = reinterpret_cast<int64_t*>(mem->buffer_ptr());
                for (size_t i = 0; i < mem->count(); ++i) {
                    out_vecs.push_back(static_cast<T>(p_mem[i]));
                }
                break;
            }
            case data_types::f16: {
                auto p_mem = reinterpret_cast<uint16_t*>(mem->buffer_ptr());
                for (size_t i = 0; i < mem->count(); ++i) {
                    out_vecs.push_back(static_cast<T>(ov::float16::from_bits(p_mem[i])));
                }
                break;
            }
            case data_types::f32: {
                auto p_mem = reinterpret_cast<float*>(mem->buffer_ptr());
                for (size_t i = 0; i < mem->count(); ++i) {
                    out_vecs.push_back(static_cast<T>(p_mem[i]));
                }
                break;
            }
            default: OPENVINO_ASSERT(false, "[GPU] read_vector: unsupported data type");
        }
    } else {
        switch (mem_dtype) {
            case data_types::i32: {
                mem_lock<int32_t, mem_lock_type::read> lock{mem, stream};
                out_vecs = std::move(std::vector<T>(lock.begin(), lock.end()));
                break;
            }
            case data_types::i64: {
                mem_lock<int64_t, mem_lock_type::read> lock{mem, stream};
                out_vecs = std::move(std::vector<T>(lock.begin(), lock.end()));
                break;
            }
            case data_types::f16: {
                mem_lock<ov::float16, mem_lock_type::read> lock{mem, stream};
                out_vecs = std::move(std::vector<T>(lock.begin(), lock.end()));
                break;
            }
            case data_types::f32: {
                mem_lock<float, mem_lock_type::read> lock{mem, stream};
                out_vecs = std::move(std::vector<T>(lock.begin(), lock.end()));
                break;
            }
            default: OPENVINO_ASSERT(false, "[GPU] read_vector: unsupported data type");
        }
    }
    return out_vecs;
}

}  // namespace cldnn
