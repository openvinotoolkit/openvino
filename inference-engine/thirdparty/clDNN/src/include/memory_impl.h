/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include "api/memory.hpp"

#include "engine_impl.h"
#include "refcounted_obj.h"

namespace cldnn {

struct memory_impl : refcounted_obj<memory_impl> {
    memory_impl(const engine_impl::ptr& engine, const layout& layout, uint32_t net_id,  allocation_type type, bool reused = false)
        : _engine(engine.get()), _layout(layout), _net_id(net_id), _bytes_count(_layout.bytes_count()), _type(type), _reused(reused) {}

    virtual ~memory_impl() {
        if (_engine != nullptr && !_reused) {
            _engine->get_memory_pool().subtract_memory_used(_bytes_count);
        }
    }
    virtual void* lock() = 0;
    virtual void unlock() = 0;
    virtual void fill(unsigned char pattern, event_impl::ptr ev) = 0;
    size_t size() const { return _bytes_count; }
    virtual shared_mem_params get_internal_params() const = 0;
    virtual bool is_allocated_by(const engine_impl& engine) const { return &engine == _engine; }
    refcounted_obj_ptr<engine_impl> get_engine() const { return engine_impl::ptr(_engine); }
    const layout& get_layout() const { return _layout; }
    uint32_t get_net_id() const { return _net_id; }
    void set_net(uint32_t id) { _net_id = id; }
    allocation_type get_allocation_type() const { return _type; }

protected:
    engine_impl *const _engine;
    const layout _layout;
    uint32_t _net_id;
    size_t _bytes_count;

private:
    // layout bytes count, needed because of traits static map destruction
    // before run of memory_impl destructor, when engine is static
    allocation_type _type;
    bool _reused;
};

struct simple_attached_memory : memory_impl {
    simple_attached_memory(const layout& layout, void* pointer, uint32_t net_id)
        : memory_impl((engine_impl::ptr) nullptr, layout, net_id, allocation_type::unknown), _pointer(pointer) {}

    void* lock() override { return _pointer; }
    void unlock() override {}
    void fill(unsigned char, event_impl::ptr) override {}
    shared_mem_params get_internal_params() const override { return { shared_mem_type::shared_mem_empty, nullptr, nullptr, nullptr,
#ifdef WIN32
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
    explicit mem_lock(memory_impl::ptr mem) : mem(mem), ptr(reinterpret_cast<T*>(mem->lock())) {}

    explicit mem_lock(memory_impl& mem) : mem_lock((memory_impl::ptr) &mem) {}

    ~mem_lock() {
        ptr = nullptr;
        mem->unlock();
    }

    size_t size() const { return mem->size() / sizeof(T); }

#if defined(_SECURE_SCL) && (_SECURE_SCL > 0)
    auto begin() & { return stdext::make_checked_array_iterator(ptr, size()); }
    auto end() & { return stdext::make_checked_array_iterator(ptr, size(), size()); }
#else
    T* begin() & { return ptr; }
    T* end() & { return ptr + size(); }
#endif

    T* data() const { return ptr; }

private:
    memory_impl::ptr mem;
    T* ptr;
};

}  // namespace cldnn
