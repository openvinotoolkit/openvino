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
#include "api/CPP/memory.hpp"

#include "api_impl.h"
#include "engine_impl.h"
#include "refcounted_obj.h"

namespace cldnn
{

struct memory_impl : refcounted_obj<memory_impl>
{
    memory_impl(const engine_impl::ptr& engine, layout layout, bool reused=false)
        : _engine(engine)
        , _layout(layout)
        , _reused(reused)
    {}

    virtual ~memory_impl()
    {
        if (_engine != nullptr && !_reused) 
        {
            _engine->get_memory_pool().subtract_memory_used(_layout.bytes_count());
        }
    }
    virtual void* lock() = 0;
    virtual void unlock() = 0;
    virtual void fill(unsigned char pattern, event_impl::ptr ev) = 0;
    size_t size() const { return _layout.bytes_count(); }
    virtual bool is_allocated_by(const engine_impl& engine) const { return &engine == _engine.get(); }
    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }
    const layout& get_layout() const { return _layout; }
protected:
    const engine_impl::ptr _engine;
    const layout _layout;
private:
    bool _reused;
};

struct simple_attached_memory : memory_impl
{
    simple_attached_memory(layout layout, void* pointer)
        : memory_impl(nullptr, layout), _pointer(pointer)
    {
    }

    void* lock() override { return _pointer; }
    void unlock() override {}
    void fill(unsigned char, event_impl::ptr) override {}
private:
    void* _pointer;
};

template <class T>
struct mem_lock
{
    mem_lock(memory_impl::ptr mem)
        : mem(mem), ptr(reinterpret_cast<T*>(mem->lock()))
    {
    }

    mem_lock(memory_impl& mem)
        : mem_lock(&mem)
    {}

    ~mem_lock()
    {
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

}

API_CAST(::cldnn_memory, cldnn::memory_impl)
