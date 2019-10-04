/*
// Copyright (c) 2019 Intel Corporation
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
#include "api/memory.hpp"
#include "memory_impl.h"
#include "engine_impl.h"

namespace cldnn {

memory memory::allocate(const engine& engine, const layout& layout, uint16_t stream_id) {
    size_t size = layout.bytes_count();
    if (size == 0)
        throw std::invalid_argument("size should be more than 0");

    return memory(engine.get()->allocate_memory(layout, stream_id).detach());
}

size_t memory::count() const {
    return get_layout().count();
}

size_t memory::size() const {
    return _impl->size();
}

const layout& memory::get_layout() const {
    return _impl->get_layout();
}

int memory::get_stream_id() const {
    return _impl->get_stream_id();
}

bool memory::is_allocated_by(const engine& engine) const {
    return _impl->is_allocated_by(*engine.get());
}

bool memory::is_the_same_buffer(const memory& other) const {
    if (_impl == other.get())
        return true;

    if (_impl->get_engine() != other.get()->get_engine())
        return false;

    // User memory, check te pointers
    if (!_impl->get_engine())
        return lock_impl() == other.lock_impl();

    // Engine memory, let it decide
    return _impl->get_engine()->is_the_same_buffer(*_impl, *other.get());
}

memory memory::attach_impl(const cldnn::layout& layout, void* ptr, uint16_t stream_id) {
    return memory(new simple_attached_memory(layout, ptr, stream_id));
}

void* memory::lock_impl() const {
    return _impl->lock();
}

void memory::unlock() const {
    _impl->unlock();
}

void memory::retain() {
    _impl->add_ref();
}
void memory::release() {
    _impl->release();
}

}  // namespace cldnn
