/*
// Copyright (c) 2017 Intel Corporation
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

#include <algorithm>
#include <fstream>
#include <vector>

#include "memory_pool.h"
#include "engine_impl.h"
#include "memory_impl.h"
#include "program_impl.h"

#include "program_node.h"

#include "gpu/memory_gpu.h"
#include <list>
#include <string>
#include <utility>
#include <set>

namespace cldnn {
memory_record::memory_record(memory_set users,
                             refcounted_obj_ptr<memory_impl>& memory,
                             uint32_t net_id,
                             uint16_t stream_id)
    : _users(users), _memory(memory), _network_id(net_id), _stream_id(stream_id) {}

memory_impl::ptr memory_pool::alloc_memory(const layout& layout, uint16_t stream_id) {
    auto context = _engine->get_context();
    if (layout.bytes_count() > context->get_engine_info().max_alloc_mem_size) {
        throw error("exceeded max size of memory object allocation", CLDNN_ALLOC_SIZE_EXCEEDED);
    }

    add_memory_used(layout.bytes_count());

    if (_max_peak_memory_used > context->get_engine_info().max_global_mem_size) {
        throw error("exceeded global device memory", CLDNN_GLOBAL_SIZE_EXCEEDED);
    }

    try {
        if (layout.format.is_image_2d()) {
            memory_impl::ptr mem_impl {new gpu::gpu_image2d(_engine, layout, stream_id), false};
            return mem_impl;
        } else {
            memory_impl::ptr mem_impl { new gpu::gpu_buffer(_engine, layout, stream_id), false };
            return mem_impl;
        }
    } catch (const cl::Error& clErr) {
        switch (clErr.err()) {
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            case CL_OUT_OF_RESOURCES:
            case CL_OUT_OF_HOST_MEMORY:
            case CL_INVALID_BUFFER_SIZE:
                throw error("out of GPU resources", CLDNN_OUT_OF_RESOURCES);
            default:
                throw error("GPU buffer allocation failed", CLDNN_ERROR);
        }
    }
}
memory_pool::~memory_pool() {}

bool memory_pool::has_conflict(const memory_set& a,
                               const std::set<primitive_id>& b,
                               uint32_t b_network_id,
                               uint16_t stream_id) {
    std::set<primitive_id> a_same_network;
    for (auto const& mem_usr : a) {
        if (mem_usr._network_id == b_network_id && mem_usr._stream_id == stream_id) {
            a_same_network.insert(mem_usr._id);
        }
    }
    std::vector<primitive_id> intersection;
    intersection.reserve(std::min(a_same_network.size(), b.size()));
    set_intersection(a_same_network.begin(),
                     a_same_network.end(),
                     b.begin(),
                     b.end(),
                     std::back_inserter(intersection));
    return !intersection.empty();
}

memory_impl::ptr memory_pool::get_from_non_padded_pool(const layout& layout,
                                                       const primitive_id& id,
                                                       uint32_t network_id,
                                                       const std::set<primitive_id>& restrictions,
                                                       uint16_t stream_id) {
    auto it = _non_padded_pool.lower_bound(layout.bytes_count());
    while (it != _non_padded_pool.end()) {
        if (it->second._stream_id == stream_id &&
            !has_conflict(it->second._users, restrictions, network_id, stream_id)) {
            it->second._users.insert(memory_user(id, network_id, stream_id));
            auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
            return ret_mem;
        } else {
            ++it;
        }
    }
    // didn't find anything for you? create new resource
    auto mem = alloc_memory(layout, stream_id);
    {
        _non_padded_pool.emplace(layout.bytes_count(),
                                 memory_record({{id, network_id, stream_id}}, mem, network_id, stream_id));
        // we don't want to store any resources with no parents so memory pool has to store weak pointer of _engine.
        _engine->release();
    }
    return mem;
}

memory_impl::ptr memory_pool::get_from_padded_pool(const layout& layout,
                                                   const primitive_id& id,
                                                   uint32_t network_id,
                                                   const std::set<primitive_id>& restrictions,
                                                   uint16_t stream_id) {
    auto first_level_cache = _padded_pool.find(layout);

    if (first_level_cache != _padded_pool.end()) {
        for (auto& rec_list : first_level_cache->second) {
            if (rec_list._stream_id == stream_id &&
                layout.size.feature[0] <= rec_list._memory->get_layout().size.feature[0] &&
                layout.size.batch[0] <= rec_list._memory->get_layout().size.batch[0] &&
                !has_conflict(rec_list._users, restrictions, network_id, stream_id)) {
                rec_list._users.insert({id, network_id, stream_id});
                auto ret_mem = _engine->reinterpret_buffer(*(rec_list._memory), layout);
                return ret_mem;
            }
        }
        auto mem = alloc_memory(layout, stream_id);
        first_level_cache->second.emplace_back(
            memory_record({{id, network_id, stream_id}}, mem, network_id, stream_id));
        // we don't want to store any resources with no parents so memory pool has to store weak pointer of _engine.
        _engine->release();
        return mem;
    }
    auto mem = alloc_memory(layout, stream_id);
    std::list<memory_record> list = {memory_record({{id, network_id, stream_id}}, mem, network_id, stream_id)};
    _padded_pool.emplace(layout, std::move(list));
    // we don't want to store any resources with no parents so memory pool has to store weak pointer of _engine.
    _engine->release();
    return mem;
}

/*
        This is not reusable within one network or it's internal micronetworks. But we can use this memory records
   between networks.
    */
memory_impl::ptr memory_pool::get_from_across_networks_pool(const layout& layout,
                                                            const primitive_id& id,
                                                            uint32_t network_id,
                                                            uint16_t stream_id) {
    auto it = _no_reusable_pool.lower_bound(layout.bytes_count());

    while (it != _no_reusable_pool.end()) {
        if (it->second._network_id != network_id &&
            it->second._stream_id == stream_id) {  // don't use non reusable resources within the same network
            if (!has_conflict(it->second._users, {}, network_id, stream_id)) {
                it->second._users.insert(memory_user(id, network_id, stream_id));
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                return ret_mem;
            }
        }
        ++it;
    }
    auto mem = alloc_memory(layout, stream_id);
    {
        _no_reusable_pool.emplace(layout.bytes_count(),
                                  memory_record({{id, network_id, stream_id}}, mem, network_id, stream_id));
        // we don't want to store any resources with no parents so memory pool has to store weak pointer of _engine.
        _engine->release();
    }
    return mem;
}

memory_impl::ptr memory_pool::get_memory(const layout& layout, uint16_t stream_id) {
    return alloc_memory(layout, stream_id);
}

memory_impl::ptr memory_pool::get_memory(const layout& layout,
                                         const primitive_id& id,
                                         uint32_t network_id,
                                         const std::set<primitive_id>& restrictions,
                                         uint16_t stream_id,
                                         bool reusable_across_network) {
    if (reusable_across_network) {
        // reusable within the same network
        if (!layout.format.is_image() && layout.data_padding == padding{{0, 0, 0, 0}, 0}) {
            // non-padded buffers
            return get_from_non_padded_pool(layout, id, network_id, restrictions, stream_id);
        } else if (!layout.format.is_image()) {
            // padded buffers
            return get_from_padded_pool(layout, id, network_id, restrictions, stream_id);
        } else {
            // images (reuse not yet implemented)
            return alloc_memory(layout, stream_id);
        }
    } else {
        return get_from_across_networks_pool(layout, id, network_id, stream_id);
    }
}

void memory_pool::clear_pool() { _non_padded_pool.clear(); }

memory_pool::memory_pool(engine_impl& engine) : _engine(&engine), _temp_memory_used(0), _max_peak_memory_used(0) {
    _engine->release();  // since engine is refcount object and there is circular dependency until context will be moved
                         // to memory pool we need to detach engine while destroying memory pool
}

void memory_pool::dump_memory_pool(const program_impl& program, std::string& path, std::string& dep) {
    using namespace std;
    ofstream log(path);

    log << "\nNon-padded pool:" << endl;
    log << "Size\tUsers:" << endl;
    for (const auto& record : _non_padded_pool) {
        log << record.first;
        for (const auto& usr : record.second._users) log << ", " << usr;
        log << endl;
    }

    log << "\n--- Padded pool: ---" << endl;
    log << "Size\tUsers:" << endl;
    for (const auto& record : _padded_pool) {
        for (const auto& mem : record.second) {
            log << mem._memory->size();
            for (const auto& usr : mem._users) log << ", " << usr;
            log << endl;
        }
    }
    log << dep;
    log.close();
    color_graph(program);
}

void memory_pool::color_graph(const program_impl& program) {
    uint32_t color = 0;
    for (const auto& record : _non_padded_pool) {
        for (const auto& usr : record.second._users) {
            if (program.has_node(usr._id))
                program.get_node(usr._id).set_reused_memory_color(color);
        }
        ++color;
    }

    for (const auto& list : _padded_pool) {
        for (const auto& record : list.second) {
            if (record._users.size() > 1) {  // one user doesn't mean reusing
                for (const auto& usr : record._users) {
                    if (program.has_node(usr._id))
                        program.get_node(usr._id).set_reused_memory_color(color);
                }
            }
            ++color;
        }
    }
}

void memory_pool::add_memory_used(size_t value) {
    _temp_memory_used += value;
    if (_temp_memory_used > _max_peak_memory_used) {
        _max_peak_memory_used = _temp_memory_used;
    }
}

void memory_pool::subtract_memory_used(size_t value) { _temp_memory_used -= value; }

}  // namespace cldnn
