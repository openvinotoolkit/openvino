// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <fstream>
#include <vector>

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/memory_pool.hpp"

#include <list>
#include <string>
#include <utility>
#include <set>
#include <stdexcept>

namespace cldnn {
memory_record::memory_record(memory_set users,
                             std::shared_ptr<memory>& memory,
                             uint32_t net_id,
                             allocation_type type)
    : _users(users), _memory(memory), _network_id(net_id), _type(type) {}

memory::ptr memory_pool::alloc_memory(const layout& layout, allocation_type type) {
    return _engine->allocate_memory(layout, type);
}

memory_pool::~memory_pool() {}

bool memory_pool::has_conflict(const memory_set& a,
                               const std::set<primitive_id>& b,
                               uint32_t b_network_id) {
    std::set<primitive_id> a_same_network;
    for (auto const& mem_usr : a) {
        if (mem_usr._network_id == b_network_id) {
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

void memory_pool::release_memory(memory* mem, const primitive_id& id, uint32_t network_id) {
    // check nonpadded pool first
    auto _layout = mem->get_layout();
    auto type = mem->get_allocation_type();

    {
        auto range = _non_padded_pool.equal_range(_layout.bytes_count());
        auto it = range.first;

        while (it != range.second && it != _non_padded_pool.end()) {
            if (it->second._network_id == network_id &&
                it->second._type == type &&
                it->second._memory.get() == mem) {
                auto user_it = it->second._users.find({ id, network_id });

                // normally there should be only one entry
                if (user_it != it->second._users.end()) {
                    user_it = it->second._users.erase(user_it);
                }
                if (it->second._users.empty()) {
                    // if this was the only user of the memory, then free it up
                    it = _non_padded_pool.erase(it);
                }

                //entry found and processed - so return
                return;
            } else {
                ++it;
            }
        }
    }
    {
        auto itr = _padded_pool.find(_layout);

        if (itr != _padded_pool.end()) {
            auto& list = itr->second;
            auto list_itr = list.begin();

            while (list_itr != list.end()) {
                if (list_itr->_memory.get() == mem &&
                    list_itr->_network_id == network_id &&
                    list_itr->_type == type) {
                    auto user_it = list_itr->_users.find({ id, network_id });

                    // normally there should be only one entry
                    if (user_it != list_itr->_users.end()) {
                        user_it = list_itr->_users.erase(user_it);
                    }
                    if (list_itr->_users.empty()) {
                        // if this was the only user of the memory, then free it up
                        list.erase(list_itr);
                    }

                    //entry found and processed - so return
                    break;
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
                _padded_pool.erase(itr);
            }
        }
    }
}

memory::ptr memory_pool::get_from_non_padded_pool(const layout& layout,
                                                  const primitive_id& id,
                                                  uint32_t network_id,
                                                  const std::set<primitive_id>& restrictions,
                                                  allocation_type type) {
    auto it = _non_padded_pool.lower_bound(layout.bytes_count());
    while (it != _non_padded_pool.end()) {
        if (it->second._network_id == network_id &&
            it->second._type == type &&
            it->second._memory->get_layout().format != format::fs_b_yx_fsv32 &&
            layout.format != format::fs_b_yx_fsv32 &&
            ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
             (layout.size.feature[0] % 32 == 0)) &&
            !has_conflict(it->second._users, restrictions, network_id)) {
            it->second._users.insert(memory_user(id, network_id));
            auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
            return ret_mem;
        } else {
            ++it;
        }
    }
    // didn't find anything for you? create new resource
    auto mem = alloc_memory(layout, type);
    {
        _non_padded_pool.emplace(layout.bytes_count(),
                                 memory_record({{id, network_id}}, mem, network_id, type));
    }
    return mem;
}

memory::ptr memory_pool::get_from_padded_pool(const layout& layout,
                                              const primitive_id& id,
                                              uint32_t network_id,
                                              const std::set<primitive_id>& restrictions,
                                              allocation_type type) {
    auto first_level_cache = _padded_pool.find(layout);

    if (first_level_cache != _padded_pool.end()) {
        for (auto& rec_list : first_level_cache->second) {
            if (rec_list._network_id == network_id &&
                rec_list._type == type &&
                ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
                 (layout.size.feature[0] % 32 == 0)) &&
                // TODO: check if this condition always correct
                layout.size.feature[0] <= rec_list._memory->get_layout().size.feature[0] &&
                layout.size.batch[0] <= rec_list._memory->get_layout().size.batch[0] &&
                rec_list._memory->get_layout().format != format::fs_b_yx_fsv32 &&
                layout.format != format::fs_b_yx_fsv32 &&
                !has_conflict(rec_list._users, restrictions, network_id)) {
                rec_list._users.insert({id, network_id});
                auto ret_mem = _engine->reinterpret_buffer(*(rec_list._memory), layout);
                return ret_mem;
            }
        }
        auto mem = alloc_memory(layout, type);
        first_level_cache->second.emplace_back(
            memory_record({{id, network_id}}, mem, network_id, type));
        return mem;
    }
    auto mem = alloc_memory(layout, type);
    std::list<memory_record> list = {memory_record({{id, network_id}}, mem, network_id, type)};
    _padded_pool.emplace(layout, std::move(list));
    return mem;
}

/*
        This is not reusable within one network or it's internal micronetworks. But we can use this memory records
   between networks.
    */
memory::ptr memory_pool::get_from_across_networks_pool(const layout& layout,
                                                       const primitive_id& id,
                                                       uint32_t network_id,
                                                       allocation_type type) {
    auto it = _no_reusable_pool.lower_bound(layout.bytes_count());

    while (it != _no_reusable_pool.end()) {
        if (it->second._network_id != network_id &&
            it->second._type == type) {  // don't use non reusable resources within the same network
            if (!has_conflict(it->second._users, {}, network_id)) {
                it->second._users.insert(memory_user(id, network_id));
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                return ret_mem;
            }
        }
        ++it;
    }
    auto mem = alloc_memory(layout, type);
    {
        _no_reusable_pool.emplace(layout.bytes_count(),
                                  memory_record({{id, network_id}}, mem, network_id, type));
    }
    return mem;
}

memory::ptr memory_pool::get_memory(const layout& layout, allocation_type type) {
    return alloc_memory(layout, type);
}

memory::ptr memory_pool::get_memory(const layout& layout,
                                    const primitive_id& id,
                                    uint32_t network_id,
                                    const std::set<primitive_id>& restrictions,
                                    allocation_type type,
                                    bool reusable_across_network) {
    if (reusable_across_network) {
        // reusable within the same network
        if (!layout.format.is_image() && layout.data_padding == padding{{0, 0, 0, 0}, 0}) {
            // non-padded buffers
            return get_from_non_padded_pool(layout, id, network_id, restrictions, type);
        } else if (!layout.format.is_image()) {
            // padded buffers
            return get_from_padded_pool(layout, id, network_id, restrictions, type);
        } else {
            // images (reuse not yet implemented)
            return alloc_memory(layout, type);
        }
    } else {
        return alloc_memory(layout, type);
    }
}

void memory_pool::clear_pool() { _non_padded_pool.clear(); }

void memory_pool::clear_pool_for_network(uint32_t network_id) {
    // free up _non_padded_pool for this network
    {
        auto itr = _non_padded_pool.begin();

        while (itr != _non_padded_pool.end()) {
            auto& record = itr->second;

            if (record._network_id == network_id) {
                itr = _non_padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

    // free up _padded_pool for this network
    {
        auto itr = _padded_pool.begin();

        while (itr != _padded_pool.end()) {
            auto& list = itr->second;
            auto list_itr = list.begin();

            while (list_itr != list.end()) {
                if (list_itr->_network_id == network_id) {
                    list_itr = list.erase(list_itr);
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
                itr = _padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

    // free up _no_reusable_pool for this network
    {
        auto itr = _no_reusable_pool.begin();

        while (itr != _no_reusable_pool.end()) {
            auto& record = itr->second;

            if (record._network_id == network_id) {
                itr = _no_reusable_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }
}

memory_pool::memory_pool(engine& engine) : _engine(&engine) { }

}  // namespace cldnn
