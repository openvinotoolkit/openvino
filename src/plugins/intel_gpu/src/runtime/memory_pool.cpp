// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <fstream>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

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

memory::ptr memory_pool::alloc_memory(const layout& layout, allocation_type type, bool reset) {
    size_t alloc_size = layout.bytes_count();

    size_t offset = 0;
    if (alloc_size <= SIZE_4KB) {
        offset = SIZE_4KB;
    } else if (alloc_size <= SIZE_16KB) {
        offset = SIZE_16KB;
    } else if (alloc_size <= SIZE_256KB) {
        offset = SIZE_256KB;
    } else if (alloc_size <= SIZE_2MB) {
        offset = SIZE_2MB;
    }


    if (type == allocation_type::usm_device && alloc_size <= SIZE_2MB) {
        std::lock_guard<std::mutex> locker(_engine->_mem_mutex);
        size_t max_idx = ALIGNED_SIZE / offset;
        auto matched_mem = get_matching_aligned_memory(alloc_size);
        auto pre_allocated_ptr = static_cast<char *>(matched_mem.first->buffer_ptr());
        auto ptr = pre_allocated_ptr + (*matched_mem.second * offset);
        // std::cout << "  -- find matching_aligned_memory : " << matched_mem.first->buffer_ptr() << ", index : " << *matched_mem.second << std::endl;
        // std::cout << "    -- offset : " << (*matched_mem.second * offset) << std::endl;
        (*matched_mem.second)++;

        if (*matched_mem.second >= max_idx) {
            // No release for checking alloc time only
            _engine->_aligned_mem_buffer.push_back(matched_mem.first);
            auto aligned_layout = cldnn::layout{{ALIGNED_SIZE}, data_types::i8, format::bfyx};
            if (alloc_size <= SIZE_4KB) {
                _engine->_aligned_mem_4KB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
                _engine->_idx_4KB_aligned = 0;
                // std::cout << "    -- Re allocate _aligned_mem_4KB : " << _engine->_aligned_mem_4KB->buffer_ptr() << std::endl;
            } else if (alloc_size <= SIZE_16KB) {
                _engine->_aligned_mem_16KB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
                _engine->_idx_16KB_aligned = 0;
                // std::cout << "    -- Re allocate _aligned_mem_16KB : " << _engine->_aligned_mem_4KB->buffer_ptr() << std::endl;
            } else if (alloc_size <= SIZE_256KB) {
                _engine->_aligned_mem_256KB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
                _engine->_idx_256KB_aligned = 0;
                // std::cout << "    -- Re allocate _aligned_mem_256KB : " << _engine->_aligned_mem_4KB->buffer_ptr() << std::endl;
            } else if (alloc_size <= SIZE_2MB) {
                _engine->_aligned_mem_2MB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
                _engine->_idx_2MB_aligned = 0;
                // std::cout << "    -- Re allocate _aligned_mem_2MB : " << _engine->_aligned_mem_4KB->buffer_ptr() << std::endl;
            }
        }

        auto ret = _engine->allocate_memory(layout, type, reset, static_cast<void *>(ptr));
        // std::cout << "    -- alloc new device mem using pre-allocated memory ptr : " << ret->buffer_ptr() << std::endl;

        return ret;
    } else {
        return _engine->allocate_memory(layout, type, reset);
    }
}

memory_pool::~memory_pool() {}

std::pair<memory_ptr, size_t *> memory_pool::get_matching_aligned_memory(size_t alloc_size) {
    if (alloc_size <= SIZE_4KB) {
        return std::make_pair(_engine->_aligned_mem_4KB, &_engine->_idx_4KB_aligned);
    } else if (alloc_size <= SIZE_16KB) {
        return std::make_pair(_engine->_aligned_mem_16KB, &_engine->_idx_16KB_aligned);
    } else if (alloc_size <= SIZE_256KB) {
        return std::make_pair(_engine->_aligned_mem_256KB, &_engine->_idx_256KB_aligned);
    } else if (alloc_size <= SIZE_2MB) {
        return std::make_pair(_engine->_aligned_mem_2MB, &_engine->_idx_2MB_aligned);
    }

    return std::make_pair(_engine->_aligned_mem_2MB, &_engine->_idx_2MB_aligned);
}

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

    if (_layout.bytes_count() <= SIZE_2MB)
        return;

    {
        auto it = _non_padded_pool.lower_bound(_layout.bytes_count());

        while (it != _non_padded_pool.end()) {
            if (it->second._network_id == network_id &&
                it->second._type == type &&
                it->second._memory->get_internal_params().mem == mem->get_internal_params().mem) {
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
                                                  allocation_type type,
                                                  bool reset) {
    auto it = _non_padded_pool.lower_bound(layout.bytes_count());
    while (it != _non_padded_pool.end()) {
        if (it->second._network_id == network_id &&
            it->second._type == type &&
            it->second._memory->get_layout().format != format::fs_b_yx_fsv32 &&
            layout.format != format::fs_b_yx_fsv32 &&
            ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
             (layout.feature() % 32 == 0)) &&
            !has_conflict(it->second._users, restrictions, network_id)) {
            it->second._users.insert(memory_user(id, network_id));
            auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
            return ret_mem;
        } else {
            ++it;
        }
    }
    GPU_DEBUG_LOG << "[" << id << ": output]" << std::endl;
    // didn't find anything for you? create new resource
    auto mem = alloc_memory(layout, type, reset);
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
                 (layout.feature() % 32 == 0)) &&
                // TODO: check if this condition always correct
                layout.feature() <= rec_list._memory->get_layout().feature() &&
                layout.batch() <= rec_list._memory->get_layout().batch() &&
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
    GPU_DEBUG_LOG << "[" << id << ": output]" << std::endl;
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

void memory_pool::init_aligned_memory() {
    auto aligned_layout = cldnn::layout{{ALIGNED_SIZE}, data_types::i8, format::bfyx};
    if (_engine->_aligned_mem_4KB == nullptr) {
        _engine->_aligned_mem_4KB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
        _engine->_idx_4KB_aligned = 0;
        // std::cout << ">> _aligned_mem_4KB : " << _engine->_aligned_mem_4KB->buffer_ptr() << std::endl;
    }
    if (_engine->_aligned_mem_16KB == nullptr) {
        _engine->_aligned_mem_16KB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
        // std::cout << ">> _aligned_mem_16KB : " << _engine->_aligned_mem_16KB->buffer_ptr() << std::endl;
        _engine->_idx_16KB_aligned = 0;
    }
    if (_engine->_aligned_mem_256KB == nullptr) {
        _engine->_aligned_mem_256KB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
        // std::cout << ">> _aligned_mem_256KB : " << _engine->_aligned_mem_256KB->buffer_ptr() << std::endl;
        _engine->_idx_256KB_aligned = 0;
    }
    if (_engine->_aligned_mem_2MB == nullptr) {
        _engine->_aligned_mem_2MB = alloc_memory(aligned_layout, allocation_type::usm_device, true);
        // std::cout << ">> _aligned_mem_2MB : " << _engine->_aligned_mem_2MB->buffer_ptr() << std::endl;
        _engine->_idx_2MB_aligned = 0;
    }



    return;
}

memory::ptr memory_pool::get_memory(const layout& layout, allocation_type type, bool reset) {
    if (type == allocation_type::usm_device)
        init_aligned_memory();

    return alloc_memory(layout, type, reset);
}

memory::ptr memory_pool::get_memory(const layout& layout,
                                    const primitive_id& id,
                                    uint32_t network_id,
                                    const std::set<primitive_id>& restrictions,
                                    allocation_type type,
                                    bool reusable_across_network,
                                    bool reset) {
    if (type == allocation_type::usm_device)
        init_aligned_memory();

    bool do_reuse = reusable_across_network;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_memory_reuse) {
        do_reuse = false;
    }
    if (do_reuse) {
        // reusable within the same network
        if (!layout.format.is_image() && layout.data_padding == padding{{0, 0, 0, 0}, 0}) {
            // non-padded buffers
            return get_from_non_padded_pool(layout, id, network_id, restrictions, type, reset);
        } else if (!layout.format.is_image()) {
            // padded buffers
            return get_from_padded_pool(layout, id, network_id, restrictions, type);
        } else {
            // images (reuse not yet implemented)
            return alloc_memory(layout, type, reset);
        }
    } else {
        return alloc_memory(layout, type, reset);
    }
}

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

memory_pool::memory_pool(engine& engine) : _engine(&engine) {
}

void memory_pool::dump(uint32_t net_id) {
    GPU_DEBUG_COUT << "Dump memory pool of network " << net_id << std::endl;
    GPU_DEBUG_COUT << "========== non-padded pool ( " << _non_padded_pool.size() << " records) ==========" << std::endl;
    for (auto mem : _non_padded_pool) {
        GPU_DEBUG_COUT << mem.second._memory->buffer_ptr() << " (size: " << mem.first << ", type: " << mem.second._type
                  << ")'s users: " << std::endl;
        for (auto user : mem.second._users) {
            GPU_DEBUG_COUT << "   -- " << user._id << std::endl;
        }
    }
    GPU_DEBUG_COUT << "========== padded pool (" << _padded_pool.size() << " records) ==========" << std::endl;
    for (auto mem : _padded_pool) {
        GPU_DEBUG_COUT << " layout: " << mem.first.to_short_string() << std::endl;
        for (auto record : mem.second) {
            GPU_DEBUG_COUT << "    " << record._memory->buffer_ptr() << ", type: " << record._type << ", users : " << std::endl;
            for (auto user : record._users) {
                GPU_DEBUG_COUT << "    --- " << user._id << std::endl;
            }
        }
    }
}
}  // namespace cldnn
