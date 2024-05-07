// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <fstream>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
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
    return _engine->allocate_memory(layout, type, reset);
}

memory_pool::~memory_pool() {}

bool memory_pool::has_conflict(const memory_set& mem_cand,
                               const std::unordered_set<size_t>& restrictions,
                               uint32_t b_network_id) {
    for (const auto& mem_usr : mem_cand) {
        if (restrictions.find(mem_usr._unique_id) != restrictions.end())
            return true;
    }
    return false;
}

void memory_pool::release_memory(memory* mem, const size_t& unique_id, primitive_id prim_id, uint32_t network_id) {
    // check nonpadded pool first
    auto _layout = mem->get_layout();
    auto type = mem->get_allocation_type();

    {
        auto it = _non_padded_pool.lower_bound(_layout.bytes_count());

        while (it != _non_padded_pool.end()) {
            if (it->second._network_id == network_id &&
                it->second._type == type &&
                it->second._memory->get_internal_params().mem == mem->get_internal_params().mem) {
                auto user_it = it->second._users.find({ unique_id, network_id, prim_id, _layout.bytes_count()});

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
                    auto user_it = list_itr->_users.find({ unique_id, network_id, prim_id, _layout.bytes_count()});

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
                                                  const primitive_id& prim_id,
                                                  size_t unique_id,
                                                  uint32_t network_id,
                                                  const std::unordered_set<size_t>& restrictions,
                                                  allocation_type type,
                                                  bool reset,
                                                  bool is_dynamic) {
    auto it = _non_padded_pool.lower_bound(layout.bytes_count());
    while (it != _non_padded_pool.end()) {
        if ((!is_dynamic || (layout.bytes_count() > it->second._memory->get_layout().bytes_count() * 0.5)) &&
            (it->second._network_id == network_id &&
            it->second._type == type &&
            it->second._memory->get_layout().format != format::fs_b_yx_fsv32 &&
            layout.format != format::fs_b_yx_fsv32 &&
            ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
             (layout.feature() % 32 == 0)) &&
            !has_conflict(it->second._users, restrictions, network_id))) {
            it->second._users.insert(memory_user(unique_id, network_id, prim_id, layout.bytes_count()));
            auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
            GPU_DEBUG_CODE(ret_mem->from_memory_pool = true);
            return ret_mem;
        } else {
            ++it;
        }
    }
    GPU_DEBUG_LOG << "[" << prim_id << "(" << unique_id << "): output]" << std::endl;
    // didn't find anything for you? create new resource
    auto mem = alloc_memory(layout, type, reset);
    {
        _non_padded_pool.emplace(layout.bytes_count(),
                                 memory_record({{unique_id, network_id, prim_id, layout.bytes_count()}}, mem, network_id, type));
    }
    return mem;
}

memory::ptr memory_pool::get_from_padded_pool(const layout& layout,
                                              const primitive_id& prim_id,
                                              size_t unique_id,
                                              uint32_t network_id,
                                              const std::unordered_set<size_t>& restrictions,
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
                rec_list._users.insert({unique_id, network_id, prim_id, layout.bytes_count()});
                auto ret_mem = _engine->reinterpret_buffer(*(rec_list._memory), layout);
                GPU_DEBUG_CODE(ret_mem->from_memory_pool = true);
                return ret_mem;
            }
        }
        auto mem = alloc_memory(layout, type);
        first_level_cache->second.emplace_back(
            memory_record({{unique_id, network_id, prim_id, layout.bytes_count()}}, mem, network_id, type));
        return mem;
    }
    GPU_DEBUG_LOG << "[" << prim_id << "(" << unique_id << ")" << ": output]" << std::endl;
    auto mem = alloc_memory(layout, type);
    std::list<memory_record> list = {memory_record({{unique_id, network_id, prim_id, layout.bytes_count()}}, mem, network_id, type)};
    _padded_pool.emplace(layout, std::move(list));
    return mem;
}

/*
        This is not reusable within one network or it's internal micronetworks. But we can use this memory records
   between networks.
    */
memory::ptr memory_pool::get_from_across_networks_pool(const layout& layout,
                                                       const primitive_id& prim_id,
                                                       size_t unique_id,
                                                       uint32_t network_id,
                                                       allocation_type type) {
    auto it = _no_reusable_pool.lower_bound(layout.bytes_count());

    while (it != _no_reusable_pool.end()) {
        if (it->second._network_id != network_id &&
            it->second._type == type) {  // don't use non reusable resources within the same network
            if (!has_conflict(it->second._users, {}, network_id)) {
                it->second._users.insert(memory_user(unique_id, network_id, prim_id, layout.bytes_count()));
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                GPU_DEBUG_CODE(ret_mem->from_memory_pool = true);
                return ret_mem;
            }
        }
        ++it;
    }
    auto mem = alloc_memory(layout, type);
    {
        _no_reusable_pool.emplace(layout.bytes_count(),
                                  memory_record({{unique_id, network_id, prim_id, layout.bytes_count()}}, mem, network_id, type));
    }
    return mem;
}

memory::ptr memory_pool::get_memory(const layout& layout, allocation_type type, bool reset) {
    return alloc_memory(layout, type, reset);
}

memory::ptr memory_pool::get_memory(const layout& layout,
                                    const primitive_id& prim_id,
                                    const size_t unique_id,
                                    uint32_t network_id,
                                    const std::unordered_set<size_t>& restrictions,
                                    allocation_type type,
                                    bool reusable_across_network,
                                    bool reset,
                                    bool is_dynamic) {
    bool do_reuse = reusable_across_network;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_memory_reuse) {
        do_reuse = false;
    }
    if (do_reuse) {
        // reusable within the same network
        if (!layout.format.is_image() && layout.data_padding == padding{{0, 0, 0, 0}, 0}) {
            // non-padded buffers
            return get_from_non_padded_pool(layout, prim_id, unique_id, network_id, restrictions, type, reset, is_dynamic);
        } else if (!layout.format.is_image()) {
            // padded buffers
            return get_from_padded_pool(layout, prim_id, unique_id, network_id, restrictions, type);
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

memory_pool::memory_pool(engine& engine) : _engine(&engine) { }

inline float get_mb_size(size_t size) {
    return (static_cast<float>(size) / (1024 * 1024));
}

inline float get_percentage(size_t size, size_t total_size) {
    return (static_cast<float>(size) * 100.0f / total_size);
}

void memory_pool::dump(uint32_t net_id, uint32_t iter, std::string dump_dir_path) {
    dump_to_screen(net_id, iter);
    if (!dump_dir_path.empty())
        dump_to_file(net_id, iter, dump_dir_path);
}

void memory_pool::dump_to_file(uint32_t net_id, uint32_t iter, std::string dump_dir_path) {
    const std::string dump_file_name = "dump_runtime_memory_pool_net_" + std::to_string(net_id) + "_iter_" + std::to_string(iter) + ".csv";
    const std::string desc = "pool_type,layout,mem_ptr,mem_type,mem_pool_szie,prim_id,unique_id,mem_size";
    const std::string dump_path = dump_dir_path + dump_file_name;
    std::ofstream of(dump_path);
    if (of.is_open()) {
        of << desc << std::endl;
        for (auto mem : _non_padded_pool) {
            for (auto user : mem.second._users) {
                of << "non_padded_pool,," << mem.second._memory->buffer_ptr() << "," << mem.second._type << ","
                    << mem.first << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
            }
        }

        for (auto mem : _padded_pool) {
            for (auto record : mem.second) {
                const size_t mem_pool_szie = record._memory->size();
                for (auto user : record._users) {
                    of << "padded_pool," << mem.first.to_short_string() << "," << record._memory->buffer_ptr() << "," << record._type << ","
                        << mem_pool_szie << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
                }
            }
        }
        std::cout << "Dump file to " << dump_path << std::endl;
    }
}

void memory_pool::dump_to_screen(uint32_t net_id, uint32_t iter) {
    GPU_DEBUG_COUT << "Dump memory pool of network (net_id : " << net_id << ", iter : " << iter << ")" << std::endl;
    GPU_DEBUG_COUT << "========== non-padded pool ( " << _non_padded_pool.size() << " records) ==========" << std::endl;
    float total_mem_request = 0.f;
    float total_mem_pool_size = 0.f;
    for (auto mem : _non_padded_pool) {
        GPU_DEBUG_COUT << mem.second._memory->buffer_ptr() << " (size: " << get_mb_size(mem.first)
            << "MB, type: " << mem.second._type << ")'s users: " << std::endl;
        total_mem_pool_size += mem.first;
        float min_percentage = 100.0f;
        float max_percentage = 0.f;
        for (auto user : mem.second._users) {
            float percentage = get_percentage(user._mem_size, mem.first);
            min_percentage = std::min(percentage, min_percentage);
            max_percentage = std::max(percentage, max_percentage);
            total_mem_request += static_cast<float>(user._mem_size);
            GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                << get_mb_size(user._mem_size) << "MB, " << percentage << "%" << std::endl;
        }
        GPU_DEBUG_COUT <<  "   - min of the memory pool entry: " << min_percentage << std::endl;
        GPU_DEBUG_COUT <<  "   - max of the memory pool entry: " << max_percentage << std::endl;
    }
    GPU_DEBUG_COUT << "========== non-padded pool efficiency: "
                    << std::to_string(static_cast<float>(total_mem_request / total_mem_pool_size))
                    << " (Total memory request : " << get_mb_size(total_mem_request) << "MB"
                    << " / Total meomry pool size : " << get_mb_size(total_mem_pool_size) << "MB) ==========" << std::endl;
    GPU_DEBUG_COUT << "========== padded pool (" << _padded_pool.size() << " records) ==========" << std::endl;
    total_mem_request = 0.f;
    total_mem_pool_size = 0.f;
    for (auto mem : _padded_pool) {
        GPU_DEBUG_COUT << " layout: " << mem.first.to_short_string() << ", records(" << mem.second.size() << ")" << std::endl;
        for (auto record : mem.second) {
            size_t mem_size = record._memory->size();
            GPU_DEBUG_COUT << "  " << record._memory->buffer_ptr() << " (size:" << get_mb_size(mem_size)
                            << "MB, type: " << record._type << ")'s usrs : " << std::endl;
            float min_percentage = 100.0f;
            float max_percentage = 0.f;
            total_mem_pool_size += mem_size;
            for (auto user : record._users) {
                float percentage = get_percentage(user._mem_size, mem_size);
                min_percentage = std::min(percentage, min_percentage);
                max_percentage = std::max(percentage, max_percentage);
                total_mem_request += static_cast<float>(user._mem_size);
                GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                    << get_mb_size(user._mem_size) << "MB, " << percentage << "%" << std::endl;
            }
            GPU_DEBUG_COUT << "   - min of the memory pool entry: " << min_percentage << std::endl;
            GPU_DEBUG_COUT << "   - max of the memory pool entry: " << max_percentage << std::endl;
        }
    }
    GPU_DEBUG_COUT << "========== padded pool efficiency: "
                    << std::to_string(static_cast<float>(total_mem_request / total_mem_pool_size))
                    << " (Total memory request : " << get_mb_size(total_mem_request) << "MB"
                    << " / Total meomry pool size : " << get_mb_size(total_mem_pool_size) << "MB) ==========" << std::endl;
}
}  // namespace cldnn
