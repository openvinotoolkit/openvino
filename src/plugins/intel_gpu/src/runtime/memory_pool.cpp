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


#ifdef GPU_DEBUG_CONFIG
#define MEM_USER(uid, nid, pid, cnt) uid, nid, pid, cnt
#else
#define MEM_USER(uid, nid, pid, cnt) uid, nid, pid
#endif
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
    // check non padded pool first
    auto _layout = mem->get_layout();
    auto type = mem->get_allocation_type();
    const auto _layout_bytes_count = _layout.bytes_count();

    GPU_DEBUG_GET_INSTANCE(debug_config);
    {
        auto it = _non_padded_pool.lower_bound(_layout_bytes_count);

        while (it != _non_padded_pool.end()) {
            if (it->second._network_id == network_id &&
                it->second._type == type &&
                it->second._memory->get_internal_params().mem == mem->get_internal_params().mem) {
                auto user_it = it->second._users.find({MEM_USER(unique_id, network_id, prim_id, _layout_bytes_count)});
                // normally there should be only one entry
                if (user_it != it->second._users.end()) {
                    user_it = it->second._users.erase(user_it);
                }
                if (it->second._users.empty()) {
#ifdef GPU_DEBUG_CONFIG
                    GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                        auto released_mem_size = it->first;
                        total_mem_size_non_padded_pool -= released_mem_size;
                        if (type == allocation_type::usm_host)
                            mem_size_non_padded_pool_host -= released_mem_size;
                    }
#endif
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
                if (list_itr->_memory.get()->get_internal_params().mem == mem->get_internal_params().mem &&
                    list_itr->_network_id == network_id &&
                    list_itr->_type == type) {
                    auto user_it = list_itr->_users.find({MEM_USER(unique_id, network_id, prim_id, _layout_bytes_count)});

                    // normally there should be only one entry
                    if (user_it != list_itr->_users.end()) {
                        user_it = list_itr->_users.erase(user_it);
                    }
                    if (list_itr->_users.empty()) {
#ifdef GPU_DEBUG_CONFIG
                        GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                            auto released_mem_size = mem->size();
                            total_mem_size_padded_pool -= released_mem_size;
                            if (type == allocation_type::usm_host)
                                mem_size_padded_pool_host -= released_mem_size;
                        }
#endif
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
#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_IF(debug_config->dump_memory_pool) {
        auto iter = std::find_if(_no_reusable_mems.begin(), _no_reusable_mems.end(), [&](const cldnn::memory_record& r) {
            return (network_id == r._network_id
                && type == r._type
                && mem->get_internal_params().mem == r._memory->get_internal_params().mem);
        });
        if (iter != _no_reusable_mems.end()) {
            GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                auto released_mem_size = iter->_users.begin()->_mem_size;
                total_mem_size_no_reusable -= released_mem_size;
                if (type == allocation_type::usm_host)
                    mem_size_no_reusable_host -= released_mem_size;
            }
            iter->_users.clear();
            _no_reusable_mems.erase(iter);
        }
    }
#endif
}

memory::ptr memory_pool::get_from_non_padded_pool(const layout& layout,
                                                  const primitive_id& prim_id,
                                                  size_t unique_id,
                                                  uint32_t network_id,
                                                  const std::unordered_set<size_t>& restrictions,
                                                  allocation_type type,
                                                  bool reset,
                                                  bool is_dynamic) {
    const auto layout_bytes_count = layout.bytes_count();
    auto it = _non_padded_pool.lower_bound(layout_bytes_count);
    while (it != _non_padded_pool.end()) {
        if ((!is_dynamic || (layout_bytes_count > it->second._memory->get_layout().bytes_count() * 0.5)) &&
            (it->second._network_id == network_id &&
            it->second._type == type &&
            it->second._memory->get_layout().format != format::fs_b_yx_fsv32 &&
            layout.format != format::fs_b_yx_fsv32 &&
            ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
             (layout.feature() % 32 == 0)) &&
            !has_conflict(it->second._users, restrictions, network_id))) {
            it->second._users.insert(memory_user(MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)));
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
        _non_padded_pool.emplace(layout_bytes_count,
                                 memory_record({{MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)}}, mem, network_id, type));
#ifdef GPU_DEBUG_CONFIG
        {
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                total_mem_size_non_padded_pool += layout_bytes_count;
                if (type == allocation_type::usm_host)
                    mem_size_non_padded_pool_host += layout_bytes_count;
            }
        }
#endif
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
                auto ret_mem = _engine->reinterpret_buffer(*(rec_list._memory), layout);
                rec_list._users.insert({MEM_USER(unique_id, network_id, prim_id, ret_mem->size())});
                GPU_DEBUG_CODE(ret_mem->from_memory_pool = true);
                return ret_mem;
            }
        }
        auto mem = alloc_memory(layout, type);
        first_level_cache->second.emplace_back(
            memory_record({{MEM_USER(unique_id, network_id, prim_id, mem->size())}}, mem, network_id, type));
#ifdef GPU_DEBUG_CONFIG
        {
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                const auto allocated_mem_size = mem->size();
                total_mem_size_padded_pool += allocated_mem_size;
                if (type == allocation_type::usm_host)
                    mem_size_padded_pool_host += allocated_mem_size;
            }
        }
#endif
        return mem;
    }
    GPU_DEBUG_LOG << "[" << prim_id << "(" << unique_id << ")" << ": output]" << std::endl;
    auto mem = alloc_memory(layout, type);
    std::list<memory_record> list = {memory_record({{MEM_USER(unique_id, network_id, prim_id, mem->size())}}, mem, network_id, type)};
    _padded_pool.emplace(layout, std::move(list));
#ifdef GPU_DEBUG_CONFIG
    {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->dump_memory_pool) {
            const auto allocated_mem_size = mem->size();
            total_mem_size_padded_pool += allocated_mem_size;
            if (type == allocation_type::usm_host)
                mem_size_padded_pool_host += allocated_mem_size;
        }
    }
#endif
    return mem;
}

/*
        This is not reusable within one network or it's internal micro networks. But we can use this memory records
   between networks.
    */
memory::ptr memory_pool::get_from_across_networks_pool(const layout& layout,
                                                       const primitive_id& prim_id,
                                                       size_t unique_id,
                                                       uint32_t network_id,
                                                       allocation_type type) {
    const auto layout_bytes_count = layout.bytes_count();
    auto it = _no_reusable_pool.lower_bound(layout_bytes_count);

    while (it != _no_reusable_pool.end()) {
        if (it->second._network_id != network_id &&
            it->second._type == type) {  // don't use non reusable resources within the same network
            if (!has_conflict(it->second._users, {}, network_id)) {
                it->second._users.insert(memory_user(MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)));
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                GPU_DEBUG_CODE(ret_mem->from_memory_pool = true);
                return ret_mem;
            }
        }
        ++it;
    }
    auto mem = alloc_memory(layout, type);
    {
        _no_reusable_pool.emplace(layout_bytes_count,
                                  memory_record({{MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)}}, mem, network_id, type));
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
        if (!layout.format.is_image() && !layout.data_padding) {
            // non-padded buffers
            return get_from_non_padded_pool(layout, prim_id, unique_id, network_id, restrictions, type, reset, is_dynamic);
        } else if (!layout.format.is_image()) {
            // padded buffers
            return get_from_padded_pool(layout, prim_id, unique_id, network_id, restrictions, type);
        } else {
            // images (reuse not yet implemented)
            auto mem = alloc_memory(layout, type, reset);
#ifdef GPU_DEBUG_CONFIG
            GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                auto allocated_mem_size = mem->size();
                _no_reusable_mems.push_back(
                                        memory_record({{MEM_USER(unique_id, network_id, prim_id, allocated_mem_size)}}, mem, network_id, type));
                total_mem_size_no_reusable += allocated_mem_size;
                if (type == allocation_type::usm_host)
                    mem_size_no_reusable_host += allocated_mem_size;
            }
#endif
            return mem;
        }
    } else {
        auto mem = alloc_memory(layout, type, reset);
#ifdef GPU_DEBUG_CONFIG
        GPU_DEBUG_IF(debug_config->dump_memory_pool) {
            auto allocated_mem_size = mem->size();
            _no_reusable_mems.push_back(
                                    memory_record({{MEM_USER(unique_id, network_id, prim_id, allocated_mem_size)}}, mem, network_id, type));
            total_mem_size_no_reusable += allocated_mem_size;
            if (type == allocation_type::usm_host)
                mem_size_no_reusable_host += allocated_mem_size;
        }
#endif
        return mem;
    }
}

void memory_pool::clear_pool_for_network(uint32_t network_id) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    // free up _non_padded_pool for this network
    {
        auto itr = _non_padded_pool.begin();

        while (itr != _non_padded_pool.end()) {
            auto& record = itr->second;

            if (record._network_id == network_id) {
#ifdef GPU_DEBUG_CONFIG
                GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                    auto released_mem_size = itr->first;
                    total_mem_size_non_padded_pool -= released_mem_size;
                    if (record._type == allocation_type::usm_host)
                        mem_size_non_padded_pool_host -= released_mem_size;
                }
#endif
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
#ifdef GPU_DEBUG_CONFIG
            auto type = list_itr->_type;
#endif
            while (list_itr != list.end()) {
                if (list_itr->_network_id == network_id) {
                    list_itr = list.erase(list_itr);
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
#ifdef GPU_DEBUG_CONFIG
                GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                    auto released_mem_size = itr->first.bytes_count();
                    total_mem_size_padded_pool -= released_mem_size;
                    if (type == allocation_type::usm_host)
                        mem_size_padded_pool_host -= released_mem_size;
                }
#endif
                itr = _padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

#ifdef GPU_DEBUG_CONFIG
    // free up _no_reusable_mems for this network
    GPU_DEBUG_IF(debug_config->dump_memory_pool) {
        auto itr = _no_reusable_mems.begin();
        while (itr != _no_reusable_mems.end()) {
            auto& record = *itr;
            if (itr->_network_id == network_id) {
                GPU_DEBUG_IF(debug_config->dump_memory_pool) {
                    auto released_mem_size = itr->_users.begin()->_mem_size;
                    total_mem_size_no_reusable -= released_mem_size;
                    if (record._type == allocation_type::usm_host)
                        mem_size_no_reusable_host -= released_mem_size;
                }
                itr = _no_reusable_mems.erase(itr);
            } else {
                itr++;
            }
        }
    }
#endif

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

#ifdef GPU_DEBUG_CONFIG
inline std::string get_mb_size(size_t size) {
    if (size == 0)
        return "0 MB";
    return std::to_string(static_cast<float>(size) / (1024 * 1024)) + " MB";
}

inline float get_utilization(size_t size, size_t total_size) {
    return (static_cast<float>(size) * 100.0f / total_size);
}
#endif

size_t memory_pool::get_total_mem_pool_size(allocation_type type) {
#ifdef GPU_DEBUG_CONFIG
    const auto host_mem_size = mem_size_no_reusable_host + mem_size_non_padded_pool_host + mem_size_padded_pool_host;
    const auto total_mem_size = total_mem_size_no_reusable + total_mem_size_non_padded_pool + total_mem_size_padded_pool;
    if (type == allocation_type::usm_host) {
        return host_mem_size;
    } else {
        return (total_mem_size - host_mem_size);
    }
#else
    return 0;
#endif
}

void memory_pool::dump(uint32_t net_id, uint32_t iter, std::string dump_dir_path) {
    dump_to_screen(net_id, iter);
    if (!dump_dir_path.empty())
        dump_to_file(net_id, iter, dump_dir_path);
}

void memory_pool::dump_to_file(uint32_t net_id, uint32_t iter, std::string dump_dir_path) {
#ifdef GPU_DEBUG_CONFIG
    const std::string dump_file_name = "dump_runtime_memory_pool_net_" + std::to_string(net_id) + "_iter_" + std::to_string(iter) + ".csv";
    const std::string desc = "pool_type,layout,mem_ptr,mem_type,mem_pool_size,prim_id,unique_id,mem_size";
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
                const size_t mem_pool_size = record._memory->size();
                for (auto user : record._users) {
                    of << "padded_pool," << mem.first.to_short_string() << "," << record._memory->buffer_ptr() << "," << record._type << ","
                        << mem_pool_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
                }
            }
        }
        for (auto mem : _no_reusable_mems) {
            for (auto user : mem._users) {
                of << "no_reusable_pool,," << mem._memory->buffer_ptr() << "," << mem._type << ","
                    << user._mem_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
            }
        }
        std::cout << "Dump file to " << dump_path << std::endl;
    }
#endif
}

void memory_pool::dump_to_screen(uint32_t net_id, uint32_t iter) {
#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_COUT << "Dump memory pool of network (net_id : " << net_id << ", iter : " << iter << ")" << std::endl;
    float total_requested_mem_non_padded_pool    = 0.f;
    float total_requested_mem_padded_pool        = 0.f;

    {
        GPU_DEBUG_COUT << "========== non-padded pool ( " << _non_padded_pool.size() << " records) ==========" << std::endl;
        for (auto mem : _non_padded_pool) {
            GPU_DEBUG_COUT << mem.second._memory->buffer_ptr() << " (size: " << get_mb_size(mem.first)
                << ", type: " << mem.second._type << ")'s users: " << std::endl;
            float min_utilization = 100.0f;
            float max_utilization = 0.f;
            for (auto user : mem.second._users) {
                float utilization = get_utilization(user._mem_size, mem.first);
                min_utilization = std::min(utilization, min_utilization);
                max_utilization = std::max(utilization, max_utilization);
                total_requested_mem_non_padded_pool += static_cast<float>(user._mem_size);
                GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                    << get_mb_size(user._mem_size) << ", " << utilization << "%" << std::endl;
            }
            GPU_DEBUG_COUT <<  "   - min utilization of the memory pool entry: " << min_utilization << " %" << std::endl;
            GPU_DEBUG_COUT <<  "   - max utilization of the memory pool entry: " << max_utilization << " %" << std::endl;
        }
    }

    {
        GPU_DEBUG_COUT << "========== padded pool (" << _padded_pool.size() << " records) ==========" << std::endl;
        for (auto mem : _padded_pool) {
            GPU_DEBUG_COUT << " layout: " << mem.first.to_short_string() << ", records(" << mem.second.size() << ")" << std::endl;
            for (auto record : mem.second) {
                size_t mem_size = record._memory->size();
                GPU_DEBUG_COUT << "  " << record._memory->buffer_ptr() << " (size:" << get_mb_size(mem_size)
                                << "MB, type: " << record._type << ")'s users : " << std::endl;
                float min_utilization = 100.0f;
                float max_utilization = 0.f;
                for (auto user : record._users) {
                    float utilization = get_utilization(user._mem_size, mem_size);
                    min_utilization = std::min(utilization, min_utilization);
                    max_utilization = std::max(utilization, max_utilization);
                    total_requested_mem_padded_pool += static_cast<float>(user._mem_size);
                    GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                        << get_mb_size(user._mem_size) << ", " << utilization << "%" << std::endl;
                }
                GPU_DEBUG_COUT << "   - min of the memory pool entry: " << min_utilization << std::endl;
                GPU_DEBUG_COUT << "   - max of the memory pool entry: " << max_utilization << std::endl;
            }
        }
    }

    {
        GPU_DEBUG_COUT << "========== no reusable memory (" << _no_reusable_mems.size() << " records) ==========" << std::endl;
        for (auto mem : _no_reusable_mems) {
            GPU_DEBUG_COUT << mem._memory->buffer_ptr() << " (type: " << mem._type << ")'s user: " << std::endl;
            for (auto user : mem._users) {
                GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                    << get_mb_size(user._mem_size) << std::endl;
            }
        }
    }

    GPU_DEBUG_COUT << "************************************************************************" << std::endl;
    GPU_DEBUG_COUT << "Memory pool footprint of the network (net_id : " << net_id << ", iter : " << iter << ")" << std::endl;
    GPU_DEBUG_COUT << "Total memory size of non_padded_pool     : " << get_mb_size(total_mem_size_non_padded_pool) << std::endl;
    if (total_mem_size_non_padded_pool > 0.f) {
        GPU_DEBUG_COUT << " * Efficiency        : "
            << std::to_string(static_cast<float>(total_requested_mem_non_padded_pool / total_mem_size_non_padded_pool))
            << " (total mem requested : " << get_mb_size(total_requested_mem_non_padded_pool)
            << " / total mem pool size : " << get_mb_size(total_mem_size_non_padded_pool) << ")" << std::endl;
        GPU_DEBUG_COUT << " * host mem size     : " << get_mb_size(mem_size_non_padded_pool_host) << std::endl;
        GPU_DEBUG_COUT << " * device mem size   : "
                            << get_mb_size(total_mem_size_non_padded_pool - mem_size_non_padded_pool_host) << std::endl;
    }
    GPU_DEBUG_COUT << "Total memory size of padded_pool memory  : " << get_mb_size(total_mem_size_padded_pool) << std::endl;
    if (total_mem_size_padded_pool > 0.f) {
        GPU_DEBUG_COUT << " * Efficiency        : "
            << std::to_string(static_cast<float>(total_requested_mem_padded_pool / total_mem_size_padded_pool))
            << " (total mem requested : " << get_mb_size(total_requested_mem_padded_pool)
            << " / total mem pool size : " << get_mb_size(total_mem_size_padded_pool) << ")" << std::endl;
        GPU_DEBUG_COUT << " * host mem size     : " << get_mb_size(mem_size_padded_pool_host) << std::endl;
        GPU_DEBUG_COUT << " * device mem size   : " << get_mb_size((total_mem_size_padded_pool - mem_size_padded_pool_host)) << std::endl;
    }
    GPU_DEBUG_COUT << "Total memory size of no reusable memory  : " << get_mb_size(total_mem_size_no_reusable) << std::endl;
    if (total_mem_size_no_reusable > 0.f) {
        GPU_DEBUG_COUT << " * host mem size     : " << get_mb_size(mem_size_no_reusable_host) << std::endl;
        GPU_DEBUG_COUT << " * device mem size   : " << get_mb_size((total_mem_size_no_reusable - mem_size_no_reusable_host)) << std::endl;
    }
    GPU_DEBUG_COUT << "************************************************************************" << std::endl;
#endif
}
}  // namespace cldnn
