// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"
#include "memory_caps.hpp"
#include "utils.hpp"

#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <list>
#include <string>
#include <atomic>

namespace cldnn {

struct memory;
struct shared_mem_params;
class engine;

using primitive_id = std::string;
using memory_ptr = std::shared_ptr<memory>;

struct memory_user {
    size_t _unique_id;
    uint32_t _network_id;
    primitive_id _prim_id;
#ifdef GPU_DEBUG_CONFIG
    size_t _mem_size;

    memory_user(size_t unique_id, uint32_t network_id, primitive_id prim_id, size_t mem_size)
        : _unique_id(unique_id), _network_id(network_id), _prim_id(prim_id), _mem_size(mem_size) {}
#endif

    memory_user(size_t unique_id, uint32_t network_id, primitive_id prim_id)
        : _unique_id(unique_id), _network_id(network_id), _prim_id(prim_id) {}

    bool operator==(const struct memory_user& rhs) const {
        return _unique_id == rhs._unique_id && _network_id == rhs._network_id;
    }

    friend std::ostream& operator<<(std::ostream& os, const memory_user& memory_user) {
        os << memory_user._prim_id << " (unique_id:" << memory_user._unique_id;
        os << ", net_id:" << memory_user._network_id << ")";
#ifdef GPU_DEBUG_CONFIG
        os << ", mem_size: " << memory_user._mem_size;
#endif
        return os;
    }
};
struct memory_set_hasher {
    size_t operator()(const memory_user& mem_user) const {
        return hash_combine(0, mem_user._unique_id);
    }
};

using memory_set = std::unordered_set<memory_user, memory_set_hasher>;

struct memory_user_comparer {
    bool operator()(const memory_user& l_mu, const memory_user& r_mu) const {
        if (l_mu._network_id != r_mu._network_id)
            return l_mu._network_id < r_mu._network_id;
        return l_mu._unique_id < r_mu._unique_id;
    }
};

struct memory_record {
    memory_set _users;  // list of primitives that already use this memory object
    memory_ptr _memory;
    uint32_t _network_id;
    allocation_type _type;

    memory_record(memory_set users, memory_ptr& memory, uint32_t net_id, allocation_type type);
};

struct padded_pool_comparer {
    bool operator()(const layout& ll, const layout& rl) const {
        if (ll.format != rl.format)
            return ll.format < rl.format;
        if (ll.data_type != rl.data_type)
            return ll.data_type < rl.data_type;
        if (ll.spatial(0) != rl.spatial(0))
            return ll.spatial(0) < rl.spatial(0);
        if (ll.spatial(1) != rl.spatial(1))
            return ll.spatial(1) < rl.spatial(1);
        return ll.data_padding < rl.data_padding;
    }
};

// memory_pool class implements memory manager that handles 4 memory pools
// - non padded buffers -
//     1 user requests for buffer with no padding.
//     2 Check if buffer with requested size exist
//     3   * yes: check if any of current users exist on request conflict list if no - return this memory, otherwise
//     goto 4
//         * no: goto 4
//     4 take next (allocations are sorted in increasing order) allocation. if there is no more allocations, create new
//     allocation otherwise go t
// - padded buffers - not implemented yet
// - images 2d - not implemented yet
// - images 2d arrays - not implemented yet
// - immutable - if user request for non reusable resource don't use pool, return

// TODO list:
// - Move from runtime to graph part
// - Improve memory consumption

class memory_pool {
    memory_pool();

    memory_ptr alloc_memory(const layout& layout, allocation_type type, bool reset = true);
    static bool has_conflict(const memory_set&, const std::unordered_set<size_t>&, uint32_t network_id);

    std::multimap<uint64_t, memory_record> _non_padded_pool;
    std::map<layout, std::list<memory_record>, padded_pool_comparer> _padded_pool;
    std::multimap<uint64_t, memory_record> _no_reusable_pool;
    engine* _engine;

public:
    explicit memory_pool(engine& engine);
    ~memory_pool();
    memory_ptr get_memory(const layout& layout,
                          const primitive_id& id,
                          size_t unique_id,
                          uint32_t network_id,
                          const std::unordered_set<size_t>& restrictions,
                          allocation_type type,
                          bool reusable = true,
                          bool reset = true,
                          bool is_dynamic = false);  // get from pool or create memory allocation
    memory_ptr get_memory(const layout& layout, allocation_type type, bool reset = true);
    memory_ptr get_from_non_padded_pool(const layout& layout,
                                        const primitive_id& prim_id,
                                        size_t unique_id,
                                        uint32_t network_id,
                                        const std::unordered_set<size_t>&,
                                        allocation_type type,
                                        bool reset = true,
                                        bool is_dynamic = false);
    memory_ptr get_from_padded_pool(const layout& layout,
                                    const primitive_id& prim_id,
                                    size_t unique_id,
                                    uint32_t network_id,
                                    const std::unordered_set<size_t>& restrictions,
                                    allocation_type type);
    memory_ptr get_from_across_networks_pool(const layout& layout,
                                             const primitive_id& id,
                                             size_t unique_id,
                                             uint32_t network_id,
                                             allocation_type type);
    void clear_pool_for_network(uint32_t network_id);
    void release_memory(memory* memory, const size_t& unique_id, primitive_id prim_id, uint32_t network_id);

    size_t get_non_padded_pool_size() {
        return _non_padded_pool.size();
    }

    void dump(uint32_t id, uint32_t iter, std::string dump_dir_path = "");
    size_t get_total_mem_pool_size(allocation_type type);

private:
    void dump_to_screen(uint32_t id, uint32_t iter);
    void dump_to_file(uint32_t id, uint32_t iter, std::string dump_dir_path);

#ifdef GPU_DEBUG_CONFIG
    std::vector<memory_record> _no_reusable_mems;

    float total_mem_size_non_padded_pool        = 0.f;
    float total_mem_size_padded_pool            = 0.f;
    float total_mem_size_no_reusable            = 0.f;
    float mem_size_non_padded_pool_host         = 0.f;
    float mem_size_padded_pool_host             = 0.f;
    float mem_size_no_reusable_host             = 0.f;
#endif
};

}  // namespace cldnn
