// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"
#include "memory_caps.hpp"

#include <vector>
#include <set>
#include <map>
#include <list>
#include <string>
#include <atomic>
#include <mutex>

namespace cldnn {

struct memory;
struct shared_mem_params;
class engine;
struct memory_user;
struct memory_user_comparer;
using memory_set = std::set<memory_user, memory_user_comparer>;
using primitive_id = std::string;

using memory_ptr = std::shared_ptr<memory>;
using memory_ptr_vector = std::vector<memory_ptr>;
const size_t ALIGNED_SIZE = (10*1024*1024);
const size_t SIZE_4KB = (4*1024);
const size_t SIZE_16KB = (16*1024);
const size_t SIZE_256KB = (256*1024);
const size_t SIZE_2MB = (2*1024*1024);
const size_t NUM_4KB = ALIGNED_SIZE / SIZE_4KB;
const size_t NUM_16KB = ALIGNED_SIZE / SIZE_16KB;
const size_t NUM_256KB = ALIGNED_SIZE / SIZE_256KB;
const size_t NUM_2MB = ALIGNED_SIZE / SIZE_2MB;

struct memory_user {
    primitive_id _id;
    uint32_t _network_id;

    memory_user(primitive_id id, uint32_t network_id)
        : _id(id), _network_id(network_id) {}

    friend std::ostream& operator<<(std::ostream& os, const memory_user& memory_user) {
        os << memory_user._id << "(" << memory_user._network_id << ")";
        return os;
    }
};

struct memory_user_comparer {
    bool operator()(const memory_user& l_mu, const memory_user& r_mu) const {
        if (l_mu._network_id != r_mu._network_id)
            return l_mu._network_id < r_mu._network_id;
        return l_mu._id < r_mu._id;
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
    static bool has_conflict(const memory_set&, const std::set<primitive_id>&, uint32_t network_id);

    std::multimap<uint64_t, memory_record> _non_padded_pool;
    std::map<layout, std::list<memory_record>, padded_pool_comparer> _padded_pool;
    std::multimap<uint64_t, memory_record> _no_reusable_pool;
    engine* _engine;

    // Allocted Aligned device memory 2MB size ahead
    // std::vector<memory_ptr> _aligned_mem_buffer;
    // memory_ptr _aligned_mem_4KB;        // Handle memalloc smaller than 4KB
    // memory_ptr _aligned_mem_16KB;       // Handle memalloc smaller than 4KB
    // memory_ptr _aligned_mem_256KB;      // Handle memalloc smaller than 4KB
    // size_t _idx_4KB_aligned;
    // size_t _idx_16KB_aligned;
    // size_t _idx_256KB_aligned;


    void init_aligned_memory();
    std::pair<memory_ptr, size_t *> get_matching_aligned_memory(size_t alloc_size);

public:
    explicit memory_pool(engine& engine);
    ~memory_pool();
    memory_ptr get_memory(const layout& layout,
                          const primitive_id& id,
                          uint32_t network_id,
                          const std::set<primitive_id>& restrictions,
                          allocation_type type,
                          bool reusable = true,
                          bool reset = true);  // get from pool or create memory allocation
    memory_ptr get_memory(const layout& layout, allocation_type type, bool reset = true);
    memory_ptr get_from_non_padded_pool(const layout& layout,
                                        const primitive_id& id,
                                        uint32_t network_id,
                                        const std::set<primitive_id>&,
                                        allocation_type type,
                                        bool reset = true);
    memory_ptr get_from_padded_pool(const layout& layout,
                                    const primitive_id& id,
                                    uint32_t network_id,
                                    const std::set<primitive_id>& restrictions,
                                    allocation_type type);
    memory_ptr get_from_across_networks_pool(const layout& layout,
                                             const primitive_id& id,
                                             uint32_t network_id,
                                             allocation_type type);
    void clear_pool_for_network(uint32_t network_id);
    void release_memory(memory* memory, const primitive_id& id, uint32_t network_id);

    size_t get_non_padded_pool_size() {
        return _non_padded_pool.size();
    }

    void dump(uint32_t id);
};

}  // namespace cldnn
