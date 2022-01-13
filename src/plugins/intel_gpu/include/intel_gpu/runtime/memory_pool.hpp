// Copyright (C) 2018-2021 Intel Corporation
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

namespace cldnn {

struct memory;
struct shared_mem_params;
class engine;
struct memory_user;
struct memory_user_comparer;
using memory_set = std::set<memory_user, memory_user_comparer>;
using primitive_id = std::string;

using memory_ptr = std::shared_ptr<memory>;

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
        if (ll.size.spatial[0] != rl.size.spatial[0])
            return ll.size.spatial[0] < rl.size.spatial[0];
        if (ll.size.spatial[1] != rl.size.spatial[1])
            return ll.size.spatial[1] < rl.size.spatial[1];
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

    memory_ptr alloc_memory(const layout& layout, allocation_type type);
    static bool has_conflict(const memory_set&, const std::set<primitive_id>&, uint32_t network_id);

    std::multimap<uint64_t, memory_record> _non_padded_pool;
    std::map<layout, std::list<memory_record>, padded_pool_comparer> _padded_pool;
    std::multimap<uint64_t, memory_record> _no_reusable_pool;
    engine* _engine;

public:
    explicit memory_pool(engine& engine);
    ~memory_pool();
    memory_ptr get_memory(const layout& layout,
                          const primitive_id& id,
                          uint32_t network_id,
                          const std::set<primitive_id>& restrictions,
                          allocation_type type,
                          bool reusable = true);  // get from pool or create memory allocation
    memory_ptr get_memory(const layout& layout, allocation_type type);
    memory_ptr get_from_non_padded_pool(const layout& layout,
                                        const primitive_id& id,
                                        uint32_t network_id,
                                        const std::set<primitive_id>&,
                                        allocation_type type);
    memory_ptr get_from_padded_pool(const layout& layout,
                                    const primitive_id& id,
                                    uint32_t network_id,
                                    const std::set<primitive_id>& restrictions,
                                    allocation_type type);
    memory_ptr get_from_across_networks_pool(const layout& layout,
                                             const primitive_id& id,
                                             uint32_t network_id,
                                             allocation_type type);
    void clear_pool();
    void clear_pool_for_network(uint32_t network_id);
    void release_memory(memory* memory, const primitive_id& id, uint32_t network_id);
};

}  // namespace cldnn
