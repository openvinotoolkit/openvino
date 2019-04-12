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
#pragma once
#include "api/CPP/layout.hpp"
#include "api/CPP/primitive.hpp"
#include "api_impl.h"

#include "refcounted_obj.h"

#include <vector>
#include <set>
#include <map>

namespace cldnn
{

struct memory_impl;
struct engine_impl;
struct program_impl;
struct memory_user;
struct memory_user_comparer;
using memory_set = std::set<memory_user, memory_user_comparer>;

struct memory_user
{
    primitive_id _id;
    uint32_t _network_id;

    memory_user(primitive_id id, uint32_t network_id) :
        _id(id) ,
        _network_id(network_id) 
    {}

    friend std::ostream& operator<<(std::ostream& os, const memory_user& memory_user)
    {
        os << memory_user._id << "(" << memory_user._network_id << ")";
        return os;
    }
};

struct memory_user_comparer
{
    bool operator()(const memory_user& l_mu, const memory_user& r_mu) const
    {
        if (l_mu._network_id != r_mu._network_id)
            return l_mu._network_id < r_mu._network_id;
        return l_mu._id < r_mu._id;
    }
};


struct memory_record
{
    memory_set _users; // list of primitives that already use this memory object
    refcounted_obj_ptr<memory_impl> _memory;
    uint32_t _network_id;

    memory_record(memory_set users, refcounted_obj_ptr<memory_impl>& memory, uint32_t net_id);
};

struct padded_pool_comparer
{
    bool operator()(const layout& ll, const layout& rl) const
    {
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
    //     3   * yes: check if any of current users exist on request conflict list if no - return this memory, otherwise goto 4
    //         * no: goto 4
    //     4 take next (allocations are sorted in increasing order) allocation. if there is no more allocations, create new allocation otherwise go t
    // - padded buffers - not implemented yet
    // - images 2d - not implemented yet
    // - images 2d arrays - not implemented yet
    // - immutable - if user request for non reusable resource don't use pool, return 

// TODO list:
// - resolve engine <--> memory_pool circular dependency
// - add padded buffers pool
// - add decreasing memory limit in gpu_buffer/image dctor
// - add support for multi networks reuse

class memory_pool
{
    memory_pool();
    
    refcounted_obj_ptr<memory_impl> alloc_memory(const layout& layout);
    static bool has_conflict(const memory_set&, const std::set<primitive_id>&, uint32_t);

    std::multimap<uint64_t, memory_record> _non_padded_pool;
    std::map<layout,std::list<memory_record>, padded_pool_comparer> _padded_pool;
    std::multimap<uint64_t, memory_record> _no_reusable_pool;
    refcounted_obj_ptr<engine_impl> _engine;
    uint64_t _temp_memory_used;
    uint64_t _max_peak_memory_used;
public:
    memory_pool(engine_impl& engine);
    ~memory_pool();
    refcounted_obj_ptr<memory_impl> get_memory(const layout& layout, const primitive_id& id, uint32_t network_id,  const std::set<primitive_id>& restrictions, bool reusable = true); // get from pool or create memory allocation
    refcounted_obj_ptr<memory_impl> get_memory(const layout& layout);
    refcounted_obj_ptr<memory_impl> get_from_non_padded_pool(const layout& layout, const primitive_id& id, uint32_t network_id, const std::set<primitive_id>&);
    refcounted_obj_ptr<memory_impl> get_from_padded_pool(const layout& layout, const primitive_id& id, uint32_t network_id, const std::set<primitive_id>& restrictions);
    refcounted_obj_ptr<memory_impl> get_from_across_networks_pool(const layout& layout, const primitive_id& id, uint32_t network_id);
    void clear_pool();
    void color_graph(const program_impl&);
    void dump_memory_pool(const program_impl&, std::string, std::string);

    uint64_t get_temp_memory_used() const { return _temp_memory_used; };
    uint64_t get_max_peak_device_memory_used() const { return _max_peak_memory_used; };
    void add_memory_used(size_t value);
    void subtract_memory_used(size_t value);
};

}
