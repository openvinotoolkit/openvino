// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/execution_config.hpp"
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
#ifdef GPU_DEBUG_CONFIG
#include <array>
#include <cstdint>
#include <mutex>
#endif

namespace cldnn {

struct memory;
struct shared_mem_params;
class engine;

using primitive_id = std::string;
using memory_ptr = std::shared_ptr<memory>;

template<typename Key>
class memory_restricter {
    private:
        const std::vector<Key>* ordered_keys_set1;  // Const reference to immutable set
        std::vector<Key> ordered_keys_set2;         // Internal mutable set

    public:
        memory_restricter() : ordered_keys_set1(nullptr) {};

        // Constructor to initialize with a const reference for set1
        explicit memory_restricter(const std::vector<Key>* externalSet)
            : ordered_keys_set1(externalSet) {
        }

        // Insert into set2 (set1 is read-only)
        void insert(const Key& key) {
            if (ordered_keys_set1) {
                auto it = std::lower_bound(ordered_keys_set1->begin(), ordered_keys_set1->end(), key);
                if (it != ordered_keys_set1->end() && *it == key) {
                    return;
                }
            }
            auto it2 = std::lower_bound(ordered_keys_set2.begin(), ordered_keys_set2.end(), key);
            if (it2 == ordered_keys_set2.end() || *it2 != key) {
                ordered_keys_set2.insert(it2, key);
            }
        }

        // Check existence in either set
        bool contains(const Key& key) const {
            if (ordered_keys_set1 && !ordered_keys_set1->empty()) {
                if (std::binary_search(ordered_keys_set1->begin(), ordered_keys_set1->end(), key)) {
                    return true;
                }
            }
            return std::binary_search(ordered_keys_set2.begin(), ordered_keys_set2.end(), key);
        }

        // Total size of both sets
        size_t size() const {
            return (ordered_keys_set1 ? ordered_keys_set1->size() : 0) + ordered_keys_set2.size();
        }

        // Check if both sets are empty
        bool empty() const {
            return (!ordered_keys_set1 || ordered_keys_set1->empty()) && ordered_keys_set2.empty();
        }

        // Iterate over both sets
        void for_each(void(*func)(const Key&)) const {
            if (ordered_keys_set1) {
                for (const auto& key : *ordered_keys_set1) func(key);
            }
            for (const auto& key : ordered_keys_set2) func(key);
        }
}; // end of memory_restricter

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
        if (l_mu._network_id != r_mu._network_id) {
            return l_mu._network_id < r_mu._network_id;
        }
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


#ifdef GPU_DEBUG_CONFIG
// Statistics caveats (also printed next to the summary in debug_helper.cpp):
//  1. "network output share" = network output bytes / cumulative allocated bytes (every allocation event, incl. reallocations), not / peak.
//  2. "foreign user records" (pool users whose uid is not a primitive of the network) is NOT the same-block probe of design 08 (F18); the
//     pool de-duplicates users by uid, so a same-uid-twice-on-one-block probe is not observable with it.
//  3. `pooled` is over-counted when the pool falls back to a direct allocation inside memory_pool::get_memory (image formats,
//     disable_memory_reuse debug flag): the hook only sees "pool enabled && reusable".
//  4. Dynamic kv dependencies (prepare_buffer_fusing.cpp, can_share_buffer(false) on kv_cache related nodes) that are not kv/variable
//     nodes themselves may be classified as sum_root (can_share_buffer(false), not constant-like).
// Allocation trace (B0/D0 minimal subset). Filled only under GPU_DEBUG when the dump_alloc_trace option is set; it never influences
// allocation behavior. Categories are mutually exclusive. Off-pool categories are assigned by the first match of this priority list
// (classification itself lives in primitive_inst.cpp, trace_allocation()):
//   shape_of > kv_variable > network_output > [internal_buffer] > inplace_concat_group > sum_root > multi_output_port > constant > other
// Allocations served by the pool are always `pooled`. `pool_disabled` is used for every allocation if the memory pool is disabled.
enum class alloc_category : uint8_t {
    pooled = 0,
    shape_of,
    kv_variable,
    network_output,
    internal_buffer,
    inplace_concat_group,
    sum_root,
    multi_output_port,
    constant,
    pool_disabled,
    other,
    count_
};

inline const char* to_string(alloc_category c) {
    switch (c) {
    case alloc_category::pooled: return "pooled";
    case alloc_category::shape_of: return "shape_of";
    case alloc_category::kv_variable: return "kv_variable";
    case alloc_category::network_output: return "network_output";
    case alloc_category::internal_buffer: return "internal_buffer";
    case alloc_category::inplace_concat_group: return "inplace_concat_group";
    case alloc_category::sum_root: return "sum_root";
    case alloc_category::multi_output_port: return "multi_output_port";
    case alloc_category::constant: return "constant";
    case alloc_category::pool_disabled: return "pool_disabled";
    case alloc_category::other: return "other";
    default: return "?";
    }
}

struct alloc_trace_record {
    primitive_id node_id;
    size_t unique_id = 0;
    size_t port = 0;           // output index, or internal buffer index if is_internal_buffer
    uint64_t bytes = 0;        // layout.bytes_count() (max shape for dynamic layouts)
    uint32_t network_id = 0;
    bool pooled = false;
    bool is_internal_buffer = false;
    bool is_output_node = false;
    bool multi_output_node = false;
    alloc_category category = alloc_category::other;
    const char* reason = "";
};

struct alloc_trace_summary {
    struct entry {
        size_t count = 0;
        uint64_t bytes = 0;
    };
    std::array<entry, static_cast<size_t>(alloc_category::count_)> by_category{};
    uint64_t total_bytes = 0;
    size_t total_count = 0;
    uint64_t network_output_bytes = 0;      // bytes of every allocation of an is_output() node, regardless of the category
    std::set<size_t> multi_output_node_uids;  // distinct nodes with more than one output that allocated outputs
    size_t non_first_port_count = 0;        // outputs with idx > 0 of multi-output nodes (pooled or not; internal buffers excluded)
    uint64_t non_first_port_bytes = 0;
    uint64_t generation = 0;                // incremented on every record, used to print only when changed
    uint64_t printed_generation = 0;
    uint64_t dropped_records = 0;           // per-allocation records not stored because the cap was reached (level 2 only)

    const entry& at(alloc_category c) const {
        return by_category[static_cast<size_t>(c)];
    }
    uint64_t off_pool_bytes() const {
        return total_bytes - at(alloc_category::pooled).bytes;
    }
};

// Held-bytes snapshot (GPU_DEBUG statistics). Every unique buffer (deduplicated by the underlying memory handle,
// get_internal_params().mem) of a network is counted once, in exactly one bucket:
//   pooled   - the buffer is a pool record; bytes are the pool BLOCK size, counted once per block (not per user)
//   ghost    - pool record that no primitive of the network references (leaked / stale pool block)
//   constant - owner is data / mutable_data / constant node
//   external - owner is input_layout, or the buffer is a user output memory (remote tensor) or an ext output block
//   variable - owner is kv_cache / read_value / assign / stateless_kv
//   off_pool[category] - everything else, by classify_off_pool_allocation() of the OWNER node
// Owner of a buffer referenced by several nodes (views, concat predecessors sharing the holder, sum writer aliasing its addend):
//   1. a concatenation node with can_be_optimized (the in-place holder);
//   2. else a node that is not can_be_optimized (the real allocator, views are skipped);
//   3. else any node;
//   ties are broken by the smallest unique id (producers are numbered before consumers).
// Bytes of a non-pool buffer: memory tracker size if present (the whole allocation), otherwise memory::size().
// Ghost limit: a leaked pool block is reported as ghost only if no live node references it; if a live node shares it, it is shown as pooled,
// so P1-L1/L2 style leaks are not detected here (they are guarded by the concat_holder_pool_leak tests B0-T1/B0-T2).
// Forward limitation (Phase 4s): sub-buffers share the parent's memory tracker (ocl_engine.cpp ~226/~232) while USM/cl_mem sub-buffers have their
// own handles. Dedup by handle + tracker size would count every sub-buffer as the whole parent block (multiple counting) once an Arena is
// sliced with create_subbuffer; this must be reworked before Phase 4s. Today the graph layer does not use sub-buffers for outputs (only
// shape_info at network.cpp ~397 and MoE internal buffers).
struct held_buffer {
    const void* handle = nullptr;
    uint64_t bytes = 0;
    primitive_id owner;
    std::string bucket;
    allocation_type type = allocation_type::unknown;
    bool referenced_by_output = false;
    // Debug attribution of the owner record (trace join key is (owner_uid, owner_port)); unset for ghost blocks.
    size_t owner_uid = 0;
    size_t owner_port = 0;
    bool owner_intermediate = false;
    int64_t layout_bytes = -1;  // owner output layout bytes_count(); -1 when n/a (internal buffer, ghost, dynamic layout)
    std::string owner_type;
};

struct held_bytes_snapshot {
    struct entry {
        size_t blocks = 0;
        uint64_t bytes = 0;
    };
    entry pooled, ghost, constant, external, variable;
    std::array<entry, static_cast<size_t>(alloc_category::count_)> off_pool{};
    uint64_t total_bytes = 0;
    size_t total_blocks = 0;
    uint64_t device_bytes = 0;          // usm_device + cl_mem part of total_bytes
    uint64_t network_output_bytes = 0;  // buffers referenced by an is_output() primitive (any bucket)
    std::vector<held_buffer> buffers;

    const entry& at_off_pool(alloc_category c) const {
        return off_pool[static_cast<size_t>(c)];
    }
    uint64_t off_pool_bytes() const {
        uint64_t b = 0;
        for (const auto& e : off_pool) {
            b += e.bytes;
        }
        return b;
    }
};

// Pool record description for the snapshot.
struct pool_block_info {
    const void* handle = nullptr;
    uint64_t bytes = 0;
    allocation_type type = allocation_type::unknown;
};
#endif  // GPU_DEBUG_CONFIG

struct padded_pool_comparer {
    bool operator()(const layout& ll, const layout& rl) const {
        if (ll.format != rl.format) {
            return ll.format < rl.format;
        }
        if (ll.data_type != rl.data_type) {
            return ll.data_type < rl.data_type;
        }
        if (ll.spatial(0) != rl.spatial(0)) {
            return ll.spatial(0) < rl.spatial(0);
        }
        if (ll.spatial(1) != rl.spatial(1)) {
            return ll.spatial(1) < rl.spatial(1);
        }
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
    memory_ptr alloc_memory(const layout& layout, allocation_type type, bool reset = true);
    static bool has_conflict(const memory_set&, const memory_restricter<uint32_t>&);

    std::multimap<uint64_t, memory_record> _non_padded_pool;
    std::map<layout, std::list<memory_record>, padded_pool_comparer> _padded_pool;
    engine* _engine;
    const ExecutionConfig& _config;
    float _mem_pool_util_threshold = 0.5f;
#ifdef GPU_DEBUG_CONFIG
    mutable std::mutex _alloc_trace_mutex;
    std::map<uint32_t, alloc_trace_summary> _alloc_trace;
    std::map<uint32_t, std::vector<alloc_trace_record>> _alloc_trace_records;
#endif

public:
    explicit memory_pool(engine& engine, const ExecutionConfig& config);
    ~memory_pool();
    memory_ptr get_memory(const layout& layout,
                          const primitive_id& id,
                          size_t unique_id,
                          uint32_t network_id,
                          const memory_restricter<uint32_t>& restrictions,
                          allocation_type type,
                          bool reusable = true,
                          bool reset = true,
                          bool is_dynamic = false);  // get from pool or create memory allocation
    memory_ptr get_memory(const layout& layout, allocation_type type, bool reset = true);
    memory_ptr get_from_non_padded_pool(const layout& layout,
                                        const primitive_id& prim_id,
                                        size_t unique_id,
                                        uint32_t network_id,
                                        const memory_restricter<uint32_t>&,
                                        allocation_type type,
                                        bool reset = true,
                                        bool is_dynamic = false);
    memory_ptr get_from_padded_pool(const layout& layout,
                                    const primitive_id& prim_id,
                                    size_t unique_id,
                                    uint32_t network_id,
                                    const memory_restricter<uint32_t>& restrictions,
                                    allocation_type type);
    void clear_pool_for_network(uint32_t network_id);
    void release_memory(memory* memory, const size_t& unique_id, primitive_id prim_id, uint32_t network_id);

    size_t get_non_padded_pool_size() {
        return _non_padded_pool.size();
    }

    // Read-only accessors for tests and statistics (do not affect pool behavior).
    // Unique ids of all users registered on the pool record(s) holding the given memory object in the given network.
    std::vector<size_t> get_user_uids(const memory& mem, uint32_t net_id) const;
    // Number of pool records (non padded + padded) in the given network that contain a user with the given unique id.
    size_t count_records_with_user(size_t uid, uint32_t net_id) const;
#ifdef GPU_DEBUG_CONFIG
    // All user unique ids (one entry per (record, user)) of the given network, for ghost-user statistics.
    std::vector<size_t> get_all_user_uids(uint32_t net_id) const;

    // Allocation trace (GPU_DEBUG statistics only).
    // Aggregated counters are always updated; the per-allocation record is stored only if keep_record is set (level 2) and the cap
    // (max_alloc_trace_records) is not reached, otherwise it is counted in alloc_trace_summary::dropped_records.
    static constexpr size_t max_alloc_trace_records = 100000;
    void record_alloc_trace(const alloc_trace_record& rec, bool keep_record = false);
    // Aggregated statistics only (no per-allocation records, cheap to copy).
    alloc_trace_summary get_alloc_trace_summary(uint32_t net_id) const;
    // Stored per-allocation records (copy; level 2 only).
    std::vector<alloc_trace_record> get_alloc_trace_records(uint32_t net_id) const;
    // Returns true (once) if the summary of the network changed since the last call; used to print only on changes.
    bool consume_alloc_trace_changes(uint32_t net_id);
    // Pool records (blocks that have at least one user) of the network.
    std::vector<pool_block_info> get_pool_blocks(uint32_t net_id) const;
#endif  // GPU_DEBUG_CONFIG

    void dump(uint32_t id, int64_t iter, std::string dump_dir_path = "");
    size_t get_total_mem_pool_size(allocation_type type);

private:
    void dump_to_screen(uint32_t id, int64_t iter);
    void dump_to_file(uint32_t id, int64_t iter, std::string dump_dir_path);

#ifdef GPU_DEBUG_CONFIG
    std::vector<memory_record> _no_reusable_mems;

    size_t total_mem_size_non_padded_pool        = 0;
    size_t total_mem_size_padded_pool            = 0;
    size_t total_mem_size_no_reusable            = 0;
    size_t mem_size_non_padded_pool_host         = 0;
    size_t mem_size_padded_pool_host             = 0;
    size_t mem_size_no_reusable_host             = 0;
#endif
};

}  // namespace cldnn
