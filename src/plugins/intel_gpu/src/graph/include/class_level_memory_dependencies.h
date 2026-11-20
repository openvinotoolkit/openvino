// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// P1-B-S: class-level memory pool identity for static in-place Concat holders (design 11 sections 3 and 7, handoff r4/r5).
//
// The alias_analysis engine partitions the output tensors into alias classes. For an admitted class the holder (the in-place concat) is the
// only allocator: it becomes the single pool user of the class under its own uid ("class root"), the slot predecessors allocate nothing,
// and the restriction sets are rewritten so that the root conflicts with everything any member conflicts with (both directions).
// Members keep their own uid (their internal buffers still use it) and gain the root uid.
//
// Layers:
//   L1 pool_admission_reject_mask(): which classes may be promoted (0 == admitted).
//   L2/L3 materialize_class_restrictions(): pure function over a restriction_store, directly testable.
//   class_level_memory_dependencies: the pass that wires both to a program (runs at the end of prepare_memory_dependencies).

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "pass_manager.h"

namespace cldnn {

namespace class_level {

enum pool_reject : uint32_t {
    rj_program_level = 1u << 0,      // L1-0: switch off / out_of_order / internal / body / memory pool disabled
    rj_placement = 1u << 1,          // L1-1: class placement is not POOL (fallback or domain reasons)
    rj_root_kind = 1u << 2,          // L1-2: root is not an in-place concat holder on port 0
    rj_root_not_shareable = 1u << 3, // L1-3: root can_share_buffer()==false (defect B marking) or is an output
    rj_member_kind = 1u << 4,        // L1-4: a member is not a plain slot / view of the class
    rj_dynamic = 1u << 5,            // L1-5: dynamic or runtime skippable member, dynamic reader
    rj_no_mempool = 1u << 6,         // L1-6: member does not use the memory pool
    rj_multi_output = 1u << 7,       // L1-7: multi output member or member on a port != 0
    rj_reused_eltw = 1u << 8,        // L1-8: oneDNN in-place writer in the class or writing into a class tensor
    rj_ambiguous_uid = 1u << 9,      // L1-9: uid shared by several nodes
    rj_external_root = 1u << 10,     // L1-10: member is an external buffer root
    rj_shape_of_reader = 1u << 11,   // L1-11: reader in a shape_of subgraph
    rj_duplicate_root_dep = 1u << 12, // L1-12: the same producer node feeds the root on several edges (any port); one slot cannot hold both
};

std::string to_string(uint32_t reject_mask);

// L1: 0 if the class is admitted. uid_is_ambiguous is injectable (alias_analysis::uid_is_ambiguous in production).
uint32_t pool_admission_reject_mask(const alias_class& k, const program& p, const std::function<bool(uint32_t)>& uid_is_ambiguous);

// Restriction sets indexed by node position. Implemented over program nodes by the pass and over plain vectors by unit tests.
class restriction_store {
public:
    virtual ~restriction_store() = default;
    virtual size_t size() const = 0;
    virtual uint32_t uid(size_t node) const = 0;
    // sorted ascending, no duplicates. The reference is invalidated by insert().
    virtual const std::vector<uint32_t>& values(size_t node) const = 0;
    virtual void insert(size_t node, uint32_t uid) = 0;
};

struct simple_restriction_store : restriction_store {
    std::vector<uint32_t> uids;
    std::vector<std::vector<uint32_t>> sets;
    size_t size() const override {
        return uids.size();
    }
    uint32_t uid(size_t node) const override {
        return uids[node];
    }
    const std::vector<uint32_t>& values(size_t node) const override {
        return sets[node];
    }
    void insert(size_t node, uint32_t id) override;
    bool contains(size_t node, uint32_t id) const;
};

struct promoted_class {
    size_t root = 0;                // node index of the holder
    std::vector<size_t> members;    // node indices, includes the root
};

// L2 + L3: (2) R(root) |= union R0(m); (3') X with a member uid in R1(X) gets the root uid and the root gets X's uid; (4) symmetrise R(root).
// The input must be the restriction state before this function (R0).
void materialize_class_restrictions(restriction_store& store, const std::vector<promoted_class>& classes);

std::vector<std::vector<uint32_t>> snapshot_restrictions(const restriction_store& store);

// Debug verifier of I-B1, I-B1-sym and I-B3 for a store after materialize_class_restrictions() given the R0 snapshot. Empty string == ok.
std::string check_class_invariants(const restriction_store& after, const std::vector<std::vector<uint32_t>>& r0, const std::vector<promoted_class>& classes);

// B4' (structured, replaces moving the legacy B4 assertion): no in-place writer, its reused dependency or the buffer owner of that dependency
// may carry a class route (POOL_AS_ROOT / ZERO_ALLOC_MEMBER); every promoted root is shareable and no promoted class has an in-place member.
// Evaluated after the flags were set. Empty string == ok.
std::string check_b4_prime(const program& p);

}  // namespace class_level

class class_level_memory_dependencies : public base_pass {
public:
    class_level_memory_dependencies() : base_pass("class_level_memory_dependencies") {}
    void run(program& p) override;
};

}  // namespace cldnn
