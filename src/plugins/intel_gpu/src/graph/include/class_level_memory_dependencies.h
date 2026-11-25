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
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    rj_holder_padding = 1u << 13,    // L1-13 (P1-B-D): dynamic holder or slot predecessor carries padding or a blocked format
    rj_mixed_dynamic = 1u << 14,     // L1-5d (P1-B-D): static member or reader in a dynamic holder class
    // P1-C (sum classes, design 18 section 3.5)
    rj_sum_provenance = 1u << 15,    // L1S-3p: the basic_memory_dependencies marks of the writers do not form a single owner chain rooted at the class root
    rj_sum_writer = 1u << 16,        // L1S-8: a writer is not a plain static frozen in-place sum writer (flags / layout / frozen entry)
    rj_sum_root_padded = 1u << 17,   // L1S-12: the root layout carries padding and kAdmitPaddedSumRoot is false
};

std::string to_string(uint32_t reject_mask);

// L1: 0 if the class is admitted. uid_is_ambiguous is injectable (alias_analysis::uid_is_ambiguous in production).
// deferred_sum_marks (P1-C-1, L1-3): nodes basic_memory_dependencies did not mark non-shareable because the sum root pool defers the mark of
// the chain root / owner of an in-place sum writer. Such a root is non-shareable exactly like with the switch off (concat roots keep the
// P1-B-S behaviour). nullptr == nothing deferred.
uint32_t pool_admission_reject_mask(const alias_class& k, const program& p, const std::function<bool(uint32_t)>& uid_is_ambiguous,
                                    const std::unordered_set<const program_node*>* deferred_sum_marks = nullptr);

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

// P1-C-1 L3 step (5) (design 18 section 3.6, L3-V): the buffer owners of the views in R(root). Given the uid u of a node, appends the uid of the owner
// of every output port k of u (resolve_buffer_owner(node, settled_only, false, k).owner) for which that owner is not the node itself. A uid shared by
// several nodes yields the union. Production: built over the program nodes by class_level_memory_dependencies::run.
using view_owner_fn = std::function<void(uint32_t u, std::vector<uint32_t>& owner_uids)>;

// The production callback: resolve_buffer_owner over the nodes of a program (memoised per uid; the settled flags must not change while it lives).
// `nodes` must outlive the callback only for the duration of this call (the callback keeps its own copy of the pointers).
view_owner_fn make_program_view_owner_fn(const std::vector<program_node*>& nodes);

// L2 + L3: (2) R(root) |= union R0(m); (3') X with a member uid in R1(X) gets the root uid and the root gets X's uid; (4) symmetrise R(root);
// (5) only with view_owners != nullptr: for every promoted root r and every u in R(r) (snapshot), every owner o of u that is not a node of r's own class
// enters R(r), and every node with uid o gets uid(r) (single pass: o is a non-view node or a concat holder, so there is no cascade).
// The input must be the restriction state before this function (R0). Returns the number of (r, o) closures that added something in step (5).
size_t materialize_class_restrictions(restriction_store& store, const std::vector<promoted_class>& classes, const view_owner_fn* view_owners = nullptr);

// I-B1-V (Debug verifier of step (5)): for every promoted root r, every u in R(r) and every owner o of u outside r's class: uid(o) in R(r) and every
// node with uid o carries uid(r). Empty string == ok.
std::string check_view_owner_closure(const restriction_store& after, const std::vector<promoted_class>& classes, const view_owner_fn& view_owners);

std::vector<std::vector<uint32_t>> snapshot_restrictions(const restriction_store& store);

// Debug verifier of I-B1, I-B1-sym and I-B3 for a store after materialize_class_restrictions() given the R0 snapshot. Empty string == ok.
std::string check_class_invariants(const restriction_store& after, const std::vector<std::vector<uint32_t>>& r0, const std::vector<promoted_class>& classes);

// B4' (structured, replaces moving the legacy B4 assertion): no in-place writer, its reused dependency or the buffer owner of that dependency
// may carry a class route (POOL_AS_ROOT / ZERO_ALLOC_MEMBER); every promoted root is shareable and no promoted class has an in-place member.
// P1-C-1: an in-place sum writer may have a POOL_AS_ROOT dependency / owner only if that node is the promoted root of the sum class of the writer;
// MUST_INPLACE members are only allowed in a promoted sum class (never in a concat class), MAY_INPLACE members never.
// Evaluated after the flags were set. Empty string == ok.
std::string check_b4_prime(const program& p);

// =================================================================================================================================
// P1-C-0 (design 18 sections 3.3, 3.5, 3.8): census of the oneDNN in-place sum classes. Read-only: nothing here writes a node flag, a
// restriction set or a pool field, and the pass runs it only for diagnostics (enable_sum_root_pool).
// =================================================================================================================================

// Program-level predicate (single implementation for the census now and the promotion in P1-C-1). Optional `why` receives the first reason
// that makes it false: switch_off / 1a_off / memory_pool_off / oooq / internal / body / no_report / 1a_skipped:<reason>.
bool sum_root_pool_active(const program& p, std::string* why = nullptr);

// Design 18 section 3.5 NB-4: padded sum roots are admitted only if C-16b (a pool block whose padding was dirtied) passes. P1-C-1: C-16b could not be
// constructed (the oneDNN sum into a padded root never formed in the unit-test pipeline), so by the rule "not passing or not constructible -> false" the
// preset true was flipped: a root with padding is rejected (rj_sum_root_padded) until a real graph proves otherwise.
inline constexpr bool kAdmitPaddedSumRoot = false;

// What basic_memory_dependencies marked for every surviving static sum writer W: (legacy_root, owner), and the inverse index.
struct sum_root_marks_index {
    std::unordered_map<const program_node*, std::pair<const program_node*, const program_node*>> of_writer;   // W -> (legacy_root, owner)
    std::unordered_map<const program_node*, std::vector<const program_node*>> writers_marking;               // n -> writers that mark n (legacy_root or owner)
};
// marked_nodes (P1-C-1): receives the writable union of {legacy_root, owner} over all writers (what basic_memory_dependencies defers), de-duplicated.
sum_root_marks_index collect_sum_root_marks(const program& p, std::vector<program_node*>* marked_nodes = nullptr);

// P1-C-1 (design 18 section 3.4): with the sum root pool active basic_memory_dependencies marks only the sum writers; the chain roots / owners
// ("marked") are written non-shareable here, for every node except the promoted sum roots, on EVERY exit of the pass (RAII), so the flags of a
// rejected class equal the ones of the switch off. apply() is idempotent and may be called early; it never throws.
struct sum_root_backfill {
    std::vector<program_node*> marked;
    std::unordered_set<const program_node*> promoted;
    bool applied = false;
    void apply();
    sum_root_backfill() = default;
    explicit sum_root_backfill(std::vector<program_node*> m) : marked(std::move(m)) {}
    sum_root_backfill(const sum_root_backfill&) = delete;
    sum_root_backfill& operator=(const sum_root_backfill&) = delete;
    ~sum_root_backfill() {
        apply();
    }
};

// A class with at least one onednn_sum member edge (hypothetical edges count when include_hypothetical; tests only).
bool is_sum_candidate_class(const alias_class& k, const alias_analysis& aa, bool include_hypothetical = false);

// L1S rules of design 18 section 3.5 (0 == the class could be promoted). Pure: every input is a parameter (marks and uid ambiguity are
// injectable). include_hypothetical: hypothetical edges are treated like onednn_sum and the frozen-entry part of L1S-8 is skipped (hand graphs).
// L1S-3 (P1-C-1): must be evaluated in the deferred state (switch on: basic_memory_dependencies did not mark the chain root / owner), where
// can_share_buffer()==false can only come from a non-sum source; the rule is the exact `!can_share_buffer() || is_output()`.
uint32_t sum_class_reject_mask(const alias_class& k, const alias_analysis& aa, const program& p, const sum_root_marks_index& marks,
                               const std::function<bool(uint32_t)>& uid_is_ambiguous, bool include_hypothetical = false,
                               bool admit_padded_root = kAdmitPaddedSumRoot);

struct sum_class_info {
    const alias_class* cls = nullptr;   // valid while the alias_analysis lives
    std::string root_id;
    size_t members = 0, writers = 0, views = 0, readers = 0;
    uint64_t root_bytes = 0;
    bool root_padded = false;
    bool downgraded_by_1a = false;   // the root is a writer that resolve_inplace_conflicts downgraded
    bool mixed_concat = false;       // some member is a concat slot (MUST_SUBREGION)
    uint32_t mask = 0;
};

struct sum_census {
    size_t candidates = 0, would_promote = 0, rejected = 0, promoted = 0;
    uint64_t would_promote_root_bytes = 0;
    size_t downgraded_roots = 0, mixed_concat = 0, padded_roots = 0, no_edge_sum = 0;
    size_t view_owner_closures = 0;   // (r, o) closures of L3 step (5) (concat and sum classes together); filled by the pass
    std::map<uint32_t, size_t> reasons;   // reject bit -> number of classes
    std::vector<sum_class_info> classes;
    int64_t us = 0;
    std::string summary() const;
};

// Walks the alias classes of `p` and reports. Needs p.get_alias_analysis() (nullptr -> empty census). Never throws on a well formed program.
// marks: precomputed collect_sum_root_marks(p) (nullptr: computed here). The census does not know which classes were promoted (the pass sets
// `promoted` and `view_owner_closures`).
sum_census run_sum_census(const program& p, const sum_root_marks_index* marks = nullptr);

}  // namespace class_level

class class_level_memory_dependencies : public base_pass {
public:
    class_level_memory_dependencies() : base_pass("class_level_memory_dependencies") {}
    void run(program& p) override;
};

}  // namespace cldnn
