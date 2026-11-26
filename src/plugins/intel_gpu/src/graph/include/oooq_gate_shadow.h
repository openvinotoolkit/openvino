// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// P2-shadow: "oooq gate" shadow experiment. OBSERVATION ONLY: it never writes a restriction, a node flag or an allocation.
//
// Question answered: if oooq_memory_dependencies were gated to out_of_order queues only, which node pairs would stop being restricted (the
// set D = S1 \ S0, S0 = restrictions after basic + skipped_branch, S1 = after oooq) and is any of them *dangerous*, i.e. do the lifetimes
// of the two nodes overlap on the in-order execution axis?
//
// The lifetime oracle uses only the existing engines: alias classes (alias_analysis) and the execution axis (exec_axis). No interval or WAR
// logic is implemented here other than the trivial "closed intervals intersect" test on the class lifetimes
// [min position of the class members, max position of members and readers] (inf for classes with a user visible output).
//
// Switch: environment variable OV_GPU_OOOQ_GATE_SHADOW=1 (ENABLE_DEBUG_CAPS builds only) or the unit-test hook set_enabled_for_test().
// Disabled = one boolean check in program::prepare_memory_dependencies, nothing is built.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cldnn {

struct program;
struct program_node;
class alias_analysis;

struct oooq_gate_shadow_report {
    // "in_order" when the experiment ran, otherwise "skipped:<reason>" (oooq | internal | body | no_analysis | no_axis)
    std::string mode;
    bool ran = false;

    // ---- summary fields (the GPU_DEBUG line) ----
    size_t pairs_total = 0;      // unordered pairs of distinct nodes that both may_use_mempool(); danger / gain are judged only for pairs of pool users
    size_t s0 = 0;               // pairs restricted (in either direction) after basic + skipped_branch
    size_t s1 = 0;               // ... after oooq
    size_t d = 0;                // S1 \ S0
    size_t danger = 0;           // lifetimes overlap although the pair is absent from S0 (either direction)
    size_t danger_in_d = 0;      // subset of danger that oooq protected (pair in D)
    size_t unverifiable = 0;     // pairs the oracle could not judge (unknown lifetime, ambiguous uid)
    size_t po_exec_mismatch = 0; // 1 when the execution axis is not the filtered processing order (not normalized), 0 otherwise

    // ---- details ----
    size_t conservative_pairs = 0;   // a node of the pair is CPU / maybe-CPU / shape_of / persistent / in a class with a fallback reason
    size_t non_pool_pairs = 0;       // a node of the pair never allocates from the pool (input/data, can_share_buffer(false), cbo, output): the pair is not a pool hazard
    size_t d_non_pool = 0;           // D pairs of that kind
    size_t same_class_pairs = 0;     // the two nodes share an alias class by design
    size_t danger_s1 = 0;            // oracle self-check: overlap although the pair is absent from S1 (LEGACY would be wrong or the oracle is imprecise)
    size_t danger_asym = 0;          // overlap, restricted in exactly one direction in S0 (the pool consults the requester's own set only)
    size_t s0_directed = 0, s1_directed = 0;
    size_t d_assessed = 0;           // D pairs the oracle could judge
    size_t d_unassessed = 0;         // D pairs that are conservative / unverifiable / same class
    size_t d_gain_pairs = 0;         // assessed D pairs with disjoint lifetimes: the real relaxation
    size_t d_gain_bytes = 0;         // sum over those of min(bytes(A), bytes(B)); dynamic layouts count 0
    size_t d_potential_bytes = 0;    // same, over all D pairs
    size_t d_bytes_unknown_pairs = 0;
    // D pairs by the oooq rule that produced them (classification of the structure only, never used to decide anything)
    size_t d_unreachable = 0;        // no DAG path between the two nodes (Pattern A conservatism)
    size_t d_fork = 0;               // path A->B exists but a user of A is unrelated to B
    size_t d_suspect = 0;            // the rest: produced by the suspect_nodes section
    size_t d_suspect_danger = 0;
    size_t d_with_input = 0;         // D pairs with an input_layout end
    size_t d_input_output_writer = 0;// D pairs with an input_layout end and a user-visible output end (may_alias sensitive)
    size_t d_first_node = 0;         // D pairs that contain the first non-constant node of the processing order
    size_t ambiguous_uid_nodes = 0;
    size_t classless_nodes = 0;
    size_t conservative_nodes = 0;

    struct pair_info {
        std::string a, b, type_a, type_b, why;
        bool in_s1 = false;
    };
    std::vector<pair_info> danger_pairs;      // capped at kMaxListed
    std::vector<pair_info> danger_in_d_pairs; // capped
    std::vector<pair_info> danger_asym_pairs; // capped
    std::vector<pair_info> danger_s1_pairs;   // capped
    std::vector<std::pair<std::string, std::string>> d_pair_ids;   // all D pairs (ids), capped at kMaxListed
    static constexpr size_t kMaxListed = 20000;

    bool in_d(const std::string& a, const std::string& b) const;
    std::string summary() const;
    // Multi-line breakdown of the dangerous pairs by (type, type); logged at most once per program when OV_GPU_OOOQ_GATE_SHADOW=2.
    std::string details() const;
};

// Directed restriction rows after basic + skipped_branch. Row i = restrictions of the node at processing-order index i, bit j = node j.
struct oooq_gate_shadow_snapshot {
    std::vector<const program_node*> nodes;                  // processing order, all nodes
    std::unordered_map<const program_node*, size_t> index;
    std::unordered_map<uint32_t, size_t> uid_to_index;       // only unambiguous uids
    std::unordered_set<size_t> ambiguous;                    // node indices whose uid is shared
    size_t words = 0;
    std::vector<std::vector<uint64_t>> rows;

    bool test(size_t from, size_t to) const {
        return (rows[from][to >> 6] >> (to & 63)) & 1ULL;
    }
    void set(size_t from, size_t to) {
        rows[from][to >> 6] |= (1ULL << (to & 63));
    }
    void clear(size_t from, size_t to) {
        rows[from][to >> 6] &= ~(1ULL << (to & 63));
    }
};


class oooq_gate_shadow {
public:
    using snapshot = oooq_gate_shadow_snapshot;

    static bool enabled();
    // -1: follow the environment, 0: force off, 1: force on. Test hook.
    static void set_enabled_for_test(int mode);
    // Test hook (T12): pairs of node ids removed from S0 (both directions) after the snapshot, so that the oracle has to report them.
    static void set_s0_erase_for_test(std::vector<std::pair<std::string, std::string>> pairs);

    // Called right before oooq_memory_dependencies. Returns nullptr when the program is out of scope (the skip is recorded).
    static std::unique_ptr<snapshot> take_snapshot(const program& p);
    // Called right after oooq and after the alias classes are built. Never throws.
    static void finish(const program& p, std::unique_ptr<snapshot> s0, const alias_analysis* aa) noexcept;

    // Most recent report of this program (nullptr if none). Reports are kept per program address until the next run on that address.
    static std::shared_ptr<const oooq_gate_shadow_report> report_for(const program& p);
    // Test hook: forget every stored report (program addresses are reused between tests).
    static void clear_reports_for_test();

    // The pure computation (also used directly by tests): s1 is read from the nodes' current restriction sets.
    static oooq_gate_shadow_report compute(const program& p, const snapshot& s0, const alias_analysis& aa);

    ~oooq_gate_shadow() = delete;
};

}  // namespace cldnn
