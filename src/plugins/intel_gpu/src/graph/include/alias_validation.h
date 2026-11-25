// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Phase 0.5b: shadow validation of the alias analysis (report only, no influence on allocation) and the war_safe oracle.
// Design: /home/ceciliapeng/repos/reports/memman/layer1/13_p05b_design.zh-CN.md (r1 + Appendix B r2).
//   * war_safe(aa, axis, query): is it safe to let writer W overwrite (in place) the tensor it reads through dependency `dep_index`?
//     Pure function of the alias engine and the execution axis. Readers are NOT taken from alias_class::readers: a "version set" is
//     built from the overwritten tensor along non-in-place alias edges, and its users are enumerated with
//     alias_analysis::for_each_tensor_user (the single implementation shared with P2).
//   * validate_alias_rules(program, aa): compile-time rules R1 (role matrix), R2, R5b, R5c, V1 and the war report of every in-place writer.
// Nothing here writes node flags, fallback / domain bits or restrictions.

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "alias_analysis.h"
#include "exec_axis.h"

namespace cldnn {

enum class war_verdict : uint8_t { Safe, Unsafe, Unknown };

enum war_reason : uint32_t {
    // Unsafe evidence
    wr_late_reader = 1u << 0,
    wr_self_read_other_port = 1u << 1,
    wr_visible_version = 1u << 2,
    wr_external_root = 1u << 3,
    wr_persistent_unregistered = 1u << 4,
    wr_via_upstream_inplace = 1u << 5,   // modifier: the external / persistent root is reached through an upstream in-place edge
    // Unknown evidence
    wr_oooq = 1u << 8,
    wr_no_axis = 1u << 9,
    wr_body_or_internal = 1u << 10,
    wr_cpu_writer = 1u << 11,
    wr_cpu_member = 1u << 12,
    wr_cpu_reader = 1u << 13,
    wr_shape_of_member = 1u << 14,
    wr_shape_of_subgraph_reader = 1u << 15,
    wr_opaque = 1u << 16,
    wr_reader_off_axis = 1u << 17,
    wr_ctrl_flow_reader = 1u << 18,
    wr_runtime_guarded = 1u << 19,
    wr_bad_query = 1u << 20,             // dependency index / tensor not part of the analysed graph
    wr_writer_off_axis = 1u << 21,       // the writer has no position on a valid axis: no order check is possible
    // Safe annotation
    wr_no_other_reader = 1u << 24,
};

constexpr uint32_t war_unsafe_mask = wr_late_reader | wr_self_read_other_port | wr_visible_version | wr_external_root | wr_persistent_unregistered;
constexpr uint32_t war_unknown_mask = 0x003fff00u;   // bits 8..21

struct war_query {
    const program_node* writer = nullptr;
    int32_t dep_index = 0;
    uint16_t out_port = 0;
};

struct war_witness {
    const program_node* reader = nullptr;
    int64_t reader_pos = -1;
    const program_node* via_node = nullptr;   // tensor of the version set through which the reader sees the overwritten bytes
    uint16_t via_port = 0;
};

struct war_result {
    war_verdict verdict = war_verdict::Unknown;
    uint32_t reasons = 0;
    int64_t writer_pos = -1;
    std::vector<war_witness> witnesses;   // at most 4
};

war_result war_safe(const alias_analysis& aa, const exec_axis& axis, const war_query& q);

// In-place class edges (kind MUST_INPLACE / MAY_INPLACE: onednn sum / mul, kv_cache, stateless_kv, lora, hypothetical) and persistent copy
// writers (assign).
std::vector<war_query> enumerate_inplace_writers(const alias_analysis& aa);

enum class rule_id : uint8_t { R1, R2, R5b, R5c, V0, V1, V2, V3 };
enum class severity : uint8_t { defect, war, info };

struct rule_hit {
    rule_id id = rule_id::R1;
    severity sev = severity::info;
    const program_node* a = nullptr;
    const program_node* b = nullptr;
    uint16_t port = 0;
    std::string cell;
};

struct alias_validation_report {
    bool ran = false;
    std::string skipped_reason;   // "internal" | "body" | "no_analysis" | "exception: ..."
    exec_axis axis;
    struct war_entry {
        war_query q;
        alias_kind kind = alias_kind::NONE;
        edge_origin origin = edge_origin::view;
        war_result r;
    };
    std::vector<war_entry> war;
    std::vector<rule_hit> hits;
    std::map<std::pair<rule_id, severity>, size_t> counts;

    size_t count(rule_id id, severity sev) const {
        auto it = counts.find({id, sev});
        return it == counts.end() ? 0 : it->second;
    }
    size_t defect_count() const;
    size_t war_count(war_verdict v) const;
    std::string summary() const;
};

// Never throws on a well-formed program; callers still wrap it (B1).
alias_validation_report validate_alias_rules(const program& p, const alias_analysis& aa);

// ------------------------------------------------------------------------------------------------------------------------------
// Phase 1a (resolve_inplace_conflicts): the planner. Design: /home/ceciliapeng/repos/reports/memman/layer1/17_p1a_design.zh-CN.md §5.
// The planner is a pure function over war_safe (the single WAR implementation) and exec_axis (the only time axis): it only maps a verdict to
// an action. It reads no kernel_impl_params and triggers no lazy oneDNN initialisation. The executor (refreeze_onednn_inplace_post_ops) applies it.
// ------------------------------------------------------------------------------------------------------------------------------

// The in-place class edge (kind MUST_INPLACE / MAY_INPLACE) that the war query `q` refers to, nullptr if there is none.
const alias_edge* inplace_edge_of(const alias_analysis& aa, const war_query& q);

enum class war_downgrade_reason : uint8_t { none, unsafe, node_unknown, no_axis, analysis_exception };
inline constexpr uint32_t war_program_level_mask = wr_oooq | wr_body_or_internal | wr_no_axis;

struct inplace_conflict_action {
    const program_node* writer = nullptr;
    int32_t mem_dep = -1;   // == q.dep_index == fused_op.outer_dep_start_idx (design H11)
    edge_origin origin = edge_origin::onednn_sum;
    war_downgrade_reason why = war_downgrade_reason::none;   // none = keep
    war_result r;
};

struct inplace_conflict_plan {
    std::vector<inplace_conflict_action> actions;   // sorted by (exec position, unique id, mem_dep); keep entries included
    size_t via_upstream_after_downgrade = 0;        // P-3 counter (upper bound, audit N5)
};

// Pure. Preconditions (caller): in_order, not internal / body, axis.valid(). `include_hypothetical` is for tests: edges with origin
// hypothetical are treated like a sum.
inplace_conflict_plan plan_inplace_conflict_resolution(const alias_analysis& aa, const exec_axis& axis, bool include_hypothetical = false);

// D6: the surviving oneDNN sum / mul_inplace writers whose verdict is not Safe (never throws); origin hypothetical only when requested.
std::vector<inplace_conflict_action> find_unsafe_surviving_inplace_writers(const alias_analysis& aa, const exec_axis& axis,
                                                                          bool include_hypothetical = false);

// Log / counter record of one resolve_inplace_conflicts run. Not serialized; entries hold node ids, never node pointers.
struct inplace_conflict_report {
    std::string skipped_reason;   // "" | "disabled" | "no_candidates" | "internal" | "body" | "oooq"
    size_t candidates = 0, kept = 0, program_level_skipped = 0, via_upstream_after_downgrade = 0;
    struct entry {
        std::string writer_id;
        int32_t mem_dep = -1;
        edge_origin origin = edge_origin::onednn_sum;
        war_downgrade_reason why = war_downgrade_reason::none;
        uint32_t reasons = 0;
    };
    std::vector<entry> entries;                  // downgraded items only
    std::map<uint32_t, size_t> downgrade_bits;   // one counter per war_reason bit
    std::string summary() const;
};

const char* to_string(war_downgrade_reason r);
const char* to_string(edge_origin o);

// Test hook (audit C3): the next resolve_inplace_conflicts run throws once inside its analysis stage. One-shot: the flag is consumed with an
// atomic exchange, so it cannot leak into a later program. Only for verifying the Release fallback.
void set_inplace_conflict_fault_for_test(bool inject);
bool take_inplace_conflict_fault_for_test();

const char* to_string(rule_id id);
const char* to_string(war_verdict v);

}  // namespace cldnn
