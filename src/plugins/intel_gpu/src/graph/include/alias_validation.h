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
    // Safe annotation
    wr_no_other_reader = 1u << 24,
};

constexpr uint32_t war_unsafe_mask = wr_late_reader | wr_self_read_other_port | wr_visible_version | wr_external_root | wr_persistent_unregistered;
constexpr uint32_t war_unknown_mask = 0x001fff00u;   // bits 8..20

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

const char* to_string(rule_id id);
const char* to_string(war_verdict v);

}  // namespace cldnn
