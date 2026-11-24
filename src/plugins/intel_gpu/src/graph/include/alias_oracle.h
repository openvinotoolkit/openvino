// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Phase 0.5b: runtime oracle of the alias analysis. Sampled once per network, at the end of the first execute() (after the stream flush,
// before reset_flags), it compares the buffers the primitives actually hold with the alias analysis. Report only, never changes anything.
//   V0: the execution axis equals network::_exec_positions.
//   V2 (structural part): two output tensors of different alias classes that share an allocation must be explained by the memory pool
//       (the actual allocators of both are users of the pool record). Lifetime disjointness (V2b) belongs to P2.
//   V3: every ALWAYS alias edge: the member holds (part of) the allocation of its provider.
// The oracle never maps, locks or synchronizes any memory. Pointers are only taken from USM allocations (never from cl_mem).
// Design: 13_p05b_design.zh-CN.md section 3.4 + Appendix B (B1, B3).

#include <cstdint>
#include <string>
#include <vector>

#include "alias_validation.h"
#include "intel_gpu/primitives/primitive.hpp"

namespace cldnn {

struct oracle_hit {
    rule_id id = rule_id::V0;
    severity sev = severity::defect;
    primitive_id a;
    primitive_id b;
    uint16_t port_a = 0;
    uint16_t port_b = 0;
    std::string detail;
};

struct alias_oracle_report {
    bool ran = false;
    std::string skipped_reason;   // "internal" | "body" | "no_analysis" | "exception: ..."
    exec_axis::status axis_status = exec_axis::status::not_in_order;
    size_t v0_checked = 0;
    size_t v2_pairs = 0;          // overlapping tensor pairs of different classes
    size_t v2_by_class = 0;       // overlapping tensors of the same class (explained by the class)
    size_t v2_by_pool = 0;        // pairs explained by a pool record
    size_t v2_unverifiable = 0;   // tensors that cannot be placed (cl_mem / image / empty)
    size_t v2_excluded = 0;       // persistent, external, skipped-for-other-reasons tensors
    size_t v3_checked = 0;
    size_t v3_unverifiable = 0;
    std::vector<oracle_hit> hits;

    size_t count(rule_id id, severity sev) const {
        size_t n = 0;
        for (const auto& h : hits) {
            n += (h.id == id && h.sev == sev) ? 1 : 0;
        }
        return n;
    }
    size_t defect_count() const {
        size_t n = 0;
        for (const auto& h : hits) {
            n += h.sev == severity::defect ? 1 : 0;
        }
        return n;
    }
};

}  // namespace cldnn
