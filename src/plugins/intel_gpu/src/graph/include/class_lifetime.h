// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// P2-a (G+): class lifetime engine. The single implementation of "when does an alias class live on the execution axis" and "may the
// class be judged by its interval alone". It is a pure function of (graph, alias_analysis, exec_axis):
//   * classes and readers come from alias_analysis (no owner / view / slot detection here),
//   * positions come from exec_axis (F9 execution order; there is no other time axis),
//   * nothing here writes a restriction, a node flag or an allocation.
// Design: /home/ceciliapeng/repos/reports/memman/layer1/19_p2_gplus_design.zh-CN.md (r2) sections 3.1 and 7 (S1).
//
// Lifetime of a class K (closed interval on the execution axis):
//   birth = minimum position of the members, death = maximum position of members and readers (shape_of readers are skipped by
//   alias_analysis). death is infinite for classes with a user visible output. A class whose member or reader has no position (data /
//   constant members that are not executable are skipped) has an unknown interval and is ineligible.
//
// The engine borrows the alias_analysis it was built from: that object must outlive the engine.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "alias_analysis.h"
#include "exec_axis.h"

namespace cldnn {

struct program;
struct program_node;

struct class_interval {
    uint32_t birth = 0;
    uint32_t death = 0;
    bool death_inf = false;
    bool known = false;
};

// Reasons a class (or a node) cannot be judged by interval alone. 0 == eligible.
enum lifetime_ineligible : uint32_t {
    li_no_position = 1u << 0,      // a member / reader / the node has no execution position (or the axis is invalid)
    li_no_class = 1u << 1,         // an output port of the node is not part of any class
    li_fallback = 1u << 2,         // alias_analysis reports a fallback reason (CPU / maybe-CPU / shape_of / role conflict / opaque / ...)
    li_external = 1u << 3,         // dm_external_root
    li_persistent = 1u << 4,       // dm_persistent
    li_cpu_reader = 1u << 5,       // a reader is (maybe) CPU; redundant with li_fallback, kept as defence in depth
    li_shape_of_reader = 1u << 6,  // a shape_of (or shape_of subgraph) reader; redundant with li_fallback, kept as defence in depth
    li_ambiguous_uid = 1u << 7,    // a member node shares its unique id with another tensor
    li_unknown_interval = 1u << 8,
    li_unmodeled_alias = 1u << 9,  // C5: selective_ssm output classes and the whole class of its input 5 (alias without an edge, F27)
};

// Program level gate of P2-a (design 19, 3.2): true iff enable_inorder_lifetime_restrictions is set AND the queue is in_order AND the engine is
// the OCL backend (ZE execution order is not established, audit 5.1) AND the program is neither internal nor a body AND the execution axis is
// valid (F9 normalized). `reason` names the first failed condition (a static string). Cheap checks run first; the axis is only read last.
struct inorder_lifetime_gate_result {
    bool open = false;
    const char* reason = "switch_off";
};
inorder_lifetime_gate_result evaluate_inorder_lifetime_gate(const program& p, const exec_axis& axis);

// Test-only mutation hooks (L32): each switch disables ONE mechanism of G+ so that a test can prove the mechanism is load bearing. The hooks are
// thread local and scoped by an RAII guard around the compilation of ONE program (no process-wide one-shot flag).
struct inorder_lifetime_mutations {
    bool no_infinity = false;              // user-output classes die at their last reader instead of at infinity
    bool no_point_sweep = false;           // no (class x node point) restrictions
    bool no_unmodeled_alias = false;       // li_unmodeled_alias does not make a class ineligible
    bool no_cpu_exclusion = false;         // (maybe-)CPU classes are judged by interval like any other
    bool no_same_class_retention = false;  // same-class pairs are filtered out of oooq too
    // E2 / E4 experiment (NOT a design change): non-root members that allocate on their own (MAY envelope) are swept with their own member_interval
    // instead of the interval of their whole class
    bool may_members_use_member_interval = false;
};
class scoped_inorder_lifetime_mutations {
public:
    explicit scoped_inorder_lifetime_mutations(const inorder_lifetime_mutations& m);
    ~scoped_inorder_lifetime_mutations();
    scoped_inorder_lifetime_mutations(const scoped_inorder_lifetime_mutations&) = delete;
    scoped_inorder_lifetime_mutations& operator=(const scoped_inorder_lifetime_mutations&) = delete;

private:
    inorder_lifetime_mutations m_saved;
};
const inorder_lifetime_mutations& current_inorder_lifetime_mutations();

class class_lifetimes;
// One line for GPU_DEBUG logs: eligible nodes / classes and the number of classes per li_* reason (a class counts once per reason).
std::string lifetimes_census(const program& p, const alias_analysis& aa, const class_lifetimes& lt);

class class_lifetimes {
public:
    static class_lifetimes build(const program& p, const alias_analysis& aa, const exec_axis& axis);

    bool valid() const {
        return m_axis.valid();
    }
    // Interval of the whole class (the allocator's block holds every member). Unknown interval for classes of another analysis.
    const class_interval& interval_of(const alias_class& k) const;
    // Interval of one member tensor on its own: from its position to the last reader of the tensor and of everything downstream that
    // aliases it (used for members that allocate on their own, MAY envelope). death_inf if any of them is a user visible output.
    class_interval member_interval(const program_node& n, uint16_t port) const;

    // position on the execution axis (the one the intervals are built on); nullopt for non-executable nodes or an invalid axis
    std::optional<uint32_t> position(const program_node& n) const {
        return m_axis.position(n);
    }
    uint32_t class_ineligible_mask(const alias_class& k) const;
    // OR over every output port class plus node level bits (no class, no position, ambiguous uid).
    uint32_t node_ineligible_mask(const program_node& n) const;
    bool eligible(const program_node& n) const {
        return node_ineligible_mask(n) == 0;
    }
    // true iff no output-port class of a equals an output-port class of b (C1: same-class pairs keep the oooq restriction)
    bool classes_disjoint(const program_node& a, const program_node& b) const;
    // Members with role allocator / may_allocate_member that may use the memory pool; each node once.
    const std::vector<const program_node*>& pool_allocators(const alias_class& k) const;

private:
    size_t index_of(const alias_class& k) const;   // SIZE_MAX if k does not belong to m_aa

    const alias_analysis* m_aa = nullptr;
    exec_axis m_axis;
    std::vector<class_interval> m_intervals;
    std::vector<uint32_t> m_masks;
    std::vector<char> m_cpu_only;   // the only fallback reasons of the class are (maybe-)CPU ones; used by the L32 hook only
    std::vector<std::vector<const program_node*>> m_allocators;
};

}  // namespace cldnn
