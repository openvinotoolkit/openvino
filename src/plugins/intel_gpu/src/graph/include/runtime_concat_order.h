// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// F9: predicates and checks shared by the order_runtime_concat_predecessors pass and network::build_exec_order.
// The two sides must never diverge, hence everything lives in this single internal header.

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "concatenation_inst.h"
#include "data_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "mutable_data_inst.h"
#include "program_node.h"

namespace cldnn {
namespace runtime_concat_order {

// A concat whose in-place fusion is decided at runtime.
inline bool is_runtime_optimized_concat(const program_node* node) {
    return node->is_dynamic() && node->is_type<concatenation>() && node->can_be_optimized();
}

// Nodes that appear in network::_exec_order (data and dependency-free mutable_data are skipped).
inline bool is_executable_node(const program_node* node) {
    return !node->is_type<data>() && (!node->is_type<mutable_data>() || !node->get_dependencies().empty());
}

// A predecessor that network::build_exec_order postpones until right before its runtime-optimized concat user.
inline bool is_allowed_pred(const program_node* node) {
    return is_executable_node(node) && node->get_users().size() == 1 && is_runtime_optimized_concat(node->get_users().front());
}

// True when the runtime in-place concat optimization of C must be cancelled because re-issuing its predecessors right before C
// could execute a predecessor twice, drop it, or mis-order it (r2-4). Pure query, does not mutate the graph.
inline bool must_cancel_runtime_concat(const program_node& c) {
    std::unordered_set<const program_node*> seen;
    for (const auto& dep : c.get_dependencies()) {
        const program_node* d = dep.first;
        if (d->is_type<data>()) {
            continue;
        }
        if (!is_allowed_pred(d) || d->get_users().front() != &c) {
            return true;
        }
        if (is_runtime_optimized_concat(d)) {  // cascade: the inner concat would lose its own predecessors
            return true;
        }
        if (!seen.insert(d).second) {  // duplicated dependency would be re-issued twice
            return true;
        }
    }
    return false;
}

// Runtime-callable check of the F9 invariants of network::_exec_order against the processing order.
//   A1: no primitive id appears twice in exec_ids.
//   A2: when the processing order is normalized (every runtime-optimized concat is directly preceded, in dependency order, by its
//       non-data dependencies, filtered to executable nodes), exec_ids must equal the filtered processing order.
// Returns std::nullopt when everything holds (or A2 is not applicable). Otherwise returns a description of the first violation.
struct check_result {
    bool normalized = false;        // A2 was applicable
    std::optional<size_t> index;    // first mismatching index (A2) or index of the first duplicate (A1)
    std::string message;            // empty when ok
    bool ok() const { return message.empty(); }
};

inline check_result check_exec_order(const program::nodes_ordering& po, const std::vector<primitive_id>& exec_ids) {
    check_result res;
    // A1
    {
        std::unordered_set<primitive_id> seen;
        for (size_t i = 0; i < exec_ids.size(); ++i) {
            if (!seen.insert(exec_ids[i]).second) {
                res.index = i;
                res.message = "A1: duplicate primitive '" + exec_ids[i] + "' in exec order at index " + std::to_string(i);
                return res;
            }
        }
    }
    // Filtered processing order.
    std::vector<const program_node*> filtered;
    for (const auto& n : po) {
        if (is_executable_node(n)) {
            filtered.push_back(n);
        }
    }
    // Normalization (C4).
    res.normalized = true;
    for (size_t i = 0; i < filtered.size() && res.normalized; ++i) {
        const auto* c = filtered[i];
        if (!is_runtime_optimized_concat(c)) {
            continue;
        }
        std::vector<const program_node*> preds;
        for (const auto& dep : c->get_dependencies()) {
            if (!dep.first->is_type<data>()) {
                preds.push_back(dep.first);
            }
        }
        if (must_cancel_runtime_concat(*c) || preds.size() > i) {
            res.normalized = false;
            break;
        }
        for (size_t k = 0; k < preds.size(); ++k) {
            if (filtered[i - preds.size() + k] != preds[k]) {
                res.normalized = false;
                break;
            }
        }
    }
    if (!res.normalized) {
        return res;
    }
    // A2
    const size_t n = std::min(filtered.size(), exec_ids.size());
    for (size_t i = 0; i < n; ++i) {
        if (filtered[i]->id() != exec_ids[i]) {
            res.index = i;
            res.message = "A2: exec order mismatch at index " + std::to_string(i) + ": exec='" + exec_ids[i] + "' po='" + filtered[i]->id() + "'";
            return res;
        }
    }
    if (filtered.size() != exec_ids.size()) {
        res.index = n;
        res.message = "A2: exec order size " + std::to_string(exec_ids.size()) + " != filtered processing order size " +
                      std::to_string(filtered.size());
    }
    return res;
}

}  // namespace runtime_concat_order
}  // namespace cldnn
