// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vk_pass.hpp"

#include "cpu_engine.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace pass {

namespace {

bool is_trivial(const ir_node& n) {
    return n.op == ir_op::parameter || n.op == ir_op::constant || n.op == ir_op::result ||
           n.op == ir_op::cache_write;  // cache_write mutates state: never dropped
}

// Nodes whose inputs are all constants with f32 payloads (quantized
// constants are skipped: folding them would defeat the in-shader dequant).
bool all_inputs_constant(const ir_graph& g, const ir_node& n) {
    if (n.inputs.empty())
        return false;
    for (const auto& in : n.inputs) {
        bool ok = false;
        for (const auto& m : g.nodes) {
            if (m.id == in) {
                ok = m.op == ir_op::constant && g.constant_data.count(in) != 0 &&
                     g.quant_constants.count(in) == 0;
                break;
            }
        }
        if (!ok)
            return false;
    }
    return true;
}

}  // namespace

ir_graph dce(const ir_graph& g) {
    // Backward reachability from the result nodes.
    std::set<std::string> keep;
    std::vector<std::string> work;
    for (const auto& n : g.nodes)
        if (n.op == ir_op::result) {
            for (const auto& in : n.inputs)
                work.push_back(in);
        }
    while (!work.empty()) {
        const std::string id = work.back();
        work.pop_back();
        if (!keep.insert(id).second)
            continue;
        for (const auto& n : g.nodes) {
            if (n.id != id || is_trivial(n))
                continue;
            for (const auto& in : n.inputs)
                work.push_back(in);
        }
    }

    ir_graph out = g;
    out.nodes.clear();
    // References across ALL remaining consumers (results included).
    std::set<std::string> referenced;
    for (const auto& n : g.nodes)
        for (const auto& in : n.inputs)
            referenced.insert(in);
    for (const auto& n : g.nodes) {
        // Orphaned constants are dropped together with their payload; a
        // constant node without data would be an unusable graph state.
        if (n.op == ir_op::constant && !referenced.count(n.id))
            continue;
        if (is_trivial(n) || keep.count(n.id))
            out.nodes.push_back(n);
    }
    std::map<std::string, std::vector<float>> cd;
    for (const auto& [id, data] : out.constant_data)
        if (referenced.count(id))
            cd[id] = data;
    out.constant_data = std::move(cd);
    std::map<std::string, ir_quant_const> qc;
    for (const auto& [id, q] : out.quant_constants)
        if (referenced.count(id))
            qc[id] = q;
    out.quant_constants = std::move(qc);
    return out;
}

ir_graph fold_constants(const ir_graph& g) {
    ir_graph cur = g;
    bool changed_any = false;
    while (true) {
        bool folded_one = false;
        for (auto& n : cur.nodes) {
            if (is_trivial(n) || !all_inputs_constant(cur, n))
                continue;
            // Build a tiny graph {constants..., node, result} and run the CPU
            // executor over it (constants must precede the consumer).
            ir_graph mini;
            for (size_t i = 0; i < n.inputs.size(); ++i) {
                ir_node c;
                c.id = n.inputs[i];
                c.op = ir_op::constant;
                mini.nodes.push_back(c);
                mini.constant_data[n.inputs[i]] = cur.constant_data.at(n.inputs[i]);
                mini.tensor_shapes[n.inputs[i]] = cur.tensor_shapes.at(n.inputs[i]);
            }
            mini.nodes.push_back(n);
            mini.tensor_shapes[n.id] = cur.tensor_shapes.at(n.id);
            // The output IS the folded node's buffer: a synthetic result node
            // would never produce a tensor, and outputs.at() would throw.
            mini.inputs.clear();
            mini.outputs = {n.id};
            std::vector<float> payload;
            try {
                const auto res = cpu_execute(mini, {});
                if (res.empty())
                    continue;
                payload = res.begin()->second;
            } catch (const std::exception&) {
                throw;
            }

            // Replace the node with a constant carrying the computed payload.
            n.op = ir_op::constant;
            n.inputs.clear();
            n.matmul_transpose_b = false;
            n.alpha = 0.0f;
            n.transpose_order.clear();
            cur.constant_data[n.id] = payload;
            folded_one = true;
            changed_any = true;
        }
        if (!folded_one)
            break;
    }
    return changed_any ? dce(cur) : cur;
}

ir_graph peephole(const ir_graph& g) {
    ir_graph cur = g;

    // Helper: rewire every consumer from |from| to |to|.
    const auto rewire = [&](const std::string& from, const std::string& to) {
        for (auto& n : cur.nodes)
            for (auto& in : n.inputs)
                if (in == from)
                    in = to;
        for (auto& id : cur.outputs)
            if (id == from)
                id = to;
    };

    // Map producer ids for quick lookup.
    std::map<std::string, size_t> by_id;
    for (size_t i = 0; i < cur.nodes.size(); ++i)
        by_id[cur.nodes[i].id] = i;

    bool changed = false;
    for (size_t i = 0; i < cur.nodes.size(); ++i) {
        const auto& n = cur.nodes[i];
        if (n.op == ir_op::relu || n.op == ir_op::sigmoid) {
            // act(act(x)) -> act(x)
            if (n.inputs.size() == 1 && by_id.count(n.inputs[0])) {
                const auto& inner = cur.nodes[by_id.at(n.inputs[0])];
                if (inner.op == n.op) {
                    rewire(n.id, inner.inputs[0]);
                    changed = true;
                }
            }
        } else if (n.op == ir_op::transpose && n.inputs.size() == 1 && by_id.count(n.inputs[0])) {
            const auto& inner = cur.nodes[by_id.at(n.inputs[0])];
            if (inner.op == ir_op::transpose && inner.inputs.size() == 1 &&
                inner.transpose_order.size() == n.transpose_order.size()) {
                // Compose the two permutations; an identity composition cancels.
                const size_t rank = n.transpose_order.size();
                std::vector<size_t> composed(rank);
                bool identity = true;
                for (size_t d = 0; d < rank; ++d) {
                    composed[d] = inner.transpose_order[n.transpose_order[d]];
                    identity &= composed[d] == d;
                }
                if (identity) {
                    rewire(n.id, inner.inputs[0]);
                } else {
                    // Keep the outer node as one composed transpose over the
                    // inner input; the inner node becomes dead and is dropped.
                    auto merged = n;
                    merged.inputs = {inner.inputs[0]};
                    merged.transpose_order = composed;
                    cur.nodes[i] = std::move(merged);
                }
                changed = true;
            }
        }
    }
    return changed ? dce(cur) : cur;
}

ir_graph optimize(const ir_graph& g) {
    ir_graph cur = g;
    for (int iter = 0; iter < 10; ++iter) {
        const auto after_fold = fold_constants(cur);
        const auto after_peep = peephole(after_fold);
        const auto after_dce = dce(after_peep);
        const bool same = after_dce.nodes.size() == cur.nodes.size() &&
                          after_dce.constant_data.size() == cur.constant_data.size();
        cur = after_dce;
        if (same)
            break;
    }
    return cur;
}

}  // namespace pass
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov

