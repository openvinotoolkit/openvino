// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "oooq_gate_shadow.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "alias_analysis.h"
#include "data_inst.h"
#include "exec_axis.h"
#include "input_layout_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "program_node.h"
#include "shape_of_inst.h"

namespace cldnn {

namespace {

constexpr int64_t kInf = std::numeric_limits<int64_t>::max();

std::mutex g_mutex;
int g_enabled_override = -1;
std::vector<std::pair<std::string, std::string>> g_s0_erase;
std::unordered_map<const program*, std::shared_ptr<const oooq_gate_shadow_report>> g_reports;

void store_report(const program& p, oooq_gate_shadow_report&& r) {
    auto sp = std::make_shared<const oooq_gate_shadow_report>(std::move(r));
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_reports[&p] = sp;
    }
    GPU_DEBUG_LOG << sp->summary() << std::endl;
#ifdef ENABLE_DEBUG_CAPS
    const char* v = std::getenv("OV_GPU_OOOQ_GATE_SHADOW");
    if (v != nullptr && v[0] == '2' && sp->ran) {
        GPU_DEBUG_LOG << sp->details() << std::endl;
    }
#endif
}

std::string type_of(const program_node& n) {
    return n.get_primitive() ? n.get_primitive()->type_string() : std::string("?");
}

size_t node_bytes(const program_node& n, bool& known) {
    size_t total = 0;
    for (size_t i = 0; i < n.get_outputs_count(); ++i) {
        const auto& l = n.get_output_layout(i);
        if (l.is_dynamic()) {
            known = false;
            continue;
        }
        total += l.bytes_count();
    }
    return total;
}

}  // namespace

bool oooq_gate_shadow_report::in_d(const std::string& a, const std::string& b) const {
    for (const auto& p : d_pair_ids) {
        if ((p.first == a && p.second == b) || (p.first == b && p.second == a)) {
            return true;
        }
    }
    return false;
}

std::string oooq_gate_shadow_report::summary() const {
    std::ostringstream s;
    s << "[oooq_gate_shadow] mode=" << mode;
    if (!ran) {
        return s.str();
    }
    s << " pairs_total=" << pairs_total << " S0=" << s0 << " S1=" << s1 << " D=" << d << " danger=" << danger
      << " danger_in_D=" << danger_in_d << " unverifiable=" << unverifiable << " po_exec_mismatch=" << po_exec_mismatch
      << " | d_gain_pairs=" << d_gain_pairs << " d_gain_bytes=" << d_gain_bytes << " d_potential_bytes=" << d_potential_bytes
      << " d_unreach=" << d_unreachable << " d_fork=" << d_fork << " d_suspect=" << d_suspect << " d_unassessed=" << d_unassessed
      << " non_pool_pairs=" << non_pool_pairs << " d_non_pool=" << d_non_pool << " conservative_pairs=" << conservative_pairs << " same_class_pairs=" << same_class_pairs << " danger_s1=" << danger_s1
      << " danger_asym=" << danger_asym << " d_with_input=" << d_with_input << " d_input_output_writer=" << d_input_output_writer
      << " d_first_node=" << d_first_node;
    return s.str();
}

std::string oooq_gate_shadow_report::details() const {
    std::ostringstream s;
    auto hist = [&](const char* title, const std::vector<pair_info>& v) {
        std::map<std::string, size_t> h;
        std::map<std::string, std::string> example;
        for (const auto& p : v) {
            const std::string key = std::min(p.type_a, p.type_b) + " / " + std::max(p.type_a, p.type_b);
            ++h[key];
            if (example.count(key) == 0) {
                example[key] = p.a + "," + p.b + " " + p.why;
            }
        }
        std::vector<std::pair<size_t, std::string>> sorted;
        for (const auto& kv : h) {
            sorted.emplace_back(kv.second, kv.first);
        }
        std::sort(sorted.rbegin(), sorted.rend());
        s << "[oooq_gate_shadow] " << title << " (" << v.size() << "):\n";
        for (size_t i = 0; i < sorted.size() && i < 12; ++i) {
            s << "    " << sorted[i].first << "  " << sorted[i].second << "   e.g. " << example[sorted[i].second] << "\n";
        }
    };
    hist("danger (overlap, absent from S0)", danger_pairs);
    hist("danger_in_D", danger_in_d_pairs);
    hist("danger_s1 (overlap, absent from S1: oracle vs LEGACY)", danger_s1_pairs);
    return s.str();
}

bool oooq_gate_shadow::enabled() {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_enabled_override >= 0) {
            return g_enabled_override == 1;
        }
    }
#ifdef ENABLE_DEBUG_CAPS
    const char* v = std::getenv("OV_GPU_OOOQ_GATE_SHADOW");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
#else
    return false;
#endif
}

void oooq_gate_shadow::set_enabled_for_test(int mode) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_enabled_override = mode;
}

void oooq_gate_shadow::set_s0_erase_for_test(std::vector<std::pair<std::string, std::string>> pairs) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_s0_erase = std::move(pairs);
}

void oooq_gate_shadow::clear_reports_for_test() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_reports.clear();
}

std::shared_ptr<const oooq_gate_shadow_report> oooq_gate_shadow::report_for(const program& p) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_reports.find(&p);
    return it == g_reports.end() ? nullptr : it->second;
}

std::unique_ptr<oooq_gate_shadow::snapshot> oooq_gate_shadow::take_snapshot(const program& p) {
    try {
        {
            // A stale report of a previous program at the same address must not be mistaken for this run.
            std::lock_guard<std::mutex> lock(g_mutex);
            g_reports.erase(&p);
        }
        const char* skip = nullptr;
        if (p.get_config().get_queue_type() != QueueTypes::in_order) {
            skip = "skipped:oooq";
        } else if (p.is_internal_program()) {
            skip = "skipped:internal";
        } else if (p.is_body_program()) {
            skip = "skipped:body";
        }
        if (skip != nullptr) {
            oooq_gate_shadow_report r;
            r.mode = skip;
            store_report(p, std::move(r));
            return nullptr;
        }

        auto s = std::make_unique<snapshot>();
        std::unordered_map<uint32_t, size_t> count;
        for (const auto* n : p.get_processing_order()) {
            s->index.emplace(n, s->nodes.size());
            s->nodes.push_back(n);
            ++count[static_cast<uint32_t>(n->get_unique_id())];
        }
        for (size_t i = 0; i < s->nodes.size(); ++i) {
            const auto uid = static_cast<uint32_t>(s->nodes[i]->get_unique_id());
            if (count[uid] > 1) {
                s->ambiguous.insert(i);
            } else {
                s->uid_to_index.emplace(uid, i);
            }
        }
        s->words = (s->nodes.size() + 63) / 64;
        s->rows.assign(s->nodes.size(), std::vector<uint64_t>(s->words, 0ULL));
        for (size_t i = 0; i < s->nodes.size(); ++i) {
            for (const uint32_t uid : s->nodes[i]->get_memory_dependencies()) {
                auto it = s->uid_to_index.find(uid);
                if (it != s->uid_to_index.end()) {
                    s->set(i, it->second);
                }
            }
        }
        std::vector<std::pair<std::string, std::string>> erase;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            erase = g_s0_erase;
        }
        for (const auto& e : erase) {
            size_t ia = s->nodes.size(), ib = s->nodes.size();
            for (size_t i = 0; i < s->nodes.size(); ++i) {
                if (s->nodes[i]->id() == e.first) {
                    ia = i;
                }
                if (s->nodes[i]->id() == e.second) {
                    ib = i;
                }
            }
            if (ia < s->nodes.size() && ib < s->nodes.size()) {
                s->clear(ia, ib);
                s->clear(ib, ia);
            }
        }
        return s;
    } catch (...) {
        return nullptr;
    }
}

namespace {

struct class_life {
    bool known = false;
    bool conservative = false;
    int64_t lo = kInf, hi = -1;
};

struct node_life {
    bool classless = false, conservative = false, unknown = false;
    std::vector<const alias_class*> cls;
    std::vector<std::pair<int64_t, int64_t>> iv;
};

bool overlap(const node_life& a, const node_life& b, std::pair<int64_t, int64_t>* ea = nullptr, std::pair<int64_t, int64_t>* eb = nullptr) {
    for (const auto& x : a.iv) {
        for (const auto& y : b.iv) {
            if (x.first <= y.second && y.first <= x.second) {
                if (ea != nullptr) {
                    *ea = x;
                }
                if (eb != nullptr) {
                    *eb = y;
                }
                return true;
            }
        }
    }
    return false;
}

// Mirrors the allocator's routing decision (primitive_inst::allocate_output: pool unless !can_share_buffer || can_be_optimized || is_output).
// Dynamic / runtime-skippable can_be_optimized nodes may still take a pool block at runtime (MAY envelope), so they count as pool users.
bool is_pool_user(const program_node& n) {
    return n.may_use_mempool() && n.can_share_buffer() && !n.is_output() && (!n.can_be_optimized() || n.is_runtime_skippable() || n.is_dynamic());
}

std::string iv_str(const std::pair<int64_t, int64_t>& v) {
    return "[" + std::to_string(v.first) + "," + (v.second == kInf ? std::string("inf") : std::to_string(v.second)) + "]";
}

}  // namespace

oooq_gate_shadow_report oooq_gate_shadow::compute(const program& p, const snapshot& s0, const alias_analysis& aa) {
    oooq_gate_shadow_report r;
    r.mode = "in_order";
    r.ran = true;

    const auto axis = exec_axis::build(p);
    r.po_exec_mismatch = axis.valid() ? 0 : 1;

    const auto& nodes = s0.nodes;
    const size_t N = nodes.size();

    // ---- lifetimes ----
    std::unordered_map<const alias_class*, class_life> clife;
    auto life_of_class = [&](const alias_class* c) -> const class_life& {
        auto it = clife.find(c);
        if (it != clife.end()) {
            return it->second;
        }
        class_life cl;
        cl.known = true;
        if ((c->fallback & ~static_cast<uint32_t>(fb_oooq)) != 0 || (c->domain & dm_persistent) != 0) {
            cl.conservative = true;
        }
        bool any_pos = false, inf = (c->domain & dm_user_output) != 0;
        auto touch = [&](const program_node* n, bool is_reader) {
            auto pos = axis.position(*n);
            if (!pos) {
                if (!is_reader && (n->is_type<data>() || n->is_constant())) {
                    return;
                }
                cl.known = false;
                return;
            }
            any_pos = true;
            cl.lo = std::min<int64_t>(cl.lo, is_reader ? cl.lo : *pos);
            cl.hi = std::max<int64_t>(cl.hi, *pos);
            if (!is_reader && n->is_output()) {
                inf = true;
            }
        };
        for (const auto& m : c->members) {
            touch(m.node, false);
        }
        for (const auto* rd : c->readers) {
            touch(rd, true);
        }
        if (!any_pos) {
            cl.known = false;
        }
        if (inf) {
            cl.hi = kInf;
        }
        return clife.emplace(c, cl).first->second;
    };

    auto own_interval = [&](const program_node& n, uint16_t port, bool& known) -> std::pair<int64_t, int64_t> {
        const auto pos = axis.position(n);
        if (!pos) {
            known = false;
            return {0, 0};
        }
        int64_t hi = *pos;
        bool inf = false;
        std::vector<std::pair<const program_node*, uint16_t>> work{{&n, port}};
        std::set<std::pair<const program_node*, uint16_t>> seen;
        while (!work.empty()) {
            const auto t = work.back();
            work.pop_back();
            if (!seen.insert(t).second) {
                continue;
            }
            inf = inf || t.first->is_output();
            if (auto tp = axis.position(*t.first)) {
                hi = std::max<int64_t>(hi, *tp);
            }
            aa.for_each_tensor_user(*t.first, t.second, [&](const program_node& u, size_t) {
                if (u.is_type<shape_of>()) {
                    return;
                }
                const auto up = axis.position(u);
                if (!up) {
                    known = false;
                } else {
                    hi = std::max<int64_t>(hi, *up);
                }
            });
            for (const size_t ei : aa.edges_into(*t.first, t.second)) {
                const auto& e = aa.edges()[ei];
                work.emplace_back(e.member_node, e.member_port);
            }
        }
        return {*pos, inf ? kInf : hi};
    };

    std::vector<node_life> life(N);
    std::vector<size_t> cand;   // nodes that may use the pool
    for (size_t i = 0; i < N; ++i) {
        const auto& n = *nodes[i];
        if (!n.may_use_mempool()) {
            continue;
        }
        cand.push_back(i);
        auto& l = life[i];
        if (!axis.valid()) {
            l.unknown = true;
            continue;
        }
        if (cpu_state(n) != cpu_none || is_persistent_type(n) || n.is_in_shape_of_subgraph()) {
            l.conservative = true;
        }
        for (size_t port = 0; port < n.get_outputs_count(); ++port) {
            const auto* c = aa.class_of(n, static_cast<int32_t>(port));
            if (c == nullptr) {
                continue;
            }
            if (std::find(l.cls.begin(), l.cls.end(), c) != l.cls.end()) {
                continue;
            }
            l.cls.push_back(c);
            const auto& cl = life_of_class(c);
            if (!cl.known) {
                l.unknown = true;
            } else if (c->root_node == &n && c->root.index == port) {
                l.iv.emplace_back(cl.lo, cl.hi);   // the allocator of the class: its block holds every member
            } else {
                // A non-root member that allocates (MAY envelope) only needs its own block from its own position until the last reader of
                // its tensor and of everything downstream that aliases it; the upstream half of the class lives in the root's block.
                bool known = true;
                const auto own = own_interval(n, static_cast<uint16_t>(port), known);
                if (known) {
                    l.iv.push_back(own);
                } else {
                    l.unknown = true;
                }
            }
            l.conservative = l.conservative || cl.conservative;
        }
        if (l.cls.empty()) {
            l.classless = true;
            l.unknown = true;
            ++r.classless_nodes;
        }
        const auto pos = axis.position(n);
        if (!pos) {
            l.unknown = true;
        } else if (!(n.can_be_optimized() && !n.is_dynamic()) && !n.is_type<input_layout>()) {
            l.iv.emplace_back(*pos, *pos);   // internal buffers: [n, n]
        }
        if (l.conservative) {
            ++r.conservative_nodes;
        }
        if (s0.ambiguous.count(i) != 0) {
            l.unknown = true;
            ++r.ambiguous_uid_nodes;
        }
    }

    // ---- reachability, for classifying D only (structure of the oooq rules, never used to decide anything) ----
    std::vector<size_t> dmap(N, std::numeric_limits<size_t>::max());   // node index -> index among non-data nodes
    std::vector<size_t> nd_nodes;
    for (size_t i = 0; i < N; ++i) {
        if (!nodes[i]->is_type<data>()) {
            dmap[i] = nd_nodes.size();
            nd_nodes.push_back(i);
        }
    }
    const size_t M = nd_nodes.size();
    const size_t W = (M + 63) / 64;
    std::vector<std::vector<uint64_t>> reach;
    bool reach_built = false;
    auto reach_test = [&](size_t a, size_t b) {
        return (reach[a][b >> 6] >> (b & 63)) & 1ULL;
    };
    auto build_reach = [&]() {
        reach_built = true;
        reach.assign(M, std::vector<uint64_t>(W, 0ULL));
        std::vector<std::vector<size_t>> adj(M);
        for (size_t a = 0; a < M; ++a) {
            for (const auto* u : nodes[nd_nodes[a]]->get_users()) {
                if (u->is_type<shape_of>()) {
                    continue;
                }
                auto it = s0.index.find(u);
                if (it == s0.index.end() || dmap[it->second] == std::numeric_limits<size_t>::max()) {
                    continue;
                }
                const size_t b = dmap[it->second];
                adj[a].push_back(b);
                reach[a][b >> 6] |= (1ULL << (b & 63));
            }
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t a = M; a-- > 0;) {
                for (const size_t b : adj[a]) {
                    for (size_t w = 0; w < W; ++w) {
                        const uint64_t nv = reach[a][w] | reach[b][w];
                        if (nv != reach[a][w]) {
                            reach[a][w] = nv;
                            changed = true;
                        }
                    }
                }
            }
        }
    };

    // first non-data node of the processing order (the node the oooq loop never pairs, see the 3(c) check)
    const size_t first_idx = nd_nodes.empty() ? std::numeric_limits<size_t>::max() : nd_nodes.front();

    auto cap_push = [&](std::vector<oooq_gate_shadow_report::pair_info>& v, const oooq_gate_shadow_report::pair_info& pi) {
        if (v.size() < oooq_gate_shadow_report::kMaxListed) {
            v.push_back(pi);
        }
    };

    for (size_t ci = 0; ci < cand.size(); ++ci) {
        for (size_t cj = ci + 1; cj < cand.size(); ++cj) {
            const size_t i = cand[ci], j = cand[cj];
            const auto &A = *nodes[i], &B = *nodes[j];
            if (A.get_unique_id() == B.get_unique_id()) {
                continue;
            }
            ++r.pairs_total;
            const bool ab0 = s0.test(i, j), ba0 = s0.test(j, i);
            const bool ab1 = A.has_memory_dependency(static_cast<uint32_t>(B.get_unique_id()));
            const bool ba1 = B.has_memory_dependency(static_cast<uint32_t>(A.get_unique_id()));
            const bool in0 = ab0 || ba0, in1 = ab1 || ba1;
            r.s0_directed += (ab0 ? 1 : 0) + (ba0 ? 1 : 0);
            r.s1_directed += (ab1 ? 1 : 0) + (ba1 ? 1 : 0);
            r.s0 += in0 ? 1 : 0;
            r.s1 += in1 ? 1 : 0;
            const bool is_d = in1 && !in0;
            const auto &la = life[i], &lb = life[j];

            bool same_class = false;
            for (const auto* c : la.cls) {
                same_class = same_class || std::find(lb.cls.begin(), lb.cls.end(), c) != lb.cls.end();
            }
            const bool conservative = la.conservative || lb.conservative;
            const bool non_pool = !is_pool_user(A) || !is_pool_user(B);
            const bool unknown = la.unknown || lb.unknown;
            bool judged = !same_class && !conservative && !non_pool && !unknown;
            if (same_class) {
                ++r.same_class_pairs;
            } else if (conservative) {
                ++r.conservative_pairs;
            } else if (non_pool) {
                ++r.non_pool_pairs;
            } else if (unknown) {
                ++r.unverifiable;
            }
            std::pair<int64_t, int64_t> ea, eb;
            const bool ov = judged && overlap(la, lb, &ea, &eb);

            if (judged && ov) {
                oooq_gate_shadow_report::pair_info pi{A.id(), B.id(), type_of(A), type_of(B), "overlap " + iv_str(ea) + " " + iv_str(eb), in1};
                if (!in0) {
                    ++r.danger;
                    cap_push(r.danger_pairs, pi);
                    if (in1) {
                        ++r.danger_in_d;
                        cap_push(r.danger_in_d_pairs, pi);
                    }
                } else if (ab0 != ba0) {
                    ++r.danger_asym;
                    pi.why += ab0 ? " restricted only in the direction first->second" : " restricted only in the direction second->first";
                    cap_push(r.danger_asym_pairs, pi);
                }
                if (!in1) {
                    ++r.danger_s1;
                    cap_push(r.danger_s1_pairs, pi);
                }
            }

            if (is_d) {
                ++r.d;
                if (r.d_pair_ids.size() < oooq_gate_shadow_report::kMaxListed) {
                    r.d_pair_ids.emplace_back(A.id(), B.id());
                }
                bool known = true;
                const size_t bytes = std::min(node_bytes(A, known), node_bytes(B, known));
                if (!known) {
                    ++r.d_bytes_unknown_pairs;
                }
                r.d_potential_bytes += known ? bytes : 0;
                if (judged) {
                    ++r.d_assessed;
                    if (!ov) {
                        ++r.d_gain_pairs;
                        r.d_gain_bytes += known ? bytes : 0;
                    }
                } else {
                    ++r.d_unassessed;
                    r.d_non_pool += (non_pool && !same_class && !conservative) ? 1 : 0;
                }
                // classification by the oooq rule
                if (!reach_built) {
                    build_reach();
                }
                const size_t a = dmap[i], b = dmap[j];
                if (a != std::numeric_limits<size_t>::max() && b != std::numeric_limits<size_t>::max()) {
                    const size_t lo = std::min(a, b), hi = std::max(a, b);
                    if (!reach_test(lo, hi)) {
                        ++r.d_unreachable;
                    } else {
                        bool fork = false;
                        for (const auto* u : nodes[nd_nodes[lo]]->get_users()) {
                            auto it = s0.index.find(u);
                            if (it == s0.index.end() || dmap[it->second] == std::numeric_limits<size_t>::max() || it->second == nd_nodes[hi]) {
                                continue;
                            }
                            const size_t ui = dmap[it->second];
                            if (!reach_test(hi, ui) && !reach_test(ui, hi)) {
                                fork = true;
                                break;
                            }
                        }
                        if (fork) {
                            ++r.d_fork;
                        } else {
                            ++r.d_suspect;
                            r.d_suspect_danger += (judged && ov) ? 1 : 0;
                        }
                    }
                }
                const bool a_in = A.is_type<input_layout>(), b_in = B.is_type<input_layout>();
                if (a_in || b_in) {
                    ++r.d_with_input;
                    if ((a_in && B.is_output()) || (b_in && A.is_output())) {
                        ++r.d_input_output_writer;
                    }
                }
                if (i == first_idx || j == first_idx) {
                    ++r.d_first_node;
                }
            }
        }
    }
    return r;
}

void oooq_gate_shadow::finish(const program& p, std::unique_ptr<snapshot> s0, const alias_analysis* aa) noexcept {
    try {
        if (!s0) {
            return;
        }
        if (aa == nullptr) {
            oooq_gate_shadow_report r;
            r.mode = "skipped:no_analysis";
            store_report(p, std::move(r));
            return;
        }
        store_report(p, compute(p, *s0, *aa));
    } catch (...) {
        try {
            oooq_gate_shadow_report r;
            r.mode = "skipped:exception";
            store_report(p, std::move(r));
        } catch (...) {
        }
    }
}

}  // namespace cldnn
