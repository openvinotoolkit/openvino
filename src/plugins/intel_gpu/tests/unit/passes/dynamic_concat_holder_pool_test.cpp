// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P1-B-D: dynamic in-place Concat holder as pool class root (first slice).
// Design: /home/ceciliapeng/repos/reports/memman/layer1/12_p1bd_design.zh-CN.md (test ids TD* follow its section 6).
//
// Switches: enable_concat_holder_pool (P1-B-S, main) and enable_dynamic_concat_holder_pool (P1-B-D, sub). Both default off.
// Tests that need oneDNN GTEST_SKIP when it is not available (SKIP == uncovered).
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "alias_analysis.h"
#include "class_level_memory_dependencies.h"
#include "concatenation_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "primitive_inst.h"
#include "primitive_inst_test_helper.h"
#include "program_node.h"
#include "program_wrapper.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

using shapes_t = std::map<std::string, ov::Shape>;
using outputs_t = std::map<std::string, std::vector<float>>;

struct sw {
    bool main = false;
    bool dyn = false;
    QueueTypes q = QueueTypes::in_order;
    int64_t reclaim_min = -1;  // < 0: default
    size_t trace = 1;
    ov::intel_gpu::ImplForcingMap forced{};
};

const sw OFF{false, false};
const sw MAIN_ONLY{true, false};
const sw BOTH{true, true};
const sw DYN_ONLY{false, true};

ExecutionConfig make_cfg(engine& eng, const sw& s) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(s.q));
    if (s.main) {
        config.set_property(ov::intel_gpu::enable_concat_holder_pool(true));
    }
    if (s.dyn) {
        config.set_property(ov::intel_gpu::enable_dynamic_concat_holder_pool(true));
    }
    if (!s.forced.empty()) {
        config.set_property(ov::intel_gpu::force_implementations(s.forced));
    }
    if (s.reclaim_min >= 0) {
        config.set_property(ov::intel_gpu::output_reclaim_min_size(static_cast<size_t>(s.reclaim_min)));
    }
    if (s.q == QueueTypes::out_of_order) {
        // T-OOOQ: oneDNN forces in_order on Arc; it has to be switched off by a USER property for the queue type to stick
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
#ifdef GPU_DEBUG_CONFIG
    config.set_user_property({ov::intel_gpu::dump_alloc_trace(s.trace)});
#endif
    return config;
}

layout dyn4(data_types dt = data_types::f32) {
    return layout{ov::PartialShape::dynamic(4), dt, format::bfyx};
}

// small integers, exact in f16 and f32
std::vector<float> vals(size_t n, int seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<float>((static_cast<int>(i) * 5 + seed * 11) % 13) - 6.f;
    }
    return v;
}

std::vector<float> read_f32(const memory::ptr& m, size_t count) {
    std::vector<float> r(count);
    if (count == 0) {
        return r;
    }
    mem_lock<float, mem_lock_type::read> lock(m, get_test_stream());
    for (size_t i = 0; i < count; i++) {
        r[i] = lock[i];
    }
    return r;
}

// One round: sets the inputs from `s` (data from vals(seed + index)), executes, returns the outputs (count taken from the output layout).
outputs_t run_round(network& net, engine& eng, const shapes_t& s, int seed) {
    int k = 0;
    for (const auto& in : s) {
        auto m = eng.allocate_memory(layout{ov::PartialShape(in.second), data_types::f32, format::bfyx});
        if (m->count() != 0) {
            set_values(m, vals(ov::shape_size(in.second), seed + k));
        }
        ++k;
        net.set_input_data(in.first, m);
    }
    auto outs = net.execute();
    outputs_t res;
    for (auto& o : outs) {
        res[o.first] = read_f32(o.second.get_memory(), net.get_output_layout(o.first).count());
    }
    return res;
}

std::vector<float> vals_of(const ov::Shape& s, int seed) {
    return vals(ov::shape_size(s), seed);
}

// reference for the topologies below: P1 = abs(x1), P2 = relu(x2), output of the concat reader = relu(concat(P1, P2)) (the reader is the identity)
std::vector<float> ref_concat2(const ov::Shape& s1, const ov::Shape& s2, int seed) {
    std::vector<float> r;
    for (float v : vals_of(s1, seed)) {
        r.push_back(std::abs(v));
    }
    for (float v : vals_of(s2, seed + 1)) {
        r.push_back(std::max(v, 0.f));
    }
    return r;
}

const void* buf_of(network& net, const std::string& id) {
    auto m = net.get_primitive(id)->output_memory_ptr();
    return m ? m->buffer_ptr() : nullptr;
}

size_t pool_records_of(network& net, const std::string& id) {
    return net.get_memory_pool().count_records_with_user(net.get_primitive(id)->get_node().get_unique_id(), net.get_id());
}

primitive_inst::alloc_route route_of(const program_node& n) {
    return primitive_inst::allocation_route(n, 0, n.can_be_optimized());
}

size_t count_roots(program& p) {
    size_t c = 0;
    for (auto* n : p.get_processing_order()) {
        c += n->is_pool_class_root() ? 1 : 0;
    }
    return c;
}

std::string dump_nodes(program& p) {
    std::ostringstream os;
    for (auto* n : p.get_processing_order()) {
        os << "  " << n->id() << " uid=" << n->get_unique_id() << " type=" << n->get_primitive()->type_string() << " dyn=" << n->is_dynamic()
           << " cbo=" << n->can_be_optimized() << " rs=" << n->is_runtime_skippable() << " cbs=" << n->can_share_buffer() << " out=" << n->is_output()
           << " root=" << n->is_pool_class_root() << " aroot=" << (n->has_alias_pool_root() ? static_cast<int64_t>(n->get_alias_pool_root_uid()) : -1)
           << " R=[";
        for (auto r : n->get_memory_dependencies()) {
            os << r << ",";
        }
        os << "]\n";
    }
    return os.str();
}

uint32_t mask_of(program& p, const std::string& id, const std::function<bool(uint32_t)>& amb = [](uint32_t) { return false; }) {
    auto aa = alias_analysis::build(p);
    const auto* c = aa.class_of(p.get_node(id));
    EXPECT_NE(c, nullptr) << id;
    return c ? class_level::pool_admission_reject_mask(*c, p, amb) : ~0u;
}

// ------------------------------------------------------------------------------------------------------------------------------
// Topologies. All activations are f32; the in-place concat is on axis 1.
// ------------------------------------------------------------------------------------------------------------------------------
// in1 -> P1 (abs) --+
//                   C (axis 1) -> A (relu, network output)
// in2 -> P2 (relu) -+
topology topo_basic(bool a_is_output_reader = true) {
    topology t;
    t.add(input_layout("in1", dyn4()));
    t.add(input_layout("in2", dyn4()));
    t.add(activation("P1", input_info("in1"), activation_func::abs));
    t.add(activation("P2", input_info("in2"), activation_func::relu));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("A", input_info("C"), activation_func::relu));
    if (!a_is_output_reader) {
        t.add(activation("Y", input_info("A"), activation_func::abs));
    }
    return t;
}

network::ptr make_net(engine& eng, topology& t, const sw& s) {
    return get_network(eng, t, make_cfg(eng, s), get_test_stream_ptr(), false);
}


// ------------------------------------------------------------------------------------------------------------------------------
// Oracle: the buffers shared by two different tensors must have disjoint lifetimes in the executed order (simulated, in_order). A fused
// in-place holder and its predecessors form ONE tensor group. Nodes that did not execute in this round (zero length) and runtime views are
// not tensors of the round.
// ------------------------------------------------------------------------------------------------------------------------------
std::vector<std::string> overlap_violations(network& net) {
    std::vector<std::string> res;
    auto prog = net.get_program();
    std::map<const program_node*, size_t> idx;
    size_t i = 0;
    for (auto* n : prog->get_processing_order()) {
        idx[n] = i++;
    }
    std::map<const program_node*, const program_node*> group;
    for (auto* n : prog->get_processing_order()) {
        if (!n->is_type<concatenation>()) {
            continue;
        }
        auto inst = net.get_primitive(n->id());
        if (!inst->can_be_optimized()) {
            continue;
        }
        for (const auto& d : n->get_dependencies()) {
            group[d.first] = n;
        }
    }
    struct tensor {
        const void* buf;
        size_t birth, death;
        std::string name;
    };
    std::map<const program_node*, tensor> groups;
    for (auto* n : prog->get_processing_order()) {
        const std::string ty = n->get_primitive()->type_string();
        if (ty == "data" || ty == "input_layout") {
            continue;
        }
        auto inst = net.get_primitive(n->id());
        auto mem = inst->output_memory_ptr();
        if (!mem) {
            continue;
        }
        const auto l = inst->get_output_layout();
        if (!l.is_static() || l.count() == 0) {
            continue;
        }
        if (inst->can_be_optimized() && !n->is_type<concatenation>()) {
            continue;
        }
        const program_node* g = group.count(n) ? group[n] : n;
        size_t birth = idx[n], death = idx[n];
        for (auto* u : n->get_users()) {
            death = std::max(death, idx[u]);
        }
        auto it = groups.find(g);
        if (it == groups.end()) {
            groups[g] = {mem->buffer_ptr(), birth, death, g->id()};
        } else {
            it->second.birth = std::min(it->second.birth, birth);
            it->second.death = std::max(it->second.death, death);
        }
    }
    std::vector<tensor> v;
    for (auto& kv : groups) {
        v.push_back(kv.second);
    }
    for (size_t a = 0; a < v.size(); a++) {
        for (size_t b = a + 1; b < v.size(); b++) {
            if (v[a].buf == v[b].buf && !(v[a].death < v[b].birth || v[b].death < v[a].birth)) {
                res.push_back(v[a].name + " [" + std::to_string(v[a].birth) + "," + std::to_string(v[a].death) + "] and " + v[b].name + " [" +
                              std::to_string(v[b].birth) + "," + std::to_string(v[b].death) + "] share a buffer");
            }
        }
    }
    return res;
}

struct round_t {
    shapes_t s;
    int seed;
};
using ref_fn = std::function<outputs_t(const shapes_t&, int)>;
using hook_fn = std::function<void(network&, size_t)>;

struct ab_nets {
    network::ptr off, on;
};

// Executes the rounds on a switch-off network and on a network built with `on_sw`; outputs must be equal to each other and to the reference,
// the lifetime oracle must be clean on the on-network after every round.
ab_nets check_rounds(engine& eng,
                     const std::function<topology()>& mk,
                     const sw& on_sw,
                     const std::vector<round_t>& rounds,
                     const ref_fn& ref,
                     const hook_fn& after_round = nullptr,
                     const hook_fn& before_round = nullptr,
                     const sw& off_sw = OFF,
                     bool compare_off = true) {
    auto t_off = mk();
    auto t_on = mk();
    ab_nets r{make_net(eng, t_off, off_sw), make_net(eng, t_on, on_sw)};
    for (size_t i = 0; i < rounds.size(); ++i) {
        if (before_round) {
            before_round(*r.on, i);
        }
        const auto out_off = run_round(*r.off, eng, rounds[i].s, rounds[i].seed);
        const auto out_on = run_round(*r.on, eng, rounds[i].s, rounds[i].seed);
        if (compare_off) {
            EXPECT_EQ(out_on, out_off) << "round " << i << ": outputs differ between switch off and on";
        }
        if (ref) {
            for (const auto* o : {&out_off, &out_on}) {
                const auto r = ref(rounds[i].s, rounds[i].seed);
                for (const auto& kv : r) {
                    const auto& got = o->at(kv.first);
                    for (size_t k = 0; k < std::min(got.size(), kv.second.size()); ++k) {
                        if (got[k] != kv.second[k]) {
                            std::cout << "[DIFF] round " << i << (o == &out_off ? " OFF" : " ON") << " first diff at " << k << " of " << got.size() << " got " << got[k]
                                      << " ref " << kv.second[k] << "\n";
                            break;
                        }
                    }
                }
            }
            if (compare_off) {
                EXPECT_EQ(out_off, ref(rounds[i].s, rounds[i].seed)) << "round " << i << ": the SWITCH-OFF output differs from the reference";
            }
            EXPECT_EQ(out_on, ref(rounds[i].s, rounds[i].seed)) << "round " << i << ": output differs from the reference";
        }
        const auto v = overlap_violations(*r.on);
        EXPECT_TRUE(v.empty()) << "round " << i << ": " << (v.empty() ? "" : v[0]);
        if (after_round) {
            after_round(*r.on, i);
        }
    }
    return r;
}

ov::Shape sh(size_t f, size_t h = 4, size_t w = 4) {
    return ov::Shape{1, f, h, w};
}

shapes_t s2(size_t f1, size_t f2, size_t h = 4, size_t w = 4) {
    return {{"in1", sh(f1, h, w)}, {"in2", sh(f2, h, w)}};
}

ref_fn ref_basic(const std::string& out_id = "A") {
    return [out_id](const shapes_t& s, int seed) {
        return outputs_t{{out_id, ref_concat2(s.at("in1"), s.at("in2"), seed)}};
    };
}

// holder-wide NaN sentinel: every byte of the holder block that a slot does not overwrite stays NaN, so uncovered bytes show up in the output
void fill_nan(network& net, const std::string& id) {
    auto m = net.get_primitive(id)->output_memory_ptr();
    if (!m) {
        return;
    }
    mem_lock<uint8_t, mem_lock_type::write> lock(m, get_test_stream());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 0; i + sizeof(float) <= lock.size(); i += sizeof(float)) {
        std::memcpy(&lock[i], &nan, sizeof(float));
    }
}

bool is_fused(network& net, const std::string& id = "C") {
    return net.get_primitive(id)->can_be_optimized();
}

}  // namespace

// ==============================================================================================================================
// Topology helpers that need the helpers above
// ==============================================================================================================================
namespace {

// P1/P2 -> C -> A -> E (sum with W) -> X (abs) -> Y (relu, output); W = relu(in3) is a third party that lives next to the holder, X comes after
// all readers of the holder.
topology topo_third_party() {
    topology t;
    t.add(input_layout("in1", dyn4()));
    t.add(input_layout("in2", dyn4()));
    t.add(input_layout("in3", dyn4()));
    t.add(activation("P1", input_info("in1"), activation_func::abs));
    t.add(activation("P2", input_info("in2"), activation_func::relu));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("A", input_info("C"), activation_func::relu));
    t.add(activation("W", input_info("in3"), activation_func::relu));
    t.add(eltwise("E", {input_info("A"), input_info("W")}, eltwise_mode::sum));
    t.add(activation("X", input_info("E"), activation_func::abs));
    t.add(activation("Y", input_info("X"), activation_func::relu));
    return t;
}

shapes_t s3(size_t f1, size_t f2, size_t h = 4, size_t w = 4) {
    auto s = s2(f1, f2, h, w);
    s["in3"] = sh(f1 + f2, h, w);
    return s;
}

ref_fn ref_third_party() {
    return [](const shapes_t& s, int seed) {
        auto c = ref_concat2(s.at("in1"), s.at("in2"), seed);
        const auto w = vals_of(s.at("in3"), seed + 2);
        for (size_t i = 0; i < c.size(); ++i) {
            c[i] += std::max(w[i], 0.f);
        }
        return outputs_t{{"Y", c}};
    };
}

}  // namespace

// ==============================================================================================================================
// TD1 (TB6). Rounds: growth, same shape, shrink, zero-length predecessors, zero-length round. Holder is flagged and a pool user,
// predecessors are never pool users, outputs of the switch off / on networks equal the reference.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td1_rounds_grow_same_shrink_zero_length) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s2(4, 4), 1},  {s2(4, 4), 2}, {s2(16, 16), 3}, {s2(2, 2), 4},
                                         {s2(8, 0), 5},  {s2(0, 8), 6}, {s2(0, 0), 7},   {s2(4, 4), 8}};
    size_t fused_rounds = 0;
    auto nets = check_rounds(eng, []() { return topo_basic(); }, BOTH, rounds, ref_basic(), [&](network& net, size_t) {
        EXPECT_NE(buf_of(net, "C"), nullptr);
        if (is_fused(net)) {
            ++fused_rounds;
        }
        EXPECT_LE(pool_records_of(net, "C"), 1u);
        EXPECT_EQ(pool_records_of(net, "P1"), 0u) << "predecessors never enter the pool";
        EXPECT_EQ(pool_records_of(net, "P2"), 0u);
    });
    auto& prog = *nets.on->get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    EXPECT_TRUE(prog.get_node("C").is_dynamic());
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
    EXPECT_EQ(route_of(prog.get_node("C")), primitive_inst::alloc_route::POOL_AS_ROOT);
    EXPECT_EQ(pool_records_of(*nets.on, "C"), 1u) << "the holder is a pool user";
    EXPECT_GT(fused_rounds, 0u);
    EXPECT_EQ(count_roots(*nets.off->get_program()), 0u);
    EXPECT_EQ(pool_records_of(*nets.off, "C"), 0u) << "switch off: the fused holder is a direct allocation";
}

// ==============================================================================================================================
// TD2 (TB7 / I-B5). A third party that lives next to the holder (W) never shares its block, X after all readers may; the oracle checks every
// round, growth and shrink included.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td2_third_party_never_shares_a_live_holder_block) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s3(4, 4), 1}, {s3(4, 4), 2}, {s3(16, 16), 3}, {s3(2, 2), 4}, {s3(8, 0), 5}, {s3(4, 12), 6}, {s3(4, 4), 7}};
    auto nets = check_rounds(eng, []() { return topo_third_party(); }, BOTH, rounds, ref_third_party(), [&](network& net, size_t i) {
        EXPECT_NE(buf_of(net, "W"), buf_of(net, "C")) << "round " << i << ": W overlaps the holder lifetime";
        for (const char* later : {"X", "Y", "E"}) {
            if (net.get_program()->has_node(later) && net.get_primitive(later)->output_memory_ptr()) {
                std::cout << "[TD2] round " << i << " " << later << " shares the holder block: " << (buf_of(net, later) == buf_of(net, "C")) << "\n";
            }
        }
    });
    EXPECT_TRUE(nets.on->get_program()->get_node("C").is_pool_class_root());
}

// ==============================================================================================================================
// TD3a. Stale pointer of a SKIP predecessor: round 2 has a zero-length P2 and a bigger P1, the holder takes a new block; P2 has to follow it.
// In round 3 P2 is back, the output must be correct (P2 writes into the holder block, not into the released old one).
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td3a_skip_predecessor_follows_the_holder_block) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s3(2, 2), 1}, {s3(8, 0), 2}, {s3(4, 4), 3}, {s3(2, 6), 4}};
    auto nets = check_rounds(eng, []() { return topo_third_party(); }, BOTH, rounds, ref_third_party(), [&](network& net, size_t i) {
        if (i == 1) {
            ASSERT_TRUE(is_fused(net));
            EXPECT_EQ(buf_of(net, "P2"), buf_of(net, "C")) << "round 1: the SKIP predecessor P2 has to be re-pointed to the current holder block";
            EXPECT_EQ(buf_of(net, "P1"), buf_of(net, "C"));
        }
    });
}

// ==============================================================================================================================
// TD5. Shrink with output_reclaim_min_size = 0: the holder drops below a tenth of its block and takes a smaller one; predecessors follow, the
// old block is free for the later nodes (X), output is correct.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td5_shrink_reclaim_rebinds_the_predecessors) {
    auto& eng = get_test_engine();
    sw on = BOTH;
    on.reclaim_min = 0;
    sw off = OFF;
    off.reclaim_min = 0;
    // rounds 1 and 4 shrink with a zero-length (SKIP) predecessor: the SKIP predecessor can only follow the holder block through the rebind
    const std::vector<round_t> rounds = {{s3(32, 32, 8, 8), 1}, {s3(1, 0, 2, 2), 2}, {s3(1, 1, 2, 2), 3}, {s3(32, 32, 8, 8), 4}, {s3(0, 1, 1, 1), 5}, {s3(1, 1, 1, 1), 6}};
    const void* prev_holder = nullptr;
    size_t block_changes = 0;
    auto nets = check_rounds(eng, []() { return topo_third_party(); }, on, rounds, ref_third_party(), [&](network& net, size_t i) {
        const void* c = buf_of(net, "C");
        if (prev_holder != nullptr && c != prev_holder) {
            ++block_changes;
        }
        prev_holder = c;
        if (is_fused(net)) {
            EXPECT_EQ(buf_of(net, "P1"), c) << "round " << i;
            EXPECT_EQ(buf_of(net, "P2"), c) << "round " << i;
        }
    }, nullptr, off);
    EXPECT_GT(block_changes, 0u) << "the holder must change its block at least once (shrink)";
}

// ==============================================================================================================================
// TD-A4. The first predecessor has length zero, the second one triggers the holder allocation. The Debug assertion A4 must not fire.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, tda4_first_predecessor_empty_second_triggers_realloc) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s2(4, 4), 1}, {s2(0, 16), 2}, {s2(8, 8), 3}, {s2(0, 32), 4}, {s2(16, 0), 5}};
    check_rounds(eng, []() { return topo_basic(); }, BOTH, rounds, ref_basic());
}

// ==============================================================================================================================
// TD-A2. Block formats are rejected by L1-13 (the last predecessor's channel tail is never written); an aligned plain format is promoted and,
// with the holder block poisoned by NaN before every round, gives the same output as the switch-off network.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, tda2_plain_format_promoted_with_nan_sentinel) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s2(4, 4), 1}, {s2(16, 3), 2}, {s2(3, 29), 3}, {s2(1, 1), 4}, {s2(8, 0), 5}, {s2(7, 7), 6}};
    auto nets = check_rounds(eng, []() { return topo_basic(); }, BOTH, rounds, ref_basic(), nullptr, [&](network& net, size_t) { fill_nan(net, "C"); });
    EXPECT_TRUE(nets.on->get_program()->get_node("C").is_pool_class_root());
}

TEST(dynamic_concat_holder_pool, tda2_blocked_format_is_rejected) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", dyn4()));
        t.add(input_layout("in2", dyn4()));
        t.add(reorder("P1", input_info("in1"), format::b_fs_yx_fsv16, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::b_fs_yx_fsv16, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(reorder("A", input_info("C"), format::bfyx, data_types::f32));
        return t;
    };
    auto t = mk();
    auto net = make_net(eng, t, BOTH);
    auto& prog = *net->get_program();
    std::cout << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_FALSE(prog.get_node("C").get_output_layout().format.block_sizes().empty()) << "precondition: blocked holder format";
    const uint32_t m = mask_of(prog, "C");
    EXPECT_NE(m & class_level::rj_holder_padding, 0u) << class_level::to_string(m);
    EXPECT_EQ(count_roots(prog), 0u) << dump_nodes(prog);
    EXPECT_FALSE(prog.get_node("C").is_pool_class_root());
    const std::vector<round_t> rounds = {{s2(4, 4), 1}, {s2(16, 5), 2}};
    auto t2 = mk();
    auto off = make_net(eng, t2, OFF);
    for (const auto& r : rounds) {
        EXPECT_EQ(run_round(*net, eng, r.s, r.seed), run_round(*off, eng, r.s, r.seed));
    }
}

// ==============================================================================================================================
// TD6a (L1-13). A predecessor with an explicit non-axis output padding: the class is rejected by rj_holder_padding, nothing is flagged,
// outputs are unchanged.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td6a_predecessor_padding_rejects_the_class) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", dyn4()));
        t.add(input_layout("in2", dyn4()));
        auto p1 = activation("P1", input_info("in1"), activation_func::abs);
        p1.output_paddings = {padding({0, 0, 1, 1}, {0, 0, 1, 1})};
        t.add(p1);
        t.add(activation("P2", input_info("in2"), activation_func::relu));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        return t;
    };
    auto t = mk();
    auto net = make_net(eng, t, BOTH);
    auto& prog = *net->get_program();
    std::cout << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << "precondition: the dynamic concat is in-place despite the predecessor padding\n" << dump_nodes(prog);
    ASSERT_TRUE(static_cast<bool>(prog.get_node("P1").get_output_layout().data_padding)) << "precondition: padded predecessor";
    const uint32_t m = mask_of(prog, "C");
    EXPECT_NE(m & class_level::rj_holder_padding, 0u) << class_level::to_string(m);
    EXPECT_EQ(count_roots(prog), 0u);
    EXPECT_EQ(pool_records_of(*net, "C"), 0u);
    auto t2 = mk();
    auto off = make_net(eng, t2, OFF);
    const auto s = s2(4, 4);
    EXPECT_EQ(run_round(*net, eng, s, 1), run_round(*off, eng, s, 1));
}

// ==============================================================================================================================
// TD6b (TB5 runtime analogue). Direct call of allocate_output with a padded layout for a flagged dynamic holder: the memory is not from the
// pool and the former pool record of the holder is released (P1-L1).
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td6b_runtime_gate_downgrades_a_padded_holder_to_direct) {
    auto& eng = get_test_engine();
    auto t = topo_basic();
    auto net = make_net(eng, t, BOTH);
    auto& prog = *net->get_program();
    const auto& c = prog.get_node("C");
    ASSERT_TRUE(c.is_pool_class_root() && c.is_dynamic()) << dump_nodes(prog);
    auto inst = net->get_primitive("C");
    auto params = *inst->get_impl_params();
    const auto plain_pshape = ov::PartialShape{1, 8, 4, 4};
    params.output_layouts[0] = layout{plain_pshape, data_types::f32, format::bfyx};
    memory_restricter<uint32_t> deps(&c.get_memory_dependencies());
    auto& pool = net->get_memory_pool();
    auto first = primitive_inst::allocate_output(eng, pool, c, params, deps, net->get_id(), false, 0, false, false, nullptr, true);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(pool_records_of(*net, "C"), 1u) << "plain layout: the holder allocates from the pool";
    auto padded = params;
    padded.output_layouts[0] = layout{plain_pshape, data_types::f32, format::bfyx, padding({0, 0, 1, 1}, {0, 0, 1, 1})};
    auto second = primitive_inst::allocate_output(eng, pool, c, padded, deps, net->get_id(), false, 0, false, false, first.get(), true);
    ASSERT_NE(second, nullptr);
    EXPECT_FALSE(second->from_memory_pool) << "padded layout: the per-round gate downgrades the route to DIRECT";
    EXPECT_EQ(pool_records_of(*net, "C"), 0u) << "P1-L1: the stale pool record of the holder must be released";
}

// ==============================================================================================================================
// TD-A6. A dynamic holder with an upper bound allocates in the constructor: it goes to the pool as the class root.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, tda6_constructor_time_pool_allocation_of_a_bounded_holder) {
    auto& eng = get_test_engine();
    auto bounded = []() { return layout{ov::PartialShape{1, ov::Dimension(0, 32), 4, 4}, data_types::f32, format::bfyx}; };
    auto mk = [&](bool padded) {
        topology t;
        t.add(input_layout("in1", bounded()));
        t.add(input_layout("in2", bounded()));
        auto p1 = activation("P1", input_info("in1"), activation_func::abs);
        if (padded) {
            p1.output_paddings = {padding({0, 0, 1, 1}, {0, 0, 1, 1})};
        }
        t.add(p1);
        t.add(activation("P2", input_info("in2"), activation_func::relu));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        return t;
    };
    {
        auto t = mk(false);
        auto net = make_net(eng, t, BOTH);
        auto& prog = *net->get_program();
        ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
        ASSERT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
        ASSERT_TRUE(net->get_primitive("C")->output_memory_ptr() != nullptr) << "bounded holder allocates in the constructor";
        EXPECT_EQ(pool_records_of(*net, "C"), 1u) << "constructor time: the holder is a pool user";
        // and the bounded network executes correctly for several rounds
        auto t2 = mk(false);
        auto off = make_net(eng, t2, OFF);
        for (const auto& r : std::vector<round_t>{{s2(4, 4), 1}, {s2(16, 2), 2}, {s2(0, 8), 3}}) {
            const auto got = run_round(*net, eng, r.s, r.seed);
            EXPECT_EQ(got, run_round(*off, eng, r.s, r.seed));
            EXPECT_EQ(got.at("A"), ref_concat2(r.s.at("in1"), r.s.at("in2"), r.seed));
        }
    }
    {
        auto t = mk(true);
        auto net = make_net(eng, t, BOTH);
        auto& prog = *net->get_program();
        EXPECT_FALSE(prog.get_node("C").is_pool_class_root()) << "padded predecessor: not promoted\n" << dump_nodes(prog);
        EXPECT_EQ(pool_records_of(*net, "C"), 0u);
    }
}

// ==============================================================================================================================
// TD10. L1 white box for dynamic classes.
// ==============================================================================================================================
namespace {
program::ptr make_hand(engine& eng, topology& t, const sw& s, const std::vector<std::string>& outs = {}) {
    auto cfg = make_cfg(eng, s);
    cfg.set_property(ov::intel_gpu::optimize_data(false));
    if (!outs.empty()) {
        cfg.set_property(ov::intel_gpu::custom_outputs(outs));
    }
    auto prog = program::build_program(eng, t, cfg, false, /*no_optimizations=*/true, false);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();
    }
    return prog;
}

// in1 -> P1 --+
//             C (axis 1, dynamic, runtime skippable, in place) -> A -> O
// in2 -> P2 --+
topology hand_dyn_topology() {
    topology t;
    t.add(input_layout("in1", dyn4()));
    t.add(input_layout("in2", dyn4()));
    t.add(activation("P1", input_info("in1"), activation_func::relu));
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("A", input_info("C"), activation_func::relu));
    t.add(activation("O", input_info("A"), activation_func::abs));
    return t;
}

program::ptr hand_dyn(engine& eng, topology& t, const sw& s, const std::vector<std::string>& outs = {"O"}) {
    auto prog = make_hand(eng, t, s, outs);
    prog->get_node("C").can_be_optimized(true);
    prog->get_node("C").set_runtime_skippable(true);  // what prepare_buffer_fusing does for a dynamic in-place concat
    for (const auto& dep : prog->get_node("C").get_dependencies()) {
        dep.first->can_share_buffer(false);
    }
    return prog;
}

void expect_no_class_flags(program& p, const std::string& msg) {
    for (auto* n : p.get_processing_order()) {
        EXPECT_FALSE(n->is_pool_class_root()) << msg << ": " << n->id();
        EXPECT_FALSE(n->has_alias_pool_root()) << msg << ": " << n->id();
    }
}
}  // namespace

TEST(dynamic_concat_holder_pool_hand, td10_l1_masks_of_a_dynamic_class) {
    auto& eng = get_test_engine();
    {
        auto t = hand_dyn_topology();
        auto prog = hand_dyn(eng, t, BOTH);
        EXPECT_EQ(mask_of(*prog, "C"), 0u) << class_level::to_string(mask_of(*prog, "C"));
        program_wrapper::prepare_memory_dependencies(*prog);
        EXPECT_EQ(count_roots(*prog), 1u);
        const uint32_t c_uid = static_cast<uint32_t>(prog->get_node("C").get_unique_id());
        EXPECT_TRUE(prog->get_node("C").is_pool_class_root());
        EXPECT_EQ(prog->get_node("C").get_alias_pool_root_uid(), c_uid);
        EXPECT_FALSE(prog->get_node("P1").has_alias_pool_root()) << "dynamic predecessors are never zero-allocation members";
        EXPECT_EQ(route_of(prog->get_node("C")), primitive_inst::alloc_route::POOL_AS_ROOT);
        EXPECT_EQ(route_of(prog->get_node("P1")), primitive_inst::alloc_route::DIRECT);
        EXPECT_EQ(class_level::check_b4_prime(*prog), "");
    }
    {  // sub switch off: exactly the P1-B-S decision
        auto t = hand_dyn_topology();
        auto prog = hand_dyn(eng, t, MAIN_ONLY);
        const uint32_t m = mask_of(*prog, "C");
        // unchanged from P1-B-S: the MAY_SUBREGION slots fail the static member kind rule and the dynamic nodes set rj_dynamic
        EXPECT_EQ(m, class_level::rj_dynamic | class_level::rj_member_kind) << class_level::to_string(m);
        program_wrapper::prepare_memory_dependencies(*prog);
        expect_no_class_flags(*prog, "sub switch off");
    }
    {  // only the sub switch: no-op
        auto t = hand_dyn_topology();
        auto prog = hand_dyn(eng, t, DYN_ONLY);
        EXPECT_NE(mask_of(*prog, "C") & class_level::rj_program_level, 0u);
        program_wrapper::prepare_memory_dependencies(*prog);
        expect_no_class_flags(*prog, "sub switch only");
    }
    {  // MAY_VIEW member of a dynamic class is accepted; a static member or a static reader is rejected with rj_mixed_dynamic
        auto t = hand_dyn_topology();
        t.add(input_layout("sin", layout{ov::PartialShape{1, 4, 4, 4}, data_types::f32, format::bfyx}));
        t.add(activation("S", input_info("sin"), activation_func::relu));
        auto prog = hand_dyn(eng, t, BOTH, {"O", "S"});
        auto aa = alias_analysis::build(*prog);
        const alias_class* c = aa.class_of(prog->get_node("C"));
        ASSERT_NE(c, nullptr);
        const auto never = [](uint32_t) { return false; };
        ASSERT_EQ(class_level::pool_admission_reject_mask(*c, *prog, never), 0u);
        {
            alias_class k = *c;
            for (auto& m : k.members) {
                if (m.node->id() == "P1") {
                    m.kind = alias_kind::MAY_VIEW;
                }
            }
            EXPECT_EQ(class_level::pool_admission_reject_mask(k, *prog, never), 0u) << "a MAY_VIEW member is accepted in a dynamic class";
            for (auto& m : k.members) {
                if (m.node->id() == "P1") {
                    m.kind = alias_kind::MUST_VIEW;
                }
            }
            EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_member_kind, 0u) << "a MUST_* kind cannot occur in a dynamic class";
        }
        {
            alias_class k = *c;
            alias_member sm;
            sm.id = tensor_id{static_cast<uint32_t>(prog->get_node("S").get_unique_id()), tensor_space::OUTPUT, 0};
            sm.kind = alias_kind::MAY_VIEW;
            sm.node = &prog->get_node("S");
            k.members.push_back(sm);
            EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_mixed_dynamic, 0u) << "static member";
        }
        {
            alias_class k = *c;
            k.readers.push_back(&prog->get_node("S"));
            EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_mixed_dynamic, 0u) << "static reader";
        }
        {  // a root that cannot fall back to non-fused execution
            prog->get_node("C").set_runtime_skippable(false);
            EXPECT_NE(class_level::pool_admission_reject_mask(*c, *prog, never) & class_level::rj_root_kind, 0u);
            prog->get_node("C").set_runtime_skippable(true);
        }
    }
    {  // maybe-CPU reader: for a dynamic node it is has_impl_for(cpu), not the preferred impl type. Find the reader types that qualify.
        size_t qualified = 0;
        for (const char* kind : {"reorder", "eltwise", "tile", "reduce", "broadcast", "select"}) {
            auto t = hand_dyn_topology();
            const std::string k = kind;
            if (k == "reorder") {
                t.add(reorder("RD", input_info("C"), format::bfyx, data_types::f16));
            } else if (k == "eltwise") {
                t.add(eltwise("RD", {input_info("C"), input_info("C")}, eltwise_mode::sum));
            } else {
                continue;  // kept to the two cheap constructions below
            }
            auto prog = hand_dyn(eng, t, BOTH, {"O", "RD"});
            const uint32_t m = mask_of(*prog, "C");
            std::cout << "[TD10] reader " << k << " mask=" << class_level::to_string(m) << "\n";
            if (m & class_level::rj_placement) {
                ++qualified;
                program_wrapper::prepare_memory_dependencies(*prog);
                expect_no_class_flags(*prog, "maybe-cpu reader");
            }
        }
        // The dynamic reader types constructible here (reorder, eltwise) report no CPU implementation (has_impl_for(cpu) is false for them), so a
        // real maybe-CPU dynamic reader could not be built in a synthetic graph: the fallback bits are injected into the alias class instead.
        std::cout << "[TD10] reader types with a cpu implementation for a dynamic node: " << qualified << "\n";
        auto t = hand_dyn_topology();
        auto prog = hand_dyn(eng, t, BOTH);
        auto aa = alias_analysis::build(*prog);
        const alias_class* c = aa.class_of(prog->get_node("C"));
        ASSERT_NE(c, nullptr);
        for (uint32_t bit : {static_cast<uint32_t>(fb_maybe_cpu), static_cast<uint32_t>(fb_cpu_impl), static_cast<uint32_t>(fb_shape_of),
                             static_cast<uint32_t>(fb_role_conflict)}) {
            alias_class k = *c;
            k.fallback |= bit;
            EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, [](uint32_t) { return false; }) & class_level::rj_placement, 0u) << "fallback bit " << bit;
        }
    }
    {  // a sure-CPU reader: a static node with the cpu preferred impl type in a (hand) dynamic class is a mixed reader and is rejected anyway
        auto t = hand_dyn_topology();
        t.add(input_layout("sin", layout{ov::PartialShape{1, 4, 4, 4}, data_types::f32, format::bfyx}));
        t.add(activation("S", input_info("sin"), activation_func::relu));
        auto prog = hand_dyn(eng, t, BOTH, {"O", "S"});
        auto aa = alias_analysis::build(*prog);
        const alias_class* c = aa.class_of(prog->get_node("C"));
        ASSERT_NE(c, nullptr);
        alias_class k = *c;
        prog->get_node("S").set_preferred_impl_type(impl_types::cpu);
        k.readers.push_back(&prog->get_node("S"));
        EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, [](uint32_t) { return false; }) & class_level::rj_mixed_dynamic, 0u);
    }
    {
        auto t = hand_dyn_topology();
        t.add(shape_of("SO", input_info("C"), data_types::i32));
        auto prog = hand_dyn(eng, t, BOTH, {"O", "SO"});
        EXPECT_NE(mask_of(*prog, "C") & class_level::rj_placement, 0u);
        program_wrapper::prepare_memory_dependencies(*prog);
        expect_no_class_flags(*prog, "shape_of reader");
    }
    {  // the L1-13 bits on the hand graph: padded slot, padded root, blocked root
        auto t = hand_dyn_topology();
        auto prog = hand_dyn(eng, t, BOTH);
        auto aa = alias_analysis::build(*prog);
        const alias_class* c = aa.class_of(prog->get_node("C"));
        ASSERT_NE(c, nullptr);
        const auto never = [](uint32_t) { return false; };
        EXPECT_FALSE(primitive_inst::holder_layout_is_plain(layout{ov::PartialShape{1, 4, 4, 4}, data_types::f32, format::bfyx, padding({0, 0, 1, 0}, {0, 0, 0, 0})}));
        EXPECT_FALSE(primitive_inst::holder_layout_is_plain(layout{ov::PartialShape{1, 16, 4, 4}, data_types::f32, format::b_fs_yx_fsv16}));
        EXPECT_TRUE(primitive_inst::holder_layout_is_plain(layout{ov::PartialShape{1, 4, 4, 4}, data_types::f32, format::bfyx}));
        // a dynamic padding mask is not padding
        padding dyn_pad({0, 0, 0, 0}, {0, 0, 0, 0}, padding::DynamicDimsMask("0010"));
        EXPECT_TRUE(primitive_inst::holder_layout_is_plain(layout{ov::PartialShape{1, 4, 4, 4}, data_types::f32, format::bfyx, dyn_pad}));
        EXPECT_EQ(class_level::pool_admission_reject_mask(*c, *prog, never), 0u);
    }
}

// ==============================================================================================================================
// TD11 (TB18). I-B1 / I-B3 on the class of a real dynamic program, built from a real restriction table plus a one-way edge injected before the
// generator runs.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool_hand, td11_invariants_on_a_dynamic_class_with_a_one_way_edge) {
    auto& eng = get_test_engine();
    auto t = hand_dyn_topology();
    t.add(input_layout("in3", dyn4()));
    t.add(activation("Z", input_info("in3"), activation_func::relu));
    t.add(eltwise("EE", {input_info("Z"), input_info("A")}, eltwise_mode::sum));
    auto prog = hand_dyn(eng, t, MAIN_ONLY, {"O", "EE"});  // the generator does not promote here (sub switch off): this is plain R0
    program_wrapper::prepare_memory_dependencies(*prog);
    auto aa = alias_analysis::build(*prog);
    const alias_class* c = aa.class_of(prog->get_node("C"));
    ASSERT_NE(c, nullptr);
    std::vector<program_node*> nodes;
    std::map<const program_node*, size_t> idx;
    for (auto* n : prog->get_processing_order()) {
        idx[n] = nodes.size();
        nodes.push_back(n);
    }
    class_level::simple_restriction_store store;
    for (auto* n : nodes) {
        store.uids.push_back(static_cast<uint32_t>(n->get_unique_id()));
        store.sets.push_back(n->get_memory_dependencies());
    }
    // one-way edge: Z lists the member P1, P1 does not list Z
    const uint32_t p1_uid = static_cast<uint32_t>(prog->get_node("P1").get_unique_id());
    store.insert(idx.at(&prog->get_node("Z")), p1_uid);
    auto& p1set = store.sets[idx.at(&prog->get_node("P1"))];
    p1set.erase(std::remove(p1set.begin(), p1set.end(), static_cast<uint32_t>(prog->get_node("Z").get_unique_id())), p1set.end());
    const auto r0 = class_level::snapshot_restrictions(store);
    class_level::promoted_class pc;
    pc.root = idx.at(&prog->get_node("C"));
    for (const auto& m : c->members) {
        pc.members.push_back(idx.at(m.node));
    }
    class_level::materialize_class_restrictions(store, {pc});
    EXPECT_EQ(class_level::check_class_invariants(store, r0, {pc}), "");
    const uint32_t c_uid = static_cast<uint32_t>(prog->get_node("C").get_unique_id());
    EXPECT_TRUE(store.contains(idx.at(&prog->get_node("Z")), c_uid)) << "Z listed the member P1: it must list the root";
    EXPECT_TRUE(store.contains(pc.root, static_cast<uint32_t>(prog->get_node("Z").get_unique_id()))) << "and the root must list Z";
}

// ==============================================================================================================================
// TD-17 (TB17). The predecessor's own uid (its internal buffers) conflicts with the holder block both ways, and the holder block never carries
// another pool user.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td17_member_uid_and_holder_block) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s3(4, 4), 1}, {s3(16, 2), 2}};
    auto nets = check_rounds(eng, []() { return topo_third_party(); }, BOTH, rounds, ref_third_party(), [&](network& net, size_t) {
        if (is_fused(net)) {
            // later (lifetime-disjoint) pool users of the same block are legal; what must hold is that the holder is a record user and that the
            // predecessors (own uid) are not users of it
            const auto uids = net.get_memory_pool().get_user_uids(*net.get_primitive("C")->output_memory_ptr(), net.get_id());
            const size_t c_uid = net.get_primitive("C")->get_node().get_unique_id();
            EXPECT_NE(std::find(uids.begin(), uids.end(), c_uid), uids.end()) << "the holder is a user of its block";
            for (auto id : {"P1", "P2"}) {
                EXPECT_EQ(std::find(uids.begin(), uids.end(), net.get_primitive(id)->get_node().get_unique_id()), uids.end()) << id << " is not a pool user";
            }
        }
    });
    auto& prog = *nets.on->get_program();
    const auto& c = prog.get_node("C");
    for (auto id : {"P1", "P2"}) {
        const auto& m = prog.get_node(id);
        EXPECT_TRUE(m.has_memory_dependency(static_cast<uint32_t>(m.get_unique_id()))) << id << " keeps its own uid";
        EXPECT_TRUE(m.has_memory_dependency(static_cast<uint32_t>(c.get_unique_id()))) << id;
        EXPECT_TRUE(c.has_memory_dependency(static_cast<uint32_t>(m.get_unique_id()))) << id;
    }
}

// ==============================================================================================================================
// TD-§2B (CLAUDE.md section 2, Pattern B dynamic item). The holder restrictions are the union of the predecessor restrictions (fusion is
// assumed to always happen); a third party that is alive next to a predecessor never shares the holder block.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td2b_root_restrictions_cover_all_predecessors) {
    auto& eng = get_test_engine();
    auto t_off = topo_third_party();
    auto off = make_net(eng, t_off, MAIN_ONLY);  // LEGACY restrictions (class rejected: dynamic)
    auto t_on = topo_third_party();
    auto on = make_net(eng, t_on, BOTH);
    auto& po = *on->get_program();
    auto& pf = *off->get_program();
    ASSERT_TRUE(po.get_node("C").is_pool_class_root());
    ASSERT_FALSE(pf.get_node("C").is_pool_class_root());
    for (auto id : {"P1", "P2"}) {
        for (auto u : pf.get_node(id).get_memory_dependencies()) {
            if (u == static_cast<uint32_t>(pf.get_node(id).get_unique_id())) {
                continue;
            }
            EXPECT_TRUE(po.get_node("C").has_memory_dependency(u)) << "R(root) must contain R(" << id << ") member " << u;
        }
    }
    const std::vector<round_t> rounds = {{s3(4, 4), 1}, {s3(8, 8), 2}, {s3(2, 2), 3}};
    for (const auto& r : rounds) {
        EXPECT_EQ(run_round(*on, eng, r.s, r.seed), ref_third_party()(r.s, r.seed));
        EXPECT_TRUE(overlap_violations(*on).empty());
        EXPECT_NE(buf_of(*on, "W"), buf_of(*on, "C"));
    }
}

// ==============================================================================================================================
// TD8 (TB9). Real out_of_order queue: nothing is promoted, restrictions and flags are those of the switch-off build.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td8_real_oooq_is_untouched) {
    auto& eng = get_test_engine();
    sw on = BOTH;
    on.q = QueueTypes::out_of_order;
    sw off = OFF;
    off.q = QueueTypes::out_of_order;
    auto t_on = topo_third_party();
    auto net_on = make_net(eng, t_on, on);
    auto t_off = topo_third_party();
    auto net_off = make_net(eng, t_off, off);
    ASSERT_EQ(net_on->get_program()->get_config().get_queue_type(), QueueTypes::out_of_order);
    ASSERT_EQ(net_off->get_program()->get_config().get_queue_type(), QueueTypes::out_of_order);
    auto& a = *net_on->get_program();
    auto& b = *net_off->get_program();
    EXPECT_EQ(count_roots(a), 0u);
    for (auto* n : a.get_processing_order()) {
        const auto& m = b.get_node(n->id());
        EXPECT_FALSE(n->has_alias_pool_root()) << n->id();
        EXPECT_EQ(n->get_memory_dependencies(), m.get_memory_dependencies()) << n->id();
        EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << n->id();
        EXPECT_EQ(n->can_be_optimized(), m.can_be_optimized()) << n->id();
    }
    const auto s = s3(4, 4);
    EXPECT_EQ(run_round(*net_on, eng, s, 1), run_round(*net_off, eng, s, 1));
}

// ==============================================================================================================================
// TD9 (neutrality). With the sub switch off (main on), and with only the sub switch on, flags, restrictions and the allocation trace over several
// rounds are those of the all-off build.
// ==============================================================================================================================
namespace {
struct trace_row {
    std::string node;
    size_t port;
    uint64_t bytes;
    bool pooled;
    std::string reason;
    bool operator==(const trace_row& o) const {
        return node == o.node && port == o.port && bytes == o.bytes && pooled == o.pooled && reason == o.reason;
    }
};
std::vector<trace_row> trace_of(network& net) {
    std::vector<trace_row> v;
#ifdef GPU_DEBUG_CONFIG
    for (const auto& r : net.get_alloc_trace_records()) {
        v.push_back({r.node_id, r.port, r.bytes, r.pooled, r.reason});
    }
#else
    (void)net;
#endif
    return v;
}
}  // namespace

TEST(dynamic_concat_holder_pool, td9_switches_off_is_neutral) {
    auto& eng = get_test_engine();
    const std::vector<round_t> rounds = {{s3(4, 4), 1}, {s3(16, 16), 2}, {s3(2, 2), 3}, {s3(8, 0), 4}, {s3(4, 4), 5}};
    sw off_t = OFF;
    off_t.trace = 2;
    sw main_t = MAIN_ONLY;
    main_t.trace = 2;
    sw dyn_t = DYN_ONLY;
    dyn_t.trace = 2;
    for (const auto& variant : {main_t, dyn_t}) {
        auto t0 = topo_third_party();
        auto base = make_net(eng, t0, off_t);
        auto t1 = topo_third_party();
        auto net = make_net(eng, t1, variant);
        for (auto* n : net->get_program()->get_processing_order()) {
            const auto& m = base->get_program()->get_node(n->id());
            EXPECT_FALSE(n->is_pool_class_root()) << n->id();
            EXPECT_FALSE(n->has_alias_pool_root()) << n->id();
            EXPECT_EQ(n->get_memory_dependencies(), m.get_memory_dependencies()) << n->id();
            EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << n->id();
            EXPECT_EQ(n->can_be_optimized(), m.can_be_optimized()) << n->id();
            EXPECT_EQ(route_of(*n), route_of(m)) << n->id();
        }
        for (const auto& r : rounds) {
            EXPECT_EQ(run_round(*net, eng, r.s, r.seed), run_round(*base, eng, r.s, r.seed));
        }
        const auto a = trace_of(*net);
        const auto b = trace_of(*base);
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_TRUE(a[i] == b[i]) << "allocation " << i << " differs: " << a[i].node << "/" << b[i].node << " " << a[i].reason << "/" << b[i].reason;
        }
#ifndef GPU_DEBUG_CONFIG
        std::cout << "[TD9] GPU_DEBUG_CONFIG off: allocation trace comparison not available\n";
#endif
    }
}

// ==============================================================================================================================
// TD13 (TB13). Cache round trip of a promoted dynamic class: the flags survive an import in a session whose switches are off (the route follows the
// serialized flags: design behaviour), the version stays 3, the output is correct.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, td13_cache_round_trip_keeps_dynamic_flags_and_version_3) {
    EXPECT_EQ(ov::intel_gpu::CompiledModel::runtime_requirements_version, 3u);
    auto& eng = get_test_engine();
    for (bool import_on : {false, true}) {
        auto t = topo_third_party();
        auto prog = program::build_program(eng, t, make_cfg(eng, BOTH));
        ASSERT_TRUE(prog->get_node("C").is_pool_class_root());
        cldnn::membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            cldnn::BinaryOutputBuffer ob = cldnn::BinaryOutputBuffer(out_mem);
            ob.set_stream(get_test_stream_ptr().get());
            prog->save(ob);
        }
        std::istream in_mem(&mem_buf);
        cldnn::BinaryInputBuffer ib = cldnn::BinaryInputBuffer(in_mem, eng);
        auto imported = std::make_shared<cldnn::program>(eng, make_cfg(eng, import_on ? BOTH : OFF));
        imported->load(ib);
        for (auto* n : prog->get_processing_order()) {
            const auto& m = imported->get_node(n->id());
            EXPECT_EQ(m.is_pool_class_root(), n->is_pool_class_root()) << n->id();
            EXPECT_EQ(m.get_alias_pool_root_uid(), n->get_alias_pool_root_uid()) << n->id();
            EXPECT_EQ(m.get_memory_dependencies(), n->get_memory_dependencies()) << n->id();
        }
        auto net = std::make_shared<network>(imported, 0);
        for (const auto& r : std::vector<round_t>{{s3(4, 4), 1}, {s3(16, 2), 2}, {s3(0, 8), 3}}) {
            EXPECT_EQ(run_round(*net, eng, r.s, r.seed), ref_third_party()(r.s, r.seed)) << "import_on=" << import_on;
            EXPECT_TRUE(overlap_violations(*net).empty());
        }
        EXPECT_EQ(route_of(net->get_program()->get_node("C")), primitive_inst::alloc_route::POOL_AS_ROOT) << "import_on=" << import_on;
        EXPECT_EQ(pool_records_of(*net, "C"), 1u);
    }
}

// ==============================================================================================================================
// TD-A1 / TD-A5. A non-reorder runtime skippable reader (dynamic permute) of the holder: MAY_VIEW member of the class, its users are in R(root),
// and in the rounds where it is not skipped it allocates its own block, which is never the holder block.
// ==============================================================================================================================
namespace {
topology topo_view_reader() {
    topology t;
    t.add(input_layout("in1", dyn4()));
    t.add(input_layout("in2", dyn4()));
    t.add(activation("P1", input_info("in1"), activation_func::abs));
    t.add(activation("P2", input_info("in2"), activation_func::relu));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(permute("V", input_info("C"), {0, 1, 3, 2}));
    t.add(activation("O", input_info("V"), activation_func::relu));
    return t;
}

ref_fn ref_view_reader() {
    return [](const shapes_t& s, int seed) {
        const auto c = ref_concat2(s.at("in1"), s.at("in2"), seed);
        const size_t f = s.at("in1")[1] + s.at("in2")[1], h = s.at("in1")[2], w = s.at("in1")[3];
        std::vector<float> o(c.size());
        for (size_t ff = 0; ff < f; ++ff) {
            for (size_t hh = 0; hh < h; ++hh) {
                for (size_t ww = 0; ww < w; ++ww) {
                    o[(ff * w + ww) * h + hh] = c[(ff * h + hh) * w + ww];
                }
            }
        }
        return outputs_t{{"O", o}};
    };
}
}  // namespace

TEST(dynamic_concat_holder_pool, tda1_non_reorder_skippable_reader_is_a_may_view_member) {
    auto& eng = get_test_engine();
    auto t = topo_view_reader();
    auto net = make_net(eng, t, BOTH);
    auto& prog = *net->get_program();
    std::cout << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("V").is_runtime_skippable()) << "precondition: the dynamic permute is runtime skippable\n" << dump_nodes(prog);
    auto aa = alias_analysis::build(prog);
    const alias_class* c = aa.class_of(prog.get_node("C"));
    ASSERT_NE(c, nullptr);
    bool found = false;
    for (const auto& m : c->members) {
        if (m.node->id() == "V") {
            found = true;
            EXPECT_EQ(m.kind, alias_kind::MAY_VIEW);
        }
    }
    EXPECT_TRUE(found) << "V is a member of the holder class";
    EXPECT_EQ(mask_of(prog, "C", [&](uint32_t u) { return aa.uid_is_ambiguous(u); }), 0u);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root());
    EXPECT_TRUE(prog.get_node("C").has_memory_dependency(static_cast<uint32_t>(prog.get_node("O").get_unique_id())))
        << "the user of the view is in R(root)\n" << dump_nodes(prog);
    // outputs identical to switch off and to the reference
    auto t2 = topo_view_reader();
    auto off = make_net(eng, t2, OFF);
    for (const auto& r : std::vector<round_t>{{s2(4, 4, 4, 4), 1}, {s2(3, 5, 4, 1), 2}, {s2(4, 4, 1, 4), 3}, {s2(2, 2, 3, 2), 4}}) {
        const auto got = run_round(*net, eng, r.s, r.seed);
        EXPECT_EQ(got, run_round(*off, eng, r.s, r.seed));
        EXPECT_EQ(got, ref_view_reader()(r.s, r.seed));
    }
}

TEST(dynamic_concat_holder_pool, tda5_unskipped_view_never_shares_the_holder_block) {
    auto& eng = get_test_engine();
    // (h, w): both > 1 -> V runs; one of them 1 -> V is skipped
    const std::vector<round_t> rounds = {{s2(4, 4, 4, 4), 1}, {s2(4, 4, 4, 1), 2}, {s2(4, 4, 4, 4), 3}, {s2(2, 6, 1, 4), 4}, {s2(8, 8, 3, 3), 5}, {s2(8, 8, 8, 1), 6}};
    std::vector<int> skipped_rounds;
    auto nets = check_rounds(eng, []() { return topo_view_reader(); }, BOTH, rounds, ref_view_reader(), [&](network& net, size_t i) {
        const bool v_skipped = net.get_primitive("V")->can_be_optimized();
        skipped_rounds.push_back(v_skipped ? 1 : 0);
        if (!v_skipped) {
            EXPECT_NE(buf_of(net, "V"), buf_of(net, "C")) << "round " << i << ": the unskipped view allocated the holder block";
        }
    });
    EXPECT_NE(std::count(skipped_rounds.begin(), skipped_rounds.end(), 1), 0) << "at least one round with a skipped V";
    EXPECT_NE(std::count(skipped_rounds.begin(), skipped_rounds.end(), 0), 0) << "at least one round with an executed V";
}

// ==============================================================================================================================
// TD7a. Outputs: a runtime skippable node that is a network output, or a view chain ending in one, puts the class in the output domain.
// Sub switch on and off give bit-identical flags and restrictions.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool_hand, td7a_output_views_keep_the_class_out_of_the_pool) {
    auto& eng = get_test_engine();
    for (int variant = 0; variant < 2; ++variant) {
        auto build = [&](const sw& s) {
            auto t = hand_dyn_topology();
            if (variant == 0) {
                t.add(reorder("R", input_info("C"), format::bfyx, data_types::f32));  // runtime skippable output reorder
            } else {
                t.add(activation("R1", input_info("C"), activation_func::relu));       // view chain C -> R1 -> R2 (output)
                t.add(activation("R2", input_info("R1"), activation_func::abs));
            }
            auto prog = hand_dyn(eng, t, s, variant == 0 ? std::vector<std::string>{"O", "R"} : std::vector<std::string>{"O", "R2"});
            if (variant == 0) {
                prog->get_node("R").can_be_optimized(true);
                prog->get_node("R").set_runtime_skippable(true);
            } else {
                for (auto id : {"R1", "R2"}) {
                    prog->get_node(id).can_be_optimized(true);
                    prog->get_node(id).set_runtime_skippable(true);
                }
            }
            return prog;
        };
        auto on = build(BOTH);
        auto sub_off = build(MAIN_ONLY);
        EXPECT_NE(mask_of(*on, "C") & class_level::rj_placement, 0u) << "variant " << variant << ": " << class_level::to_string(mask_of(*on, "C"));
        program_wrapper::prepare_memory_dependencies(*on);
        program_wrapper::prepare_memory_dependencies(*sub_off);
        expect_no_class_flags(*on, "output view");
        for (auto* n : on->get_processing_order()) {
            EXPECT_EQ(n->get_memory_dependencies(), sub_off->get_node(n->id()).get_memory_dependencies()) << n->id();
            EXPECT_EQ(n->can_share_buffer(), sub_off->get_node(n->id()).can_share_buffer()) << n->id();
        }
    }
}

// ==============================================================================================================================
// TD-rs / test (i). A dynamic strided_slice and a dynamic crop directly as concat predecessors. Records what the class admission decides today.
// ==============================================================================================================================
namespace {
std::string add_slice(topology& t, engine& eng, const std::string& name, const std::string& in, int64_t begin_f, int64_t end_f) {
    auto mk = [&](const std::string& id, std::vector<int64_t> v) {
        auto m = eng.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
        set_values<int64_t>(m, v);
        t.add(cldnn::data(id, m));
    };
    mk(name + "_b", {0, begin_f, 0, 0});
    mk(name + "_e", {1, end_f, 1 << 20, 1 << 20});
    mk(name + "_s", {1, 1, 1, 1});
    t.add(strided_slice(name, input_info(in), input_info(name + "_b"), input_info(name + "_e"), input_info(name + "_s"), {}, {}, {}, {}, {}, {}));
    return name;
}

// x (8 features at runtime) -> S1 = x[:, 0:4], S2 = x[:, 4:8] -> [act1, act2 ->] C -> A
topology topo_slice_concat(engine& eng, bool with_activation) {
    topology t;
    t.add(input_layout("x", dyn4()));
    add_slice(t, eng, "S1", "x", 0, 4);
    add_slice(t, eng, "S2", "x", 4, 8);
    if (with_activation) {
        // an eltwise with a constant (an activation would be fused into the strided_slice and disappear as a node): N1 = S1 + 1, N2 = S2 - 1
        auto k = [&](const std::string& id, float v) {
            auto m = eng.allocate_memory(layout{ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx});
            set_values<float>(m, {v});
            t.add(cldnn::data(id, m));
        };
        k("K1", 1.f);
        k("K2", -1.f);
        t.add(eltwise("N1", {input_info("S1"), input_info("K1")}, eltwise_mode::sum));
        t.add(eltwise("N2", {input_info("S2"), input_info("K2")}, eltwise_mode::sum));
        t.add(concatenation("C", {input_info("N1"), input_info("N2")}, 1));
    } else {
        t.add(concatenation("C", {input_info("S1"), input_info("S2")}, 1));
    }
    t.add(activation("A", input_info("C"), activation_func::relu));
    return t;
}

std::vector<float> ref_slice_concat(const ov::Shape& s, int seed, bool with_activation) {
    const auto x = vals_of(s, seed);
    const size_t half = x.size() / 2;
    std::vector<float> r;
    for (size_t i = 0; i < half; ++i) {
        r.push_back(with_activation ? x[i] + 1.f : x[i]);
    }
    for (size_t i = half; i < x.size(); ++i) {
        r.push_back(with_activation ? x[i] - 1.f : x[i]);
    }
    for (auto& v : r) {
        v = std::max(v, 0.f);  // the reader A (relu)
    }
    return r;
}
}  // namespace

TEST(dynamic_concat_holder_pool, tdrs_strided_slice_predecessor_admission_today) {
    auto& eng = get_test_engine();
    auto t = topo_slice_concat(eng, false);
    auto net = make_net(eng, t, BOTH);
    auto& prog = *net->get_program();
    std::cout << dump_nodes(prog);
    const bool fused = prog.get_node("C").can_be_optimized();
    uint32_t mask = 0;
    if (fused) {
        mask = mask_of(prog, "C");
    }
    std::cout << "[TD-rs] strided_slice predecessors: C in place=" << fused << " mask=" << class_level::to_string(mask)
              << " promoted=" << prog.get_node("C").is_pool_class_root() << " S1 rs=" << prog.get_node("S1").is_runtime_skippable()
              << " S1 cbo=" << prog.get_node("S1").can_be_optimized() << "\n";
    // Decision recorded today (comment for the P1-B-D Step 8 / MiniMax-H3 question Q2): the strided_slice predecessors built here have a real
    // (non no-op) slice, so mark_runtime_skippable_nodes does not mark them (rs=0, cbo=0); the class has no role conflict and is PROMOTED. NOTE: the
    // 'user is an in-place concat' guard of that pass only protects STATIC strided_slice nodes; a DYNAMIC strided_slice of no-op shape (full-range end
    // mask) can be marked cbo+rs even when it feeds a concat (see tdrs_noop_strided_slice_predecessor below). Whether the Slice nodes of the real
    // MiniMax-H3 graph look like this is not known.
    EXPECT_TRUE(fused);
    EXPECT_EQ(mask, 0u) << class_level::to_string(mask);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root());
    EXPECT_FALSE(prog.get_node("S1").is_runtime_skippable());
    // outputs identical to the switch-off network and to the reference whatever the admission decision is
    auto t2 = topo_slice_concat(eng, false);
    auto off = make_net(eng, t2, OFF);
    for (const auto& s : {sh(8, 4, 4), sh(8, 2, 3), sh(8, 1, 1), sh(8, 6, 6)}) {
        const shapes_t in = {{"x", s}};
        const auto got = run_round(*net, eng, in, 3);
        EXPECT_EQ(got, run_round(*off, eng, in, 3));
        EXPECT_EQ(got.at("A"), ref_slice_concat(s, 3, false));
    }
}

TEST(dynamic_concat_holder_pool, tdrs_crop_predecessor_admission_today) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("x", dyn4()));
        t.add(crop("S1", input_info("x"), tensor(1, 4, 1, 1), tensor(0, 0, 0, 0)));
        t.add(crop("S2", input_info("x"), tensor(1, 4, 1, 1), tensor(0, 4, 0, 0)));
        t.add(concatenation("C", {input_info("S1"), input_info("S2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        return t;
    };
    auto t = mk();
    cldnn::network::ptr net;
    try {
        net = make_net(eng, t, BOTH);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "dynamic crop with this construction is not supported: " << e.what();
    }
    auto& prog = *net->get_program();
    std::cout << dump_nodes(prog);
    const bool fused = prog.get_node("C").can_be_optimized();
    uint32_t mask = fused ? mask_of(prog, "C") : 0u;
    std::cout << "[TD-rs] crop predecessors: C in place=" << fused << " mask=" << class_level::to_string(mask) << " promoted=" << prog.get_node("C").is_pool_class_root() << "\n";
    // Decision recorded today: the crop predecessors are plain compute nodes here (rs=0, cbo=0), the class is promoted
    EXPECT_TRUE(fused);
    EXPECT_EQ(mask, 0u) << class_level::to_string(mask);
    auto t2 = mk();
    auto off = make_net(eng, t2, OFF);
    const shapes_t in = {{"x", sh(8, 4, 4)}};
    EXPECT_EQ(run_round(*net, eng, in, 5), run_round(*off, eng, in, 5));
}

// ==============================================================================================================================
// Test (ii). Slice -> activation -> Concat (the Pattern B shape of the MiniMax-H3 AdaLN case) in in_order: the activations are plain compute
// nodes, so the class is admitted although the slices in front of them are runtime skippable views of the (external) input.
// ==============================================================================================================================
TEST(dynamic_concat_holder_pool, tdii_slice_activation_concat) {
    auto& eng = get_test_engine();
    auto mk = [&]() { return topo_slice_concat(eng, true); };
    auto t = mk();
    auto net_probe = make_net(eng, t, BOTH);
    auto& prog = *net_probe->get_program();
    std::cout << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_TRUE(prog.has_node("N1") && prog.has_node("N2")) << "precondition: the eltwise nodes between the slices and the concat exist\n" << dump_nodes(prog);
    const uint32_t mask = mask_of(prog, "C");
    std::cout << "[TD-ii] Slice->act->Concat: mask=" << class_level::to_string(mask) << " promoted=" << prog.get_node("C").is_pool_class_root() << "\n";
    EXPECT_EQ(mask, 0u) << class_level::to_string(mask);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root());
    const std::vector<round_t> rounds = {{{{"x", sh(8, 4, 4)}}, 1}, {{{"x", sh(8, 8, 8)}}, 2}, {{{"x", sh(8, 2, 2)}}, 3}, {{{"x", sh(8, 1, 1)}}, 4}, {{{"x", sh(8, 4, 4)}}, 5}};
    auto nets = check_rounds(eng, mk, BOTH, rounds, [](const shapes_t& s, int seed) { return outputs_t{{"A", ref_slice_concat(s.at("x"), seed, true)}}; },
                             [&](network& net, size_t) { EXPECT_EQ(pool_records_of(net, "N1"), 0u); EXPECT_EQ(pool_records_of(net, "N2"), 0u); });
    EXPECT_EQ(pool_records_of(*nets.on, "C"), 1u);
}

// ==============================================================================================================================
// TD12. A/B on K serial Slice->activation->Concat instances: pooled / off-pool bytes, held snapshot, allocation count; outputs equal.
// ==============================================================================================================================
namespace {
topology topo_serial(engine& eng, size_t k_stages) {
    topology t;
    t.add(input_layout("x", dyn4()));
    std::string cur = "x";
    for (size_t k = 0; k < k_stages; ++k) {
        const std::string sfx = std::to_string(k);
        add_slice(t, eng, "S1_" + sfx, cur, 0, 4);
        add_slice(t, eng, "S2_" + sfx, cur, 4, 8);
        t.add(activation("N1_" + sfx, input_info("S1_" + sfx), activation_func::abs));
        t.add(activation("N2_" + sfx, input_info("S2_" + sfx), activation_func::relu));
        t.add(concatenation("C_" + sfx, {input_info("N1_" + sfx), input_info("N2_" + sfx)}, 1));
        t.add(activation("A_" + sfx, input_info("C_" + sfx), activation_func::relu));
        cur = "A_" + sfx;
    }
    return t;
}

std::vector<float> ref_serial(const ov::Shape& s, int seed, size_t k_stages) {
    auto x = vals_of(s, seed);
    for (size_t k = 0; k < k_stages; ++k) {
        const size_t half = x.size() / 2;
        std::vector<float> r(x.size());
        for (size_t i = 0; i < half; ++i) {
            r[i] = std::max(std::abs(x[i]), 0.f);
        }
        for (size_t i = half; i < x.size(); ++i) {
            r[i] = std::max(x[i], 0.f);
        }
        x = r;
    }
    return x;
}
}  // namespace

TEST(dynamic_concat_holder_pool, td12_ab_serial_slice_activation_concat) {
    auto& eng = get_test_engine();
    const size_t K = 3;
    const std::string last = "A_" + std::to_string(K - 1);
    auto mk = [&]() { return topo_serial(eng, K); };
    const std::vector<round_t> rounds = {{{{"x", sh(8, 4, 4)}}, 1}, {{{"x", sh(8, 8, 8)}}, 2}, {{{"x", sh(8, 2, 2)}}, 3}, {{{"x", sh(8, 4, 4)}}, 4}};
    auto nets = check_rounds(eng, mk, BOTH, rounds, [&](const shapes_t& s, int seed) { return outputs_t{{last, ref_serial(s.at("x"), seed, K)}}; });
    EXPECT_EQ(count_roots(*nets.on->get_program()), K);
    EXPECT_EQ(count_roots(*nets.off->get_program()), 0u);
#ifdef GPU_DEBUG_CONFIG
    const auto s_on = nets.on->get_held_bytes_snapshot();
    const auto s_off = nets.off->get_held_bytes_snapshot();
    const auto m_on = nets.on->get_alloc_trace_summary();
    const auto m_off = nets.off->get_alloc_trace_summary();
    auto row = [&](const char* tag, const held_bytes_snapshot& s, const alloc_trace_summary& m) {
        std::cout << "[AB] TD12 switch=" << tag << " pooled_bytes=" << s.pooled.bytes << " off_pool_bytes=" << s.off_pool_bytes()
                  << " off_pool_inplace_concat_group_bytes=" << s.at_off_pool(alloc_category::inplace_concat_group).bytes << " total_held_bytes=" << s.total_bytes
                  << " allocs=" << m.total_count << " pooled_allocs=" << m.at(alloc_category::pooled).count << "\n";
    };
    row("off", s_off, m_off);
    row("on ", s_on, m_on);
    EXPECT_LE(s_on.at_off_pool(alloc_category::inplace_concat_group).bytes, s_off.at_off_pool(alloc_category::inplace_concat_group).bytes)
        << "the in-place concat group bytes outside the pool must not grow";
#else
    GTEST_SKIP() << "GPU_DEBUG_CONFIG is off: the byte statistics of the A/B run are NOT covered (correctness part passed)";
#endif
}

// ==============================================================================================================================
// oneDNN dependent tests (runtime rejection of the in-place concat through the 64 byte alignment rule of a oneDNN predecessor).
// P1 / P2: reorder f32 -> f16, P2 is forced to a oneDNN preferred impl. P1 with an odd feature count has a byte size that is not a multiple of 64,
// which makes the runtime match() of the dynamic concat fail (prepare_buffer_fusing.cpp: onednn_byte_offset % 64 != 0).
// ==============================================================================================================================
namespace {
topology topo_onednn() {
    topology t;
    t.add(input_layout("in1", dyn4()));
    t.add(input_layout("in2", dyn4()));
    t.add(input_layout("in3", dyn4()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("A", input_info("C"), format::bfyx, data_types::f32));
    t.add(activation("W", input_info("in3"), activation_func::relu));
    t.add(eltwise("E", {input_info("A"), input_info("W")}, eltwise_mode::sum));
    t.add(activation("X", input_info("E"), activation_func::abs));
    t.add(activation("Y", input_info("X"), activation_func::relu));
    return t;
}

ref_fn ref_onednn() {
    return [](const shapes_t& s, int seed) {
        std::vector<float> c = vals_of(s.at("in1"), seed);
        const auto b = vals_of(s.at("in2"), seed + 1);
        c.insert(c.end(), b.begin(), b.end());
        const auto w = vals_of(s.at("in3"), seed + 2);
        for (size_t i = 0; i < c.size(); ++i) {
            c[i] += std::max(w[i], 0.f);
        }
        for (auto& v : c) {
            v = std::max(std::abs(v), 0.f);
        }
        return outputs_t{{"Y", c}};
    };
}

sw onednn_sw(const sw& base) {
    sw s = base;
    s.forced = {{"P2", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn}}};
    return s;
}

bool onednn_available(engine& eng) {
    return eng.get_device_info().supports_immad;
}
}  // namespace

TEST(dynamic_concat_holder_pool_onednn, td3b_non_fused_round_with_a_skip_predecessor) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false: the oneDNN driven runtime rejection is NOT covered";
    }
    // R1 fused (2+2), R2 fused (2+0: P2 SKIP), R3 non-fused: P1 has 9 features (288 bytes, not a multiple of 64) in front of the oneDNN P2, the holder grows
    // R0 non-fused with a non-empty P2 (P2 owns a buffer: count > 0), R1 fused (P2 aliases the holder block, count stays stale),
    // R2 non-fused with a SKIP P2 and a bigger holder (the holder takes a new block), later rounds flip again
    const std::vector<round_t> rounds = {{s3(9, 3), 1}, {s3(4, 4), 2}, {s3(21, 0), 3}, {s3(3, 3), 4}, {s3(2, 0), 5}, {s3(9, 0), 6}, {s3(4, 4), 7}};
    std::vector<int> fused_log;
    size_t checked = 0;
    auto nets = check_rounds(eng, []() { return topo_onednn(); }, onednn_sw(BOTH), rounds, ref_onednn(), [&](network& net, size_t i) {
        const bool fused = is_fused(net);
        fused_log.push_back(fused ? 1 : 0);
        auto p2 = net.get_primitive("P2");
        const bool p2_empty = p2->get_output_layout().is_static() && p2->get_output_layout().count() == 0;
        if (!fused && p2_empty && i == 2) {
            ++checked;
            EXPECT_NE(buf_of(net, "P1"), buf_of(net, "C")) << "non-fused round: P1 owns its buffer";
            EXPECT_EQ(buf_of(net, "P2"), buf_of(net, "C")) << "A3(a): the SKIP predecessor points to the CURRENT holder block ...";
            EXPECT_EQ(p2->get_max_output_layout_count(), 0u) << "... with count 0 (never reused as an own buffer)";
        }
        // (a SKIP predecessor that already aliased an unchanged holder block keeps its stale count in later non-fused rounds: that state is reachable
        // without P1-B-D as well and is guarded by the same-buffer check of realloc_outputs; only the re-pointing of a replaced block clears the count)
    }, nullptr, onednn_sw(OFF));
    std::cout << "[TD3b] fused per round:";
    for (int f : fused_log) {
        std::cout << " " << f;
    }
    std::cout << "\n";
    if (std::count(fused_log.begin(), fused_log.end(), 0) == 0) {
        GTEST_SKIP() << "O1: the runtime rejection could not be triggered on this device/construction: the non-fused rounds are NOT covered";
    }
    EXPECT_GT(checked, 0u);
}

// TD-A3a. As TD3b with the holder block poisoned by NaN before every round and rounds that follow a non-fused round with a SKIP predecessor by a
// non-fused round in which that predecessor is back: a predecessor that kept (or reused) a holder-sized block would corrupt the output.
TEST(dynamic_concat_holder_pool_onednn, tda3a_nan_sentinel_around_the_non_fused_skip_predecessor) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false: the oneDNN driven runtime rejection is NOT covered";
    }
    const std::vector<round_t> rounds = {{s3(2, 2), 1}, {s3(2, 0), 2}, {s3(9, 0), 3}, {s3(9, 3), 4}, {s3(4, 4), 5}, {s3(9, 0), 6}, {s3(3, 9), 7}, {s3(2, 2), 8}};
    check_rounds(eng, []() { return topo_onednn(); }, onednn_sw(BOTH), rounds, ref_onednn(), nullptr, [&](network& net, size_t) { fill_nan(net, "C"); }, onednn_sw(OFF));
}

TEST(dynamic_concat_holder_pool_onednn, td4_fused_nonfused_fused_flip_with_holder_block_change) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false: the oneDNN driven runtime rejection is NOT covered";
    }
    sw on = onednn_sw(BOTH);
    on.reclaim_min = 0;
    sw off = onednn_sw(OFF);
    off.reclaim_min = 0;
    const std::vector<round_t> rounds = {{s3(16, 16, 4, 4), 1}, {s3(3, 0, 1, 1), 2}, {s3(16, 16, 4, 4), 3}, {s3(1, 0, 1, 1), 4}, {s3(5, 5, 2, 2), 5}, {s3(8, 8, 4, 4), 6}};
    std::vector<int> fused_log;
    check_rounds(eng, []() { return topo_onednn(); }, on, rounds, ref_onednn(), [&](network& net, size_t) { fused_log.push_back(is_fused(net) ? 1 : 0); }, nullptr, off);
    std::cout << "[TD4] fused per round:";
    for (int f : fused_log) {
        std::cout << " " << f;
    }
    std::cout << "\n";
    if (std::count(fused_log.begin(), fused_log.end(), 0) == 0 || std::count(fused_log.begin(), fused_log.end(), 1) == 0) {
        GTEST_SKIP() << "O1: no fused/non-fused flip could be triggered: the flip is NOT covered";
    }
}

TEST(dynamic_concat_holder_pool_onednn, tda3b_all_predecessors_own_memory_after_fused_to_nonfused) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false: the oneDNN driven runtime rejection is NOT covered";
    }
    // fused with the big holder, then non-fused (9 + 3: P2 oneDNN behind a 288 byte P1) with a different holder size: every non-empty predecessor must
    // own memory, none may keep the replaced holder block; the Debug assertion A3(b) must not fire
    sw on = onednn_sw(BOTH);
    on.reclaim_min = 0;
    sw off = onednn_sw(OFF);
    off.reclaim_min = 0;
    const std::vector<round_t> rounds = {{s3(16, 16), 1}, {s3(9, 3), 2}, {s3(16, 16), 3}, {s3(9, 21), 4}};
    bool saw_nonfused = false;
    check_rounds(eng, []() { return topo_onednn(); }, on, rounds, ref_onednn(), [&](network& net, size_t) {
        if (!is_fused(net)) {
            saw_nonfused = true;
            EXPECT_NE(buf_of(net, "P1"), buf_of(net, "C"));
            EXPECT_NE(buf_of(net, "P2"), buf_of(net, "C"));
            EXPECT_NE(buf_of(net, "P1"), nullptr);
            EXPECT_NE(buf_of(net, "P2"), nullptr);
        }
    }, nullptr, off);
    if (!saw_nonfused) {
        GTEST_SKIP() << "O1: the runtime rejection could not be triggered: NOT covered";
    }
}

TEST(dynamic_concat_holder_pool_onednn, td7b_third_party_never_shares_the_holder_block_in_non_fused_rounds) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false: the oneDNN driven runtime rejection is NOT covered";
    }
    const std::vector<round_t> rounds = {{s3(2, 2), 1}, {s3(9, 3), 2}, {s3(9, 21), 3}, {s3(4, 4), 4}, {s3(3, 3), 5}};
    bool saw_nonfused = false;
    check_rounds(eng, []() { return topo_onednn(); }, onednn_sw(BOTH), rounds, ref_onednn(), [&](network& net, size_t i) {
        EXPECT_NE(buf_of(net, "W"), buf_of(net, "C")) << "round " << i;
        saw_nonfused = saw_nonfused || !is_fused(net);
    }, nullptr, onednn_sw(OFF));
    if (!saw_nonfused) {
        GTEST_SKIP() << "O1: the runtime rejection could not be triggered: the non-fused rounds are NOT covered";
    }
}

// TD-A3c (B1, review round). P2 (oneDNN) behind P1: round 0 (P1 empty, P2 b) is fused and P2 aliases the holder block; round 1 (P1 9 features: 288
// bytes, P2 loses its 64 byte alignment) is rejected at runtime, the holder needs a bigger block. P2's runtime padding does not change (only the mask),
// so without the fix P2 keeps writing into the old holder block at offset 0 while the holder moved on: the concat reads unwritten bytes.
TEST(dynamic_concat_holder_pool_onednn, tda3c_aliased_predecessor_gets_its_own_buffer_when_the_runtime_concat_is_rejected) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false: the oneDNN driven runtime rejection is NOT covered";
    }
    // NOTE: the order avoids the mirrored LEGACY state (rejected round followed by a fused round with the same, zero, paddings: the predecessor keeps its
    // own buffer while the fused holder reads the holder block; the switch-off network is wrong there too, see P1BD-O3 below)
    const std::vector<round_t> rounds = {{s3(0, 4), 1}, {s3(9, 4), 2}, {s3(4, 4), 3}, {s3(0, 4), 4}, {s3(9, 4), 5}, {s3(4, 4), 6}};
    std::vector<int> fused_log;
    check_rounds(eng, []() { return topo_onednn(); }, onednn_sw(BOTH), rounds, ref_onednn(), [&](network& net, size_t i) {
        fused_log.push_back(is_fused(net) ? 1 : 0);
        if (!is_fused(net)) {
            EXPECT_NE(buf_of(net, "P1"), buf_of(net, "C")) << "round " << i;
            EXPECT_NE(buf_of(net, "P2"), buf_of(net, "C")) << "round " << i << ": P2 must own a buffer, not alias the holder block";
        }
    }, [&](network& net, size_t) { fill_nan(net, "C"); }, onednn_sw(OFF), /*compare_off=*/false);  // the LEGACY output is wrong in this sequence (P1BD-O3)
    std::cout << "[TD-A3c] fused per round:";
    for (int f : fused_log) {
        std::cout << " " << f;
    }
    std::cout << "\n";
    if (std::count(fused_log.begin(), fused_log.end(), 0) == 0) {
        GTEST_SKIP() << "O1: the runtime rejection could not be triggered: NOT covered";
    }
}

// C4. A dynamic reader forced to the cpu impl is "maybe CPU" (has_impl_for returns true for forced impls): the class carries fb_maybe_cpu and is rejected.
TEST(dynamic_concat_holder_pool_hand, c4_forced_cpu_reader_is_maybe_cpu) {
    auto& eng = get_test_engine();
    sw s = BOTH;
    s.forced = {{"A", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::cpu}}};
    auto t = hand_dyn_topology();
    auto prog = hand_dyn(eng, t, s);
    auto aa = alias_analysis::build(*prog);
    const alias_class* c = aa.class_of(prog->get_node("C"));
    ASSERT_NE(c, nullptr);
    EXPECT_NE(c->fallback & fb_maybe_cpu, 0u) << "fallback=" << c->fallback;
    const uint32_t m = class_level::pool_admission_reject_mask(*c, *prog, [](uint32_t) { return false; });
    EXPECT_NE(m & class_level::rj_placement, 0u) << class_level::to_string(m);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "forced cpu reader");
}

// C5. A DYNAMIC strided_slice of no-op shape (full-range end mask, zero begin) as a concat predecessor. Records what the passes do.
TEST(dynamic_concat_holder_pool, tdrs_noop_strided_slice_predecessor) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("x1", dyn4()));
        t.add(input_layout("x2", dyn4()));
        auto mkd = [&](const std::string& id, std::vector<int64_t> v) {
            auto m = eng.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
            set_values<int64_t>(m, v);
            t.add(cldnn::data(id, m));
        };
        mkd("nb", {0, 0, 0, 0});
        mkd("ne", {0, 0, 0, 0});
        mkd("ns", {1, 1, 1, 1});
        t.add(strided_slice("S1", input_info("x1"), input_info("nb"), input_info("ne"), input_info("ns"), {0, 0, 0, 0}, {1, 1, 1, 1}, {}, {}, {}, {}));
        t.add(activation("P2", input_info("x2"), activation_func::relu));
        t.add(concatenation("C", {input_info("S1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        return t;
    };
    auto t = mk();
    auto net = make_net(eng, t, BOTH);
    auto& prog = *net->get_program();
    std::cout << dump_nodes(prog);
    const bool s1_rs = prog.get_node("S1").is_runtime_skippable();
    const bool s1_cbo = prog.get_node("S1").can_be_optimized();
    const bool c_fused = prog.get_node("C").can_be_optimized();
    uint32_t mask = c_fused ? mask_of(prog, "C") : 0u;
    std::cout << "[TD-rs noop] S1 rs=" << s1_rs << " cbo=" << s1_cbo << " C in place=" << c_fused << " promoted=" << prog.get_node("C").is_pool_class_root()
              << " mask=" << class_level::to_string(mask) << "\n";
    // what the code really does (asserted from the observed run): see the printed line; the invariant that must hold in every case is that a
    // class with a view / skippable predecessor is never promoted and that outputs equal the switch-off network
    if (s1_rs || s1_cbo) {
        EXPECT_FALSE(prog.get_node("C").is_pool_class_root()) << "a runtime skippable predecessor is a role conflict";
        if (c_fused) {
            EXPECT_NE(mask & class_level::rj_placement, 0u) << class_level::to_string(mask);
        }
    }
    auto t2 = mk();
    auto off = make_net(eng, t2, OFF);
    const shapes_t in = {{"x1", sh(4, 4, 4)}, {"x2", sh(4, 4, 4)}};
    EXPECT_EQ(run_round(*net, eng, in, 3), run_round(*off, eng, in, 3));
}

// ==============================================================================================================================
// Observation tests (CLAUDE.md rule: DISABLED characterisation only)
// ==============================================================================================================================
// P1BD-O1 (candidate, reachability not established). When the concat output shape does not change, do_runtime_in_place_concat returns before the
// padding update. If the predecessor lengths change in opposite directions with an unchanged total, the slot offsets (paddings) stay stale.
TEST(dynamic_concat_holder_pool, DISABLED_dynamic_concat_compensating_pred_lengths_keep_stale_paddings) {
    auto& eng = get_test_engine();
    // switch off: this is an existing behaviour of the runtime in-place concat, independent of the pool
    const std::vector<round_t> rounds = {{s2(6, 2), 1}, {s2(2, 6), 2}, {s2(4, 4), 3}, {s2(7, 1), 4}};
    auto t = topo_basic();
    auto net = make_net(eng, t, OFF);
    for (const auto& r : rounds) {
        EXPECT_EQ(run_round(*net, eng, r.s, r.seed), ref_basic()(r.s, r.seed)) << "compensating lengths";
    }
}

// P1BD-O2 (waste, not a numeric defect). A dynamic concat that never fuses at runtime keeps all its predecessors permanently can_share_buffer(false):
// they allocate directly instead of using the pool. Characterisation: with a predecessor that is always oneDNN-rejected the predecessors stay off pool.
TEST(dynamic_concat_holder_pool, DISABLED_never_fused_dynamic_concat_preds_stay_off_pool) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false";
    }
    auto t = topo_onednn();
    auto net = make_net(eng, t, onednn_sw(BOTH));
    for (const auto& r : std::vector<round_t>{{s3(9, 3), 1}, {s3(9, 21), 2}}) {
        run_round(*net, eng, r.s, r.seed);
        ASSERT_FALSE(is_fused(*net));
        EXPECT_FALSE(net->get_program()->get_node("P1").can_share_buffer()) << "P1 is permanently off the pool (prepare_buffer_fusing sets can_share_buffer(false))";
        EXPECT_EQ(pool_records_of(*net, "P1"), 0u);
        EXPECT_EQ(pool_records_of(*net, "P2"), 0u);
    }
}

// P1BD-O3 (candidate, LEGACY half of B1, switch OFF). Round 0 (P1 empty, P2 16 features) is fused: P2 aliases the holder block. Round 1 (P1 9 features,
// P2 4 features) is rejected at runtime (P2 oneDNN loses its 64 byte alignment) while the holder keeps its block: P2 keeps writing into the holder block
// at [0, b) and the concat kernel reads it from there while writing [0, 9 + b) of the same buffer in one kernel: read/write overlap. (With a block change
// the predecessor keeps its own old DIRECT block, which should be correct.) Not a P1-B-D regression; characterisation only. Observed while writing
// TD-A3c: the switch-OFF network gives a wrong P2 part (first diff at the P2 slot of the (9,4) round after the (0,4) round) even though the holder
// changes its block there, i.e. the 'correct with a block change' half does not hold on this device either (TD-A3c compares only the switch-on output).
TEST(dynamic_concat_holder_pool_onednn, DISABLED_legacy_rejected_runtime_concat_predecessor_aliases_holder_block) {
    auto& eng = get_test_engine();
    if (!onednn_available(eng)) {
        GTEST_SKIP() << "supports_immad is false";
    }
    auto t = topo_onednn();
    auto net = make_net(eng, t, onednn_sw(OFF));
    const std::vector<round_t> rounds = {{s3(0, 16), 1}, {s3(9, 4), 2}, {s3(0, 4), 3}};  // round 2: the mirrored state (rejected -> fused, paddings equal)
    for (size_t i = 0; i < rounds.size(); ++i) {
        const auto got = run_round(*net, eng, rounds[i].s, rounds[i].seed);
        EXPECT_EQ(got, ref_onednn()(rounds[i].s, rounds[i].seed)) << "round " << i << " fused=" << is_fused(*net);
        if (i == 1) {  // (round 2 is the mirrored state: the output differs from the reference in the switch-off network as well)
            EXPECT_FALSE(is_fused(*net)) << "the runtime rejection did not trigger";
            EXPECT_NE(buf_of(*net, "P2"), buf_of(*net, "C")) << "the rejected-round predecessor must not alias the holder block";
        }
    }
}
