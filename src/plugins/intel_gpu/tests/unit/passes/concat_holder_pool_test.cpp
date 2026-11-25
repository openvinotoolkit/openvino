// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P1-B-S: static Concat holder admission to the memory pool (class-level restriction generator).
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "class_level_memory_dependencies.h"
#include "concatenation_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/pooling.hpp"
#include "alias_analysis.h"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/assign.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "primitive_inst.h"
#include "program_node.h"
#include "program_wrapper.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

layout in_l(size_t f = 16) {
    return layout{ov::PartialShape{1, static_cast<int64_t>(f), 32, 32}, data_types::f32, format::bfyx};
}

ExecutionConfig make_cfg(engine& eng, bool holder_pool, QueueTypes q = QueueTypes::in_order, size_t trace_level = 1) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (holder_pool) {
        config.set_property(ov::intel_gpu::enable_concat_holder_pool(true));
    }
    if (q == QueueTypes::out_of_order) {
        // see T-OOOQ: oneDNN forces in_order, so it has to be switched off by a user property
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
#ifdef GPU_DEBUG_CONFIG
    config.set_user_property({ov::intel_gpu::dump_alloc_trace(trace_level)});
#endif
    return config;
}

[[maybe_unused]] std::vector<float> vals(size_t n, int seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<float>((static_cast<int>(i) * 5 + seed * 11) % 13) - 6.f;  // small integers, exact in f16
    }
    return v;
}

[[maybe_unused]] std::string dump_nodes(program& p) {
    std::ostringstream os;
    for (auto* n : p.get_processing_order()) {
        os << "  " << n->id() << " uid=" << n->get_unique_id() << " type=" << n->get_primitive()->type_string() << " cbo=" << n->can_be_optimized()
           << " cbs=" << n->can_share_buffer() << " out=" << n->is_output() << " root=" << n->is_pool_class_root()
           << " aroot=" << (n->has_alias_pool_root() ? static_cast<int64_t>(n->get_alias_pool_root_uid()) : -1) << " R=[";
        for (auto r : n->get_memory_dependencies()) {
            os << r << ",";
        }
        os << "]\n";
    }
    return os.str();
}


// -------------------------------------------------------------------------------------------------------------------------------
// Execution, statistics, oracle
// -------------------------------------------------------------------------------------------------------------------------------
struct run_input {
    std::string name;
    ov::Shape shape;
    int seed;
};

std::vector<float> read_f32(memory::ptr m) {
    std::vector<float> r(m->get_layout().count());
    if (m->get_layout().data_type == data_types::f16) {
        mem_lock<ov::float16, mem_lock_type::read> lock(m, get_test_stream());
        for (size_t i = 0; i < r.size(); i++) {
            r[i] = static_cast<float>(lock[i]);
        }
    } else {
        mem_lock<float, mem_lock_type::read> lock(m, get_test_stream());
        for (size_t i = 0; i < r.size(); i++) {
            r[i] = lock[i];
        }
    }
    return r;
}

// Executes the network and returns the outputs by id (f32 or f16 converted to float).
std::map<std::string, std::vector<float>> run_net(network& net, engine& eng, const std::vector<run_input>& inputs) {
    for (const auto& in : inputs) {
        auto m = eng.allocate_memory(layout{ov::PartialShape(in.shape), data_types::f32, format::bfyx});
        set_values(m, vals(ov::shape_size(in.shape), in.seed));
        net.set_input_data(in.name, m);
    }
    auto outs = net.execute();
    std::map<std::string, std::vector<float>> res;
    for (auto& o : outs) {
        res[o.first] = read_f32(o.second.get_memory());
    }
    return res;
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

// Pattern A/B oracle (simulated execution order = processing order of a static in_order program): two alias classes that live in the same
// buffer must have disjoint [first member, last reader] intervals. Returns the violations (empty == ok).
std::vector<std::string> lifetime_violations(network& net) {
    std::vector<std::string> res;
    auto prog = net.get_program();
    const auto* aa = prog->get_alias_analysis();
    std::unique_ptr<alias_analysis> local_aa;  // an imported program does not run prepare_memory_dependencies: build the analysis here
    if (aa == nullptr) {
        local_aa = std::make_unique<alias_analysis>(alias_analysis::build(*prog));
        aa = local_aa.get();
    }
    std::map<const program_node*, size_t> idx;
    size_t i = 0;
    for (auto* n : prog->get_processing_order()) {
        idx[n] = i++;
    }
    struct cls_info {
        const void* buf;
        size_t birth, death;
        std::string name;
    };
    std::vector<cls_info> infos;
    for (const auto& c : aa->classes()) {
        if (c.domain & (dm_external_root | dm_persistent)) {
            continue;
        }
        const void* buf = buf_of(net, c.root_node->id());
        if (buf == nullptr) {
            continue;
        }
        size_t birth = std::numeric_limits<size_t>::max(), death = 0;
        for (const auto& m : c.members) {
            birth = std::min(birth, idx.at(m.node));
            death = std::max(death, idx.at(m.node));
            for (const auto* u : m.node->get_users()) {
                death = std::max(death, idx.at(u));
            }
        }
        infos.push_back({buf, birth, death, c.root_node->id()});
    }
    for (size_t a = 0; a < infos.size(); a++) {
        for (size_t b = a + 1; b < infos.size(); b++) {
            if (infos[a].buf != infos[b].buf) {
                continue;
            }
            if (!(infos[a].death < infos[b].birth || infos[b].death < infos[a].birth)) {
                res.push_back("classes " + infos[a].name + " [" + std::to_string(infos[a].birth) + "," + std::to_string(infos[a].death) + "] and " +
                              infos[b].name + " [" + std::to_string(infos[b].birth) + "," + std::to_string(infos[b].death) + "] share a buffer");
            }
        }
    }
    return res;
}

struct ab_stats {
    size_t promoted = 0;
    uint64_t pooled_bytes = 0, off_pool_bytes = 0, holder_group_bytes = 0, total_bytes = 0, device_bytes = 0;
    size_t alloc_count = 0, pooled_alloc_count = 0;
    bool gpu_debug = false;
    std::map<std::string, std::vector<float>> out;
    std::vector<std::string> violations;
};

ab_stats collect(network& net, engine& eng, const std::vector<run_input>& inputs) {
    ab_stats s;
    s.promoted = count_roots(*net.get_program());
    s.out = run_net(net, eng, inputs);
    s.violations = lifetime_violations(net);
#ifdef GPU_DEBUG_CONFIG
    s.gpu_debug = true;
    const auto snap = net.get_held_bytes_snapshot();
    s.pooled_bytes = snap.pooled.bytes;
    s.off_pool_bytes = snap.off_pool_bytes();
    s.holder_group_bytes = snap.at_off_pool(alloc_category::inplace_concat_group).bytes;
    s.total_bytes = snap.total_bytes;
    s.device_bytes = snap.device_bytes;
    const auto sm = net.get_alloc_trace_summary();
    s.alloc_count = sm.total_count;
    s.pooled_alloc_count = sm.at(alloc_category::pooled).count;
#endif
    return s;
}

void print_ab(const std::string& name, const ab_stats& off, const ab_stats& on) {
    auto row = [&](const char* sw, const ab_stats& s) {
        std::cout << "[AB] " << name << " switch=" << sw << " promoted=" << s.promoted << " pooled_bytes=" << s.pooled_bytes
                  << " off_pool_bytes=" << s.off_pool_bytes << " off_pool_inplace_concat_group_bytes=" << s.holder_group_bytes
                  << " total_held_bytes=" << s.total_bytes << " allocs=" << s.alloc_count << " pooled_allocs=" << s.pooled_alloc_count
                  << (s.gpu_debug ? "" : " (GPU_DEBUG_CONFIG off: byte stats n/a)") << "\n";
    };
    row("off", off);
    row("on ", on);
    std::cout << "[AB] " << name << " outputs_elementwise_equal=" << (off.out == on.out ? "yes" : "NO") << "\n" << std::flush;
}

// Builds the same topology with the switch off and on, executes both, checks outputs and lifetimes and prints the A/B row.
struct ab_result {
    network::ptr off_net, on_net;
    ab_stats off, on;
};

ab_result run_ab(const std::string& name, engine& eng, const std::function<topology()>& make_topology, const std::vector<run_input>& inputs,
                 QueueTypes q = QueueTypes::in_order) {
    ab_result r;
    // a topology is consumed by the program build (graph initialization rewrites its primitives): one instance per build
    auto t_off = make_topology();
    auto t_on = make_topology();
    r.off_net = get_network(eng, t_off, make_cfg(eng, false, q), get_test_stream_ptr(), false);
    r.on_net = get_network(eng, t_on, make_cfg(eng, true, q), get_test_stream_ptr(), false);
    r.off = collect(*r.off_net, eng, inputs);
    r.on = collect(*r.on_net, eng, inputs);
    print_ab(name, r.off, r.on);
    EXPECT_EQ(r.off.out, r.on.out) << name << ": outputs differ between switch off and on";
    EXPECT_TRUE(r.off.violations.empty()) << name << " (off): " << (r.off.violations.empty() ? "" : r.off.violations[0]);
    EXPECT_TRUE(r.on.violations.empty()) << name << " (on): " << (r.on.violations.empty() ? "" : r.on.violations[0]);
    return r;
}

memory::ptr const_f16(engine& eng, size_t f, float v) {
    auto m = eng.allocate_memory(layout{ov::PartialShape{1, static_cast<int64_t>(f), 32, 32}, data_types::f16, format::bfyx});
    set_values(m, std::vector<ov::float16>(m->count(), ov::float16(v)));
    return m;
}

// in1 -> P1 (reorder f16) --+
//                           C (axis 1) -> A (relu) -> B (sum with const) -> out (reorder f32, network output)
// in2 -> P2 (reorder f16) --+
// All of C, A, B have the same size. In the legacy path the holder C is a direct allocation; with the class root in the pool, B (which only
// has to avoid A) can take the block of C.
topology topo_chain(engine& eng) {
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(input_layout("in2", in_l()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("A", input_info("C"), activation_func::relu));
    t.add(cldnn::data("kb", const_f16(eng, 32, 1.f)));
    t.add(eltwise("B", {input_info("A"), input_info("kb")}, eltwise_mode::sum));
    t.add(reorder("out", input_info("B"), format::bfyx, data_types::f32));
    return t;
}


// "Hand" graph (no optimization passes): flags are set by the test, selected impls are not chosen, restrictions come from
// program_wrapper::prepare_memory_dependencies(). The switch is part of the program config.
program::ptr make_hand(engine& eng, topology& t, bool holder_pool, QueueTypes q = QueueTypes::in_order, bool internal = false, bool body = false,
                       const std::vector<std::string>& outs = {}) {
    auto cfg = make_cfg(eng, holder_pool, q);
    cfg.set_property(ov::intel_gpu::optimize_data(false));
    if (!outs.empty()) {
        cfg.set_property(ov::intel_gpu::custom_outputs(outs));
    }
    auto prog = program::build_program(eng, t, cfg, internal, /*no_optimizations=*/true, body);
    for (auto* n : prog->get_processing_order()) {
        n->get_output_layouts(false);
        n->set_unique_id();  // compile_graph assigns the ids; without that stage every node would have id 0
    }
    return prog;
}

// in1 -> P1 --+
//             C (axis 1) -> A -> O
// in2 -> P2 --+
topology hand_concat_topology() {
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(input_layout("in2", in_l()));
    t.add(activation("P1", input_info("in1"), activation_func::relu));
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(activation("A", input_info("C"), activation_func::relu));
    t.add(activation("O", input_info("A"), activation_func::abs));
    return t;
}

}  // namespace

// ===============================================================================================================================
// TB1. The holder enters the pool as the class root, a later node shares its block, bytes go down, outputs are identical.
// ===============================================================================================================================
TEST(concat_holder_pool, tb1_holder_enters_pool_and_later_node_shares_its_block) {
    auto& eng = get_test_engine();
    auto mk = [&]() { return topo_chain(eng); };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB1", eng, mk, ins);
    auto& off = *r.off_net;
    auto& on = *r.on_net;
    [[maybe_unused]] const auto holder_bytes = on.get_program()->get_node("C").get_output_layout().bytes_count();

    // off: legacy behaviour, nothing touched
    EXPECT_EQ(count_roots(*off.get_program()), 0u);
    EXPECT_EQ(route_of(off.get_program()->get_node("C")), primitive_inst::alloc_route::DIRECT);
    EXPECT_EQ(pool_records_of(off, "C"), 0u);
    EXPECT_NE(buf_of(off, "B"), buf_of(off, "C")) << "legacy: the holder is outside the pool, nobody can take its block";

    // on: C is the only pool user of its class, P1/P2 allocate nothing and live in its buffer, B reuses the block of the dead holder
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized());
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root());
    EXPECT_EQ(route_of(prog.get_node("C")), primitive_inst::alloc_route::POOL_AS_ROOT);
    const uint32_t c_uid = static_cast<uint32_t>(prog.get_node("C").get_unique_id());
    for (auto id : {"P1", "P2"}) {
        EXPECT_EQ(prog.get_node(id).get_alias_pool_root_uid(), c_uid) << id;
        EXPECT_EQ(route_of(prog.get_node(id)), primitive_inst::alloc_route::ZERO_ALLOC_MEMBER) << id;
        EXPECT_FALSE(on.get_primitive(id)->mem_allocated()) << id;
        EXPECT_EQ(buf_of(on, id), buf_of(on, "C")) << id << " must live in the holder buffer";
    }
    EXPECT_EQ(pool_records_of(on, "C"), 1u);
    EXPECT_EQ(buf_of(on, "B"), buf_of(on, "C")) << "B must reuse the pool block of the dead holder";
    EXPECT_NE(buf_of(on, "A"), buf_of(on, "C"));
    EXPECT_EQ(r.on.promoted, 1u);
#ifdef GPU_DEBUG_CONFIG
    EXPECT_EQ(r.off.holder_group_bytes, holder_bytes);
    EXPECT_EQ(r.on.holder_group_bytes, 0u);
    EXPECT_EQ(r.off.total_bytes - r.on.total_bytes, holder_bytes) << "exactly one holder block is saved";
    EXPECT_EQ(r.off.alloc_count - r.on.alloc_count, 2u) << "the transient direct allocations of the reorder predecessors disappear";
#endif
    // I-B3 / TB17 on the real graph: members keep their restrictions (own uid included) and gain the root uid; the root gains the member uids
    for (auto id : {"P1", "P2"}) {
        const auto& m = prog.get_node(id);
        EXPECT_TRUE(m.has_memory_dependency(static_cast<uint32_t>(m.get_unique_id()))) << id << " keeps its own uid (internal buffers)";
        EXPECT_TRUE(m.has_memory_dependency(c_uid)) << id;
        EXPECT_TRUE(prog.get_node("C").has_memory_dependency(static_cast<uint32_t>(m.get_unique_id()))) << id << " is restricted for the root";
    }
}

// ===============================================================================================================================
// TB2. Pattern A (parallel branches): two holders of structurally unreachable branches. The restrictions between them stay conservative
// (they are mutually restricted, so they get different blocks); the oracle (simulated execution order) finds no lifetime overlap in a shared buffer.
// ===============================================================================================================================
TEST(concat_holder_pool, tb2_parallel_branches_two_holders) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        for (auto id : {"in1", "in2", "in3", "in4"}) {
            t.add(input_layout(id, in_l()));
        }
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(reorder("Q1", input_info("in3"), format::bfyx, data_types::f16));
        t.add(reorder("Q2", input_info("in4"), format::bfyx, data_types::f16));
        t.add(concatenation("C1", {input_info("P1"), input_info("P2")}, 1));
        t.add(concatenation("C2", {input_info("Q1"), input_info("Q2")}, 1));
        t.add(activation("A1", input_info("C1"), activation_func::relu));
        t.add(activation("A2", input_info("C2"), activation_func::abs));
        t.add(eltwise("E", {input_info("A1"), input_info("A2")}, eltwise_mode::sum));
        t.add(reorder("out", input_info("E"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 16, 32, 32}, 3}, {"in4", {1, 16, 32, 32}, 4}};
    auto r = run_ab("TB2", eng, mk, ins);
    auto& prog = *r.on_net->get_program();
    EXPECT_EQ(r.on.promoted, 2u) << dump_nodes(prog);
    const auto& c1 = prog.get_node("C1");
    const auto& c2 = prog.get_node("C2");
    EXPECT_TRUE(c1.is_pool_class_root() && c2.is_pool_class_root());
    EXPECT_TRUE(c1.has_memory_dependency(static_cast<uint32_t>(c2.get_unique_id())) && c2.has_memory_dependency(static_cast<uint32_t>(c1.get_unique_id())))
        << "I-B1 across classes: the two roots must restrict each other\n"
        << dump_nodes(prog);
    // members of one branch are restricted against the other root as well (I-B1, (3'))
    for (auto id : {"Q1", "Q2"}) {
        EXPECT_TRUE(prog.get_node(id).has_memory_dependency(static_cast<uint32_t>(c2.get_unique_id()))) << id;
    }
    EXPECT_NE(buf_of(*r.on_net, "C1"), buf_of(*r.on_net, "C2"));
}

// ===============================================================================================================================
// TB3. A predecessor with a second (plain) user: it is a zero-allocation slot nevertheless; the second user reads the slot.
// ===============================================================================================================================
TEST(concat_holder_pool, tb3_two_user_predecessor) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        t.add(activation("X", input_info("P1"), activation_func::abs));  // second user of P1
        t.add(reorder("out2", input_info("X"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB3", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_EQ(prog.get_node("P1").get_users().size(), 2u);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
    EXPECT_EQ(route_of(prog.get_node("P1")), primitive_inst::alloc_route::ZERO_ALLOC_MEMBER) << "a plain second user does not prevent the zero allocation";
    EXPECT_EQ(buf_of(on, "P1"), buf_of(on, "C"));
    EXPECT_TRUE(r.off_net->get_primitive("P1")->mem_allocated()) << "legacy: the two-user predecessor is allocated directly and overwritten by the holder";
    EXPECT_FALSE(on.get_primitive("P1")->mem_allocated());
}

// ===============================================================================================================================
// TB4. Cascaded concats: the inner holder is a slot of the outer one (single user and two users).
// ===============================================================================================================================
namespace {
topology topo_cascade(bool inner_two_users) {
    topology t;
    for (auto id : {"in1", "in2", "in3"}) {
        t.add(input_layout(id, in_l()));
    }
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(reorder("P3", input_info("in3"), format::bfyx, data_types::f16));
    t.add(concatenation("Cin", {input_info("P1"), input_info("P2")}, 1));
    t.add(concatenation("Cout", {input_info("Cin"), input_info("P3")}, 1));
    t.add(activation("A", input_info("Cout"), activation_func::relu));
    t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
    if (inner_two_users) {
        t.add(activation("X", input_info("Cin"), activation_func::abs));
        t.add(reorder("out2", input_info("X"), format::bfyx, data_types::f32));
    }
    return t;
}
}  // namespace

TEST(concat_holder_pool, tb4a_cascade_single_user_inner) {
    auto& eng = get_test_engine();
    auto mk = []() { return topo_cascade(false); };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 16, 32, 32}, 3}};
    auto r = run_ab("TB4a", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("Cin").can_be_optimized() && prog.get_node("Cout").can_be_optimized()) << dump_nodes(prog);
    EXPECT_EQ(r.on.promoted, 1u) << "only the outer holder is a class root\n" << dump_nodes(prog);
    EXPECT_TRUE(prog.get_node("Cout").is_pool_class_root());
    EXPECT_FALSE(prog.get_node("Cin").is_pool_class_root());
    const uint32_t root = static_cast<uint32_t>(prog.get_node("Cout").get_unique_id());
    for (auto id : {"Cin", "P1", "P2", "P3"}) {
        EXPECT_EQ(prog.get_node(id).get_alias_pool_root_uid(), root) << id;
        EXPECT_EQ(buf_of(on, id), buf_of(on, "Cout")) << id;
    }
}

TEST(concat_holder_pool, tb4b_cascade_two_user_inner) {
    auto& eng = get_test_engine();
    auto mk = []() { return topo_cascade(true); };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 16, 32, 32}, 3}};
    auto r = run_ab("TB4b", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("Cin").can_be_optimized() && prog.get_node("Cout").can_be_optimized()) << dump_nodes(prog);
    ASSERT_EQ(prog.get_node("Cin").get_users().size(), 2u);
    EXPECT_EQ(r.on.promoted, 1u) << dump_nodes(prog);
    const uint32_t root = static_cast<uint32_t>(prog.get_node("Cout").get_unique_id());
    EXPECT_EQ(prog.get_node("Cin").get_alias_pool_root_uid(), root);
    EXPECT_EQ(buf_of(on, "Cin"), buf_of(on, "Cout"));
    EXPECT_EQ(buf_of(on, "P1"), buf_of(on, "Cout"));
}

// ===============================================================================================================================
// TB5. Holder with its own padding (E10): the class is promoted, the padded holder layout allocates from the pool, outputs are identical.
// ===============================================================================================================================
TEST(concat_holder_pool, tb5_padded_holder) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        auto c = concatenation("C", {input_info("P1"), input_info("P2")}, 1);
        c.output_paddings = {padding({0, 0, 1, 1}, {0, 0, 1, 1})};
        t.add(c);
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB5", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_TRUE(static_cast<bool>(prog.get_node("C").get_output_layout().data_padding)) << "precondition: the holder has padding";
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
    EXPECT_EQ(pool_records_of(on, "C"), 1u);
    EXPECT_EQ(route_of(prog.get_node("P1")), primitive_inst::alloc_route::ZERO_ALLOC_MEMBER);
    EXPECT_EQ(buf_of(on, "P1"), buf_of(on, "C"));
    EXPECT_EQ(pool_records_of(*r.off_net, "C"), 0u);
}

// ===============================================================================================================================
// TB8. Crop: its holder is the producer, which already is a pool user. The switch changes nothing: no class is promoted, flags and
// restrictions are identical to the switch-off build, outputs are identical.
// ===============================================================================================================================
TEST(concat_holder_pool, tb8_crop_graph_is_untouched) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("in", in_l(32)));
        t.add(activation("PR", input_info("in"), activation_func::relu));
        t.add(crop("crop1", input_info("PR"), tensor(1, 16, 32, 32), tensor(0, 0, 0, 0)));
        t.add(crop("crop2", input_info("PR"), tensor(1, 16, 32, 32), tensor(0, 16, 0, 0)));
        t.add(reorder("out1", input_info("crop1"), format::bfyx, data_types::f32));
        t.add(reorder("out2", input_info("crop2"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in", {1, 32, 32, 32}, 1}};
    auto r = run_ab("TB8", eng, mk, ins);
    auto& off = *r.off_net->get_program();
    auto& on = *r.on_net->get_program();
    ASSERT_TRUE(on.get_node("crop1").can_be_optimized()) << "precondition: in-place crop\n" << dump_nodes(on);
    EXPECT_EQ(r.on.promoted, 0u);
    for (auto* n : on.get_processing_order()) {
        const auto& m = off.get_node(n->id());
        EXPECT_FALSE(n->has_alias_pool_root()) << n->id();
        EXPECT_EQ(n->get_memory_dependencies(), m.get_memory_dependencies()) << n->id() << " restrictions differ";
        EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << n->id();
        EXPECT_EQ(n->can_be_optimized(), m.can_be_optimized()) << n->id();
    }
    EXPECT_EQ(r.off.alloc_count, r.on.alloc_count);
    EXPECT_EQ(r.off.total_bytes, r.on.total_bytes);
}

// ===============================================================================================================================
// TB9. out_of_order queue (T-OOOQ style: oneDNN switched off by a user property; the user property is also what makes the queue type stick on
// an oneDNN device): the switch is ignored and everything is bit-identical to the switch-off build. In a real out_of_order build the
// optimizer does not even produce an in-place concat here, so the generator is additionally driven on a hand-flagged in-place concat.
// ===============================================================================================================================
TEST(concat_holder_pool, tb9_oooq_is_untouched) {
    auto& eng = get_test_engine();
    auto mk = [&]() { return topo_chain(eng); };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB9(ooo)", eng, mk, ins, QueueTypes::out_of_order);
    ASSERT_EQ(r.on_net->get_program()->get_config().get_queue_type(), QueueTypes::out_of_order);
    auto& off = *r.off_net->get_program();
    auto& on = *r.on_net->get_program();
    EXPECT_EQ(r.on.promoted, 0u);
    for (auto* n : on.get_processing_order()) {
        const auto& m = off.get_node(n->id());
        EXPECT_FALSE(n->has_alias_pool_root()) << n->id();
        EXPECT_EQ(n->get_memory_dependencies(), m.get_memory_dependencies()) << n->id();
        EXPECT_EQ(n->can_share_buffer(), m.can_share_buffer()) << n->id();
    }
    EXPECT_EQ(r.off.alloc_count, r.on.alloc_count);
    EXPECT_EQ(r.off.total_bytes, r.on.total_bytes);
}

TEST(concat_holder_pool, tb9b_oooq_hand_flagged_in_place_concat) {
    auto& eng = get_test_engine();
    for (bool on : {false, true}) {
        auto t = hand_concat_topology();
        auto prog = make_hand(eng, t, on, QueueTypes::out_of_order);
        ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
        prog->get_node("C").can_be_optimized(true);
        program_wrapper::prepare_memory_dependencies(*prog);
        EXPECT_EQ(count_roots(*prog), 0u) << "switch=" << on << ": out_of_order must keep the legacy path";
        for (auto* n : prog->get_processing_order()) {
            EXPECT_FALSE(n->has_alias_pool_root()) << n->id();
        }
        const auto* aa = prog->get_alias_analysis();
        ASSERT_NE(aa, nullptr);
        const auto mask = class_level::pool_admission_reject_mask(*aa->class_of(prog->get_node("C")), *prog, [](uint32_t) { return false; });
        EXPECT_TRUE(mask & class_level::rj_program_level) << class_level::to_string(mask);
    }
}

// ===============================================================================================================================
// TB16 / TB17 / TB18. The pure restriction core (L2/L3) on plain restriction tables.
// ===============================================================================================================================
namespace {

using class_level::promoted_class;
using class_level::simple_restriction_store;

simple_restriction_store make_store(const std::vector<uint32_t>& uids, const std::vector<std::vector<uint32_t>>& sets) {
    simple_restriction_store s;
    s.uids = uids;
    s.sets = sets;
    for (auto& v : s.sets) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return s;
}

}  // namespace

// NOTE (S-1): real graphs cannot tell the three generator steps apart: the legacy restrictions (oooq closure) are symmetric, so (2), (3') and
// (4) each re-derive what the others add. The single-step discrimination lives in TB16-TB18 (asymmetric tables) and in the Debug verifier
// check_class_invariants(); a failure of the whole generator on real graphs is detected by TB28/TB29/TB30 and the lifetime oracle. No real-graph
// construction with an asymmetric restriction was found, so none is claimed.
// nodes: 0:P1(10) 1:P2(11) 2:C(12, root) 3:X(20) 4:Y(21) 5:Z(22)
TEST(concat_holder_pool_core, tb16_ib1_both_directions_and_noop_for_unrelated) {
    auto store = make_store({10, 11, 12, 20, 21, 22}, {{10, 12, 20}, {11, 12}, {10, 11, 12, 21}, {10, 20}, {12, 21}, {22}});
    const auto r0 = class_level::snapshot_restrictions(store);
    const std::vector<promoted_class> classes = {{2, {0, 1, 2}}};
    class_level::materialize_class_restrictions(store, classes);
    EXPECT_EQ(store.sets[2], (std::vector<uint32_t>{10, 11, 12, 20, 21})) << "(2)+(3')+(4): the root conflicts with everything a member conflicts with";
    EXPECT_EQ(store.sets[3], (std::vector<uint32_t>{10, 12, 20})) << "X conflicted with member P1: it gets the root uid";
    EXPECT_EQ(store.sets[4], (std::vector<uint32_t>{12, 21}));
    EXPECT_EQ(store.sets[0], (std::vector<uint32_t>{10, 12, 20})) << "members keep their own uid (internal buffers) and gain the root uid";
    EXPECT_EQ(store.sets[5], (std::vector<uint32_t>{22})) << "unrelated node is untouched";
    EXPECT_EQ(class_level::check_class_invariants(store, r0, classes), "");
}

// two classes: the roots must contain each other as soon as any member of one is restricted against any member of the other
// nodes: 0:A1(1) 1:R1(2, root) 2:A2(3) 3:R2(4, root)
TEST(concat_holder_pool_core, tb16b_ib1_cross_class_roots_contain_each_other) {
    auto store = make_store({1, 2, 3, 4}, {{1, 2, 3}, {1, 2}, {1, 3, 4}, {3, 4}});  // A1 and A2 restrict each other, nothing else relates the classes
    const auto r0 = class_level::snapshot_restrictions(store);
    const std::vector<promoted_class> classes = {{1, {0, 1}}, {3, {2, 3}}};
    class_level::materialize_class_restrictions(store, classes);
    EXPECT_TRUE(std::binary_search(store.sets[1].begin(), store.sets[1].end(), 4u)) << "r1 must contain r2";
    EXPECT_TRUE(std::binary_search(store.sets[3].begin(), store.sets[3].end(), 2u)) << "r2 must contain r1";
    EXPECT_EQ(class_level::check_class_invariants(store, r0, classes), "");
}

// TB17: the member's own uid stays in its set, so the member's internal buffers (registered under the member uid) conflict with the root
// block and vice versa.
TEST(concat_holder_pool_core, tb17_member_internal_buffer_never_shares_the_root_block) {
    auto store = make_store({1, 2}, {{1}, {2}});  // PMD puts every node's own uid into its set
    const auto r0 = class_level::snapshot_restrictions(store);
    const std::vector<promoted_class> classes = {{1, {0, 1}}};
    class_level::materialize_class_restrictions(store, classes);
    EXPECT_TRUE(std::binary_search(store.sets[0].begin(), store.sets[0].end(), 1u)) << "member keeps its own uid";
    EXPECT_TRUE(std::binary_search(store.sets[0].begin(), store.sets[0].end(), 2u)) << "member internal buffer request must conflict with the root block";
    EXPECT_TRUE(std::binary_search(store.sets[1].begin(), store.sets[1].end(), 1u)) << "root request must conflict with a block holding member internal buffers";
    EXPECT_EQ(class_level::check_class_invariants(store, r0, classes), "");
}

// TB18: one-way restriction edges (member lists X but X does not list the member and the opposite) become two-way, and the verifier
// flags a table in which the second half ((3') / (4)) was not applied.
TEST(concat_holder_pool_core, tb18_one_way_edges_and_verifier_is_discriminating) {
    // 0:P(1, member) 1:C(2, root) 2:U(3) 3:V(4)
    auto store = make_store({1, 2, 3, 4}, {{1, 2, 4}, {2}, {3, 1}, {4}});  // R0[U] lists member P only; R0[P] lists V only
    const auto r0 = class_level::snapshot_restrictions(store);
    const std::vector<promoted_class> classes = {{1, {0, 1}}};
    class_level::materialize_class_restrictions(store, classes);
    EXPECT_TRUE(std::binary_search(store.sets[2].begin(), store.sets[2].end(), 2u)) << "U listed the member: it must list the root";
    EXPECT_TRUE(std::binary_search(store.sets[1].begin(), store.sets[1].end(), 3u)) << "and the root must list U";
    EXPECT_TRUE(std::binary_search(store.sets[1].begin(), store.sets[1].end(), 4u)) << "member listed V: the root must list V";
    EXPECT_TRUE(std::binary_search(store.sets[3].begin(), store.sets[3].end(), 2u)) << "and V must list the root (step 4)";
    EXPECT_EQ(class_level::check_class_invariants(store, r0, classes), "");

    auto without_4 = store;  // damage: drop what only step (4) provides
    without_4.sets[3].erase(std::remove(without_4.sets[3].begin(), without_4.sets[3].end(), 2u), without_4.sets[3].end());
    EXPECT_NE(class_level::check_class_invariants(without_4, r0, classes), "") << "I-B1 verifier must detect the missing root uid in R(V)";
    auto without_3 = store;  // damage: drop what only step (3') provides
    without_3.sets[2].erase(std::remove(without_3.sets[2].begin(), without_3.sets[2].end(), 2u), without_3.sets[2].end());
    EXPECT_NE(class_level::check_class_invariants(without_3, r0, classes), "");
    auto widened = store;  // damage: a non-root node gained a non-root uid
    widened.sets[2].push_back(99);
    std::sort(widened.sets[2].begin(), widened.sets[2].end());
    EXPECT_NE(class_level::check_class_invariants(widened, r0, classes), "") << "I-B3: non-root nodes may only gain root uids";
}

// ===============================================================================================================================
// Hand graph tests: L1 admission bits and the pass on flagged graphs.
// ===============================================================================================================================
namespace {

const alias_class* class_of_node(const alias_analysis& aa, program& p, const std::string& id) {
    return aa.class_of(p.get_node(id));
}

uint32_t mask_for(program& p, const std::string& id, const std::function<bool(uint32_t)>& amb = [](uint32_t) { return false; }) {
    auto aa = alias_analysis::build(p);
    const auto* c = class_of_node(aa, p, id);
    EXPECT_NE(c, nullptr) << id;
    return c ? class_level::pool_admission_reject_mask(*c, p, amb) : ~0u;
}

// A hand concat program: C flagged in place, switch on, memory dependencies built (the pass runs inside).
program::ptr hand_promoted(engine& eng, topology& t, const std::vector<std::string>& outs = {"O"}) {
    auto prog = make_hand(eng, t, true, QueueTypes::in_order, false, false, outs);
    prog->get_node("C").can_be_optimized(true);
    // what prepare_buffer_fusing does for every predecessor of an in-place concat
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

TEST(concat_holder_pool_hand, promoted_on_the_plain_graph_and_flags_follow_the_class) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    auto prog = hand_promoted(eng, t);
    program_wrapper::prepare_memory_dependencies(*prog);
    EXPECT_EQ(mask_for(*prog, "C"), 0u);
    EXPECT_EQ(count_roots(*prog), 1u);
    const uint32_t c_uid = static_cast<uint32_t>(prog->get_node("C").get_unique_id());
    EXPECT_TRUE(prog->get_node("C").is_pool_class_root());
    EXPECT_EQ(prog->get_node("C").get_alias_pool_root_uid(), c_uid);
    EXPECT_EQ(prog->get_node("P1").get_alias_pool_root_uid(), c_uid);
    EXPECT_EQ(prog->get_node("P2").get_alias_pool_root_uid(), c_uid);
    EXPECT_FALSE(prog->get_node("A").has_alias_pool_root());
    EXPECT_FALSE(prog->get_node("in1").has_alias_pool_root());
    EXPECT_EQ(class_level::check_b4_prime(*prog), "");
}

// TB10 a-e. Rigid exclusions: the class stays on the legacy path and no flag is written.
TEST(concat_holder_pool_hand, tb10a_cpu_reader) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    auto prog = hand_promoted(eng, t);
    prog->get_node("A").set_preferred_impl_type(impl_types::cpu);
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_placement, 0u);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "cpu reader");
}

TEST(concat_holder_pool_hand, tb10b_shape_of_reader) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    t.add(shape_of("S", input_info("C"), data_types::i32));
    auto prog = hand_promoted(eng, t, {"O", "S"});
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_placement, 0u);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "shape_of reader");
}

TEST(concat_holder_pool_hand, tb10c_cbo_reshape_view_is_a_network_output) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    t.add(reshape("R", input_info("C"), false, std::vector<int64_t>{1, 32, 1024}, ov::PartialShape{1, 32, 1024}));
    auto prog = hand_promoted(eng, t, {"O", "R"});
    prog->get_node("R").can_be_optimized(true);
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_placement, 0u) << "a view of the holder is a Result: the class is in the output domain";
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "output view");
}

TEST(concat_holder_pool_hand, tb10d_mutable_data_propagated_output) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    t.add(mutable_data("md", {input_info("C")}, eng.allocate_memory(layout{ov::PartialShape{1, 32, 32, 32}, data_types::f32, format::bfyx})));
    auto prog = hand_promoted(eng, t, {"O", "md"});
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_placement, 0u) << "C becomes an output through the mutable_data user";
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "mutable_data output");
}

TEST(concat_holder_pool_hand, tb10e_reader_in_a_shape_of_subgraph) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    auto prog = hand_promoted(eng, t);
    prog->get_node("A").set_in_shape_of_subgraph(true);
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_shape_of_reader, 0u);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "shape_of subgraph reader");
}

// Individual L1 bits (a fabricated or mutated class is judged by the same mask function).
TEST(concat_holder_pool_hand, l1_individual_reject_bits) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    auto prog = hand_promoted(eng, t);
    auto aa = alias_analysis::build(*prog);
    const alias_class* c = class_of_node(aa, *prog, "C");
    ASSERT_NE(c, nullptr);
    const auto never = [](uint32_t) { return false; };
    ASSERT_EQ(class_level::pool_admission_reject_mask(*c, *prog, never), 0u);
    {  // L1-4: a MAY edge member (or an in-place member) is not allowed
        alias_class k = *c;
        for (auto& m : k.members) {
            if (m.node->id() == "P1") {
                m.kind = alias_kind::MAY_VIEW;
            }
        }
        EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_member_kind, 0u);
        for (auto& m : k.members) {
            if (m.node->id() == "P1") {
                m.kind = alias_kind::MUST_INPLACE;
            }
        }
        EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_member_kind, 0u);
    }
    {  // L1-7: member on a port != 0
        alias_class k = *c;
        k.members[0].id.index = 1;
        EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_multi_output, 0u);
    }
    {  // L1-2: the root has to be an in-place concat
        alias_class k = *c;
        k.root_node = &prog->get_node("A");
        EXPECT_NE(class_level::pool_admission_reject_mask(k, *prog, never) & class_level::rj_root_kind, 0u);
    }
    {  // L1-9: ambiguous uid of a member
        const uint32_t p1 = static_cast<uint32_t>(prog->get_node("P1").get_unique_id());
        EXPECT_NE(class_level::pool_admission_reject_mask(*c, *prog, [&](uint32_t u) { return u == p1; }) & class_level::rj_ambiguous_uid, 0u);
    }
    // L1-3 (defect B marking / output holder)
    prog->get_node("C").can_share_buffer(false);
    EXPECT_NE(class_level::pool_admission_reject_mask(*c, *prog, never) & class_level::rj_root_not_shareable, 0u);
    prog->get_node("C").can_share_buffer(true);
    prog->get_node("C").set_output(true);
    EXPECT_NE(class_level::pool_admission_reject_mask(*c, *prog, never) & class_level::rj_root_not_shareable, 0u);
    prog->get_node("C").set_output(false);
    // L1-5: runtime skippable member
    prog->get_node("P1").set_runtime_skippable(true);
    EXPECT_NE(class_level::pool_admission_reject_mask(*c, *prog, never) & class_level::rj_dynamic, 0u);
    prog->get_node("P1").set_runtime_skippable(false);
    EXPECT_EQ(class_level::pool_admission_reject_mask(*c, *prog, never), 0u) << "all temporary mutations were undone";
}

// TB21. A member that does not use the pool (constant data as a slot) and a view with a MAY edge.
TEST(concat_holder_pool_hand, tb21_data_member_and_skippable_view_member) {
    auto& eng = get_test_engine();
    {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(cldnn::data("K", eng.allocate_memory(in_l())));
        t.add(activation("P1", input_info("in1"), activation_func::relu));
        t.add(concatenation("C", {input_info("P1"), input_info("K")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(activation("O", input_info("A"), activation_func::abs));
        auto prog = hand_promoted(eng, t);
        const auto mask = mask_for(*prog, "C");
        EXPECT_NE(mask & class_level::rj_no_mempool, 0u) << class_level::to_string(mask);
        EXPECT_NE(mask & class_level::rj_external_root, 0u) << class_level::to_string(mask);
        program_wrapper::prepare_memory_dependencies(*prog);
        expect_no_class_flags(*prog, "constant slot");
    }
    {   // a runtime-skippable view of the holder: MAY_VIEW member, dynamic-like decision
        auto t = hand_concat_topology();
        auto prog = hand_promoted(eng, t);
        prog->get_node("A").can_be_optimized(true);
        prog->get_node("A").set_runtime_skippable(true);
        const auto mask = mask_for(*prog, "C");
        EXPECT_NE(mask & (class_level::rj_member_kind | class_level::rj_dynamic), 0u) << class_level::to_string(mask);
        program_wrapper::prepare_memory_dependencies(*prog);
        expect_no_class_flags(*prog, "skippable view");
    }
}

// TB15. Internal and body programs never run the generator.
TEST(concat_holder_pool_hand, tb15_internal_and_body_programs) {
    auto& eng = get_test_engine();
    for (int kind = 0; kind < 2; ++kind) {
        auto t = hand_concat_topology();
        auto prog = make_hand(eng, t, true, QueueTypes::in_order, /*internal=*/kind == 0, /*body=*/kind == 1, {"O"});
        prog->get_node("C").can_be_optimized(true);
        program_wrapper::prepare_memory_dependencies(*prog);
        expect_no_class_flags(*prog, kind == 0 ? "internal" : "body");
        const auto* aa = prog->get_alias_analysis();
        ASSERT_NE(aa, nullptr);
        const auto mask = class_level::pool_admission_reject_mask(*aa->class_of(prog->get_node("C")), *prog, [](uint32_t) { return false; });
        EXPECT_NE(mask & class_level::rj_program_level, 0u);
    }
}

// TB24. uid collisions: the engine reports the uid as ambiguous, the class is rejected, no flag is written.
TEST(concat_holder_pool_hand, tb24_duplicate_uid_rejects_the_class) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    auto prog = hand_promoted(eng, t);
    prog->get_node("O").set_unique_id(prog->get_node("P1").get_unique_id());  // O collides with the slot P1
    auto aa = alias_analysis::build(*prog);
    const uint32_t uid = static_cast<uint32_t>(prog->get_node("P1").get_unique_id());
    EXPECT_TRUE(aa.uid_is_ambiguous(uid));
    EXPECT_GE(aa.ambiguous_uid_count(), 1u);
    EXPECT_NE(class_level::pool_admission_reject_mask(*aa.class_of(prog->get_node("C")), *prog, [&](uint32_t u) { return aa.uid_is_ambiguous(u); }) &
                  class_level::rj_ambiguous_uid,
              0u);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "duplicate uid");
}

// r5-1. Zero allocation needs every user other than the holder to be a plain, non-skippable, non-optimized node (alias_analysis only builds
// view edges for input 0, a cbo multi-input user may re-bind input_memory(k) of the slot at construction).
TEST(concat_holder_pool_hand, r51_zero_alloc_requires_plain_second_users) {
    auto& eng = get_test_engine();
    for (int variant = 0; variant < 3; ++variant) {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(input_layout("in3", in_l()));
        t.add(activation("P1", input_info("in1"), activation_func::relu));
        t.add(activation("P2", input_info("in2"), activation_func::abs));
        t.add(activation("X", input_info("in3"), activation_func::relu));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(activation("O", input_info("A"), activation_func::abs));
        // second user of P1 reads it through port 1 (not an input-0 view edge)
        t.add(eltwise("E", {input_info("X"), input_info("P1")}, eltwise_mode::sum));
        t.add(activation("OE", input_info("E"), activation_func::abs));
        auto prog = hand_promoted(eng, t, {"O", "OE"});
        if (variant == 1) {
            prog->get_node("E").can_be_optimized(true);
        } else if (variant == 2) {
            prog->get_node("E").set_runtime_skippable(true);
        }
        program_wrapper::prepare_memory_dependencies(*prog);
        ASSERT_TRUE(prog->get_node("C").is_pool_class_root()) << "variant " << variant << ": the class itself is promoted";
        const uint32_t c_uid = static_cast<uint32_t>(prog->get_node("C").get_unique_id());
        EXPECT_EQ(prog->get_node("P2").get_alias_pool_root_uid(), c_uid) << "P2 has no other user: zero allocation";
        if (variant == 0) {
            EXPECT_EQ(prog->get_node("P1").get_alias_pool_root_uid(), c_uid) << "plain second user: zero allocation";
        } else {
            EXPECT_FALSE(prog->get_node("P1").has_alias_pool_root())
                << "variant " << variant << ": a cbo / skippable second user may re-bind the slot buffer: P1 must keep allocating";
        }
    }
}

// TB21 (placeholder, DISABLED): a multi-output member of the class. No real in-place concat predecessor with two outputs exists in the
// optimizer today, so the rule L1-7 (member on port != 0 / get_outputs_count() != 1) is only exercised through the fabricated class in
// l1_individual_reject_bits. Kept as a characterisation point for the MO problem cluster (P1-M, postponed).
TEST(concat_holder_pool_hand, DISABLED_multi_output_slot_is_never_promoted) {
    FAIL() << "not constructible through build_program; see l1_individual_reject_bits for the mask-level check";
}

// TB23 (oneDNN free variants). B4': the structured assertion fails when a promoted root is not shareable or the class has an in-place member.
TEST(concat_holder_pool_hand, tb23_b4_prime_detects_corrupted_state) {
    auto& eng = get_test_engine();
    auto t = hand_concat_topology();
    auto prog = hand_promoted(eng, t);
    program_wrapper::prepare_memory_dependencies(*prog);
    ASSERT_TRUE(prog->get_node("C").is_pool_class_root());
    EXPECT_EQ(class_level::check_b4_prime(*prog), "");
    prog->get_node("C").can_share_buffer(false);  // defect B marking after promotion
    EXPECT_NE(class_level::check_b4_prime(*prog), "") << "a promoted root must stay shareable";
    prog->get_node("C").can_share_buffer(true);
    EXPECT_EQ(class_level::check_b4_prime(*prog), "");
}

// ===============================================================================================================================
// Real-pipeline tests
// ===============================================================================================================================

// TB19. A second user that is larger than the holder is constructed before it (allocation order is by bytes, descending). Whatever the
// construction order, the final binding of the slot and the data must be correct.
TEST(concat_holder_pool, tb19_big_second_user_is_constructed_first) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l(32)));
        t.add(input_layout("in2", in_l(16)));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        t.add(activation("X", input_info("P1"), activation_func::abs));       // second user of P1
        t.add(reorder("outX", input_info("X"), format::bfyx, data_types::f32));  // f32: bigger than the holder
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 32, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB19", eng, mk, ins);
    auto& prog = *r.on_net->get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_EQ(prog.get_node("P1").get_users().size(), 2u);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
    EXPECT_EQ(buf_of(*r.on_net, "P1"), buf_of(*r.on_net, "C"));
    EXPECT_EQ(buf_of(*r.on_net, "P2"), buf_of(*r.on_net, "C"));
}

// TB20. reorder and permute predecessors (single / two users): no direct allocation record with the switch on, one with the switch off.
TEST(concat_holder_pool, tb20_reorder_and_permute_predecessors_have_no_direct_allocation) {
#ifndef GPU_DEBUG_CONFIG
    GTEST_SKIP() << "allocation trace records need GPU_DEBUG_CONFIG";
#else
    auto& eng = get_test_engine();
    std::map<std::string, int> in_place_runs;
    for (int kind = 0; kind < 3; ++kind) {        // 0: reorder, 1: permute, 2: reshape (allocates in its own constructor: never a zero-allocation slot, M1)
        for (int two_users = 0; two_users < 2; ++two_users) {
            const std::string what = std::string(kind == 0 ? "reorder" : kind == 1 ? "permute" : "reshape") + (two_users ? " two users" : " single user");
            auto mk = [&]() {
                topology t;
                const layout l1 = kind == 2 ? layout{ov::PartialShape{1, 16, 1024, 1}, data_types::f32, format::bfyx} : in_l();
                t.add(input_layout("in1", l1));
                t.add(input_layout("in2", l1));
                if (kind == 0) {
                    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
                    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
                } else if (kind == 1) {
                    t.add(permute("P1", input_info("in1"), std::vector<uint16_t>{0, 1, 3, 2}));
                    t.add(permute("P2", input_info("in2"), std::vector<uint16_t>{0, 1, 3, 2}));
                } else {
                    t.add(reshape("P1", input_info("in1"), false, std::vector<int64_t>{1, 16, 32, 32}, ov::PartialShape{1, 16, 32, 32}));
                    t.add(reshape("P2", input_info("in2"), false, std::vector<int64_t>{1, 16, 32, 32}, ov::PartialShape{1, 16, 32, 32}));
                }
                t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
                t.add(activation("A", input_info("C"), activation_func::relu));
                t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
                if (two_users) {
                    t.add(activation("X", input_info("P1"), activation_func::abs));
                    t.add(reorder("outX", input_info("X"), format::bfyx, data_types::f32));
                }
                return t;
            };
            auto t_off = mk();
            auto t_on = mk();
            auto off = get_network(eng, t_off, make_cfg(eng, false, QueueTypes::in_order, 2), get_test_stream_ptr(), false);
            auto on = get_network(eng, t_on, make_cfg(eng, true, QueueTypes::in_order, 2), get_test_stream_ptr(), false);
            const std::vector<run_input> ins = kind == 2 ? std::vector<run_input>{{"in1", {1, 16, 1024, 1}, 1}, {"in2", {1, 16, 1024, 1}, 2}}
                                                          : std::vector<run_input>{{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
            auto o_off = run_net(*off, eng, ins);
            auto o_on = run_net(*on, eng, ins);  // must build and run without tripping the Debug assertions
            EXPECT_EQ(o_off, o_on) << what;
            if (!on->get_program()->get_node("C").can_be_optimized()) {
                std::cout << "[TB20] " << what << ": the concat is not in place in this graph (iteration skipped, other combinations still run)\n";
                continue;
            }
            ++in_place_runs[what];
            auto has_record = [](network& n, const std::string& id) {
                for (const auto& rec : n.get_alloc_trace_records()) {
                    if (rec.node_id == id && !rec.is_internal_buffer) {
                        return true;
                    }
                }
                return false;
            };
            for (const char* id : {"P1", "P2"}) {
                if (!on->get_program()->has_node(id)) {
                    continue;
                }
                if (kind == 2) {
                    EXPECT_FALSE(on->get_program()->get_node(id).has_alias_pool_root()) << what << " " << id << ": reshape allocates in its constructor, no zero allocation";
                    continue;
                }
                EXPECT_TRUE(on->get_program()->get_node(id).has_alias_pool_root()) << what << " " << id;
                EXPECT_FALSE(has_record(*on, id)) << what << ": " << id << " must not allocate with the switch on";
                if (kind == 0) {
                    EXPECT_TRUE(has_record(*off, id)) << what << ": reorder predecessors allocate transiently in the legacy path: " << id;
                }
            }
        }
    }
    for (const char* k : {"reorder single user", "reorder two users", "permute single user", "permute two users"}) {
        EXPECT_EQ(in_place_runs[k], 1) << k << " must have been exercised in place";
    }
    std::cout << "[TB20] reshape in-place runs: " << in_place_runs["reshape single user"] + in_place_runs["reshape two users"] << " of 2\n";
#endif
}

// M1 (hand): a non-cbo reshape as a slot predecessor is never a zero-allocation member, whatever the pipeline decides about it.
TEST(concat_holder_pool_hand, m1_reshape_predecessor_is_not_zero_alloc) {
    auto& eng = get_test_engine();
    for (int two_users = 0; two_users < 2; ++two_users) {
        topology t;
        t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 1024, 1}, data_types::f32, format::bfyx}));
        t.add(input_layout("in2", in_l()));
        t.add(reshape("P1", input_info("in1"), false, std::vector<int64_t>{1, 16, 32, 32}, ov::PartialShape{1, 16, 32, 32}));
        t.add(activation("P2", input_info("in2"), activation_func::abs));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(activation("O", input_info("A"), activation_func::abs));
        std::vector<std::string> outs = {"O"};
        if (two_users) {
            t.add(activation("X", input_info("P1"), activation_func::abs));
            outs.push_back("X");
        }
        auto prog = hand_promoted(eng, t, outs);
        program_wrapper::prepare_memory_dependencies(*prog);
        // In this hand graph the reshape (a view-capable type) also has a view edge to its input, so the class is rejected as a role conflict before
        // M1 matters; the M1 exclusion in is_zero_alloc_member is defense in depth. Both outcomes must leave the reshape allocating.
        std::cout << "[M1] two_users=" << two_users << " promoted=" << prog->get_node("C").is_pool_class_root()
                  << " mask=" << class_level::to_string(mask_for(*prog, "C")) << "\n";
        EXPECT_FALSE(prog->get_node("P1").has_alias_pool_root()) << "reshape keeps allocating (two_users=" << two_users << ")";
    }
}

// TB22. A dynamic reader of a static holder: the class is not promoted (decision moved to P1-B-D), outputs are identical.
// P1-B-D note (dynamic_concat_holder_pool_test.cpp, TD15): this stays true after P1-B-D, with or without enable_dynamic_concat_holder_pool: the
// sub-switch only admits classes whose ROOT is dynamic; a class with a static root keeps the static L1-5 rule (any dynamic reader sets rj_dynamic).
TEST(concat_holder_pool, tb22_dynamic_reader_keeps_the_legacy_path) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(input_layout("ind", layout{ov::PartialShape{1, 32, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_types::f32, format::bfyx}));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(reorder("Dq", input_info("ind"), format::bfyx, data_types::f16));
        t.add(eltwise("E", {input_info("C"), input_info("Dq")}, eltwise_mode::sum));  // dynamic direct reader of the static holder
        t.add(reorder("out", input_info("E"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"ind", {1, 32, 32, 32}, 3}};
    auto r = run_ab("TB22", eng, mk, ins);
    auto& prog = *r.on_net->get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << "precondition: static in-place holder\n" << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("E").is_dynamic()) << dump_nodes(prog);
    EXPECT_FALSE(prog.get_node("C").is_dynamic());
    EXPECT_EQ(r.on.promoted, 0u) << "a dynamic reader may re-decide at runtime: P1-B-D";
    const auto* aa = prog.get_alias_analysis();
    ASSERT_NE(aa, nullptr);
    EXPECT_NE(class_level::pool_admission_reject_mask(*aa->class_of(prog.get_node("C")), prog, [](uint32_t) { return false; }) & class_level::rj_dynamic, 0u);
}

// TB25. A second user that is a view (runtime skippable reshape) of the predecessor: the view is a MAY member of the class, the class is rejected.
TEST(concat_holder_pool, tb25_second_user_is_a_view) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        t.add(reshape("V", input_info("P1"), false, std::vector<int64_t>{1, 16, 1024}, ov::PartialShape{1, 16, 1024}));  // view of the slot
        t.add(activation("X", input_info("V"), activation_func::abs));
        t.add(reorder("outX", input_info("X"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB25", eng, mk, ins);
    auto& prog = *r.on_net->get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    const auto* aa = prog.get_alias_analysis();
    ASSERT_NE(aa, nullptr);
    const auto mask = class_level::pool_admission_reject_mask(*aa->class_of(prog.get_node("C")), prog, [&](uint32_t u) { return aa->uid_is_ambiguous(u); });
    // the reshape of the slot is a runtime skippable view: a MAY_VIEW member of the class, so the whole class is rejected
    EXPECT_NE(mask & class_level::rj_member_kind, 0u) << class_level::to_string(mask) << "\n" << dump_nodes(prog);
    EXPECT_EQ(r.on.promoted, 0u);
    EXPECT_FALSE(prog.get_node("P1").has_alias_pool_root());
    EXPECT_EQ(r.on.alloc_count, r.off.alloc_count) << "legacy path: identical allocation sequence";
    EXPECT_EQ(r.on.total_bytes, r.off.total_bytes);
}

// TB26. The switch off (explicit false == default): no class flags anywhere, the route of every node/port equals the pre-P1-B-S inline predicate.
TEST(concat_holder_pool, tb26_switch_off_is_bit_identical_to_the_legacy_predicate) {
    auto& eng = get_test_engine();
    for (auto mk : {std::function<topology()>([&]() { return topo_chain(eng); }), std::function<topology()>([]() { return topo_cascade(true); })}) {
        auto t_default = mk();
        auto net_default = get_network(eng, t_default, make_cfg(eng, false), get_test_stream_ptr(), false);  // the switch is never set: default
        auto t_off = mk();
        auto cfg_off = make_cfg(eng, false);
        cfg_off.set_property(ov::intel_gpu::enable_concat_holder_pool(false));                            // explicit false
        auto net_off = get_network(eng, t_off, cfg_off, get_test_stream_ptr(), false);
        ASSERT_FALSE(net_default->get_program()->get_config().get_enable_concat_holder_pool()) << "default is off";
        for (auto* n : net_off->get_program()->get_processing_order()) {
            const auto& d = net_default->get_program()->get_node(n->id());
            EXPECT_FALSE(n->is_pool_class_root()) << n->id();
            EXPECT_FALSE(n->has_alias_pool_root()) << n->id();
            EXPECT_EQ(n->get_memory_dependencies(), d.get_memory_dependencies()) << n->id();
            const auto route = primitive_inst::allocation_route(*n, 0, n->can_be_optimized());
            const bool legacy_direct = !n->can_share_buffer() || n->can_be_optimized() || n->is_output();
            EXPECT_EQ(route, legacy_direct ? primitive_inst::alloc_route::DIRECT : primitive_inst::alloc_route::LEGACY_POOL) << n->id();
            // non zero ports follow the legacy predicate as well
            EXPECT_EQ(primitive_inst::allocation_route(*n, 1, n->can_be_optimized()), legacy_direct ? primitive_inst::alloc_route::DIRECT : primitive_inst::alloc_route::LEGACY_POOL);
        }
        // same allocation sequence (trace) and the same buffers sharing pattern
        const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 16, 32, 32}, 3}};
        const auto s_off = collect(*net_off, eng, ins.size() == 3 && net_off->get_program()->has_node("in3") ? ins : std::vector<run_input>{ins[0], ins[1]});
        const auto s_def = collect(*net_default, eng, ins.size() == 3 && net_default->get_program()->has_node("in3") ? ins : std::vector<run_input>{ins[0], ins[1]});
        EXPECT_EQ(s_off.out, s_def.out);
        EXPECT_EQ(s_off.alloc_count, s_def.alloc_count);
        EXPECT_EQ(s_off.pooled_bytes, s_def.pooled_bytes);
        EXPECT_EQ(s_off.off_pool_bytes, s_def.off_pool_bytes);
        EXPECT_EQ(s_off.total_bytes, s_def.total_bytes);
    }
}

// TB13. Cache round trip: the new flags are serialized, the import does not re-run the generator, the version is 3. Each of the four
// export/import combinations also builds a network from the imported program and executes it: outputs equal the uncached result, classes that
// share a buffer have disjoint lifetimes and zero-allocation members live in the root buffer. With the switch off at export every route is legacy.
TEST(concat_holder_pool, tb13_cache_round_trip_keeps_flags_and_version_is_3) {
    EXPECT_EQ(ov::intel_gpu::CompiledModel::runtime_requirements_version, 3u) << "the node blob changed (pool_class_root, alias_pool_root_uid)";
    auto& eng = get_test_engine();
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    for (bool export_on : {false, true}) {
        for (bool import_on : {false, true}) {
            const std::string tag = std::string("export=") + (export_on ? "on" : "off") + " import=" + (import_on ? "on" : "off");
            auto t = topo_chain(eng);
            auto cfg = make_cfg(eng, export_on);
            auto prog = program::build_program(eng, t, cfg);
            ASSERT_TRUE(prog->get_node("C").can_be_optimized());
            ASSERT_EQ(count_roots(*prog), export_on ? 1u : 0u);
            cldnn::membuf mem_buf;
            {
                std::ostream out_mem(&mem_buf);
                cldnn::BinaryOutputBuffer ob = cldnn::BinaryOutputBuffer(out_mem);
                ob.set_stream(get_test_stream_ptr().get());
                prog->save(ob);
            }
            std::istream in_mem(&mem_buf);
            cldnn::BinaryInputBuffer ib = cldnn::BinaryInputBuffer(in_mem, eng);
            auto imported = std::make_shared<cldnn::program>(eng, make_cfg(eng, import_on));
            imported->load(ib);
            for (auto* n : prog->get_processing_order()) {
                const auto& m = imported->get_node(n->id());
                EXPECT_EQ(m.is_pool_class_root(), n->is_pool_class_root()) << tag << " " << n->id();
                EXPECT_EQ(m.get_alias_pool_root_uid(), n->get_alias_pool_root_uid()) << tag << " " << n->id();
                EXPECT_EQ(m.get_memory_dependencies(), n->get_memory_dependencies()) << tag << " " << n->id();
                if (!export_on) {
                    const auto route = primitive_inst::allocation_route(m, 0, m.can_be_optimized());
                    const bool legacy_direct = !m.can_share_buffer() || m.can_be_optimized() || m.is_output();
                    EXPECT_EQ(route, legacy_direct ? primitive_inst::alloc_route::DIRECT : primitive_inst::alloc_route::LEGACY_POOL) << tag << " " << n->id();
                }
            }
            EXPECT_EQ(imported->get_alias_analysis(), nullptr) << "the import does not run prepare_memory_dependencies";
            // reference: the uncached program of the same export configuration
            auto t_ref = topo_chain(eng);
            auto ref_net = get_network(eng, t_ref, make_cfg(eng, export_on), get_test_stream_ptr(), false);
            const auto ref = run_net(*ref_net, eng, ins);
            auto net = std::make_shared<network>(imported, 0);
            const auto got = run_net(*net, eng, ins);
            EXPECT_EQ(got, ref) << tag << ": imported program output differs from the uncached result";
            EXPECT_TRUE(lifetime_violations(*net).empty()) << tag;
            auto& ip = *net->get_program();
            if (export_on) {
                for (auto id : {"P1", "P2"}) {
                    EXPECT_EQ(route_of(ip.get_node(id)), primitive_inst::alloc_route::ZERO_ALLOC_MEMBER) << tag << " " << id;
                    EXPECT_EQ(buf_of(*net, id), buf_of(*net, "C")) << tag << " " << id << " must be bound to the root buffer";
                }
                EXPECT_EQ(route_of(ip.get_node("C")), primitive_inst::alloc_route::POOL_AS_ROOT) << tag;
            }
        }
    }
}

// TB27. Estimation (program::get_estimated_device_mem_usage). It is measured against the engine-wide usm_device counter, which includes
// weights, so only the DIFFERENCE between the switch-off and switch-on programs is compared, never the absolute value (r5-4).
// HEAD's estimator skips every can_be_optimized node, i.e. it ignores the direct holder allocation entirely (existing deviation, not fixed
// by this phase): switch-off estimate == real held bytes - holder bytes; switch-on estimate counts the pooled holder.
TEST(concat_holder_pool, tb27_estimate_differs_by_exactly_the_pooled_holder) {
#ifndef GPU_DEBUG_CONFIG
    GTEST_SKIP() << "the held-bytes snapshot used as the reference needs GPU_DEBUG_CONFIG";
#else
    auto& eng = get_test_engine();
    if (eng.get_device_info().dev_type != device_type::discrete_gpu && eng.get_device_info().dev_type != device_type::integrated_gpu) {
        GTEST_SKIP() << "unknown device type";
    }
    auto t_off = topo_chain(eng);
    auto t_on = topo_chain(eng);
    auto prog_off = program::build_program(eng, t_off, make_cfg(eng, false));
    auto prog_on = program::build_program(eng, t_on, make_cfg(eng, true));
    const auto holder_bytes = static_cast<int64_t>(prog_on->get_node("C").get_output_layout().bytes_count());
    ASSERT_TRUE(prog_on->get_node("C").is_pool_class_root());
    const auto e_off = prog_off->get_estimated_device_mem_usage();
    const auto e_on = prog_on->get_estimated_device_mem_usage();
    ASSERT_GE(e_off.second, 0);
    ASSERT_GE(e_on.second, 0);
    prog_off.reset();
    prog_on.reset();
    auto n_off_t = topo_chain(eng);
    auto n_on_t = topo_chain(eng);
    auto net_off = get_network(eng, n_off_t, make_cfg(eng, false), get_test_stream_ptr(), false);
    auto net_on = get_network(eng, n_on_t, make_cfg(eng, true), get_test_stream_ptr(), false);
    const auto h_off = static_cast<int64_t>(net_off->get_held_bytes_snapshot().device_bytes);
    const auto h_on = static_cast<int64_t>(net_on->get_held_bytes_snapshot().device_bytes);
    std::cout << "[TB27] estimate off=" << e_off.second << " on=" << e_on.second << " delta=" << (e_on.second - e_off.second) << "; held device bytes off=" << h_off
              << " on=" << h_on << " delta=" << (h_on - h_off) << "; holder bytes=" << holder_bytes << std::endl;
    // estimate(on) - estimate(off) == (held(on) - held(off)) + holder_bytes: the legacy estimate omits the direct holder block that the held snapshot counts
    EXPECT_EQ(e_on.second - e_off.second, (h_on - h_off) + holder_bytes);
#endif
}

// Q6 (r5-3). Candidate P1-Q6, only registered: a constant non-data node (may_use_mempool() == false, empty restriction set) that would be
// allocated from the pool of the main network (route LEGACY_POOL, cbs true, not cbo, not an output). The probe below shows the situation is not
// reachable through build_program: propagate_constants replaces every constant node with a data node unless it is an output (DIRECT) and no pass
// inserts nodes after it. The DISABLED test documents the unreachable case.
TEST(concat_holder_pool_q6, probe_constant_non_data_nodes_do_not_survive_propagate_constants) {
    auto& eng = get_test_engine();
    layout l{ov::PartialShape{1, 3, 4, 4}, data_types::f32, format::bfyx};
    auto c0 = eng.allocate_memory(l);
    auto c1 = eng.allocate_memory(l);
    topology t;
    t.add(input_layout("in", l));
    t.add(cldnn::data("c0", c0));
    t.add(cldnn::data("c1", c1));
    t.add(activation("ca", input_info("c0"), activation_func::abs));
    t.add(eltwise("ce", {input_info("ca"), input_info("c1")}, eltwise_mode::sum));
    t.add(reorder("cr", input_info("ce"), format::bfyx, data_types::f16));
    t.add(reorder("cr2", input_info("cr"), format::bfyx, data_types::f32));
    t.add(concatenation("cc", {input_info("cr2"), input_info("in")}, 1));
    t.add(activation("a1", input_info("cc"), activation_func::relu));
    t.add(concatenation("cc2", {input_info("ce"), input_info("ce")}, 1));
    t.add(eltwise("e2", {input_info("a1"), input_info("cc2")}, eltwise_mode::sum));
    t.add(reorder("out", input_info("e2"), format::bfyx, data_types::f32));
    auto prog = program::build_program(eng, t, make_cfg(eng, true));
    size_t surviving = 0;
    for (auto* n : prog->get_processing_order()) {
        if (n->is_constant() && !n->is_type<data>() && !n->is_output()) {
            ++surviving;
        }
    }
    EXPECT_EQ(surviving, 0u) << "a constant non-data node survived propagate_constants and is not an output";
}

TEST(concat_holder_pool_q6, DISABLED_constant_non_data_node_pools_with_empty_restriction) {
    // Candidate P1-Q6 (unreachable today, see the probe above). If a constant non-data node ever reaches allocate_output through the pool route,
    // add_memory_dependency(const program_node&) skips it (may_use_mempool() == false), its restriction set is empty and it can share any block.
    FAIL() << "unreachable: no constructor path found; keep as a characterisation placeholder";
}

// ===============================================================================================================================
// TB28-TB33: class lifetime edge cases on real networks (oracle: lifetime_violations + exact buffer binding checks)
// ===============================================================================================================================
namespace {

size_t po_index(program& p, const std::string& id) {
    size_t i = 0;
    for (auto* n : p.get_processing_order()) {
        if (n->id() == id) {
            return i;
        }
        ++i;
    }
    return std::numeric_limits<size_t>::max();
}

}  // namespace

// TB28. X runs between the first slot producer and the holder, dies before the holder, is at least as large as the holder and is only
// related to P1 (not to P2 or C). Without the class merge the holder could take X's block (X is dead when C starts); the slot of P1 lives
// in the holder buffer from P1 on, so that would corrupt P1's data.
TEST(concat_holder_pool, tb28_dead_before_holder_but_alive_over_a_slot) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("in3", in_l(16)));  // declared first: ends up after in1 in the processing order
        t.add(input_layout("in1", in_l(16)));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(activation("Xa", input_info("in3"), activation_func::relu));
        t.add(softmax("X", input_info("Xa"), 1));                           // f32, 16 features: as large as the holder
        t.add(cldnn::data("ky", [&]() { auto m = eng.allocate_memory(in_l(16)); set_values(m, std::vector<float>(m->count(), 1.f)); return m; }()));
        t.add(eltwise("Y", {input_info("X"), input_info("ky")}, eltwise_mode::sum));
        t.add(reorder("P2", input_info("Y"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in3", {1, 16, 32, 32}, 3}};
    auto r = run_ab("TB28", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
    ASSERT_LT(po_index(prog, "P1"), po_index(prog, "X")) << "precondition: X runs after the first slot producer\n" << dump_nodes(prog);
    ASSERT_LT(po_index(prog, "Y"), po_index(prog, "C")) << "precondition: X/Y are dead before the holder\n" << dump_nodes(prog);
    ASSERT_GE(prog.get_node("X").get_output_layout().bytes_count(), prog.get_node("C").get_output_layout().bytes_count());
    const uint32_t c_uid = static_cast<uint32_t>(prog.get_node("C").get_unique_id());
    for (auto id : {"X", "Y"}) {
        EXPECT_NE(buf_of(on, id), buf_of(on, "C")) << id << " must not share the holder block";
    }
    EXPECT_TRUE(prog.get_node("X").has_memory_dependency(c_uid)) << "X is restricted against the root (I-B1 via its relation to P1)\n" << dump_nodes(prog);
    EXPECT_TRUE(prog.get_node("C").has_memory_dependency(static_cast<uint32_t>(prog.get_node("X").get_unique_id())));
}

// TB29. The reader of the holder is a view of a view; X runs between the intermediate view and the final reader.
TEST(concat_holder_pool, tb29_view_of_view_reader_with_x_in_between) {
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("in3", in_l(32)));
        t.add(input_layout("in2", in_l(16)));
        t.add(input_layout("in1", in_l(16)));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(crop("V1", input_info("C"), tensor(1, 24, 32, 32), tensor(0, 4, 0, 0)));
        t.add(crop("V2", input_info("V1"), tensor(1, 16, 32, 32), tensor(0, 4, 0, 0)));
        t.add(softmax("X", input_info("in3"), 1));                                                 // f32 32 features: larger than the holder
        t.add(cldnn::data("kx", [&]() { auto m = eng.allocate_memory(in_l(32)); set_values(m, std::vector<float>(m->count(), 1.f)); return m; }()));
        t.add(eltwise("X2", {input_info("X"), input_info("kx")}, eltwise_mode::sum));
        t.add(softmax("X3", input_info("X2"), 1));
        t.add(eltwise("X4", {input_info("X3"), input_info("kx")}, eltwise_mode::prod));
        t.add(softmax("Xc", input_info("X4"), 1));                                                 // X chain dies here
        t.add(reorder("Xr", input_info("Xc"), format::bfyx, data_types::f16));
        t.add(crop("Xk", input_info("Xr"), tensor(1, 16, 32, 32), tensor(0, 0, 0, 0)));
        t.add(eltwise("Rd", {input_info("V2"), input_info("Xk")}, eltwise_mode::sum));  // final reader of the view chain
        t.add(reorder("out", input_info("Rd"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 32, 32, 32}, 3}};
    auto r = run_ab("TB29", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    const auto* aa = prog.get_alias_analysis();
    const auto* c = aa->class_of(prog.get_node("C"));
    ASSERT_NE(c, nullptr);
    std::cout << "[TB29] mask="
              << class_level::to_string(class_level::pool_admission_reject_mask(*c, prog, [&](uint32_t u) { return aa->uid_is_ambiguous(u); })) << " members=" << c->members.size()
              << " V1cbo=" << (prog.has_node("V1") && prog.get_node("V1").can_be_optimized()) << "\n"
              << dump_nodes(prog);
    ASSERT_TRUE(prog.has_node("V1") && prog.get_node("V1").can_be_optimized() && prog.get_node("V2").can_be_optimized()) << "precondition: view chain\n" << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("C").is_pool_class_root());
    ASSERT_LT(po_index(prog, "V1"), po_index(prog, "X4")) << "precondition: part of the X chain runs after the intermediate view\n" << dump_nodes(prog);
    ASSERT_LT(po_index(prog, "Xc"), po_index(prog, "Rd")) << "precondition: X chain dies before the final reader";
    for (auto id : {"X2", "X3", "X4", "Xc"}) {
        EXPECT_NE(buf_of(on, id), buf_of(on, "C")) << id;
    }
    // the view chain is part of the class and its last reader extends the class lifetime
    auto in_class = [&](const char* id) {
        for (const auto& m : c->members) {
            if (m.node->id() == id) {
                return true;
            }
        }
        return false;
    };
    EXPECT_TRUE(in_class("V1") && in_class("V2"));
    EXPECT_TRUE(std::find(c->readers.begin(), c->readers.end(), &prog.get_node("Rd")) != c->readers.end()) << "Rd is a reader of the class";
    EXPECT_TRUE(prog.get_node("Rd").has_memory_dependency(static_cast<uint32_t>(prog.get_node("C").get_unique_id())));
}

// TB30. Cross class: a slot predecessor of the second holder reads the first holder. Both classes are promoted, the roots contain each other and
// the two holders live in different blocks.
TEST(concat_holder_pool, tb30_two_classes_where_a_slot_reads_the_other_holder) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        for (auto id : {"in1", "in2", "in3"}) {
            t.add(input_layout(id, in_l()));
        }
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C1", {input_info("P1"), input_info("P2")}, 1));        // K1 (32 features f16)
        t.add(activation("Q1", input_info("C1"), activation_func::relu));          // reads K1 and is a slot of K2
        t.add(reorder("Q2", input_info("in3"), format::bfyx, data_types::f16));
        t.add(concatenation("C2", {input_info("Q1"), input_info("Q2")}, 1));        // K2 (48 features f16)
        t.add(activation("A", input_info("C2"), activation_func::abs));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 16, 32, 32}, 3}};
    auto r = run_ab("TB30", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C1").can_be_optimized() && prog.get_node("C2").can_be_optimized()) << dump_nodes(prog);
    EXPECT_EQ(r.on.promoted, 2u) << dump_nodes(prog);
    const uint32_t u1 = static_cast<uint32_t>(prog.get_node("C1").get_unique_id());
    const uint32_t u2 = static_cast<uint32_t>(prog.get_node("C2").get_unique_id());
    EXPECT_TRUE(prog.get_node("C1").has_memory_dependency(u2)) << "r1 contains r2";
    EXPECT_TRUE(prog.get_node("C2").has_memory_dependency(u1)) << "r2 contains r1";
    EXPECT_NE(buf_of(on, "C1"), buf_of(on, "C2"));
    EXPECT_EQ(buf_of(on, "Q1"), buf_of(on, "C2")) << "Q1 is a slot of the second holder";
}

// TB31. The same predecessor feeds two concats (an in-place concat is restricted to a single holder per slot): no class with a role conflict
// is ever promoted. One of the two holders is padded, the other is not.
TEST(concat_holder_pool, tb31_shared_predecessor_two_holders) {
    auto& eng = get_test_engine();
    for (int padded = 0; padded < 2; ++padded) {
        auto mk = [&]() {
            topology t;
            for (auto id : {"in1", "in2", "in3"}) {
                t.add(input_layout(id, in_l()));
            }
            t.add(reorder("P", input_info("in1"), format::bfyx, data_types::f16));
            t.add(reorder("Q1", input_info("in2"), format::bfyx, data_types::f16));
            t.add(reorder("Q2", input_info("in3"), format::bfyx, data_types::f16));
            t.add(concatenation("C1", {input_info("P"), input_info("Q1")}, 1));
            auto c2 = concatenation("C2", {input_info("P"), input_info("Q2")}, 1);
            if (padded) {
                c2.output_paddings = {padding({0, 0, 1, 1}, {0, 0, 1, 1})};
            }
            t.add(c2);
            t.add(activation("A1", input_info("C1"), activation_func::relu));
            t.add(activation("A2", input_info("C2"), activation_func::abs));
            t.add(reorder("out1", input_info("A1"), format::bfyx, data_types::f32));
            t.add(reorder("out2", input_info("A2"), format::bfyx, data_types::f32));
            return t;
        };
        const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"in3", {1, 16, 32, 32}, 3}};
        auto r = run_ab(std::string("TB31(padded=") + (padded ? "1" : "0") + ")", eng, mk, ins);
        auto& prog = *r.on_net->get_program();
        const bool both = prog.get_node("C1").can_be_optimized() && prog.get_node("C2").can_be_optimized();
        std::cout << "[TB31] padded=" << padded << " C1 cbo=" << prog.get_node("C1").can_be_optimized() << " C2 cbo=" << prog.get_node("C2").can_be_optimized() << " roots=" << r.on.promoted << "\n";
        if (both) {
            EXPECT_EQ(r.on.promoted, 0u) << "a predecessor with two in-place holders is a role conflict: no class is promoted\n" << dump_nodes(prog);
            const auto* aa = prog.get_alias_analysis();
            EXPECT_NE(aa->class_of(prog.get_node("C1"))->fallback & fb_role_conflict, 0u);
        }
        // a promoted class never contains a node of a role-conflict class
        for (auto* n : prog.get_processing_order()) {
            if (n->is_pool_class_root()) {
                EXPECT_EQ(prog.get_alias_analysis()->class_of(*n)->fallback, 0u) << n->id();
            }
        }
    }
}

// TB32. Mixed network: a static holder and a dynamic branch whose outputs are (re)allocated at run time from the same pool. The dynamic
// nodes must never be handed the block of the holder while the holder is alive.
TEST(concat_holder_pool, tb32_mixed_dynamic_branch_never_takes_the_holder_block) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(input_layout("ind", layout{ov::PartialShape{1, 16, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_types::f32, format::bfyx}));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        t.add(activation("D1", input_info("ind"), activation_func::relu));  // dynamic nodes
        t.add(softmax("D2", input_info("D1"), 1));
        t.add(reorder("outd", input_info("D2"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}, {"ind", {1, 16, 32, 32}, 3}};
    auto r = run_ab("TB32", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_TRUE(prog.get_node("D1").is_dynamic());
    EXPECT_EQ(r.on.promoted, 1u) << dump_nodes(prog);
    const uint32_t root = static_cast<uint32_t>(prog.get_node("C").get_unique_id());
    for (auto id : {"D1", "D2"}) {
        EXPECT_TRUE(prog.get_node(id).has_memory_dependency(root)) << id << ": root must be in R(X) of a runtime-allocated dynamic node\n" << dump_nodes(prog);
        EXPECT_NE(buf_of(on, id), buf_of(on, "C")) << id << " must not be handed the holder block";
    }
    // second round with another dynamic shape (reallocation from the pool)
    const std::vector<run_input> ins2 = {{"in1", {1, 16, 32, 32}, 4}, {"in2", {1, 16, 32, 32}, 5}, {"ind", {1, 16, 48, 48}, 6}};
    const auto o_on = run_net(on, eng, ins2);
    const auto o_off = run_net(*r.off_net, eng, ins2);
    EXPECT_EQ(o_on, o_off);
    for (auto id : {"D1", "D2"}) {
        EXPECT_NE(buf_of(on, id), buf_of(on, "C")) << id << " (second round)";
    }
    EXPECT_TRUE(lifetime_violations(on).empty());
}

// TB33. Neutrality: a network output that is a direct reader and a reader of a reader of the holder.
TEST(concat_holder_pool, tb33_outputs_as_reader_and_reader_of_reader) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
        t.add(reorder("out1", input_info("C"), format::bfyx, data_types::f32));       // direct reader, network output
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out2", input_info("A"), format::bfyx, data_types::f32));       // reader of a reader, network output
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}};
    auto r = run_ab("TB33", eng, mk, ins);
    auto& on = *r.on_net;
    auto& prog = *on.get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root()) << "output readers do not make the holder an output";
    for (auto id : {"out1", "out2"}) {
        EXPECT_TRUE(prog.get_node(id).is_output());
        EXPECT_NE(buf_of(on, id), buf_of(on, "C")) << id;
        EXPECT_FALSE(prog.get_node(id).has_alias_pool_root());
    }
}

// TB31b (hand): a predecessor shared by two in-place holders (role conflict in alias_analysis): neither class is promoted. One holder is padded.
TEST(concat_holder_pool_hand, tb31b_shared_predecessor_two_flagged_holders_padded_and_plain) {
    auto& eng = get_test_engine();
    for (int padded = 0; padded < 2; ++padded) {
        topology t;
        for (auto id : {"in1", "in2", "in3"}) {
            t.add(input_layout(id, in_l()));
        }
        t.add(activation("P", input_info("in1"), activation_func::relu));
        t.add(activation("Q1", input_info("in2"), activation_func::relu));
        t.add(activation("Q2", input_info("in3"), activation_func::relu));
        t.add(concatenation("C1", {input_info("P"), input_info("Q1")}, 1));
        t.add(concatenation("C2", {input_info("P"), input_info("Q2")}, 1));
        t.add(activation("O1", input_info("C1"), activation_func::relu));
        t.add(activation("O2", input_info("C2"), activation_func::abs));
        auto prog = make_hand(eng, t, true, QueueTypes::in_order, false, false, {"O1", "O2"});
        prog->get_node("C1").can_be_optimized(true);
        prog->get_node("C2").can_be_optimized(true);
        if (padded) {
            prog->get_node("C2").set_output_padding(padding({0, 0, 1, 1}, {0, 0, 1, 1}));
        }
        for (const char* c : {"C1", "C2"}) {
            for (const auto& d : prog->get_node(c).get_dependencies()) {
                d.first->can_share_buffer(false);
            }
        }
        program_wrapper::prepare_memory_dependencies(*prog);
        const auto* aa = prog->get_alias_analysis();
        ASSERT_NE(aa, nullptr);
        EXPECT_EQ(aa->class_of(prog->get_node("C1")), aa->class_of(prog->get_node("C2"))) << "one class: the shared predecessor links both holders";
        EXPECT_NE(aa->class_of(prog->get_node("C1"))->fallback & fb_role_conflict, 0u);
        expect_no_class_flags(*prog, padded ? "padded + plain holder" : "two plain holders");
    }
}

// The same producer on two edges of one concat: the generator must not make the output wrong (intel_cpu rejects this case explicitly).
TEST(concat_holder_pool, tb_same_producer_on_two_edges_of_one_concat) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(reorder("P", input_info("in1"), format::bfyx, data_types::f16));
        t.add(concatenation("C", {input_info("P"), input_info("P")}, 1));
        t.add(activation("A", input_info("C"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        return t;
    };
    const std::vector<run_input> ins = {{"in1", {1, 16, 32, 32}, 1}};
    auto r = run_ab("TBdup", eng, mk, ins);
    auto& prog = *r.on_net->get_program();
    std::cout << "[TBdup] C cbo=" << prog.get_node("C").can_be_optimized() << " promoted=" << r.on.promoted << "\n";
    // reference: relu(P) repeated twice along the feature axis
    const auto in = vals(16 * 32 * 32, 1);
    std::vector<float> expect;
    for (int rep = 0; rep < 2; ++rep) {
        for (float v : in) {
            expect.push_back(std::max(0.f, v));
        }
    }
    EXPECT_EQ(r.on.out.at("out"), expect) << "the concat of the same producer twice must be correct";
    EXPECT_EQ(r.off.out.at("out"), expect);
    // unconditional: whether the optimizer keeps the concat in place or not, the same producer on two edges is never promoted
    EXPECT_EQ(r.on.promoted, 0u) << "same producer on two edges: must not be promoted (C cbo=" << prog.get_node("C").can_be_optimized() << ")";
}

// ===============================================================================================================================
// oneDNN in-place sum interplay (TB11 / TB12 / TB23 oneDNN variants). GTEST_SKIP without supports_immad (a SKIP counts as "not covered").
// ===============================================================================================================================
namespace {

#define P1B_SKIP_IF_NO_IMMAD()                                                                    \
    do {                                                                                          \
        if (!get_test_engine().get_device_info().supports_immad) {                                \
            GTEST_SKIP() << "the oneDNN in-place sum needs a oneDNN device (supports_immad)";    \
        }                                                                                         \
    } while (0)

layout f16_l(size_t f) {
    return layout{ov::PartialShape{1, static_cast<int64_t>(f), 32, 32}, data_types::f16, format::bfyx};
}

memory::ptr make_weights_f16(engine& eng, size_t O, size_t I, size_t K) {
    auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(O), static_cast<int64_t>(I), static_cast<int64_t>(K), static_cast<int64_t>(K)},
                                          data_types::f16, format::bfyx});
    std::vector<ov::float16> w(O * I * K * K);
    for (size_t o = 0; o < O; o++) {
        for (size_t i = 0; i < I; i++) {
            for (size_t ky = 0; ky < K; ky++) {
                for (size_t kx = 0; kx < K; kx++) {
                    const bool centre = (ky == K / 2) && (kx == K / 2);
                    w[((o * I + i) * K + ky) * K + kx] = ov::float16((o == i && centre) ? 0.5f : (((o + 2 * i + ky + kx) % 7 == 0) ? 0.125f : 0.f));
                }
            }
        }
    }
    set_values(mem, w);
    return mem;
}

ov::intel_gpu::ImplementationDesc impl_desc(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", t};
}

ExecutionConfig onednn_cfg(engine& eng, bool holder_pool, const ov::intel_gpu::ImplForcingMap& forced) {
    auto cfg = make_cfg(eng, holder_pool);
    cfg.set_property(ov::intel_gpu::force_implementations(forced));
    return cfg;
}

void add_conv(topology& t, const primitive_id& id, const primitive_id& in, const primitive_id& w, int pad) {
    t.add(convolution(id, input_info(in), w, "", 1, {1, 1}, {1, 1}, {pad, pad}, {pad, pad}, false));
}

memory::ptr f16_input(engine& eng, size_t f, int variant) {
    auto mem = eng.allocate_memory(f16_l(f));
    std::vector<ov::float16> v(mem->count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(variant == 1 ? static_cast<float>(i % 17) * 0.125f - 1.0f : static_cast<float>((i * 3) % 13) * 0.0625f - 0.5f);
    }
    set_values(mem, v);
    return mem;
}

std::map<std::string, std::vector<float>> run_f16(network& net, const std::map<std::string, memory::ptr>& inputs) {
    for (const auto& kv : inputs) {
        net.set_input_data(kv.first, kv.second);
    }
    auto outs = net.execute();
    std::map<std::string, std::vector<float>> r;
    for (auto& o : outs) {
        r[o.first] = read_f32(o.second.get_memory());
    }
    return r;
}

// sum_into_concat = false: sum with an unrelated activation X; true: the sum dependency is the concat C (F10 control graph)
topology topo_onednn_sum(engine& eng, bool sum_into_concat) {
    topology t;
    t.add(input_layout("in1", f16_l(16)));
    t.add(input_layout("in3", f16_l(32)));
    t.add(cldnn::data("w_pq", make_weights_f16(eng, 16, 16, 1)));
    add_conv(t, "P", "in1", "w_pq", 0);
    add_conv(t, "Q", "in1", "w_pq", 0);
    t.add(concatenation("C", {input_info("P"), input_info("Q")}, 1));
    t.add(cldnn::data("w_w", make_weights_f16(eng, 32, 32, 3)));
    add_conv(t, "W", "in3", "w_w", 1);
    if (sum_into_concat) {
        t.add(eltwise("S", {input_info("W"), input_info("C")}, eltwise_mode::sum));
    } else {
        t.add(input_layout("in2", f16_l(32)));
        t.add(activation("X", input_info("in2"), activation_func::relu));
        t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
        t.add(reorder("out_C", input_info("C"), format::bfyx, data_types::f32));
    }
    t.add(reorder("out_S", input_info("S"), format::bfyx, data_types::f32));
    return t;
}

ov::intel_gpu::ImplForcingMap onednn_forced(bool sum_into_concat) {
    ov::intel_gpu::ImplForcingMap m = {{"P", impl_desc(impl_types::ocl)}, {"Q", impl_desc(impl_types::ocl)}, {"C", impl_desc(impl_types::ocl)}, {"W", impl_desc(impl_types::onednn)}};
    if (!sum_into_concat) {
        m.emplace("X", impl_desc(impl_types::ocl));
    }
    return m;
}

std::map<std::string, memory::ptr> onednn_inputs(engine& eng, bool sum_into_concat) {
    std::map<std::string, memory::ptr> in{{"in1", f16_input(eng, 16, 1)}, {"in3", f16_input(eng, 32, 2)}};
    if (!sum_into_concat) {
        in.emplace("in2", f16_input(eng, 32, 3));
    }
    return in;
}

}  // namespace

// TB11. A promoted holder next to an unrelated oneDNN in-place sum: the class is promoted, the sum writer / dependency / owner carry no class
// route (B4'), outputs are identical with the switch on and off.
TEST(concat_holder_pool_onednn, tb11_promoted_holder_next_to_an_unrelated_sum) {
    P1B_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto t_off = topo_onednn_sum(eng, false);
    auto t_on = topo_onednn_sum(eng, false);
    auto net_off = get_network(eng, t_off, onednn_cfg(eng, false, onednn_forced(false)), get_test_stream_ptr(), false);
    auto net_on = get_network(eng, t_on, onednn_cfg(eng, true, onednn_forced(false)), get_test_stream_ptr(), false);
    auto& prog = *net_on->get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    ASSERT_NE(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog.get_node("W")), -1) << "precondition: W carries the in-place sum\n" << dump_nodes(prog);
    EXPECT_TRUE(prog.get_node("C").is_pool_class_root()) << dump_nodes(prog);
    EXPECT_EQ(class_level::check_b4_prime(prog), "");
    for (auto id : {"W", "X"}) {
        EXPECT_NE(route_of(prog.get_node(id)), primitive_inst::alloc_route::POOL_AS_ROOT) << id;
        EXPECT_NE(route_of(prog.get_node(id)), primitive_inst::alloc_route::ZERO_ALLOC_MEMBER) << id;
    }
    const auto o_off = run_f16(*net_off, onednn_inputs(eng, false));
    const auto o_on = run_f16(*net_on, onednn_inputs(eng, false));
    EXPECT_EQ(o_off, o_on);
    EXPECT_TRUE(lifetime_violations(*net_on).empty());
}

// TB12. The oneDNN sum writes into the in-place concat itself (F10 control graph: sum kept): the holder is the owner of a sum chain (defect B
// marking, can_share_buffer(false)), its class has an in-place member and is never promoted. Outputs are identical.
TEST(concat_holder_pool_onednn, tb12_sum_into_the_holder_blocks_the_promotion) {
    P1B_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto t_off = topo_onednn_sum(eng, true);
    auto t_on = topo_onednn_sum(eng, true);
    auto net_off = get_network(eng, t_off, onednn_cfg(eng, false, onednn_forced(true)), get_test_stream_ptr(), false);
    auto net_on = get_network(eng, t_on, onednn_cfg(eng, true, onednn_forced(true)), get_test_stream_ptr(), false);
    auto& prog = *net_on->get_program();
    ASSERT_TRUE(prog.get_node("C").can_be_optimized()) << dump_nodes(prog);
    const bool sum_kept = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog.get_node("W")) != -1;
    std::cout << "[TB12] sum kept=" << sum_kept << " C cbs=" << prog.get_node("C").can_share_buffer() << "\n";
    ASSERT_TRUE(sum_kept) << "precondition: the sum into the concat is legal in the control graph\n" << dump_nodes(prog);
    EXPECT_FALSE(prog.get_node("C").is_pool_class_root()) << "the holder of a sum chain stays outside the pool\n" << dump_nodes(prog);
    EXPECT_FALSE(prog.get_node("P").has_alias_pool_root());
    EXPECT_EQ(class_level::check_b4_prime(prog), "");
    const auto* aa = prog.get_alias_analysis();
    const auto* c = aa->class_of(prog.get_node("C"));
    ASSERT_NE(c, nullptr);
    const auto mask = class_level::pool_admission_reject_mask(*c, prog, [&](uint32_t u) { return aa->uid_is_ambiguous(u); });
    EXPECT_NE(mask & (class_level::rj_member_kind | class_level::rj_root_not_shareable | class_level::rj_reused_eltw), 0u) << class_level::to_string(mask);
    const auto o_off = run_f16(*net_off, onednn_inputs(eng, true));
    const auto o_on = run_f16(*net_on, onednn_inputs(eng, true));
    EXPECT_EQ(o_off, o_on);

    // TB23 (oneDNN variant a): force a root flag on a class with an in-place member: B4' must report it
    prog.get_node("C").set_pool_class_root(true);
    prog.get_node("C").can_share_buffer(true);
    EXPECT_NE(class_level::check_b4_prime(prog), "") << "a promoted class with an in-place member (or a writer with a class route) must be reported";
}

// TB23 (oneDNN variant b): the dependency of a sum writer carries a class route -> B4' reports it.
TEST(concat_holder_pool_onednn, tb23_b4_prime_reports_a_class_route_on_the_sum_dependency) {
    P1B_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto t_on = topo_onednn_sum(eng, false);
    auto net_on = get_network(eng, t_on, onednn_cfg(eng, true, onednn_forced(false)), get_test_stream_ptr(), false);
    auto& prog = *net_on->get_program();
    ASSERT_NE(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog.get_node("W")), -1);
    ASSERT_EQ(class_level::check_b4_prime(prog), "");
    prog.get_node("X").set_alias_pool_root_uid(static_cast<uint32_t>(prog.get_node("C").get_unique_id()));  // X becomes a zero-allocation member
    EXPECT_NE(class_level::check_b4_prime(prog), "") << "dependency of the sum writer must never be a class member";
    prog.get_node("X").set_alias_pool_root_uid(program_node::invalid_alias_pool_root_uid);
    EXPECT_EQ(class_level::check_b4_prime(prog), "");
}

// L1-12 (a): a hand-flagged in-place concat(P, P): rejected, P keeps allocating, data is correct (reference computed on the host).
TEST(concat_holder_pool_hand, tb_dup_a_same_producer_twice_is_rejected_and_data_correct) {
    auto& eng = get_test_engine();
    auto build = [&](bool holder_pool, topology& t) {
        auto prog = make_hand(eng, t, holder_pool, QueueTypes::in_order, false, false, {"O"});
        prog->get_node("C").can_be_optimized(true);
        for (const auto& d : prog->get_node("C").get_dependencies()) {
            d.first->can_share_buffer(false);
        }
        program_wrapper::prepare_memory_dependencies(*prog);
        return prog;
    };
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(activation("P", input_info("in1"), activation_func::relu));
    t.add(concatenation("C", {input_info("P"), input_info("P")}, 1));
    t.add(activation("O", input_info("C"), activation_func::abs));
    auto prog = build(true, t);
    const auto* aa = prog->get_alias_analysis();
    ASSERT_NE(aa, nullptr);
    const auto mask = class_level::pool_admission_reject_mask(*aa->class_of(prog->get_node("C")), *prog, [](uint32_t) { return false; });
    EXPECT_NE(mask & class_level::rj_duplicate_root_dep, 0u) << class_level::to_string(mask);
    EXPECT_EQ(count_roots(*prog), 0u);
    EXPECT_FALSE(prog->get_node("P").has_alias_pool_root()) << "P must keep allocating";
    EXPECT_EQ(route_of(prog->get_node("P")), primitive_inst::alloc_route::DIRECT);
}

// (a, data): the same shape through the real pipeline built with the switch on: outputs equal the host reference.
TEST(concat_holder_pool, tb_dup_a_data_matches_reference_with_switch_on) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(reorder("P", input_info("in1"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P"), input_info("P")}, 1));
    t.add(activation("A", input_info("C"), activation_func::relu));
    t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
    auto net = get_network(eng, t, make_cfg(eng, true), get_test_stream_ptr(), false);
    const auto o = run_net(*net, eng, {{"in1", {1, 16, 32, 32}, 1}});
    const auto in = vals(16 * 32 * 32, 1);
    std::vector<float> expect;
    for (int rep = 0; rep < 2; ++rep) {
        for (float v : in) {
            expect.push_back(std::max(0.f, v));
        }
    }
    EXPECT_EQ(o.at("out"), expect);
    EXPECT_EQ(count_roots(*net->get_program()), 0u);
}

// (b): the mask bit on a fabricated duplicate dependency of the root (white box).
TEST(concat_holder_pool_hand, tb_dup_b_mask_bit) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(activation("P", input_info("in1"), activation_func::relu));
    t.add(concatenation("C", {input_info("P"), input_info("P")}, 1));
    t.add(activation("O", input_info("C"), activation_func::abs));
    auto prog = make_hand(eng, t, true, QueueTypes::in_order, false, false, {"O"});
    prog->get_node("C").can_be_optimized(true);
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_duplicate_root_dep, 0u);
    auto t2 = hand_concat_topology();
    auto prog2 = hand_promoted(eng, t2);
    EXPECT_EQ(mask_for(*prog2, "C") & class_level::rj_duplicate_root_dep, 0u) << "distinct producers: bit not set";
}

// M2 (hand): the duplicate-producer rule covers cascaded concats and a producer next to its own view.
TEST(concat_holder_pool_hand, tb_dup_c_cascade_inner_and_outer_share_a_producer) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(input_layout("in2", in_l()));
    t.add(activation("P", input_info("in1"), activation_func::relu));
    t.add(activation("Q", input_info("in2"), activation_func::relu));
    t.add(concatenation("Cin", {input_info("P"), input_info("Q")}, 1));
    t.add(concatenation("Cout", {input_info("Cin"), input_info("P")}, 1));  // P feeds the inner and the outer concat
    t.add(activation("O", input_info("Cout"), activation_func::abs));
    auto prog = make_hand(eng, t, true, QueueTypes::in_order, false, false, {"O"});
    prog->get_node("Cin").can_be_optimized(true);
    prog->get_node("Cout").can_be_optimized(true);
    for (const char* c : {"Cin", "Cout"}) {
        for (const auto& d : prog->get_node(c).get_dependencies()) {
            d.first->can_share_buffer(false);
        }
    }
    auto aa = alias_analysis::build(*prog);
    const auto* c = aa.class_of(prog->get_node("Cout"));
    ASSERT_NE(c, nullptr);
    const auto mask = class_level::pool_admission_reject_mask(*c, *prog, [](uint32_t) { return false; });
    EXPECT_NE(mask & class_level::rj_duplicate_root_dep, 0u) << class_level::to_string(mask);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "cascade with a shared producer");
}

TEST(concat_holder_pool_hand, tb_dup_d_producer_and_its_own_view_in_two_slots) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(activation("P", input_info("in1"), activation_func::relu));
    t.add(reshape("V", input_info("P"), false, std::vector<int64_t>{1, 16, 32, 32}, ov::PartialShape{1, 16, 32, 32}));
    t.add(concatenation("C", {input_info("P"), input_info("V")}, 1));
    t.add(activation("O", input_info("C"), activation_func::abs));
    auto prog = make_hand(eng, t, true, QueueTypes::in_order, false, false, {"O"});
    prog->get_node("C").can_be_optimized(true);
    prog->get_node("V").can_be_optimized(true);  // V is a settled view of P
    EXPECT_NE(mask_for(*prog, "C") & class_level::rj_duplicate_root_dep, 0u);
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "producer and its view");
}

// C-6 (a): a read_value as slot predecessor. (kv_cache is not constructible in a hand graph without a variable setup; read_value shares the persistent /
// external-root classification that rejects kv classes.)
TEST(concat_holder_pool_hand, c6a_read_value_slot_is_rejected_and_no_flag_written) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", in_l()));
    t.add(input_layout("in2", in_l()));
    t.add(read_value{"RV", {input_info("in1")}, "v0", {in_l()}});
    t.add(activation("P2", input_info("in2"), activation_func::abs));
    t.add(concatenation("C", {input_info("RV"), input_info("P2")}, 1));
    t.add(activation("O", input_info("C"), activation_func::abs));
    t.add(assign("as", {input_info("P2")}, "v0", in_l()));
    auto prog = make_hand(eng, t, true, QueueTypes::in_order, false, false, {"O", "as"});
    prog->get_node("C").can_be_optimized(true);
    const auto mask = mask_for(*prog, "C");
    EXPECT_NE(mask & class_level::rj_placement, 0u) << class_level::to_string(mask) << " (L1-1: persistent class)";
    EXPECT_NE(mask & class_level::rj_external_root, 0u) << class_level::to_string(mask) << " (L1-10)";
    program_wrapper::prepare_memory_dependencies(*prog);
    expect_no_class_flags(*prog, "read_value slot");
}

// C-6 (b): a binary_mul_inplace writer (oneDNN FC with a fused eltwise prod) whose second operand is the holder. F17 normally refuses a concat
// as reused dependency (reused index -1, then the holder stays promotable); if the alias were ever allowed, L1-8 has to reject the class.
TEST(concat_holder_pool_onednn, c6b_mul_inplace_writer_with_the_holder_as_dependency) {
    P1B_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto mk = [&]() {
        topology t;
        t.add(input_layout("in3", layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx}));
        t.add(input_layout("in4", layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx}));
        t.add(input_layout("inw", layout{ov::PartialShape{32, 16}, data_types::f16, format::bfyx}));
        t.add(activation("CP", input_info("in3"), activation_func::abs));
        t.add(activation("CQ", input_info("in4"), activation_func::relu));
        t.add(concatenation("C", {input_info("CP"), input_info("CQ")}, 0));
        auto wm = eng.allocate_memory(layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx});
        set_values(wm, std::vector<ov::float16>(wm->count(), ov::float16(0.0625f)));
        t.add(cldnn::data("wW", wm));
        t.add(fully_connected("W", input_info("inw"), "wW"));
        t.add(eltwise("P", {input_info("W"), input_info("C")}, eltwise_mode::prod));
        t.add(reorder("out", input_info("P"), format::bfyx, data_types::f32));
        return t;
    };
    ov::intel_gpu::ImplForcingMap forced = {{"CP", impl_desc(impl_types::ocl)}, {"CQ", impl_desc(impl_types::ocl)}, {"C", impl_desc(impl_types::ocl)},
                                            {"W", impl_desc(impl_types::onednn)}};
    auto t_off = mk();
    auto t_on = mk();
    auto net_off = get_network(eng, t_off, onednn_cfg(eng, false, forced), get_test_stream_ptr(), false);
    auto net_on = get_network(eng, t_on, onednn_cfg(eng, true, forced), get_test_stream_ptr(), false);
    auto& prog = *net_on->get_program();
    if (!prog.get_node("C").can_be_optimized()) {
        GTEST_SKIP() << "the optimizer does not keep this 2D concat in place (reorders around it): the writer/holder combination is not constructible here (NOT covered)";
    }
    const int32_t reused = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(prog.get_node("W"));
    std::cout << "[C6b] W reused eltw idx=" << reused << " C promoted=" << prog.get_node("C").is_pool_class_root() << "\n";
    if (reused != -1) {
        EXPECT_FALSE(prog.get_node("C").is_pool_class_root()) << "a writer aliasing the holder must keep the class outside the pool";
    }
    EXPECT_EQ(class_level::check_b4_prime(prog), "");
    auto run = [&](network& n) {
        auto m1 = eng.allocate_memory(layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx});
        auto m2 = eng.allocate_memory(layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx});
        auto m3 = eng.allocate_memory(layout{ov::PartialShape{32, 16}, data_types::f16, format::bfyx});
        set_values(m1, std::vector<ov::float16>(m1->count(), ov::float16(1.f)));
        set_values(m2, std::vector<ov::float16>(m2->count(), ov::float16(2.f)));
        set_values(m3, std::vector<ov::float16>(m3->count(), ov::float16(0.5f)));
        return run_f16(n, {{"in3", m1}, {"in4", m2}, {"inw", m3}});
    };
    EXPECT_EQ(run(*net_off), run(*net_on));
}

// M2 (pipeline): the same producer in an inner and the outer concat. Whatever the optimizer does, the output is right and nothing is promoted.
TEST(concat_holder_pool, tb_dup_e_cascade_shared_producer_pipeline) {
    auto& eng = get_test_engine();
    auto mk = []() {
        topology t;
        t.add(input_layout("in1", in_l()));
        t.add(input_layout("in2", in_l()));
        t.add(reorder("P", input_info("in1"), format::bfyx, data_types::f16));
        t.add(reorder("Q", input_info("in2"), format::bfyx, data_types::f16));
        t.add(concatenation("Cin", {input_info("P"), input_info("Q")}, 1));
        t.add(concatenation("Cout", {input_info("Cin"), input_info("P")}, 1));
        t.add(activation("A", input_info("Cout"), activation_func::relu));
        t.add(reorder("out", input_info("A"), format::bfyx, data_types::f32));
        return t;
    };
    auto r = run_ab("TBdupE", eng, mk, {{"in1", {1, 16, 32, 32}, 1}, {"in2", {1, 16, 32, 32}, 2}});
    auto& prog = *r.on_net->get_program();
    std::cout << "[TBdupE] Cin cbo=" << prog.get_node("Cin").can_be_optimized() << " Cout cbo=" << prog.get_node("Cout").can_be_optimized() << " promoted=" << r.on.promoted << "\n";
    const auto in1 = vals(16 * 32 * 32, 1), in2 = vals(16 * 32 * 32, 2);
    std::vector<float> expect;
    for (const auto* v : {&in1, &in2, &in1}) {
        for (float x : *v) {
            expect.push_back(std::max(0.f, x));
        }
    }
    EXPECT_EQ(r.on.out.at("out"), expect);
    EXPECT_EQ(r.on.promoted, 0u) << dump_nodes(prog);
}

// =================================================================================================================================
// P1-C-1 (design 18 r3, CL-2): the output-view owner gap that L3 step (5) closes, for an in-place concat holder.
// Y -> r1 (reorder, network output, compatible-but-not-identical layout), Y -> r2 (identical to r1, merged away by remove_redundant_reorders: the
// kept r1 inherits r2's user) -> U -> Z, P (two poolings) -> H = concat(Z, P) (in-place holder). U (not a slot) reads the view and both slots are
// reachable from Y, so the union of the members' restrictions (L3 step 2) does not contain Y by itself: only step (5) closes the gap. After the merge y = r1 is a can_be_optimized + requires_reinterpret
// view of Y that is an output AND has a user, and Y has no other user: oooq and skipped_branch do not relate Y and H, the direct insertion of r1
// by past_outputs skips the fold of r1 to Y (pass_manager.h:347-350).
// =================================================================================================================================
namespace {

struct out_view_topo {
    topology t;
    std::vector<std::string> outs;
};

out_view_topo topo_output_view_owner(bool plain_user_after) {
    out_view_topo r;
    auto& t = r.t;
    t.add(input_layout("in_y", layout{ov::PartialShape{1, 32, 32, 32}, data_types::f32, format::bfyx}));
    t.add(activation("Y", input_info("in_y"), activation_func::abs));
    t.add(reorder("r1", input_info("Y"), format::bfzyx, data_types::f32));
    t.add(reorder("r2", input_info("Y"), format::bfzyx, data_types::f32));
    t.add(activation("U", input_info("r2"), activation_func::abs));   // reads the (merged) output view; the slot Z does not read it directly
    t.add(eltwise("Z", {input_info("U"), input_info("U")}, eltwise_mode::sum, data_types::f16));   // 2|x| as f16: half the bytes of Y
    t.add(eltwise("P", {input_info("U"), input_info("U")}, eltwise_mode::prod, data_types::f16));   // x^2 as f16; both slots are fed by U (reachable from Y)
    t.add(concatenation("H", {input_info("Z"), input_info("P")}, 1));   // 64 x 32 x 32 f16 == the bytes of Y
    t.add(activation("A", input_info("H"), activation_func::abs));
    r.outs = {"r1", "A"};   // A is a network output: it never takes a pool block, so only H can expose the gap
    if (plain_user_after) {   // LEGACY characterisation: A is an ordinary pool user followed by an output reorder
        t.add(reorder("out_A", input_info("A"), format::bfyx, data_types::f32));
        r.outs = {"r1", "out_A"};
    }
    return r;
}

}  // namespace

namespace {

std::map<std::string, std::vector<float>> run_out_view(network& net, engine& eng) {
    auto y = eng.allocate_memory(layout{ov::PartialShape{1, 32, 32, 32}, data_types::f32, format::bfyx});
    set_values(y, vals(32 * 32 * 32, 1));
    net.set_input_data("in_y", y);
    auto outs = net.execute();
    std::map<std::string, std::vector<float>> res;
    for (auto& o : outs) {
        res[o.first] = read_f32(o.second.get_memory());
    }
    return res;
}

network::ptr out_view_network(engine& eng, bool holder_pool, bool plain_user_after, std::string* why = nullptr) {
    auto tp = topo_output_view_owner(plain_user_after);
    auto cfg = make_cfg(eng, holder_pool);
    cfg.set_property(ov::intel_gpu::custom_outputs(tp.outs));
    auto net = get_network(eng, tp.t, cfg, get_test_stream_ptr(), false);
    (void)why;
    return net;
}

// the preconditions of the output-view owner gap on a built program (design 18 r3 C-19): y = r1 is an output with a user, a cbo + requires_reinterpret view
// of Y, Y has no other user. Returns an empty string if they hold, otherwise the first one that does not.
std::string out_view_construction_gap(program& p) {
    auto& y = p.get_node("r1");
    auto& Y = p.get_node("Y");
    if (!y.is_output()) {
        return "r1 is not a network output";
    }
    if (y.get_users().empty()) {
        return "r1 has no user (the duplicate reorder was not merged into it)";
    }
    if (!y.can_be_optimized() || !y.as<reorder>().requires_reinterpret()) {
        return "r1 is not a can_be_optimized + requires_reinterpret view";
    }
    if (Y.get_users().size() != 1) {
        return "Y has more than one user";
    }
    return "";
}

}  // namespace

// C-19c (CL-2, concat holder variant): the owner Y of the output view y = r1 is not related to the holder H by oooq / skipped_branch (the gap exists in the
// switch-off restriction), the L3 step (5) closes it: H and Y restrict each other, never share a block, the lifetime oracle is clean, outputs equal the
// switch-off run and the reference.
TEST(concat_holder_pool, C_19c_output_view_owner_before_holder_never_shares) {
    auto& eng = get_test_engine();
    auto net_off = out_view_network(eng, false, false);
    auto net_on = out_view_network(eng, true, false);
    auto& off = *net_off->get_program();
    auto& on = *net_on->get_program();
    if (const auto why = out_view_construction_gap(on); !why.empty()) {
        GTEST_SKIP() << "construction not formed: " << why << "\n" << dump_nodes(on);
    }
    if (!on.get_node("H").is_pool_class_root()) {
        GTEST_SKIP() << "construction not formed: the holder H was not promoted\n" << dump_nodes(on);
    }
    const bool gap_in_legacy = !off.get_node("H").has_memory_dependency(off.get_node("Y").get_unique_id());
    if (!gap_in_legacy) {
        GTEST_SKIP() << "construction not formed: the LEGACY restriction of H already contains Y (the gap is not present)\n" << dump_nodes(off);
    }
    EXPECT_TRUE(on.get_node("H").has_memory_dependency(on.get_node("Y").get_unique_id())) << "L3 step (5) must add the owner Y of the view r1 to R(H)\n" << dump_nodes(on);
    EXPECT_TRUE(on.get_node("Y").has_memory_dependency(on.get_node("H").get_unique_id())) << "... and H to R(Y)";
    const auto o_off = run_out_view(*net_off, eng);
    const auto o_on = run_out_view(*net_on, eng);
    const auto xin = vals(32 * 32 * 32, 1);
    // output A (a plain output of the concat): equal between the two configurations. (The absolute value is not compared with a host reference: the f32 ->
    // f16 eltwise on a bfzyx-formatted 4D view is not the subject of this test and was not validated as a reference.)
    EXPECT_EQ(o_on.at("A"), o_off.at("A")) << "output A equals the switch-off run";
    std::vector<float> expect_r1;
    for (float x : xin) {
        expect_r1.push_back(std::fabs(x));
    }
    EXPECT_EQ(o_on.at("r1"), expect_r1) << "the output view r1 of Y keeps the data of Y: H did not take the block of Y";
    EXPECT_NE(buf_of(*net_on, "H"), buf_of(*net_on, "Y")) << "H must not take the block of Y while the output view of Y is alive";
    EXPECT_TRUE(lifetime_violations(*net_on).empty());
}

// C-19L (NB-V3, LEGACY characterisation, DISABLED): with every switch off, a plain pool user that comes after r1 in processing order (A, or N) has r1 in its
// restriction through past_outputs but not Y, and neither oooq nor skipped_branch relates it to Y (Y's only user is the output view r1, which reaches it): it can
// take the block of Y while the output view r1 of Y is alive, and the network output r1 is overwritten. REPRODUCED on this graph (the output differs from
// abs(in_y); A is the node that shares the block, see the printed line). The same gap is what L3 step (5) closes for promoted roots (sum root, concat holder).
TEST(concat_holder_pool, DISABLED_output_view_owner_skipped_fold_shares_block) {
    auto& eng = get_test_engine();
    auto net = out_view_network(eng, false, true);
    auto& prog = *net->get_program();
    if (const auto why = out_view_construction_gap(prog); !why.empty()) {
        GTEST_SKIP() << "construction not formed: " << why;
    }
    const auto out = run_out_view(*net, eng);
    std::vector<float> expect;
    for (float x : vals(32 * 32 * 32, 1)) {
        expect.push_back(std::fabs(x));
    }
    for (const char* id : {"A", "H", "Z", "U", "P", "out_A"}) {
        std::cout << "[C-19L] " << id << " shares the buffer of Y: " << (buf_of(*net, id) == buf_of(*net, "Y")) << " restriction has Y: "
                  << prog.get_node(id).has_memory_dependency(prog.get_node("Y").get_unique_id()) << "\n";
    }
    EXPECT_EQ(out.at("r1"), expect) << "LEGACY: the output view r1 of Y is corrupted when a later pool user takes the block of Y";
}
