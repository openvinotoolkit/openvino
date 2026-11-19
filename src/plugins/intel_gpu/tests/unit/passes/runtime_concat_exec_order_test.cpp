// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Pre-0 F9: unify the processing order with network::_exec_order for runtime in-place concats.
// Design: /home/ceciliapeng/repos/reports/memman/layer1/07_pre0_f9_design.zh-CN.md (r2 rulings in section 9).
//
// "Before F9" baselines are produced on the test side (restoring the old processing order / flags), never by a source switch.

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "concatenation_inst.h"
#include "eltwise_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "pass_manager.h"
#include "primitive_inst.h"
#include "primitive_inst_test_helper.h"
#include "program_node.h"
#include "program_wrapper.h"
#include "runtime_concat_order.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;
using namespace cldnn::runtime_concat_order;

namespace {

// ------------------------------------------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------------------------------------------
ExecutionConfig make_cfg(engine& eng, QueueTypes q, const std::vector<std::string>& outputs = {}) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (!outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(outputs));
    }
    if (q == QueueTypes::out_of_order) {
        // With oneDNN enabled (Arc), ExecutionConfig::finalize() forces the in-order queue, so oneDNN must be switched off. It has to be a
        // *user* property: set_property() with the default value is not recorded as user-set and finalize() would re-enable oneDNN.
        config.set_user_property({ov::intel_gpu::use_onednn(false)}, ov::OptionVisibility::RELEASE_INTERNAL);
    }
    return config;
}

layout dyn_layout(data_types dt = data_types::f32) {
    return layout{ov::PartialShape::dynamic(4), dt, format::bfyx};
}

// in1 -> P1 (reorder f16) --+
//                          concat C (axis 1) -> output (reorder f32)
// in2 -> P2 (reorder f16) --+
topology basic_topology() {
    topology t;
    t.add(input_layout("in1", dyn_layout()));
    t.add(input_layout("in2", dyn_layout()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    return t;
}

// basic_topology plus a consumer Y that is rewired (by the test, after prepare_buffer_fusing) to read P1 as well, i.e. P1 gets a
// second user and must not be re-issued by network::build_exec_order.
topology injected_topology() {
    topology t = basic_topology();
    t.add(input_layout("in3", dyn_layout()));
    t.add(input_layout("in4", dyn_layout()));
    t.add(reorder("Z", input_info("in3"), format::bfyx, data_types::f16));
    t.add(reorder("W", input_info("in4"), format::bfyx, data_types::f16));
    t.add(eltwise("Y", {input_info("Z"), input_info("W")}, eltwise_mode::sum));
    return t;
}

// in1 -> D (in1 + in1) -> P0 --+
//                             C -> output
// in2 -> Z (in2 + in2) -> P1 --+
topology branch_topology() {
    topology t;
    t.add(input_layout("in1", dyn_layout()));
    t.add(input_layout("in2", dyn_layout()));
    t.add(eltwise("D", {input_info("in1"), input_info("in1")}, eltwise_mode::sum));
    t.add(eltwise("Z", {input_info("in2"), input_info("in2")}, eltwise_mode::sum));
    t.add(reorder("P0", input_info("D"), format::bfyx, data_types::f16));
    t.add(reorder("P1", input_info("Z"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P0"), input_info("P1")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    return t;
}

using hook_t = std::function<void(program&)>;

enum class upto { compile, post, full };

// Same stages as program::build_program(), with a hook between pre_optimize_graph and graph compilation.
program::ptr build_staged(engine& eng, const topology& t, const ExecutionConfig& cfg, const hook_t& hook = nullptr, upto stage = upto::full) {
    auto prog = program::build_program(eng, t, cfg, /*is_internal=*/false, /*no_optimizations=*/true);
    program_wrapper::pre_optimize_graph(*prog, false);
    // The hook runs after prepare_buffer_fusing (inside pre_optimize_graph) and before compile_graph, so that implementations of
    // rewired nodes are selected for the rewired structure.
    if (hook) {
        hook(*prog);
    }
    program_wrapper::run_graph_compilation(*prog);
    if (stage == upto::compile) {
        return prog;
    }
    program_wrapper::post_optimize_graph(*prog, false);
    if (stage == upto::post) {
        return prog;
    }
    program_wrapper::prepare_memory_dependencies(*prog);
    program_wrapper::apply_opt_pass<build_implementations>(*prog);
    if (eng.get_device_info().has_separate_cache) {
        program_wrapper::transfer_memory_to_device(*prog);
    }
    return prog;
}

std::vector<primitive_id> po_ids(program& prog, bool filtered = false) {
    std::vector<primitive_id> ids;
    for (const auto* n : prog.get_processing_order()) {
        if (!filtered || is_executable_node(n)) {
            ids.push_back(n->id());
        }
    }
    return ids;
}

std::string join(const std::vector<primitive_id>& v) {
    std::string s;
    for (const auto& x : v) {
        s += x + " ";
    }
    return s;
}

// Rearranges the listed nodes so that they are contiguous in the given order (the position of the last one is kept).
void force_order(program& prog, const std::vector<primitive_id>& ids) {
    auto& po = prog.get_processing_order();
    for (size_t i = ids.size() - 1; i-- > 0;) {
        auto& node = prog.get_node(ids[i]);
        auto& next = prog.get_node(ids[i + 1]);
        po.erase(&node);
        po.insert(&next, &node);
    }
}

// Moves one node back before `before` (used to restore the pre-F9 order after the pass has run).
void move_before(program& prog, const primitive_id& id, const primitive_id& before) {
    auto& po = prog.get_processing_order();
    auto& node = prog.get_node(id);
    po.erase(&node);
    po.insert(&prog.get_node(before), &node);
}

size_t count_id(const std::vector<primitive_id>& ids, const primitive_id& id) {
    return static_cast<size_t>(std::count(ids.begin(), ids.end(), id));
}

// Small integers are exact in f16, so references are exact as well.
std::vector<float> make_ints(size_t n, int seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = static_cast<float>((static_cast<int>(i) * 7 + seed * 13) % 16);
    }
    return v;
}

memory::ptr make_input(engine& eng, const ov::Shape& s, const std::vector<float>& data) {
    auto l = layout{ov::PartialShape(s), data_types::f32, format::bfyx};
    auto m = eng.allocate_memory(l);
    if (!data.empty()) {
        set_values<float>(m, data);
    }
    return m;
}

// The memory object behind an output may be larger than the logical output (shape predictor pre-allocation), so counts are taken from the
// output layout of the network, not from memory->get_layout().
template <class Outs>
std::vector<float> read_out_f32(network& net, Outs& outs, const primitive_id& id) {
    auto m = outs.at(id).get_memory();
    std::vector<float> r(net.get_output_layout(id).count());
    if (r.empty()) {
        return r;
    }
    mem_lock<float> lock(m, get_test_stream());
    std::copy(lock.begin(), lock.begin() + r.size(), r.begin());
    return r;
}

template <class Outs>
std::vector<float> read_out_f16(network& net, Outs& outs, const primitive_id& id) {
    auto m = outs.at(id).get_memory();
    std::vector<float> r(net.get_output_layout(id).count());
    if (r.empty()) {
        return r;
    }
    mem_lock<ov::float16> lock(m, get_test_stream());
    for (size_t i = 0; i < r.size(); i++) {
        r[i] = static_cast<float>(lock[i]);
    }
    return r;
}

// Runs one round of the basic / branch / injected topologies and checks the outputs. `with_y`: injected topology, Y = P1 + W.
void run_round_and_check(network& net, engine& eng, const ov::Shape& s1, const ov::Shape& s2, int seed, bool with_y, bool branch = false) {
    const auto d1 = make_ints(ov::shape_size(s1), seed);
    const auto d2 = make_ints(ov::shape_size(s2), seed + 1);
    net.set_input_data("in1", make_input(eng, s1, d1));
    net.set_input_data("in2", make_input(eng, s2, d2));
    std::vector<float> d4;
    if (with_y) {
        d4 = make_ints(ov::shape_size(s1), seed + 3);
        net.set_input_data("in3", make_input(eng, s1, make_ints(ov::shape_size(s1), seed + 2)));
        net.set_input_data("in4", make_input(eng, s1, d4));
    }
    auto outs = net.execute();
    auto out = read_out_f32(net, outs, "output");

    std::vector<float> ref;
    for (auto v : d1) {
        ref.push_back(branch ? 2 * v : v);
    }
    for (auto v : d2) {
        ref.push_back(branch ? 2 * v : v);
    }
    if (ref.empty()) {
        return;  // fully empty round: the network leaves the previous output buffer in place, nothing to compare
    }
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(out[i], ref[i]) << "output[" << i << "] shape1=" << s1.to_string() << " shape2=" << s2.to_string();
    }
    if (with_y) {
        auto y = read_out_f16(net, outs, "Y");
        ASSERT_EQ(y.size(), d1.size());
        for (size_t i = 0; i < y.size(); i++) {
            ASSERT_EQ(y[i], d1[i] + d4[i]) << "Y[" << i << "]";
        }
    }
}

program::ptr build_injected(engine& eng, QueueTypes q) {
    return build_staged(eng, injected_topology(), make_cfg(eng, q, {"output", "Y"}), [](program& p) {
        auto& y = p.get_node("Y");
        y.replace_dependency(0, p.get_node("P1"), false);
    });
}

}  // namespace

// ==============================================================================================================================
// T1. Predecessor with a second user: executed twice before F9, concat cancelled after F9.
// ==============================================================================================================================

// C10: natural reproduction attempt via the "same input, same reorder" merge in remove_redundant_reorders (no dynamic guard).
// Whatever the merge does, the invariants must hold: no duplicate in the exec order, A2 is consistent.
TEST(runtime_concat_exec_order, t1_natural_merge_of_identical_reorders) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", dyn_layout()));
    t.add(reorder("P1", input_info("in1"), format::bfyx, data_types::f16));
    t.add(reorder("P2", input_info("in1"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    auto prog = program::build_program(eng, t, make_cfg(eng, QueueTypes::in_order), false, false);
    ASSERT_NE(prog, nullptr);
    const bool p1_exists = prog->has_node("P1");
    const bool p2_exists = prog->has_node("P2");
    std::cout << "[C10] P1 present=" << p1_exists << " P2 present=" << p2_exists << " C optimized=" << prog->get_node("C").can_be_optimized() << " C deps="
              << prog->get_node("C").get_dependencies().size() << std::endl;
    if (p1_exists && p2_exists) {
        std::cout << "[C10] P1 users=" << prog->get_node("P1").get_users().size() << " P2 users=" << prog->get_node("P2").get_users().size()
                  << " (merge not reproduced: two distinct single-user predecessors)" << std::endl;
    }
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    for (const auto& id : ids) {
        EXPECT_EQ(count_id(ids, id), 1u) << id << " executed more than once";
    }
    EXPECT_TRUE(check_exec_order(prog->get_processing_order(), ids).ok());
}

TEST(runtime_concat_exec_order, t1_multi_user_predecessor_cancelled) {
    auto& eng = get_test_engine();
    auto prog = build_injected(eng, QueueTypes::in_order);
    auto& c = prog->get_node("C");
    ASSERT_EQ(prog->get_node("P1").get_users().size(), 2u) << "injection did not take effect";

    // After F9: the unsafe runtime concat is cancelled.
    EXPECT_FALSE(c.can_be_optimized());
    EXPECT_FALSE(c.is_runtime_skippable());
    EXPECT_TRUE(must_cancel_runtime_concat(c) || !is_runtime_optimized_concat(&c));

    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    EXPECT_EQ(count_id(ids, "P1"), 1u) << join(ids);
    EXPECT_EQ(count_id(ids, "P2"), 1u) << join(ids);
    const auto res = check_exec_order(prog->get_processing_order(), ids);
    EXPECT_TRUE(res.ok()) << res.message;

    run_round_and_check(net, eng, {1, 16, 4, 4}, {1, 16, 4, 4}, 1, true);
    run_round_and_check(net, eng, {1, 8, 4, 4}, {1, 8, 4, 4}, 2, true);
}

// The injected structure reproduces the defect when the cancel rule is undone on the test side (the old behaviour): P1 appears twice.
TEST(runtime_concat_exec_order, t1_pre_f9_baseline_executes_predecessor_twice) {
    auto& eng = get_test_engine();
    auto prog = build_injected(eng, QueueTypes::in_order);
    auto& c = prog->get_node("C");
    ASSERT_FALSE(c.can_be_optimized());
    // Pre-F9 state: concat keeps the runtime in-place flags (they are what pbf produced).
    c.can_be_optimized(true);
    c.set_runtime_skippable(true);
    ASSERT_TRUE(is_runtime_optimized_concat(&c));
    ASSERT_TRUE(must_cancel_runtime_concat(c));

    // Debug build (NDEBUG undefined): the F9 assertion in network::build_exec_order rejects the structure, the message must carry the A1 tag.
    // Release build: no assertion, the defect is visible as P1 executed twice and check_exec_order reports A1.
    std::vector<primitive_id> ids;
    bool threw = false;
    try {
        network net(prog, 0);
        ids = net.get_executed_primitive_ids();
    } catch (const ov::Exception& e) {
        threw = true;
        const std::string msg = e.what();
        std::cout << "[T1 pre-F9] network ctor rejected the structure (Debug path): " << msg << std::endl;
        EXPECT_TRUE(msg.find("F9") != std::string::npos) << msg;
        EXPECT_TRUE(msg.find("A1") != std::string::npos || msg.find("present twice") != std::string::npos) << msg;
    }
    if (threw) {
        return;
    }
    EXPECT_EQ(count_id(ids, "P1"), 2u) << join(ids);
    const auto res = check_exec_order(prog->get_processing_order(), ids);
    EXPECT_FALSE(res.ok());
    EXPECT_TRUE(res.message.find("A1") != std::string::npos) << res.message;
    std::cout << "[T1 pre-F9] exec order: " << join(ids) << " | check: " << res.message << std::endl;
}

// ==============================================================================================================================
// T2. The "shape updated by someone else" flag must not outlive a round (normal path, zero-length rounds included).
// ==============================================================================================================================
TEST(runtime_concat_exec_order, t2_flags_clear_after_rounds_including_zero_length) {
    auto& eng = get_test_engine();
    auto prog = build_staged(eng, basic_topology(), make_cfg(eng, QueueTypes::in_order));
    ASSERT_TRUE(prog->get_node("C").can_be_optimized());
    network net(prog, 0);
    const std::vector<std::pair<ov::Shape, ov::Shape>> rounds = {
        {{1, 16, 4, 4}, {1, 16, 4, 4}},
        {{1, 16, 4, 4}, {1, 0, 4, 4}},  // zero-length second predecessor
        {{1, 0, 4, 4}, {1, 0, 4, 4}},   // zero-length round
        {{1, 8, 2, 2}, {1, 8, 2, 2}},
        {{1, 24, 4, 4}, {1, 4, 4, 4}},
    };
    int seed = 1;
    for (const auto& r : rounds) {
        run_round_and_check(net, eng, r.first, r.second, seed++, false);
        for (const char* id : {"P1", "P2", "C"}) {
            EXPECT_FALSE(PrimitiveInstTestHelper::get_update_shape_done_by_other(net.get_primitive(id)))
                << id << " keeps _update_shape_done_by_other after round with shapes " << r.first.to_string() << "/" << r.second.to_string();
            // C legitimately keeps _allocation_done_by_other == true while its shape is unchanged (it is only reset on SHAPE_CHANGED, see
            // primitive_inst.cpp do_runtime_in_place_concat), so the flag is asserted for the predecessors only.
            if (std::string(id) != "C") {
                EXPECT_FALSE(PrimitiveInstTestHelper::get_allocation_done_by_other(net.get_primitive(id)))
                    << id << " keeps _allocation_done_by_other after round with shapes " << r.first.to_string() << "/" << r.second.to_string();
            }
        }
    }
}

// ==============================================================================================================================
// T3. skipped_branch sees the real time axis.
// ==============================================================================================================================
namespace {
struct branch_state {
    bool d_z = false;   // D and Z are mutually exclusive
    bool z_d = false;
};

branch_state skipped_branch_state(engine& eng, bool apply_f9, bool full_prepare) {
    auto prog = build_staged(eng, branch_topology(), make_cfg(eng, QueueTypes::in_order), nullptr, upto::compile);
    // Original (DFS-like) order that leaves Z between D's slot and C: P0 sits right behind D, ahead of Z.
    force_order(*prog, {"in1", "D", "P0", "in2", "Z", "P1", "C"});
    EXPECT_TRUE(prog->get_node("C").can_be_optimized());
    if (apply_f9) {
        program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*prog);
    }
    if (full_prepare) {
        program_wrapper::prepare_memory_dependencies(*prog);
    } else {
        program_wrapper::apply_opt_pass<skipped_branch_memory_dependencies>(*prog);
    }
    auto& d = prog->get_node("D");
    auto& z = prog->get_node("Z");
    branch_state st;
    st.d_z = d.has_memory_dependency(static_cast<uint32_t>(z.get_unique_id()));
    st.z_d = z.has_memory_dependency(static_cast<uint32_t>(d.get_unique_id()));
    return st;
}
}  // namespace

TEST(runtime_concat_exec_order, t3_skipped_branch_alone_pre_and_post) {
    auto& eng = get_test_engine();
    const auto pre = skipped_branch_state(eng, false, false);
    const auto post = skipped_branch_state(eng, true, false);
    std::cout << "[T3] skipped_branch alone: pre-F9 D-Z=" << pre.d_z << "/" << pre.z_d << " post-F9 D-Z=" << post.d_z << "/" << post.z_d << std::endl;
    EXPECT_FALSE(pre.d_z);
    EXPECT_FALSE(pre.z_d);
    EXPECT_TRUE(post.d_z);
    EXPECT_TRUE(post.z_d);
}

TEST(runtime_concat_exec_order, t3_full_prepare_memory_dependencies_masks_the_difference) {
    auto& eng = get_test_engine();
    const auto pre = skipped_branch_state(eng, false, true);
    const auto post = skipped_branch_state(eng, true, true);
    std::cout << "[T3] full prepare_memory_dependencies: pre-F9 D-Z=" << pre.d_z << "/" << pre.z_d << " post-F9 D-Z=" << post.d_z << "/" << post.z_d
              << std::endl;
    // oooq closes the D-Z gap in both cases (D and Z have no path between them), so the F9 difference is masked.
    EXPECT_TRUE(pre.d_z);
    EXPECT_TRUE(pre.z_d);
    EXPECT_TRUE(post.d_z);
    EXPECT_TRUE(post.z_d);
}

// ==============================================================================================================================
// T4. Unification, idempotency, cache round trip, simulated old cache.
// ==============================================================================================================================
namespace {
program::ptr roundtrip(engine& eng, program& prog, const ExecutionConfig& cfg) {
    cldnn::membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        cldnn::BinaryOutputBuffer ob = cldnn::BinaryOutputBuffer(out_mem);
        ob.set_stream(get_test_stream_ptr().get());
        prog.save(ob);
    }
    std::istream in_mem(&mem_buf);
    cldnn::BinaryInputBuffer ib = cldnn::BinaryInputBuffer(in_mem, eng);
    auto imported = std::make_shared<cldnn::program>(eng, cfg);
    imported->load(ib);
    return imported;
}
}  // namespace

TEST(runtime_concat_exec_order, t4_exec_order_equals_filtered_po_and_pass_is_idempotent) {
    auto& eng = get_test_engine();
    {
        // Precondition: before the pass the PO is NOT normalized (explicit order that leaves Z between P0 and C), after it is.
        auto pre = build_staged(eng, branch_topology(), make_cfg(eng, QueueTypes::in_order), nullptr, upto::compile);
        force_order(*pre, {"in1", "D", "P0", "in2", "Z", "P1", "C"});
        ASSERT_TRUE(pre->get_node("C").can_be_optimized());
        EXPECT_FALSE(check_exec_order(pre->get_processing_order(), po_ids(*pre, true)).normalized);
        program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*pre);
        EXPECT_TRUE(check_exec_order(pre->get_processing_order(), po_ids(*pre, true)).normalized);
    }
    auto prog = build_staged(eng, branch_topology(), make_cfg(eng, QueueTypes::in_order));
    ASSERT_TRUE(prog->get_node("C").can_be_optimized());

    const auto po = po_ids(*prog);
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    EXPECT_EQ(ids, po_ids(*prog, true)) << "exec: " << join(ids) << " po: " << join(po_ids(*prog, true));
    const auto res = check_exec_order(prog->get_processing_order(), ids);
    EXPECT_TRUE(res.normalized);
    EXPECT_TRUE(res.ok()) << res.message;

    // predecessors sit directly in front of C, in dependency order
    const auto c_pos = std::find(ids.begin(), ids.end(), "C") - ids.begin();
    ASSERT_GE(c_pos, 2);
    EXPECT_EQ(ids[c_pos - 2], "P0");
    EXPECT_EQ(ids[c_pos - 1], "P1");

    // idempotency
    program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*prog);
    program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*prog);
    EXPECT_EQ(po_ids(*prog), po);

    run_round_and_check(net, eng, {1, 8, 3, 3}, {1, 4, 3, 3}, 1, false, true);
    run_round_and_check(net, eng, {1, 2, 5, 5}, {1, 12, 5, 5}, 2, false, true);
}

TEST(runtime_concat_exec_order, t4_cache_round_trip_keeps_exec_order_and_outputs) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order);
    auto prog = build_staged(eng, branch_topology(), cfg);
    network net_ref(prog, 0);
    const auto ref_ids = net_ref.get_executed_primitive_ids();

    program::ptr imported;
    ASSERT_NO_THROW(imported = roundtrip(eng, *prog, cfg));
    EXPECT_EQ(po_ids(*imported), po_ids(*prog));
    network net(imported, 0);
    EXPECT_EQ(net.get_executed_primitive_ids(), ref_ids);
    run_round_and_check(net, eng, {1, 8, 3, 3}, {1, 4, 3, 3}, 1, false, true);
    run_round_and_check(net, eng, {1, 3, 2, 2}, {1, 5, 2, 2}, 2, false, true);
}

// Old cache: the processing order was saved before F9, i.e. P0 sits ahead of Z. A2 must be skipped and the exec order must be the same
// as the one produced with the F9 order (the network side still performs the move).
TEST(runtime_concat_exec_order, t4_simulated_old_cache_skips_a2_and_keeps_exec_order) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order);

    auto new_prog = build_staged(eng, branch_topology(), cfg);
    network net_new(new_prog, 0);
    const auto new_ids = net_new.get_executed_primitive_ids();

    auto old_prog = build_staged(eng, branch_topology(), cfg);
    // restore the order of an old cache: P0 right behind D, ahead of Z/P1
    move_before(*old_prog, "P0", "Z");
    auto old_po = po_ids(*old_prog, true);
    const auto pos_p0 = std::find(old_po.begin(), old_po.end(), "P0");
    const auto pos_z = std::find(old_po.begin(), old_po.end(), "Z");
    ASSERT_LT(pos_p0, pos_z);

    network net_old(old_prog, 0);  // would throw in Debug if A2 was wrongly applied to the un-normalized order
    const auto old_ids = net_old.get_executed_primitive_ids();
    const auto res = check_exec_order(old_prog->get_processing_order(), old_ids);
    EXPECT_FALSE(res.normalized);
    EXPECT_TRUE(res.ok()) << res.message;
    EXPECT_EQ(old_ids, new_ids) << "old: " << join(old_ids) << " new: " << join(new_ids);
    run_round_and_check(net_old, eng, {1, 8, 3, 3}, {1, 4, 3, 3}, 1, false, true);
}

// Negative checks of the helper itself (A1/A2 do fire).
TEST(runtime_concat_exec_order, t4_check_exec_order_detects_violations) {
    auto& eng = get_test_engine();
    auto prog = build_staged(eng, branch_topology(), make_cfg(eng, QueueTypes::in_order));
    network net(prog, 0);
    auto ids = net.get_executed_primitive_ids();
    ASSERT_TRUE(check_exec_order(prog->get_processing_order(), ids).ok());

    auto dup = ids;
    dup.push_back(ids.front());
    const auto r1 = check_exec_order(prog->get_processing_order(), dup);
    EXPECT_FALSE(r1.ok());
    EXPECT_NE(r1.message.find("A1"), std::string::npos) << r1.message;

    auto swapped = ids;
    ASSERT_GE(swapped.size(), 4u);
    std::swap(swapped[1], swapped[2]);
    const auto r2 = check_exec_order(prog->get_processing_order(), swapped);
    EXPECT_FALSE(r2.ok());
    EXPECT_NE(r2.message.find("A2"), std::string::npos) << r2.message;

    auto shorter = ids;
    shorter.pop_back();
    EXPECT_FALSE(check_exec_order(prog->get_processing_order(), shorter).ok());
}

// ==============================================================================================================================
// T5. Cancel path: several rounds with changing shapes (also covers the oneDNN static concat branch of primitive_inst when available).
// ==============================================================================================================================
TEST(runtime_concat_exec_order, t5_cancel_path_multi_round_outputs) {
    auto& eng = get_test_engine();
    auto prog = build_injected(eng, QueueTypes::in_order);
    ASSERT_FALSE(prog->get_node("C").can_be_optimized());
    network net(prog, 0);
    const std::vector<ov::Shape> shapes = {{1, 16, 4, 4}, {1, 16, 4, 4}, {1, 32, 6, 6}, {1, 8, 2, 3}, {1, 24, 5, 5}, {1, 16, 4, 4}, {1, 1, 1, 1}};
    int seed = 1;
    for (const auto& s : shapes) {
        run_round_and_check(net, eng, s, s, seed++, true);
        EXPECT_FALSE(PrimitiveInstTestHelper::get_update_shape_done_by_other(net.get_primitive("P1")));
        EXPECT_FALSE(PrimitiveInstTestHelper::get_update_shape_done_by_other(net.get_primitive("C")));
        for (const char* id : {"P1", "C"}) {
            EXPECT_FALSE(PrimitiveInstTestHelper::get_allocation_done_by_other(net.get_primitive(id))) << id;
        }
    }
    // The cancelled concat owns its output buffer (no aliasing with a predecessor).
    auto c_mem = net.get_primitive("C")->output_memory_ptr();
    EXPECT_FALSE(eng.is_the_same_buffer(*c_mem, *net.get_primitive("P1")->output_memory_ptr()));
}

TEST(runtime_concat_exec_order, t5_cancel_path_onednn_static_concat_branch) {
    auto& eng = get_test_engine();
    if (!eng.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN is not available on this device: the static oneDNN concat branch is NOT covered";
    }
    auto prog = build_injected(eng, QueueTypes::in_order);
    auto& c = prog->get_node("C");
    ASSERT_FALSE(c.can_be_optimized());
    ASSERT_FALSE(c.is_runtime_skippable());
    network net(prog, 0);
    const std::vector<ov::Shape> shapes = {{1, 16, 4, 4}, {1, 16, 4, 4}, {1, 32, 6, 6}, {1, 8, 2, 3}, {1, 16, 4, 4}};
    int seed = 1;
    for (const auto& s : shapes) {
        run_round_and_check(net, eng, s, s, seed++, true);
    }
    const auto& impl = net.get_primitive("C")->get_impl();
    std::cout << "[T5] concat impl after rounds: " << (impl ? impl->get_kernel_name() : std::string("<none>")) << " is_onednn="
              << (impl && impl->is_onednn()) << " is_dynamic=" << (impl && impl->is_dynamic()) << std::endl;
}

// C5: the static oneDNN concat branch of primitive_inst (needs i8/u8, planar bfyx, static non-batch dims and *no* dynamic padding on any
// concat input layout). The predecessors of a cancelled concat keep their dynamic padding mask (R3), so this test records whether the
// branch is reachable from the cancel path and, if it is, that the outputs stay correct.
TEST(runtime_concat_exec_order, t5_cancel_path_i8_static_onednn_concat_eligibility) {
    auto& eng = get_test_engine();
    if (!eng.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN is not available on this device: the static oneDNN concat branch is NOT covered";
    }
    const layout dyn_i8{ov::PartialShape{-1, 8, 4, 4}, data_types::i8, format::bfyx};
    topology t;
    t.add(input_layout("in1", dyn_i8));
    t.add(input_layout("in2", dyn_i8));
    t.add(input_layout("in3", dyn_i8));
    t.add(input_layout("in4", dyn_i8));
    t.add(eltwise("P1", {input_info("in1"), input_info("in1")}, eltwise_mode::sum));
    t.add(eltwise("P2", {input_info("in2"), input_info("in2")}, eltwise_mode::sum));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    t.add(eltwise("Z", {input_info("in3"), input_info("in3")}, eltwise_mode::sum));
    t.add(eltwise("W", {input_info("in4"), input_info("in4")}, eltwise_mode::sum));
    t.add(eltwise("Y", {input_info("Z"), input_info("W")}, eltwise_mode::sum));
    auto prog = build_staged(eng, t, make_cfg(eng, QueueTypes::in_order, {"output", "Y"}), [](program& p) {
        std::cout << "[T5 i8] C optimized after pbf=" << p.get_node("C").can_be_optimized() << std::endl;
        p.get_node("Y").replace_dependency(0, p.get_node("P1"), false);
    });
    auto& c = prog->get_node("C");
    std::cout << "[T5 i8] C optimized after F9=" << c.can_be_optimized() << " runtime_skippable=" << c.is_runtime_skippable()
              << " P1 users=" << prog->get_node("P1").get_users().size() << std::endl;
    for (size_t i = 0; i < c.get_dependencies().size(); i++) {
        const auto l = c.get_input_layout(i);
        std::cout << "[T5 i8] C input " << i << " layout " << l.to_short_string() << " dynamic padding=" << l.data_padding.is_dynamic() << std::endl;
    }
    network net(prog, 0);
    const std::vector<int64_t> batches = {1, 1, 2, 3, 1};
    int seed = 1;
    for (auto b : batches) {
        const ov::Shape s{static_cast<size_t>(b), 8, 4, 4};
        const size_t n = ov::shape_size(s);
        auto mk = [&](int sd) {
            auto m = eng.allocate_memory(layout{ov::PartialShape(s), data_types::i8, format::bfyx});
            std::vector<int8_t> v(n);
            for (size_t i = 0; i < n; i++) {
                v[i] = static_cast<int8_t>((i * 3 + sd) % 7);
            }
            set_values<int8_t>(m, v);
            return std::make_pair(m, v);
        };
        auto a1 = mk(seed), a2 = mk(seed + 1), a3 = mk(seed + 2), a4 = mk(seed + 3);
        net.set_input_data("in1", a1.first);
        net.set_input_data("in2", a2.first);
        net.set_input_data("in3", a3.first);
        net.set_input_data("in4", a4.first);
        auto outs = net.execute();
        auto out = read_out_f32(net, outs, "output");
        ASSERT_EQ(out.size(), 2 * n);
        // axis-1 concat of [b,8,4,4] tensors: per batch element, 8*16 values of P1 followed by 8*16 values of P2
        const size_t per_b = n / static_cast<size_t>(b);
        for (int64_t bi = 0; bi < b; bi++) {
            for (size_t i = 0; i < per_b; i++) {
                ASSERT_EQ(out[bi * 2 * per_b + i], 2.f * a1.second[bi * per_b + i]);
                ASSERT_EQ(out[bi * 2 * per_b + per_b + i], 2.f * a2.second[bi * per_b + i]);
            }
        }
        seed += 4;
        net.get_program()->get_compilation_context().wait_all();
    }
    const auto& impl = net.get_primitive("C")->get_impl();
    std::cout << "[T5 i8] concat impl after rounds: " << (impl ? impl->get_kernel_name() : std::string("<none>")) << " is_onednn="
              << (impl && impl->is_onednn()) << std::endl;
}

// ==============================================================================================================================
// T6. out_of_order: F9 does not touch the processing order (the move stays in network::build_exec_order).
// ==============================================================================================================================
TEST(runtime_concat_exec_order, t6_ooo_queue_po_untouched_and_restrictions_identical) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::out_of_order);
    auto before = build_staged(eng, branch_topology(), cfg, nullptr, upto::compile);
    ASSERT_EQ(before->get_config().get_queue_type(), QueueTypes::out_of_order);
    // an order in which F9 would move P0 behind Z if it was allowed to
    force_order(*before, {"in1", "D", "P0", "in2", "Z", "P1", "C"});
    const auto po_before = po_ids(*before);
    program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*before);
    EXPECT_EQ(po_ids(*before), po_before) << "the pass moved nodes under out_of_order";
    EXPECT_TRUE(before->get_node("C").can_be_optimized());

    // Full pipeline: PO is whatever the BFS order produced; applying the pass again must not change it, nor the restrictions.
    auto prog = build_staged(eng, branch_topology(), cfg);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
    const auto po = po_ids(*prog);
    std::map<primitive_id, std::vector<uint32_t>> restr;
    for (const auto* n : prog->get_processing_order()) {
        restr[n->id()] = n->get_memory_dependencies();
    }
    program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*prog);
    program_wrapper::prepare_memory_dependencies(*prog);
    EXPECT_EQ(po_ids(*prog), po);
    for (const auto* n : prog->get_processing_order()) {
        EXPECT_EQ(n->get_memory_dependencies(), restr.at(n->id())) << n->id();
    }
    network net(prog, 0);
    run_round_and_check(net, eng, {1, 8, 3, 3}, {1, 4, 3, 3}, 1, false, true);
    run_round_and_check(net, eng, {1, 2, 5, 5}, {1, 12, 5, 5}, 2, false, true);
}

// OOOQ with an unsafe concat: the cancel rule is queue independent.
TEST(runtime_concat_exec_order, t6_ooo_queue_unsafe_concat_still_cancelled) {
    auto& eng = get_test_engine();
    auto prog = build_injected(eng, QueueTypes::out_of_order);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
    EXPECT_FALSE(prog->get_node("C").can_be_optimized());
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    EXPECT_EQ(count_id(ids, "P1"), 1u) << join(ids);
    run_round_and_check(net, eng, {1, 16, 4, 4}, {1, 16, 4, 4}, 1, true);
}

// ==============================================================================================================================
// T7. Static network untouched; loop-body style program passes A1/A2.
// ==============================================================================================================================
TEST(runtime_concat_exec_order, t7_static_network_untouched) {
    auto& eng = get_test_engine();
    const layout l{ov::PartialShape{1, 8, 3, 3}, data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("in1", l));
    t.add(input_layout("in2", l));
    t.add(eltwise("D", {input_info("in1"), input_info("in1")}, eltwise_mode::sum));
    t.add(eltwise("Z", {input_info("in2"), input_info("in2")}, eltwise_mode::sum));
    t.add(reorder("P0", input_info("D"), format::bfyx, data_types::f16));
    t.add(reorder("P1", input_info("Z"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P0"), input_info("P1")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    for (auto q : {QueueTypes::in_order, QueueTypes::out_of_order}) {
        auto prog = build_staged(eng, t, make_cfg(eng, q), nullptr, upto::compile);
        ASSERT_EQ(prog->get_config().get_queue_type(), q);
        const auto po_before = po_ids(*prog);
        program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*prog);
        EXPECT_EQ(po_ids(*prog), po_before);

        auto full = build_staged(eng, t, make_cfg(eng, q));
        ASSERT_EQ(full->get_config().get_queue_type(), q);
        network net(full, 0);
        EXPECT_EQ(net.get_executed_primitive_ids(), po_ids(*full, true));
        EXPECT_TRUE(check_exec_order(full->get_processing_order(), net.get_executed_primitive_ids()).ok());
        const auto d1 = make_ints(72, 1);
        const auto d2 = make_ints(72, 2);
        net.set_input_data("in1", make_input(eng, {1, 8, 3, 3}, d1));
        net.set_input_data("in2", make_input(eng, {1, 8, 3, 3}, d2));
        auto outs_t = net.execute();
        auto out = read_out_f32(net, outs_t, "output");
        ASSERT_EQ(out.size(), 144u);
        for (size_t i = 0; i < 72; i++) {
            ASSERT_EQ(out[i], 2 * d1[i]);
            ASSERT_EQ(out[72 + i], 2 * d2[i]);
        }
    }
}

TEST(runtime_concat_exec_order, t7_body_program_with_dynamic_concat_passes_a1_a2) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order);
    auto prog = program::build_program(eng, branch_topology(), cfg, /*is_internal=*/true, /*no_optimizations=*/false, /*is_body_program=*/true);
    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->get_node("C").can_be_optimized());
    network net(prog, /*is_internal=*/true, /*is_primary_stream=*/true);  // Debug A1/A2 run inside the constructor
    const auto res = check_exec_order(prog->get_processing_order(), net.get_executed_primitive_ids());
    EXPECT_TRUE(res.normalized);
    EXPECT_TRUE(res.ok()) << res.message;
}

// ==============================================================================================================================
// Extra: a pool-eligible node B that is first in the processing order, with its user U between P's original slot and C.
// ==============================================================================================================================
namespace {
// B = D = in1 + in1 is not the first PO node itself (input_layouts come first), so the "first node" role is played by the first
// non-data node that exists in this graph; the test records which one it is.
topology first_node_topology() {
    topology t;
    t.add(input_layout("in1", dyn_layout()));
    t.add(input_layout("in2", dyn_layout()));
    t.add(eltwise("B", {input_info("in1"), input_info("in1")}, eltwise_mode::sum));      // pool-eligible
    t.add(reorder("P0", input_info("B"), format::bfyx, data_types::f16));                  // predecessor of C, user of B
    t.add(eltwise("U", {input_info("B"), input_info("in2")}, eltwise_mode::sum));          // second user of B
    t.add(reorder("P1", input_info("in2"), format::bfyx, data_types::f16));
    t.add(concatenation("C", {input_info("P0"), input_info("P1")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    return t;
}
}  // namespace

TEST(runtime_concat_exec_order, first_node_b_poolable_user_between_p_and_c_in_order) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::in_order, {"output", "U"});
    auto prog = build_staged(eng, first_node_topology(), cfg, [](program& p) {
        // original order: B, P0, U, in2-side, C -> U sits between P0's slot and C
        force_order(p, {"in1", "B", "P0", "U", "P1", "C"});
    });
    // The hook ran before post_optimize (BFS/F9); restore the shape of the scenario if the pipeline reordered it.
    std::cout << "[first_node] PO after build: " << join(po_ids(*prog, true)) << std::endl;
    network net(prog, 0);
    const auto ids = net.get_executed_primitive_ids();
    EXPECT_TRUE(check_exec_order(prog->get_processing_order(), ids).ok());
    const std::vector<ov::Shape> shapes = {{1, 8, 4, 4}, {1, 3, 2, 2}, {1, 8, 4, 4}};
    int seed = 1;
    for (const auto& s : shapes) {
        const auto d1 = make_ints(ov::shape_size(s), seed);
        const auto d2 = make_ints(ov::shape_size(s), seed + 1);
        net.set_input_data("in1", make_input(eng, s, d1));
        net.set_input_data("in2", make_input(eng, s, d2));
        auto outs = net.execute();
        auto out = read_out_f32(net, outs, "output");
        auto u = read_out_f32(net, outs, "U");
        ASSERT_EQ(out.size(), 2 * d1.size());
        ASSERT_EQ(u.size(), d1.size());
        for (size_t i = 0; i < d1.size(); i++) {
            ASSERT_EQ(out[i], 2 * d1[i]);
            ASSERT_EQ(out[d1.size() + i], d2[i]);
            ASSERT_EQ(u[i], 2 * d1[i] + d2[i]);
        }
        seed += 2;
    }
}

TEST(runtime_concat_exec_order, first_node_b_poolable_user_between_p_and_c_ooo_po_untouched) {
    auto& eng = get_test_engine();
    auto cfg = make_cfg(eng, QueueTypes::out_of_order, {"output", "U"});
    auto prog = build_staged(eng, first_node_topology(), cfg, nullptr, upto::compile);
    ASSERT_EQ(prog->get_config().get_queue_type(), QueueTypes::out_of_order);
    // original order: B, P0, U, P1, C -> U sits between P0's slot and C; an in_order F9 would move P0 behind U
    force_order(*prog, {"in1", "B", "P0", "U", "P1", "C"});
    const auto po_before = po_ids(*prog);
    program_wrapper::apply_opt_pass<order_runtime_concat_predecessors>(*prog);
    EXPECT_EQ(po_ids(*prog), po_before);
}

// ==============================================================================================================================
// C9. Allocation report (Step 8 substitute): pool block count, peak device memory and output buffers per T-model, in_order dynamic.
// Prints one line per model; the numbers of the pre-F9 binary are compared with the post-F9 ones by the caller.
// ==============================================================================================================================
namespace {
void report_allocation(engine& eng, const std::string& name, const topology& t, const std::vector<std::string>& outs, bool with_y,
                       bool injected, bool branch) {
    auto cfg = make_cfg(eng, QueueTypes::in_order, outs);
    const auto used_dev_before = eng.get_used_device_memory(allocation_type::usm_device);
    const auto used_host_before = eng.get_used_device_memory(allocation_type::usm_host);
    program::ptr prog;
    if (injected) {
        prog = build_staged(eng, t, cfg, [](program& p) { p.get_node("Y").replace_dependency(0, p.get_node("P1"), false); });
    } else {
        prog = program::build_program(eng, t, cfg, false, false);
    }
    network net(prog, 0);
    const std::vector<ov::Shape> shapes = {{1, 16, 4, 4}, {1, 32, 6, 6}, {1, 8, 2, 3}, {1, 16, 4, 4}};
    std::string outputs;
    int seed = 1;
    for (const auto& s : shapes) {
        run_round_and_check(net, eng, s, s, seed++, with_y, branch);
    }
    for (const auto& id : outs) {
        auto m = net.get_primitive(id)->output_memory_ptr();
        outputs += id + "(size=" + std::to_string(m->size()) + ",ptr_shared_with_input_buffers=" + "n/a" + ") ";
    }
    std::cout << "[C9] model=" << name << " used_dev_delta=" << (eng.get_used_device_memory(allocation_type::usm_device) - used_dev_before)
              << " used_host_delta=" << (eng.get_used_device_memory(allocation_type::usm_host) - used_host_before) << " pool_blocks=" << net.get_memory_pool().get_non_padded_pool_size() << " pool_bytes_usm_device="
              << net.get_memory_pool().get_total_mem_pool_size(allocation_type::usm_device) << " pool_bytes_usm_host="
              << net.get_memory_pool().get_total_mem_pool_size(allocation_type::usm_host) << " max_used_device_mem=" << eng.get_max_used_device_memory()
              << " outputs: " << outputs << std::endl;
}
}  // namespace

TEST(runtime_concat_exec_order, c9_allocation_report) {
    auto& eng = get_test_engine();
    report_allocation(eng, "basic", basic_topology(), {"output"}, false, false, false);
    report_allocation(eng, "branch", branch_topology(), {"output"}, false, false, true);
    report_allocation(eng, "injected", injected_topology(), {"output", "Y"}, true, true, false);
}

// ==============================================================================================================================
// F24 (candidate, not fixed by F9): a dynamic strided_slice / resample used as a runtime in-place concat predecessor may be skipped at
// runtime, while network keeps it in the predecessors set. Characterization only: the reproduction is NOT established yet. Forced on
// (--gtest_also_run_disabled_tests) this simple construction passes, i.e. it does not trigger the candidate defect.
// ==============================================================================================================================
TEST(runtime_concat_exec_order, DISABLED_f24_strided_slice_predecessor_with_zero_length_sibling) {
    auto& eng = get_test_engine();
    topology t;
    t.add(input_layout("in1", dyn_layout()));
    t.add(input_layout("in2", dyn_layout()));
    t.add(data("begin", [&] {
        auto m = eng.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
        set_values<int64_t>(m, {0, 0, 0, 0});
        return m;
    }()));
    t.add(data("end", [&] {
        auto m = eng.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
        set_values<int64_t>(m, {1, 1 << 20, 1 << 20, 1 << 20});
        return m;
    }()));
    t.add(data("stride", [&] {
        auto m = eng.allocate_memory({ov::PartialShape{4}, data_types::i64, format::bfyx});
        set_values<int64_t>(m, {1, 1, 1, 1});
        return m;
    }()));
    t.add(strided_slice("P1", input_info("in1"), input_info("begin"), input_info("end"), input_info("stride"), {}, {}, {}, {}, {}, {}));
    t.add(reorder("P2", input_info("in2"), format::bfyx, data_types::f32));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));
    auto prog = program::build_program(eng, t, make_cfg(eng, QueueTypes::in_order), false, false);
    network net(prog, 0);
    // zero-length sibling: expected (once F24 is fixed) a correct output with P1 executed exactly once
    net.set_input_data("in1", make_input(eng, {1, 4, 2, 2}, make_ints(16, 1)));
    net.set_input_data("in2", make_input(eng, {1, 0, 2, 2}, {}));
    auto outs_t = net.execute();
        auto out = read_out_f32(net, outs_t, "output");
    const auto ref = make_ints(16, 1);
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(out[i], ref[i]);
    }
}
