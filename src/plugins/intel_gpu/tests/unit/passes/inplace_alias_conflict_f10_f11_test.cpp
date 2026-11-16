// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression tests for
//   F10: a oneDNN sum whose dependency is an in-place concat while a concat predecessor has another reader;
//   F11: a oneDNN sum whose writer also reads the sum dependency's buffer through a view.
// Every test asserts the CORRECT behavior and the 4-way consistency chain
//   frozen_sum(W) == (get_add_fusing_type(W, desc) == sum) == (get_reused_eltwmem_idx(W) != -1)
//                 == is_the_same_buffer(W.out, dep.out)
//
// NOTE: until F5 lands, the frozen value (add_onednn_optimization_attributes) and the live value can still
// diverge for reasons unrelated to F10/F11 (users count, padding, identity test after reorder removal).
// The F10 structural clause is the Pre-0 placeholder for the Phase 1a WAR rule; any relaxation of it
// must be applied to the structural predicate S at the freeze, because F5 cannot upgrade later.
//
// F11b (self read through an in-place crop view of the sum dependency) is N/A: such a topology cannot
// be built. prepare_buffer_fusing.cpp (is_optimizable_padding_for_crop) rejects an onednn user of an in-place
// crop, and the fsv16 crop rules reject the channel-offset/size cases; so the view is never a settled
// crop view in front of a oneDNN writer. The helper still handles it structurally.

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "openvino/op/interpolate.hpp"

#include "program_node.h"
#include "program_helpers.h"
#include "primitive_inst.h"

#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

using fusing = onednn_eltwise_fusing_helpers;

std::vector<float> read_f32(memory::ptr mem) {
    mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
    return std::vector<float>(lock.begin(), lock.end());
}

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " can_be_optimized=" << n->can_be_optimized()
           << " runtime_skippable=" << n->is_runtime_skippable() << " is_output=" << n->is_output()
           << " fused=" << n->get_fused_primitives().size() << " users=" << n->get_users().size()
           << " impl=" << static_cast<int>(n->get_preferred_impl_type())
           << " fmt=" << n->get_output_layout().format.to_string() << "\n";
    }
    return os.str();
}

bool frozen_sum(const program_node& p) {
    for (const auto& d : p.get_fused_primitives_onednn()) {
        if (d.op_type == onednn_post_op_type::sum) {
            return true;
        }
    }
    return false;
}

const fused_primitive_desc* fused_sum(const program_node& p) {
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::sum && fd.deps.size() == 1 && fd.has_outer_dep()) {
            return &fd;
        }
    }
    return nullptr;
}

std::vector<ov::float16> make_weights_values(size_t O, size_t I, size_t K) {
    std::vector<ov::float16> w(O * I * K * K);
    for (size_t o = 0; o < O; o++) {
        for (size_t i = 0; i < I; i++) {
            for (size_t ky = 0; ky < K; ky++) {
                for (size_t kx = 0; kx < K; kx++) {
                    const bool centre = (ky == K / 2) && (kx == K / 2);
                    float v = (o == i && centre) ? 0.5f : (((o + 2 * i + ky + kx) % 7 == 0) ? 0.125f : 0.f);
                    w[((o * I + i) * K + ky) * K + kx] = ov::float16(v);
                }
            }
        }
    }
    return w;
}

memory::ptr make_weights(engine& eng, size_t O, size_t I, size_t K) {
    auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(O), static_cast<int64_t>(I), static_cast<int64_t>(K), static_cast<int64_t>(K)},
                                          data_types::f16, format::bfyx});
    set_values(mem, make_weights_values(O, I, K));
    return mem;
}

memory::ptr make_input(engine& eng, size_t C, int variant) {
    layout l{ov::PartialShape{1, static_cast<int64_t>(C), 32, 32}, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(variant == 1 ? static_cast<float>(i % 17) * 0.125f - 1.0f
                                        : static_cast<float>((i * 3) % 13) * 0.0625f - 0.5f);
    }
    set_values(mem, v);
    return mem;
}

void add_conv(topology& t, const primitive_id& id, const primitive_id& in, const primitive_id& w, int pad) {
    t.add(convolution(id, input_info(in), w, "", 1, {1, 1}, {1, 1}, {pad, pad}, {pad, pad}, false));
}

ov::intel_gpu::ImplementationDesc impl_desc(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", t};
}

ExecutionConfig opt_config(engine& eng, const ov::intel_gpu::ImplForcingMap& forced) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    return config;
}

ExecutionConfig ref_config(engine& eng) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    return config;
}

using host_map = std::map<std::string, std::vector<float>>;

host_map run_net(network& net, const std::map<std::string, memory::ptr>& inputs, const std::vector<std::string>& out_ids) {
    for (const auto& kv : inputs) {
        net.set_input_data(kv.first, kv.second);
    }
    auto outs = net.execute();
    host_map r;
    for (const auto& id : out_ids) {
        r[id] = read_f32(outs.at(id).get_memory());
    }
    return r;
}

void expect_outputs_match(const host_map& got, const host_map& ref) {
    for (const auto& kv : ref) {
        const auto& g = got.at(kv.first);
        ASSERT_EQ(g.size(), kv.second.size()) << kv.first;
        float max_diff = 0.f;
        size_t bad = 0;
        for (size_t i = 0; i < g.size(); i++) {
            const float d = std::fabs(g[i] - kv.second[i]);
            max_diff = std::max(max_diff, d);
            if (d > 0.02f + 0.01f * std::fabs(kv.second[i])) {
                bad++;
            }
        }
        std::cout << "[CMP] " << kv.first << " max_diff=" << max_diff << " mismatches=" << bad << "/" << g.size() << std::endl;
        EXPECT_EQ(bad, 0u) << "output " << kv.first << " does not match the reference (max_diff=" << max_diff << ")";
    }
}

// The 4-way consistency chain. `expect_sum` is the expected (post-fix) value of every link.
void check_chain(network& net, const primitive_id& w_id, const primitive_id& dep_id, bool expect_sum, const fused_primitive_desc& desc) {
    auto& eng = get_test_engine();
    auto& w = net.get_program()->get_node(w_id);
    const bool frozen = frozen_sum(w);
    const bool live = fusing::get_add_fusing_type(w, desc) == add_fusing_type::sum;
    const bool reused = fusing::get_reused_eltwmem_idx(w) != -1;
    const bool same = eng.is_the_same_buffer(*net.get_primitive(w_id)->output_memory_ptr(), *net.get_primitive(dep_id)->output_memory_ptr());
    std::cout << "[CHAIN] frozen_sum=" << frozen << " live_sum=" << live << " reused=" << reused << " same_buffer=" << same
              << " (expected " << expect_sum << ")" << std::endl;
    EXPECT_EQ(frozen, live) << "frozen/live disagree";
    EXPECT_EQ(live, reused) << "live/reused_idx disagree";
    EXPECT_EQ(reused, same) << "reused_idx/shared buffer disagree";
    EXPECT_EQ(frozen, expect_sum) << "frozen sum differs from expectation";
    EXPECT_EQ(live, expect_sum);
    if (!expect_sum) {
        EXPECT_EQ(fusing::get_reused_eltwmem_idx(w), -1);
        EXPECT_TRUE(fusing::get_add_fusing_type(w, desc) == add_fusing_type::binary_per_tensor);
    }
}

// Common F10 skeleton. mode 0: late reader (R = prod(P, Z)); mode 1: control (no R); mode 2: conservative (R = abs(P) feeds W).
struct f10_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
};

f10_graph build_f10(engine& eng, int mode) {
    f10_graph g;
    auto& t = g.t;
    const bool conservative = mode == 2;
    t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx}));
    t.add(data("w_pq", make_weights(eng, 16, 16, 1)));
    add_conv(t, "P", "in1", "w_pq", 0);
    add_conv(t, "Q", "in1", "w_pq", 0);
    t.add(concatenation("C", {input_info("P"), input_info("Q")}, 1));
    if (conservative) {
        t.add(activation("R", input_info("P"), activation_func::abs));
        t.add(data("w_w", make_weights(eng, 32, 16, 3)));
        add_conv(t, "W", "R", "w_w", 1);
    } else {
        t.add(input_layout("in2", layout{ov::PartialShape{1, 32, 32, 32}, data_types::f16, format::bfyx}));
        t.add(data("w_w", make_weights(eng, 32, 32, 3)));
        add_conv(t, "W", "in2", "w_w", 1);
    }
    t.add(eltwise("S", {input_info("W"), input_info("C")}, eltwise_mode::sum));
    t.add(reorder("out_S", input_info("S"), format::bfyx, data_types::f32));
    g.outs.push_back("out_S");
    if (!conservative) {
        t.add(data("w_z", make_weights(eng, 16, 32, 1)));
        add_conv(t, "Z", "S", "w_z", 0);
        t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
        g.outs.push_back("out_Z");
        if (mode == 0) {
            t.add(eltwise("R", {input_info("P"), input_info("Z")}, eltwise_mode::prod));
            t.add(reorder("out_R", input_info("R"), format::bfyx, data_types::f32));
            g.outs.push_back("out_R");
        }
    }
    g.forced = {{"P", impl_desc(impl_types::ocl)},
                {"Q", impl_desc(impl_types::ocl)},
                {"C", impl_desc(impl_types::ocl)},
                {"W", impl_desc(impl_types::onednn)}};
    if (!conservative) {
        g.forced.emplace("Z", impl_desc(impl_types::ocl));
        if (mode == 0) {
            g.forced.emplace("R", impl_desc(impl_types::ocl));
        }
    }
    return g;
}

void run_f10(int mode, bool expect_sum) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum needs supports_immad";
    }
    auto g = build_f10(engine, mode);

    std::map<std::string, memory::ptr> inputs{{"in1", make_input(engine, 16, 1)}};
    if (mode != 2) {
        inputs.emplace("in2", make_input(engine, 32, 2));
    }

    ExecutionConfig config = opt_config(engine, g.forced);
    ASSERT_EQ(config.get_queue_type(), QueueTypes::in_order);
    auto prog = program::build_program(engine, g.t, config);
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;

    // ---- graph-shape preconditions (stop and report if any fails; do not retune the topology)
    auto& P = prog->get_node("P");
    auto& C = prog->get_node("C");
    auto& W = prog->get_node("W");
    ASSERT_TRUE(C.can_be_optimized()) << flags;
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    ASSERT_TRUE(P.get_fused_primitives().empty()) << "R must not be fused into P\n" << flags;
    if (mode == 1) {
        ASSERT_EQ(P.get_users().size(), 1u) << flags;
    } else {
        ASSERT_EQ(P.get_users().size(), 2u) << flags;
        std::set<std::string> u;
        for (auto* x : P.get_users()) {
            u.insert(x->id());
        }
        ASSERT_EQ(u, (std::set<std::string>{"C", "R"})) << flags;
    }
    const auto* desc = fused_sum(W);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise sum\n" << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "C") << flags;
    const auto& po = prog->get_processing_order();
    if (mode != 1) {
        auto& R = prog->get_node("R");
        if (mode == 2) {
            ASSERT_LT(po.get_processing_number(&R), po.get_processing_number(&W)) << flags;
        } else {
            ASSERT_GT(po.get_processing_number(&R), po.get_processing_number(&W)) << flags;
        }
    }

    auto res = run_net(net, inputs, g.outs);

    // The reference graph needs a topology copy built without forced impls.
    auto gref = build_f10(engine, mode);
    network net_ref(engine, gref.t, ref_config(engine));
    auto ref = run_net(net_ref, inputs, gref.outs);

    check_chain(net, "W", "C", expect_sum, *desc);
    expect_outputs_match(res, ref);
}

}  // namespace

// Pre-fix: W overwrites the slots of C while R still reads P's slot afterwards.
TEST(inplace_alias_conflict_f10_f11, F10_concat_slot_late_reader) {
    run_f10(0, /*expect_sum=*/false);
}

// Control: P has only the concat as user (same topology without R), so sum into the concat is legal and must be kept.
TEST(inplace_alias_conflict_f10_f11, F10_control_single_user_slots) {
    run_f10(1, /*expect_sum=*/true);
}

// R always runs before W, so the sum is in fact safe; Pre-0 rejects it conservatively.
TEST(inplace_alias_conflict_f10_f11, F10_conservative_reader_before_writer) {
    // conservative Pre-0 rejection; Phase 1 war_safe may relax
    run_f10(2, /*expect_sum=*/false);
}

// W reads D through V (a no-op resample that only becomes a view after the freeze) while summing into D.
TEST(inplace_alias_conflict_f10_f11, F11a_self_read_through_post_freeze_resample_view) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN sum needs supports_immad";
    }
    topology t;
    t.add(input_layout("in1", layout{ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx}));
    t.add(data("w_d", make_weights(engine, 16, 16, 1)));
    add_conv(t, "D", "in1", "w_d", 0);
    t.add(resample("V", input_info("D"), tensor(1, 16, 32, 32), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST));
    t.add(data("w_w", make_weights(engine, 16, 16, 3)));
    add_conv(t, "W", "V", "w_w", 1);
    t.add(eltwise("S", {input_info("W"), input_info("D")}, eltwise_mode::sum));
    t.add(reorder("out_S", input_info("S"), format::bfyx, data_types::f32));

    ov::intel_gpu::ImplForcingMap forced{{"D", impl_desc(impl_types::onednn)},
                                         {"V", impl_desc(impl_types::ocl)},
                                         {"W", impl_desc(impl_types::onednn)}};
    std::map<std::string, memory::ptr> inputs{{"in1", make_input(engine, 16, 1)}};

    ExecutionConfig config = opt_config(engine, forced);
    ASSERT_EQ(config.get_queue_type(), QueueTypes::in_order);
    auto prog = program::build_program(engine, t, config);
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << flags;

    auto& D = prog->get_node("D");
    auto& V = prog->get_node("V");
    auto& W = prog->get_node("W");
    ASSERT_TRUE(V.can_be_optimized() && V.is_runtime_skippable()) << flags;
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    ASSERT_EQ(D.get_users().size(), 2u) << flags;
    const auto* desc = fused_sum(W);
    ASSERT_NE(desc, nullptr) << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "D") << flags;

    auto res = run_net(net, inputs, {"out_S"});
    network net_ref(engine, t, ref_config(engine));
    auto ref = run_net(net_ref, inputs, {"out_S"});

    check_chain(net, "W", "D", /*expect_sum=*/false, *desc);
    expect_outputs_match(res, ref);
}
