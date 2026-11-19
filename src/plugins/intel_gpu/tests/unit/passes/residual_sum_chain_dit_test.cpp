// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Characterization / regression tests for oneDNN `sum` post-op residual chains whose residual source is
// far away (along dep(0)) from the fused FC, modelled on the SeedVR2 DiT block. In the real graph the attn
// residual sits 25 dep(0) hops above the FC (MLP residual: 3 hops) while is_direct_ancestor() only walks
// max_ancestor_walk_depth - 1 hops, so the attn residual is demoted to binary_add and the sum chain breaks.
//
// The assertions marked "FLIP AFTER Phase 3a" describe the CURRENT behavior and must be inverted once the
// bounded walk is replaced by a liveness-based decision.

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/primitives/reorder.hpp"

#include "program_node.h"
#include "primitive_inst.h"
#include "fully_connected_inst.h"
#include "eltwise_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"
#include "program_helpers.h"
#include "to_string_utils.h"

#include <iostream>
#include <sstream>
#include <map>

using namespace cldnn;
using namespace ::tests;

#ifdef ENABLE_ONEDNN_FOR_GPU

namespace {

// Mirrors program_helpers.cpp::is_direct_ancestor()::max_ancestor_walk_depth, which is a function-local
// static constexpr and therefore not reachable from tests, so it is hard-coded here. The value is pinned by
// add_onednn_optimization_attributes.ancestor_walk_depth_boundary.
constexpr int kAssumedMaxAncestorWalkDepth = 7;

// The walk performs at most kAssumedMaxAncestorWalkDepth iterations and checks the target before stepping,
// so the farthest reachable target is kAssumedMaxAncestorWalkDepth - 1 hops away.
bool bound_predicts_sum(int hops) {
    return hops <= kAssumedMaxAncestorWalkDepth - 1;
}

struct Hop {
    std::string id;
    std::string type;
    bool cbo;
    std::string layout;
};

Hop make_hop(const program_node& n) {
    return Hop{n.id(), n.get_primitive()->type_string(), n.can_be_optimized(), n.get_output_layout().to_short_string()};
}

// Same walk as is_direct_ancestor() (compare first, then step to dep(0)) but without the depth bound.
// Returns the hop count when t is reached, -1 when the chain ends early or cap is exceeded.
int walk_dep0(const program_node& p, const program_node& t, std::vector<Hop>& path, int cap = 64) {
    const program_node* it = &p;
    for (int i = 0; i <= cap; i++) {
        path.push_back(make_hop(*it));
        if (it == &t)
            return i;
        if (it->get_dependencies().empty())
            return -1;
        it = &it->get_dependency(0);
    }
    return -1;
}

std::string dump(const std::vector<Hop>& path) {
    std::ostringstream os;
    for (size_t i = 0; i < path.size(); i++) {
        os << "  [" << i << "] " << path[i].id << " (" << path[i].type << ")"
           << " cbo=" << path[i].cbo << " " << path[i].layout << "\n";
    }
    return os.str();
}

bool path_has_type(const std::vector<Hop>& path, const std::string& type) {
    for (const auto& h : path)
        if (h.type == type)
            return true;
    return false;
}

const fused_primitive_desc* fused_sum(const program_node& p) {
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::sum && fd.deps.size() == 1)
            return &fd;
    }
    return nullptr;
}

bool frozen_sum(const program_node& p) {
    for (const auto& d : p.get_fused_primitives_onednn())
        if (d.op_type == onednn_post_op_type::sum)
            return true;
    return false;
}

const char* fusing_str(add_fusing_type t) {
    switch (t) {
    case add_fusing_type::sum: return "sum";
    case add_fusing_type::binary_per_tensor: return "binary_per_tensor";
    case add_fusing_type::binary_per_oc: return "binary_per_oc";
    default: return "not_supported";
    }
}


// F5: frozen in-place sum <=> compiled impl slot is a sum <=> get_reused_eltwmem_idx names the dependency, and a frozen sum has no
// downgrade reason. Checked on programs built by the full pipeline (the re-freeze pass already ran).
void expect_f5_consistent(const program_node& p, bool expect_sum) {
    const auto* op = fused_sum(p);
    ASSERT_NE(op, nullptr) << p.id();
    const int32_t dep = op->outer_dep_start_idx;
    const bool frozen = onednn_eltwise_fusing_helpers::is_frozen_inplace_post_op(p, dep, eltwise_mode::sum);
    const auto slots = get_compiled_post_op_slots(p.get_fused_primitives_onednn(), static_cast<size_t>(dep),
                                                  {onednn_post_op_type::sum, onednn_post_op_type::optimized_sum});
    const bool impl_sum = p.get_selected_impl() && slots.slot && p.get_selected_impl()->is_sum_post_op(slots.kinds, *slots.slot);
    EXPECT_EQ(frozen, expect_sum) << p.id();
    EXPECT_EQ(impl_sum, expect_sum) << p.id();
    EXPECT_EQ(onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(p) == dep, expect_sum) << p.id();
    if (frozen) {
        EXPECT_TRUE(onednn_eltwise_fusing_helpers::frozen_sum_downgrade_reason(p, dep) ==
                    onednn_eltwise_fusing_helpers::sum_downgrade_reason::none)
            << p.id();
    }
}

ExecutionConfig make_config(engine& engine) {
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    return config;
}

memory::ptr make_mem(engine& engine, const ov::PartialShape& shape, data_types dt, float v) {
    auto m = engine.allocate_memory(layout{shape, dt, format::bfyx});
    auto& stream = get_test_stream();
    if (dt == data_types::f32) {
        mem_lock<float> l(m, stream);
        for (size_t i = 0; i < l.size(); i++) l[i] = v;
    } else if (dt == data_types::f16) {
        mem_lock<ov::float16> l(m, stream);
        for (size_t i = 0; i < l.size(); i++) l[i] = ov::float16(v);
    } else {
        OPENVINO_THROW("unsupported dt in make_mem");
    }
    return m;
}

// Non-identity permutation of [0, n), n = 32.
memory::ptr make_perm_idx(engine& engine, int n) {
    auto m = engine.allocate_memory(layout{ov::PartialShape{n}, data_types::i32, format::bfyx});
    mem_lock<int32_t> l(m, get_test_stream());
    for (int i = 0; i < n; i++) l[i] = (i * 7 + 3) % n;
    return m;
}

reshape make_reshape(const std::string& id, const std::string& in, const std::vector<int64_t>& dims) {
    return reshape(id, input_info(in), false, dims, ov::PartialShape(dims));
}

void add_fc(topology& t, engine& engine, const std::string& id, const std::string& in, int64_t ofm, int64_t ifm,
            data_types dt = data_types::f32, bool out_f32 = false) {
    t.add(data(id + "_w", make_mem(engine, ov::PartialShape{ofm, ifm}, dt, 0.01f)));
    if (out_f32)
        t.add(fully_connected(id, input_info(in), id + "_w", "", data_types::f32));
    else
        t.add(fully_connected(id, input_info(in), id + "_w"));
}

// Shared prerequisites on a fused FC p whose residual source is target.
void check_prereq(program_node& p, program_node& target) {
    ASSERT_EQ(p.get_preferred_impl_type(), impl_types::onednn) << "node " << p.id() << " is not a oneDNN node";
    ASSERT_NE(fused_sum(p), nullptr) << "node " << p.id() << " has no fused eltwise sum with a single dep";
    ASSERT_EQ(target.get_users().size(), 2u) << "target " << target.id() << " must have exactly 2 users";
}

// ---------------------------------------------------------------- T1

struct T1Param {
    std::string family;  // "F", "D", "Dp"
    int k;
};

struct T1Obs {
    int hops = -1;
    add_fusing_type live = add_fusing_type::not_supported;
    bool frozen = false;
    bool x_share = false;
};

void run_t1(engine& engine, const T1Param& prm, T1Obs& obs) {
    topology t;
    t.add(input_layout("in", layout{ov::PartialShape{32, 64}, data_types::f32, format::bfyx}));
    add_fc(t, engine, "x_proj", "in", 64, 64);

    std::string cur = "x_proj";
    for (int i = 1; i <= prm.k; i++) {
        const std::string id = "fc_" + std::to_string(i);
        add_fc(t, engine, id, cur, 64, 64);
        cur = id;
    }

    auto add_crop_concat_gather = [&](const std::string& src) {
        t.add(crop("crop0", input_info(src), tensor(16, 64, 1, 1), tensor(0, 0, 0, 0)));
        t.add(crop("crop1", input_info(src), tensor(16, 64, 1, 1), tensor(16, 0, 0, 0)));
        t.add(concatenation("concat", {input_info("crop0"), input_info("crop1")}, 0));
        t.add(data("gidx", make_perm_idx(engine, 32)));
        t.add(gather("gather", input_info("concat"), input_info("gidx"), 0, 2, ov::Shape{32, 64}));
        cur = "gather";
    };

    if (prm.family == "D") {
        t.add(make_reshape("rs1", cur, {2, 32, 32}));
        t.add(permute("perm", input_info("rs1"), {1, 0, 2}));
        t.add(make_reshape("rs2", "perm", {32, 64}));
        add_crop_concat_gather("rs2");
    } else if (prm.family == "Dp") {
        add_crop_concat_gather(cur);
    }  // family F: p directly follows FC_k

    add_fc(t, engine, "p", cur, 64, 64);
    t.add(eltwise("res", input_info("p"), input_info("x_proj"), eltwise_mode::sum));
    // softmax as the output: a reorder output could be fused/removed which would make p an output node.
    t.add(softmax("out", input_info("res"), 1));

    auto prog = program::build_program(engine, t, make_config(engine), false, false);
    ASSERT_NE(prog, nullptr);

    auto& p = prog->get_node("p");
    auto& X = prog->get_node("x_proj");
    ASSERT_NO_FATAL_FAILURE(check_prereq(p, X));

    std::vector<Hop> path;
    obs.hops = walk_dep0(p, X, path);
    ASSERT_GE(obs.hops, 0) << "dep(0) walk from p did not reach x_proj\n" << dump(path);

    obs.live = onednn_eltwise_fusing_helpers::get_add_fusing_type(p, *fused_sum(p));
    obs.frozen = frozen_sum(p);
    obs.x_share = X.can_share_buffer();

    std::cout << "[T1] family=" << prm.family << " k=" << prm.k << " hops=" << obs.hops << " live=" << fusing_str(obs.live)
              << " frozen_sum=" << obs.frozen << " X.can_share_buffer=" << obs.x_share << "\n" << dump(path);

    EXPECT_EQ(obs.frozen, obs.live == add_fusing_type::sum) << "frozen/live disagree\n" << dump(path);
    expect_f5_consistent(p, obs.frozen);
    EXPECT_EQ(obs.live == add_fusing_type::sum, bound_predicts_sum(obs.hops)) << "hops=" << obs.hops << "\n" << dump(path);
    if (obs.live != add_fusing_type::sum) {
        EXPECT_EQ(obs.live, add_fusing_type::binary_per_tensor) << dump(path);
    }
    EXPECT_EQ(obs.x_share, !bound_predicts_sum(obs.hops)) << "hops=" << obs.hops << "\n" << dump(path);
}

std::vector<T1Param> t1_params() {
    std::vector<T1Param> v;
    for (int k = 4; k <= 11; k++) v.push_back({"F", k});
    for (int k : {0, 1, 3, 5}) v.push_back({"D", k});
    for (int k : {1, 2, 3}) v.push_back({"Dp", k});
    return v;
}

}  // namespace

class residual_sum_chain_dit_hops : public ::testing::TestWithParam<T1Param> {};

TEST_P(residual_sum_chain_dit_hops, bound_matches_measured_hops) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN sum post-op requires supports_immad";
    T1Obs obs;
    ASSERT_NO_FATAL_FAILURE(run_t1(engine, GetParam(), obs));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         residual_sum_chain_dit_hops,
                         ::testing::ValuesIn(t1_params()),
                         [](const ::testing::TestParamInfo<T1Param>& info) {
                             return info.param.family + "_k" + std::to_string(info.param.k);
                         });

TEST(residual_sum_chain_dit, T1_boundary_observed) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN sum post-op requires supports_immad";
    bool seen_sum_at_6 = false;
    bool seen_binary_at_7 = false;
    for (const auto& prm : t1_params()) {
        T1Obs obs;
        ASSERT_NO_FATAL_FAILURE(run_t1(engine, prm, obs));
        if (obs.hops == kAssumedMaxAncestorWalkDepth - 1 && obs.live == add_fusing_type::sum)
            seen_sum_at_6 = true;
        if (obs.hops == kAssumedMaxAncestorWalkDepth && obs.live != add_fusing_type::sum)
            seen_binary_at_7 = true;
    }
    EXPECT_TRUE(seen_sum_at_6) << "no parameter produced a measured 6-hop chain classified as sum";
    EXPECT_TRUE(seen_binary_at_7) << "no parameter produced a measured 7-hop chain classified as binary";
}

// ---------------------------------------------------------------- T2 / T2c

namespace {

struct DitOpts {
    bool no_rope = false;
    bool no_sdpa = false;
    bool fc_in_reorder = false;
};

struct DitGraph {
    topology t;
    DitOpts opts;
};

// S=32, C=64, H=2, D=32.
void build_dit_mha_chain(topology& t, engine& engine, const DitOpts& o) {
    const auto f32 = data_types::f32;
    const auto f16 = data_types::f16;
    t.add(input_layout("in", layout{ov::PartialShape{32, 64}, f32, format::bfyx}));
    t.add(input_layout("gate", layout{ov::PartialShape{1, 64}, f32, format::bfyx}));
    add_fc(t, engine, "x_proj", "in", 64, 64);
    t.add(data("gamma_a", make_mem(engine, ov::PartialShape{64}, f32, 1.f)));
    t.add(rms("rms_a", input_info("x_proj"), input_info("gamma_a"), 1e-6f));
    add_fc(t, engine, "qkv", "rms_a", 192, 64);
    t.add(crop("crop_q", input_info("qkv"), tensor(32, 64, 1, 1), tensor(0, 0, 0, 0)));
    t.add(crop("crop_k", input_info("qkv"), tensor(32, 64, 1, 1), tensor(0, 64, 0, 0)));
    t.add(crop("crop_v", input_info("qkv"), tensor(32, 64, 1, 1), tensor(0, 128, 0, 0)));

    // q path
    t.add(make_reshape("rs_q1", "crop_q", {32, 2, 32}));
    t.add(data("idx_q", make_perm_idx(engine, 32)));
    t.add(gather("gather_q", input_info("rs_q1"), input_info("idx_q"), 0, 3, ov::Shape{32, 2, 32}));
    t.add(data("gamma_q", make_mem(engine, ov::PartialShape{32}, f32, 1.f)));
    t.add(rms("rms_q", input_info("gather_q"), input_info("gamma_q"), 1e-6f));
    t.add(make_reshape("rs_q2", "rms_q", {1, 32, 2, 32}));
    t.add(reorder("q_f16", input_info("rs_q2"), format::bfyx, f16));
    t.add(permute("q_perm", input_info("q_f16"), {0, 2, 1, 3}));
    t.add(crop("crop_w0", input_info("q_perm"), tensor(1, 2, 32, 16), tensor(0, 0, 0, 0)));
    t.add(crop("crop_w1", input_info("q_perm"), tensor(1, 2, 32, 16), tensor(0, 0, 0, 16)));
    t.add(make_reshape("rs_w0", "crop_w0", {1, 2, 16, 16, 2}));
    t.add(permute("perm5", input_info("rs_w0"), {0, 1, 2, 4, 3}));
    t.add(make_reshape("rs_w0b", "perm5", {1, 2, 16, 32}));
    std::string rope_id = "rope";
    if (o.no_rope) {
        t.add(softmax("rope", input_info("rs_w0b"), -1));
    } else {
        ov::op::internal::RoPE::Config cfg;
        cfg.head_cnt = 2;
        cfg.head_size = 32;
        cfg.rotary_ndims = 32;
        t.add(data("cos", make_mem(engine, ov::PartialShape{1, 1, 16, 32}, f16, 0.5f)));
        t.add(data("sin", make_mem(engine, ov::PartialShape{1, 1, 16, 32}, f16, 0.5f)));
        t.add(rope("rope", {input_info("rs_w0b"), input_info("cos"), input_info("sin")}, cfg));
    }
    t.add(concatenation("q_cat1", {input_info("rope"), input_info("crop_w1")}, 2));
    t.add(crop("q_c0", input_info("q_cat1"), tensor(1, 2, 32, 16), tensor(0, 0, 0, 0)));
    t.add(crop("q_c1", input_info("q_cat1"), tensor(1, 2, 32, 16), tensor(0, 0, 0, 16)));
    t.add(concatenation("q_cat2", {input_info("q_c0"), input_info("q_c1")}, 2));

    std::string attn_in;
    if (o.no_sdpa) {
        t.add(softmax("sdpa", input_info("q_cat2"), -1));
    } else {
        for (const std::string n : {"k", "v"}) {
            t.add(make_reshape("rs_" + n, "crop_" + n, {1, 32, 2, 32}));
            t.add(reorder(n + "_f16", input_info("rs_" + n), format::bfyx, f16));
            t.add(permute(n + "_perm", input_info(n + "_f16"), {0, 2, 1, 3}));
        }
        const std::vector<int64_t> id4 = {0, 1, 2, 3};
        t.add(scaled_dot_product_attention("sdpa", {input_info("q_cat2"), input_info("k_perm"), input_info("v_perm")},
                                           false, -1, id4, id4, id4, id4));
    }
    t.add(make_reshape("rs_o1", "sdpa", {2, 32, 32}));
    t.add(permute("perm_o", input_info("rs_o1"), {1, 0, 2}));
    t.add(make_reshape("rs_o2", "perm_o", {32, 64}));
    t.add(crop("o_c0", input_info("rs_o2"), tensor(16, 64, 1, 1), tensor(0, 0, 0, 0)));
    t.add(crop("o_c1", input_info("rs_o2"), tensor(16, 64, 1, 1), tensor(16, 0, 0, 0)));
    t.add(concatenation("o_cat", {input_info("o_c0"), input_info("o_c1")}, 0));
    t.add(data("idx_o", make_perm_idx(engine, 32)));
    t.add(gather("o_gather", input_info("o_cat"), input_info("idx_o"), 0, 2, ov::Shape{32, 64}));
    std::string fc_in = "o_gather";
    if (o.fc_in_reorder) {
        t.add(reorder("o_f32", input_info("o_gather"), format::bfyx, f32));
        fc_in = "o_f32";
    }
    add_fc(t, engine, "attn_out", fc_in, 64, 64, o.fc_in_reorder ? f32 : f16, !o.fc_in_reorder);
}

void build_dit_mha6(topology& t, engine& engine) {
    const auto f32 = data_types::f32;
    t.add(input_layout("in", layout{ov::PartialShape{32, 64}, f32, format::bfyx}));
    t.add(input_layout("gate", layout{ov::PartialShape{1, 64}, f32, format::bfyx}));
    add_fc(t, engine, "x_proj", "in", 64, 64);
    t.add(data("gamma_a", make_mem(engine, ov::PartialShape{64}, f32, 1.f)));
    t.add(rms("rms_a", input_info("x_proj"), input_info("gamma_a"), 1e-6f));
    for (const std::string n : {"q", "k", "v"}) {
        add_fc(t, engine, "fc_" + n, "rms_a", 64, 64);
        t.add(make_reshape("rs_" + n, "fc_" + n, {1, 32, 2, 32}));
    }
    const std::vector<int64_t> bshd = {0, 2, 1, 3};
    t.add(scaled_dot_product_attention("sdpa", {input_info("rs_q"), input_info("rs_k"), input_info("rs_v")},
                                       false, -1, bshd, bshd, bshd, bshd));
    t.add(make_reshape("rs_o", "sdpa", {32, 64}));
    add_fc(t, engine, "attn_out", "rs_o", 64, 64);
}

void build_dit_tail(topology& t, engine& engine) {
    const auto f32 = data_types::f32;
    t.add(eltwise("prod", input_info("attn_out"), input_info("gate"), eltwise_mode::prod));
    t.add(eltwise("sum_attn", input_info("prod"), input_info("x_proj"), eltwise_mode::sum));
    t.add(data("gamma_m", make_mem(engine, ov::PartialShape{64}, f32, 1.f)));
    t.add(rms("rms_m", input_info("sum_attn"), input_info("gamma_m"), 1e-6f));
    add_fc(t, engine, "up", "rms_m", 128, 64);
    add_fc(t, engine, "down", "up", 64, 128);
    t.add(eltwise("sum_mlp", input_info("down"), input_info("sum_attn"), eltwise_mode::sum));
    t.add(softmax("out", input_info("sum_mlp"), 1));
}

program::ptr build_with_fallback(engine& engine, bool mha6, DitOpts& used, std::vector<std::string>& log) {
    std::vector<DitOpts> attempts;
    if (mha6) {
        attempts.push_back({});
    } else {
        attempts = {DitOpts{false, false, false}, DitOpts{false, false, true}, DitOpts{true, false, false},
                    DitOpts{true, true, false}, DitOpts{true, true, true}};
    }
    for (const auto& o : attempts) {
        try {
            topology t;
            if (mha6)
                build_dit_mha6(t, engine);
            else
                build_dit_mha_chain(t, engine, o);
            build_dit_tail(t, engine);
            auto prog = program::build_program(engine, t, make_config(engine), false, false);
            used = o;
            return prog;
        } catch (const std::exception& e) {
            std::ostringstream os;
            os << "attempt no_rope=" << o.no_rope << " no_sdpa=" << o.no_sdpa << " fc_in_reorder=" << o.fc_in_reorder
               << " failed: " << e.what();
            log.push_back(os.str());
            std::cout << "[T2] " << os.str() << std::endl;
        }
    }
    return nullptr;
}

void run_dit(bool mha6) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN sum post-op requires supports_immad";

    DitOpts used;
    std::vector<std::string> log;
    auto prog = build_with_fallback(engine, mha6, used, log);
    ASSERT_NE(prog, nullptr) << "all build attempts failed";
    ::testing::Test::RecordProperty("subst_no_rope", static_cast<int>(used.no_rope));
    ::testing::Test::RecordProperty("subst_no_sdpa", static_cast<int>(used.no_sdpa));
    ::testing::Test::RecordProperty("subst_fc_in_reorder", static_cast<int>(used.fc_in_reorder));
    std::cout << "[T2] mha6=" << mha6 << " substitutions: no_rope=" << used.no_rope << " no_sdpa=" << used.no_sdpa
              << " fc_in_reorder=" << used.fc_in_reorder << "\n";

    auto& X = prog->get_node("x_proj");
    auto& Y = prog->get_node("attn_out");
    auto& Z = prog->get_node("down");

    // (a) attn residual
    ASSERT_NO_FATAL_FAILURE(check_prereq(Y, X));
    std::vector<Hop> attn_path;
    const int attn_hops = walk_dep0(Y, X, attn_path);
    ASSERT_GE(attn_hops, 0) << dump(attn_path);
    ::testing::Test::RecordProperty("attn_hops", attn_hops);
    std::cout << "[T2] attn_hops=" << attn_hops << "\n" << dump(attn_path);

    const auto attn_live = onednn_eltwise_fusing_helpers::get_add_fusing_type(Y, *fused_sum(Y));
    const bool attn_frozen = frozen_sum(Y);
    std::cout << "[T2] attn live=" << fusing_str(attn_live) << " frozen_sum=" << attn_frozen << "\n";

    // (b) MLP residual
    ASSERT_NO_FATAL_FAILURE(check_prereq(Z, Y));
    std::vector<Hop> mlp_path;
    const int mlp_hops = walk_dep0(Z, Y, mlp_path);
    ASSERT_GE(mlp_hops, 0) << dump(mlp_path);
    ::testing::Test::RecordProperty("mlp_hops", mlp_hops);
    std::cout << "[T2] mlp_hops=" << mlp_hops << "\n" << dump(mlp_path);
    const auto mlp_live = onednn_eltwise_fusing_helpers::get_add_fusing_type(Z, *fused_sum(Z));
    const bool mlp_frozen = frozen_sum(Z);
    std::cout << "[T2] mlp live=" << fusing_str(mlp_live) << " frozen_sum=" << mlp_frozen << "\n";

    EXPECT_EQ(attn_frozen, attn_live == add_fusing_type::sum) << dump(attn_path);
    EXPECT_EQ(mlp_frozen, mlp_live == add_fusing_type::sum) << dump(mlp_path);
    expect_f5_consistent(Y, attn_frozen);
    expect_f5_consistent(Z, mlp_frozen);
    EXPECT_EQ(mlp_hops, 3) << dump(mlp_path);
    EXPECT_EQ(mlp_live, add_fusing_type::sum) << dump(mlp_path);
    EXPECT_TRUE(mlp_frozen);

    if (!mha6) {
        ASSERT_GE(attn_hops, 7) << "fidelity precondition: the attn chain must be beyond the walk bound\n" << dump(attn_path);
        for (const std::string ty : {"gather", "concatenation", "crop", "reshape", "permute", "rms", "fully_connected"})
            EXPECT_TRUE(path_has_type(attn_path, ty)) << "attn path lacks type " << ty << "\n" << dump(attn_path);
        if (!used.no_sdpa) {
            EXPECT_TRUE(path_has_type(attn_path, "scaled_dot_product_attention")) << dump(attn_path);
        }
        if (!used.no_rope) {
            EXPECT_TRUE(path_has_type(attn_path, "rope")) << dump(attn_path);
        }
        // FLIP AFTER Phase 3a: attn becomes sum (live == sum, frozen == true).
        EXPECT_EQ(attn_live, add_fusing_type::binary_per_tensor) << dump(attn_path);
        EXPECT_FALSE(attn_frozen) << dump(attn_path);
        // FLIP AFTER Phase 3a: X.can_share_buffer() becomes false (X is the root of the Y chain).
        EXPECT_TRUE(X.can_share_buffer());
    } else {
        EXPECT_EQ(attn_hops, 6) << dump(attn_path);
        EXPECT_EQ(attn_live, add_fusing_type::sum) << dump(attn_path);
        EXPECT_TRUE(attn_frozen);
        EXPECT_FALSE(X.can_share_buffer());
    }
    // Y is the root of the Z sum chain, Z is the chain tail.
    EXPECT_FALSE(Y.can_share_buffer());
    EXPECT_FALSE(Z.can_share_buffer());

    // (d) runtime aliasing
    network net(prog, 0);
    net.set_input_data("in", make_mem(engine, ov::PartialShape{32, 64}, data_types::f32, 0.1f));
    net.set_input_data("gate", make_mem(engine, ov::PartialShape{1, 64}, data_types::f32, 0.5f));
    net.execute();
    auto x_mem = net.get_primitive("x_proj")->output_memory_ptr();
    auto y_mem = net.get_primitive("attn_out")->output_memory_ptr();
    auto z_mem = net.get_primitive("down")->output_memory_ptr();
    EXPECT_TRUE(engine.is_the_same_buffer(*z_mem, *y_mem)) << "Z must accumulate in place into Y";
    if (!mha6) {
        // FLIP AFTER Phase 3a: Y shares the buffer with X.
        EXPECT_FALSE(engine.is_the_same_buffer(*y_mem, *x_mem)) << "Y must not alias X while the attn residual is binary_add";
    } else {
        EXPECT_TRUE(engine.is_the_same_buffer(*y_mem, *x_mem)) << "Y must alias X (sum chain)";
        EXPECT_TRUE(engine.is_the_same_buffer(*z_mem, *x_mem)) << "Z must alias X (sum chain)";
    }
}

}  // namespace

TEST(residual_sum_chain_dit, T2_dit_block_25_hops) {
    run_dit(false);
}

TEST(residual_sum_chain_dit, T2c_mha_6_hops_control) {
    run_dit(true);
}

#endif  // ENABLE_ONEDNN_FOR_GPU
