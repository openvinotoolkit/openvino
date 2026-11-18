// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Pre-0 F17: alias eligibility of binary_mul_inplace (oneDNN FC with a fused eltwise prod; dst may be bound to the post-op src1).
//
// Fix under test:
//   * can_use_mul_inplace rules M0..M7 (dep must own its buffer, not be an external root / output / concat / view, single port-0
//     reader, ...), M3b/M3c on the forward view chain of the writer;
//   * M8: for a static node the frozen post-op type of the fused op must agree (frozen AND live);
//   * primitive_inst::reused_eltw_alias_allowed: dst may only be bound while the ACTUAL impl is a oneDNN impl compiled with
//     binary_mul_inplace for that post-op, and a stale alias is dropped on every non-binding path (explicit flag).
//
// Every "rejected" case asserts, in this order:
//   1. vacuity: the pre-F17 conditions (M0) on the dependency accept the fusion, i.e. the test would not pass vacuously. Exception T5b:
//      there W is already an output after build_program (basic_memory_dependencies promoted it), so the LIVE M0 rejects it; what T5b
//      checks is that the FROZEN attrs (computed before that promotion) carry no binary_mul_inplace;
//   2. the oneDNN attrs frozen for W carry no binary_mul_inplace;
//   3. get_reused_eltwmem_idx(W) == -1 and W.out is not the dependency buffer;
//   4. outputs equal the non-optimized reference on two rounds, and user-visible input/state buffers are untouched.
//
// Tests that could not be constructed with the available primitives (T9 other-port variant, T10, T12, T15 variants and the static
// resample view of T3) are listed at the end of the file with the reason.

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

#include "pass_manager.h"
#include "program_helpers.h"
#include "program_node.h"
#include "primitive_inst.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

using fusing = onednn_eltwise_fusing_helpers;

const ov::PartialShape kShape{32, 16};

#define F17_SKIP_IF_NO_IMMAD()                                                       \
    do {                                                                             \
        if (!get_test_engine().get_device_info().supports_immad) {                   \
            GTEST_SKIP() << "binary_mul_inplace needs a oneDNN device (supports_immad)"; \
        }                                                                            \
    } while (0)

// Reads the logical tensor only: a dynamic output buffer can be larger than the tensor it currently holds.
std::vector<float> to_f32(memory::ptr mem) {
    std::vector<float> r;
    const size_t count = mem->get_layout().count();
    if (mem->get_layout().data_type == data_types::f16) {
        mem_lock<ov::float16, mem_lock_type::read> lock(mem, get_test_stream());
        for (size_t i = 0; i < count && i < lock.size(); i++) {
            r.push_back(static_cast<float>(lock[i]));
        }
    } else {
        mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
        for (size_t i = 0; i < count && i < lock.size(); i++) {
            r.push_back(lock[i]);
        }
    }
    return r;
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, int variant) {
    layout l{shape, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(static_cast<float>(((i * (3 + variant)) % 13)) * 0.0625f + 0.125f * variant);
    }
    set_values(mem, v);
    return mem;
}

memory::ptr make_weights(engine& eng, int64_t o, int64_t i, int variant) {
    layout l{ov::PartialShape{o, i}, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t k = 0; k < v.size(); k++) {
        v[k] = ov::float16(static_cast<float>((k * (5 + variant)) % 7) * 0.03125f - 0.0625f);
    }
    set_values(mem, v);
    return mem;
}

ov::intel_gpu::ImplementationDesc impl_desc(impl_types t) {
    return ov::intel_gpu::ImplementationDesc{format::bfyx, "", t};
}

ExecutionConfig opt_config(engine& eng, const ov::intel_gpu::ImplForcingMap& forced, const std::vector<std::string>& custom_outputs,
                           QueueTypes q = QueueTypes::in_order) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

ExecutionConfig ref_config(engine& eng, const std::vector<std::string>& custom_outputs) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

const fused_primitive_desc* fused_prod(const program_node& p) {
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::prod && fd.deps.size() == 1 && fd.has_outer_dep()) {
            return &fd;
        }
    }
    return nullptr;
}

bool frozen_has_mul_inplace(const program_node& p) {
    for (const auto& e : p.get_fused_primitives_onednn()) {
        if (e.op_type == onednn_post_op_type::binary_mul_inplace) {
            return true;
        }
    }
    return false;
}

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " can_be_optimized=" << n->can_be_optimized()
           << " runtime_skippable=" << n->is_runtime_skippable() << " is_output=" << n->is_output()
           << " fused=" << n->get_fused_primitives().size() << " users=" << n->get_users().size()
           << " dyn=" << n->is_dynamic() << " impl=" << static_cast<int>(n->get_preferred_impl_type()) << "\n";
    }
    return os.str();
}

struct mul_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;
    std::vector<std::string> custom_outputs;
    // id -> shape of every input_layout, fed with deterministic data per round
    std::vector<std::pair<std::string, ov::PartialShape>> inputs;
    std::string dep_id;
};

using memmap = std::map<std::string, memory::ptr>;

// A dynamic output memory can be larger than the tensor it currently holds (shape-predictor preallocation) and its layout may report
// the capacity; the tail is not defined, so compare only the logical elements.
void trim(std::vector<float>& v, size_t n) {
    if (v.size() > n) {
        v.resize(n);
    }
}
void trim_map(std::map<std::string, std::vector<float>>& m, size_t n) {
    for (auto& kv : m) {
        trim(kv.second, n);
    }
}

// false if either buffer is not allocated yet (an input_layout has no buffer before the first set_input_data)
bool same_buf(engine& eng, const primitive_inst& a, const primitive_inst& b) {
    auto ma = a.output_memory_ptr();
    auto mb = b.output_memory_ptr();
    return ma && mb && eng.is_the_same_buffer(*ma, *mb);
}

std::map<std::string, std::vector<float>> run_net(network& net, const memmap& ins, const std::vector<std::string>& outs) {
    for (const auto& in : ins) {
        net.set_input_data(in.first, in.second);
    }
    auto res = net.execute();
    std::map<std::string, std::vector<float>> r;
    for (const auto& id : outs) {
        r[id] = to_f32(res.at(id).get_memory());
    }
    return r;
}

void expect_match(const std::map<std::string, std::vector<float>>& got, const std::map<std::string, std::vector<float>>& ref,
                  const std::string& tag) {
    for (const auto& kv : ref) {
        const auto& g = got.at(kv.first);
        ASSERT_EQ(g.size(), kv.second.size()) << kv.first;
        size_t bad = 0;
        float max_diff = 0.f;
        for (size_t i = 0; i < g.size(); i++) {
            const float d = std::fabs(g[i] - kv.second[i]);
            max_diff = std::max(max_diff, d);
            if (d > 0.02f + 0.01f * std::fabs(kv.second[i])) {
                bad++;
            }
        }
        std::cout << "[CMP " << tag << "] " << kv.first << " max_diff=" << max_diff << " mismatches=" << bad << "/" << g.size() << std::endl;
        EXPECT_EQ(bad, 0u) << tag << ": output " << kv.first << " does not match the reference";
    }
}

// ----------------------------------------------------------------------------------------------------------------------------
// Static rejected cases
// ----------------------------------------------------------------------------------------------------------------------------

enum class dep_kind {
    input,                  // T2a: dep is a single-user input_layout
    view_of_input,          // T2b: dep is reshape(input)
    view_of_o_with_reader,  // T3b: dep is a no-op strided_slice of O, O has a later reader
    mutable_data_dep,       // T6a
    concat_holder,          // T7: dep is an in-place concat
    output_dep,             // T5a: dep is a network output
    writer_md_output        // T5b: W becomes an output through a mutable_data user
};

mul_graph build_static(engine& eng, dep_kind kind, memory::ptr md_mem) {
    mul_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs.push_back({"in1", kShape});
    std::string w_in = "in1";
    switch (kind) {
    case dep_kind::input:
        t.add(input_layout("in2", layout{kShape, data_types::f16, format::bfyx}));
        g.inputs.push_back({"in2", kShape});
        g.dep_id = "in2";
        break;
    case dep_kind::view_of_input:
        t.add(input_layout("in2", layout{ov::PartialShape{1, 32, 16}, data_types::f16, format::bfyx}));
        g.inputs.push_back({"in2", ov::PartialShape{1, 32, 16}});
        t.add(reshape("Rs", input_info("in2"), false, std::vector<int64_t>{32, 16}, kShape));
        g.dep_id = "Rs";
        break;
    case dep_kind::view_of_o_with_reader:
        t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
        g.inputs.push_back({"in3", kShape});
        t.add(data("wO", make_weights(eng, 16, 16, 1)));
        t.add(fully_connected("O", input_info("in3"), "wO"));
        t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
        g.forced.emplace("O", impl_desc(impl_types::ocl));
        g.dep_id = "V";
        break;
    case dep_kind::mutable_data_dep:
        t.add(mutable_data("m", md_mem));
        g.dep_id = "m";
        break;
    case dep_kind::concat_holder:
        t.add(input_layout("in3", layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx}));
        t.add(input_layout("in4", layout{ov::PartialShape{16, 16}, data_types::f16, format::bfyx}));
        g.inputs.push_back({"in3", ov::PartialShape{16, 16}});
        g.inputs.push_back({"in4", ov::PartialShape{16, 16}});
        t.add(activation("CP", input_info("in3"), activation_func::abs));
        t.add(activation("CQ", input_info("in4"), activation_func::relu));
        t.add(concatenation("C", {input_info("CP"), input_info("CQ")}, 0));
        g.forced.emplace("CP", impl_desc(impl_types::ocl));
        g.forced.emplace("CQ", impl_desc(impl_types::ocl));
        g.forced.emplace("C", impl_desc(impl_types::ocl));
        g.dep_id = "C";
        break;
    case dep_kind::output_dep:
    case dep_kind::writer_md_output:
        t.add(data("wD", make_weights(eng, 16, 16, 1)));
        t.add(fully_connected("D", input_info("in1"), "wD"));
        g.forced.emplace("D", impl_desc(impl_types::ocl));
        g.dep_id = "D";
        break;
    }
    t.add(data("wW", make_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info(w_in), "wW"));
    t.add(eltwise("P", {input_info("W"), input_info(g.dep_id)}, eltwise_mode::prod));
    t.add(data("wZ", make_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("P"), "wZ"));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    if (kind == dep_kind::view_of_o_with_reader) {
        // second reader of O that runs after W (it also depends on Z)
        t.add(data("wQ", make_weights(eng, 16, 16, 4)));
        t.add(fully_connected("Q", input_info("O"), "wQ"));
        t.add(eltwise("R", {input_info("Z"), input_info("Q")}, eltwise_mode::sum));
        t.add(reorder("out_R", input_info("R"), format::bfyx, data_types::f32));
        g.forced.emplace("Q", impl_desc(impl_types::ocl));
        g.outs = {"out_R"};
    } else {
        t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
        g.outs = {"out_Z"};
    }
    if (kind == dep_kind::output_dep) {
        g.outs.push_back("D");
        g.custom_outputs = {"D", g.outs.front()};
    }
    if (kind == dep_kind::writer_md_output) {
        t.add(mutable_data("md", {input_info("P")}, make_f16(eng, kShape, 3)));
        g.custom_outputs = {"md", g.outs.front()};
    }
    return g;
}

void run_static_rejected(dep_kind kind, const std::string& tag, QueueTypes queue = QueueTypes::in_order) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();

    auto md_mem = make_f16(engine, kShape, 3);
    const auto md_snapshot = kind == dep_kind::mutable_data_dep ? to_f32(md_mem) : std::vector<float>{};
    auto g = build_static(engine, kind, md_mem);

    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, g.custom_outputs, queue));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    std::cout << "[" << tag << "]\n" << flags;

    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_prod(W);
    ASSERT_NE(desc, nullptr) << "W has no fused eltwise prod\n" << flags;
    auto& DEP = W.get_dependency(desc->outer_dep_start_idx);
    ASSERT_EQ(DEP.id(), g.dep_id) << flags;
    // vacuity: M0 (the pre-F17 condition set) accepts
    ASSERT_EQ(DEP.get_users().size(), 1u) << flags;
    ASSERT_FALSE(DEP.is_constant()) << flags;
    if (kind == dep_kind::view_of_input || kind == dep_kind::view_of_o_with_reader) {
        ASSERT_TRUE(DEP.can_be_optimized() || DEP.is_runtime_skippable()) << flags;
    }
    if (kind == dep_kind::concat_holder) {
        ASSERT_TRUE(DEP.can_be_optimized()) << "concat is expected to be in-place\n" << flags;
    }
    if (kind == dep_kind::output_dep) {
        ASSERT_TRUE(DEP.is_output()) << flags;
    }
    if (kind == dep_kind::writer_md_output) {
        ASSERT_TRUE(W.is_output()) << "W is expected to become an output through mutable_data\n" << flags;
    }

    // A no-op strided_slice only becomes a view AFTER the attribute freeze, so the freeze (settled_only M1) may still have compiled
    // binary_mul_inplace for it. That is the harmless direction (dst stays private, src1 is passed as an ordinary operand); the live
    // predicate and the binding points must reject it.
    if (kind != dep_kind::view_of_o_with_reader) {
        EXPECT_FALSE(frozen_has_mul_inplace(W)) << "attrs were frozen with binary_mul_inplace for a rejected alias\n" << flags;
    }
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1) << flags;
    auto w = net.get_primitive("W");
    EXPECT_EQ(w->get_reused_eltw_dep_idx(), -1) << "F19 (sum only) must not record a mul writer";
    if (kind != dep_kind::writer_md_output) {
        auto dep = net.get_primitive(g.dep_id);
        if (dep->outputs_allocated() && w->outputs_allocated()) {
            EXPECT_FALSE(engine.is_the_same_buffer(*w->output_memory_ptr(), *dep->output_memory_ptr()))
                << "W dst is bound to the dependency buffer (before the first execute)";
        }
    }

    // two rounds with different input data
    memmap inA, inB;
    std::map<std::string, std::vector<float>> snapA, snapB;
    int v = 1;
    for (const auto& in : g.inputs) {
        inA[in.first] = make_f16(engine, in.second, v);
        inB[in.first] = make_f16(engine, in.second, v + 3);
        snapA[in.first] = to_f32(inA[in.first]);
        snapB[in.first] = to_f32(inB[in.first]);
        v++;
    }
    auto resA = run_net(net, inA, g.outs);
    auto resB = run_net(net, inB, g.outs);

    auto rmd = make_f16(engine, kShape, 3);
    auto gref = build_static(engine, kind, rmd);
    network net_ref(engine, gref.t, ref_config(engine, gref.custom_outputs));
    auto refA = run_net(net_ref, inA, gref.outs);
    auto refB = run_net(net_ref, inB, gref.outs);
    expect_match(resA, refA, tag + "/A");
    expect_match(resB, refB, tag + "/B");

    for (const auto& in : g.inputs) {
        EXPECT_EQ(to_f32(inA[in.first]), snapA[in.first]) << "user input " << in.first << " (round A) was modified";
        EXPECT_EQ(to_f32(inB[in.first]), snapB[in.first]) << "user input " << in.first << " (round B) was modified";
    }
    if (kind == dep_kind::mutable_data_dep) {
        EXPECT_EQ(to_f32(md_mem), md_snapshot) << "user-attached mutable_data was modified";
    }
}

TEST(inplace_alias_conflict_f17, T2a_static_dep_is_input_layout) {
    run_static_rejected(dep_kind::input, "T2a");
}
TEST(inplace_alias_conflict_f17, T2b_static_dep_is_view_of_input) {
    run_static_rejected(dep_kind::view_of_input, "T2b");
}
TEST(inplace_alias_conflict_f17, T3b_static_dep_is_view_of_o_with_later_reader) {
    run_static_rejected(dep_kind::view_of_o_with_reader, "T3b");
}
TEST(inplace_alias_conflict_f17, T5a_static_dep_is_output) {
    run_static_rejected(dep_kind::output_dep, "T5a");
}
TEST(inplace_alias_conflict_f17, T5b_static_writer_becomes_output_through_mutable_data) {
    run_static_rejected(dep_kind::writer_md_output, "T5b");
}
TEST(inplace_alias_conflict_f17, T6a_static_dep_is_mutable_data) {
    run_static_rejected(dep_kind::mutable_data_dep, "T6a");
}
TEST(inplace_alias_conflict_f17, T7_static_dep_is_in_place_concat) {
    run_static_rejected(dep_kind::concat_holder, "T7");
}
TEST(inplace_alias_conflict_f17, T11_ooo_queue_dep_is_input_layout) {
    run_static_rejected(dep_kind::input, "T11", QueueTypes::out_of_order);
}

// T6b: dep is a read_value. Structural only (no variable set up): the freeze and the live predicate must both reject.
TEST(inplace_alias_conflict_f17, T6b_static_dep_is_read_value) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    mul_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(read_value{"rv", {input_info("in1")}, "v0", {layout{kShape, data_types::f16, format::bfyx}}});
    t.add(data("wW", make_weights(engine, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("P", {input_info("W"), input_info("rv")}, eltwise_mode::prod));
    t.add(data("wZ", make_weights(engine, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("P"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, {}));
    const std::string flags = dump_flags(*prog);
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_prod(W);
    ASSERT_NE(desc, nullptr) << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "rv") << flags;
    EXPECT_FALSE(frozen_has_mul_inplace(W)) << flags;
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1) << flags;
}

// ----------------------------------------------------------------------------------------------------------------------------
// Accepted cases: C-1 (pool test topology plus reused_eltw_alias_allowed) and C-2 (gated MLP), static and dynamic.
// ----------------------------------------------------------------------------------------------------------------------------

struct accept_ctx {
    ov::Dimension batch_dim;
    bool dynamic = false;
};

accept_ctx make_ctx(bool dynamic, int64_t static_batch) {
    accept_ctx c;
    c.dynamic = dynamic;
    c.batch_dim = dynamic ? ov::Dimension::dynamic() : ov::Dimension(static_batch);
    if (dynamic) {
        c.batch_dim.set_symbol(std::make_shared<ov::Symbol>());
    }
    return c;
}

ExecutionConfig accept_config(engine& engine, bool pool, const ov::intel_gpu::ImplForcingMap& forced) {
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::enable_memory_pool(pool));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    return config;
}

// Same topology as basic_memory_dependencies_mul_inplace.fc_output_is_not_reused. Returns W's dependency id.
topology make_c1_topology(const accept_ctx& c, memory::ptr weights_mem) {
    const int64_t ifm = 64, ofm = 32;
    return topology(input_layout("input", layout{ov::PartialShape{c.batch_dim, ifm}, data_types::f16, format::bfyx}),
                    input_layout("mul_in", layout{ov::PartialShape{c.batch_dim, ofm}, data_types::f16, format::bfyx}),
                    data("weights", weights_mem),
                    activation("mul_data", input_info("mul_in"), activation_func::abs),
                    fully_connected("fc", input_info("input"), "weights", "", 2),
                    eltwise("mul", input_info("fc"), input_info("mul_data"), eltwise_mode::prod),
                    softmax("softmax", input_info("mul"), 1),
                    softmax("x", input_info("softmax"), 1),
                    eltwise("sum", input_info("mul"), input_info("x"), eltwise_mode::sum),
                    reorder("out", input_info("sum"), format::bfyx, data_types::f32));
}

ov::intel_gpu::ImplForcingMap fc_onednn() {
    return ov::intel_gpu::ImplForcingMap{{"fc", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn}}};
}

class f17_c1 : public ::testing::TestWithParam<bool> {};
TEST_P(f17_c1, mul_inplace_still_taken_and_allowed) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    const bool dynamic = GetParam();
    const std::vector<int64_t> batches = dynamic ? std::vector<int64_t>{16, 32, 8} : std::vector<int64_t>{16};
    auto c = make_ctx(dynamic, batches[0]);
    auto weights_mem = make_weights(engine, 32, 64, 0);

    network net_ref(engine, make_c1_topology(c, weights_mem), accept_config(engine, false, fc_onednn()));
    network net(program::build_program(engine, make_c1_topology(c, weights_mem), accept_config(engine, true, fc_onednn())), 0);

    auto& fc_node = net.get_program()->get_node("fc");
    const auto idx = fusing::get_reused_eltwmem_idx(fc_node);
    ASSERT_NE(idx, -1) << "F17 must keep the in-place mul for a private, single-reader dependency";
    ASSERT_EQ(fc_node.get_dependency(idx).id(), "mul_data");
    // dynamic nodes build their attrs and impl at the first execute; the frozen list and the impl exist only for static nodes here
    if (!dynamic) {
        EXPECT_TRUE(frozen_has_mul_inplace(fc_node));
        ASSERT_TRUE(net.get_primitive("fc")->reused_eltw_alias_allowed(idx));
    }

    for (const auto batch : batches) {
        auto in_mem = make_f16(engine, ov::PartialShape{batch, 64}, 1);
        auto mul_mem = make_f16(engine, ov::PartialShape{batch, 32}, 2);
        net_ref.set_input_data("input", in_mem);
        net_ref.set_input_data("mul_in", mul_mem);
        auto ref = to_f32(net_ref.execute().at("out").get_memory());
        net.set_input_data("input", in_mem);
        net.set_input_data("mul_in", mul_mem);
        auto out = to_f32(net.execute().at("out").get_memory());
        trim(ref, batch * 32);
        trim(out, batch * 32);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(ref[i], out[i], 1e-2f) << "batch=" << batch << " index " << i;
        }
        EXPECT_TRUE(engine.is_the_same_buffer(*net.get_primitive("fc")->output_memory_ptr(), *net.get_primitive("mul_data")->output_memory_ptr()))
            << "batch=" << batch << ": FC dst is expected to alias the binary_mul_inplace input";
        EXPECT_FALSE(engine.is_the_same_buffer(*net.get_primitive("fc")->output_memory_ptr(), *net.get_primitive("x")->output_memory_ptr()));
        EXPECT_TRUE(net.get_primitive("fc")->reused_eltw_alias_allowed(idx)) << "batch=" << batch << ": the actual impl must be compiled in place";
    }
}
INSTANTIATE_TEST_SUITE_P(smoke, f17_c1, ::testing::Bool(), [](const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "dynamic" : "static";
});

// C-2: gated MLP: act(gate(x)) * up(x) -> down. The prod dependency is the gate branch (private, single reader).
topology make_gated_mlp(const accept_ctx& c, memory::ptr wg, memory::ptr wu, memory::ptr wd) {
    return topology(input_layout("x_in", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                    data("wg", wg),
                    data("wu", wu),
                    data("wd", wd),
                    fully_connected("gate", input_info("x_in"), "wg", "", 2),
                    activation("act", input_info("gate"), activation_func::swish),
                    fully_connected("up", input_info("x_in"), "wu", "", 2),
                    eltwise("gated", input_info("up"), input_info("act"), eltwise_mode::prod),
                    fully_connected("down", input_info("gated"), "wd", "", 2),
                    reorder("out", input_info("down"), format::bfyx, data_types::f32));
}

class f17_c2 : public ::testing::TestWithParam<bool> {};
TEST_P(f17_c2, gated_mlp_keeps_inplace) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    const bool dynamic = GetParam();
    const std::vector<int64_t> batches = dynamic ? std::vector<int64_t>{16, 32, 8} : std::vector<int64_t>{16};
    auto c = make_ctx(dynamic, batches[0]);
    auto wg = make_weights(engine, 32, 64, 0);
    auto wu = make_weights(engine, 32, 64, 1);
    auto wd = make_weights(engine, 16, 32, 2);
    ov::intel_gpu::ImplForcingMap forced{{"gate", impl_desc(impl_types::onednn)}, {"up", impl_desc(impl_types::onednn)},
                                         {"down", impl_desc(impl_types::ocl)}};

    network net_ref(engine, make_gated_mlp(c, wg, wu, wd), accept_config(engine, false, forced));
    auto prog = program::build_program(engine, make_gated_mlp(c, wg, wu, wd), accept_config(engine, true, forced));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);

    // The prod is fused into whichever FC the fusing pass chose; find the writer.
    program_node* writer = nullptr;
    for (const char* id : {"up", "gate"}) {
        auto& n = prog->get_node(id);
        if (n.get_preferred_impl_type() == impl_types::onednn && fused_prod(n) != nullptr) {
            writer = &n;
        }
    }
    ASSERT_NE(writer, nullptr) << "the gating prod was not fused into an FC\n" << flags;
    const auto idx = fusing::get_reused_eltwmem_idx(*writer);
    ASSERT_NE(idx, -1) << "F17 must keep the in-place mul of the gated MLP\n" << flags;
    auto w_inst = net.get_primitive(writer->id());
    if (!dynamic) {
        EXPECT_TRUE(frozen_has_mul_inplace(*writer));
        ASSERT_TRUE(w_inst->reused_eltw_alias_allowed(idx));
    }

    for (const auto batch : batches) {
        auto in_mem = make_f16(engine, ov::PartialShape{batch, 64}, 1);
        net_ref.set_input_data("x_in", in_mem);
        auto ref = to_f32(net_ref.execute().at("out").get_memory());
        net.set_input_data("x_in", in_mem);
        auto out = to_f32(net.execute().at("out").get_memory());
        trim(ref, batch * 16);
        trim(out, batch * 16);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(ref[i], out[i], 2e-2f + 1e-2f * std::fabs(ref[i])) << "batch=" << batch << " index " << i;
        }
        auto dep_inst = net.get_primitive(writer->get_dependency(idx).id());
        EXPECT_TRUE(engine.is_the_same_buffer(*w_inst->output_memory_ptr(), *dep_inst->output_memory_ptr()))
            << "batch=" << batch << ": dst is expected to alias the gate branch buffer";
        EXPECT_TRUE(w_inst->reused_eltw_alias_allowed(idx)) << "batch=" << batch << ": the actual impl must be compiled in place";
    }
}
INSTANTIATE_TEST_SUITE_P(smoke, f17_c2, ::testing::Bool(), [](const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "dynamic" : "static";
});

// ----------------------------------------------------------------------------------------------------------------------------
// Dynamic rejected cases
// ----------------------------------------------------------------------------------------------------------------------------

// T1: dynamic, dep is a single-user input_layout. Pre-fix, realloc binds dst to the user input memory and the FC writes into it.
// Also covers "stale write": the same shape on the next round writes into the previous round's user memory.
TEST(inplace_alias_conflict_f17, T1_dynamic_dep_is_input_layout) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(true, 0);
    auto weights_mem = make_weights(engine, 32, 64, 0);
    auto make_topo = [&]() {
        return topology(input_layout("input", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        input_layout("mul_in", layout{ov::PartialShape{c.batch_dim, 32}, data_types::f16, format::bfyx}),
                        data("weights", weights_mem),
                        fully_connected("fc", input_info("input"), "weights", "", 2),
                        eltwise("mul", input_info("fc"), input_info("mul_in"), eltwise_mode::prod),
                        softmax("sm", input_info("mul"), 1),  // keeps the writer from being the network output
                        reorder("out", input_info("sm"), format::bfyx, data_types::f32));
    };
    network net_ref(engine, make_topo(), accept_config(engine, false, fc_onednn()));
    auto prog = program::build_program(engine, make_topo(), accept_config(engine, true, fc_onednn()));
    network net(prog, 0);
    program_node* fcp = nullptr;
    for (auto* n : prog->get_processing_order()) {
        if (n->is_type<fully_connected>() && fused_prod(*n) != nullptr) {
            fcp = n;
        }
    }
    ASSERT_NE(fcp, nullptr) << dump_flags(*prog);
    const std::string fc_id = fcp->id();
    std::cout << "[T1] writer=" << fc_id << " reused_idx=" << fusing::get_reused_eltwmem_idx(*fcp) << "\n" << dump_flags(*prog);
    ASSERT_EQ(fcp->get_preferred_impl_type(), impl_types::onednn);
    EXPECT_FALSE(frozen_has_mul_inplace(*fcp));
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(*fcp), -1);

    // rounds: 16, 16 (same shape: stale write), 32 (grow), 8 (shrink)
    for (const auto batch : std::vector<int64_t>{16, 16, 32, 8}) {
        auto in_mem = make_f16(engine, ov::PartialShape{batch, 64}, 1);
        auto mul_mem = make_f16(engine, ov::PartialShape{batch, 32}, 2);
        const auto snap = to_f32(mul_mem);
        net_ref.set_input_data("input", in_mem);
        net_ref.set_input_data("mul_in", mul_mem);
        auto ref = to_f32(net_ref.execute().at("out").get_memory());
        net.set_input_data("input", in_mem);
        net.set_input_data("mul_in", mul_mem);
        auto out = to_f32(net.execute().at("out").get_memory());
        trim(ref, batch * 32);
        trim(out, batch * 32);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(ref[i], out[i], 1e-2f) << "batch=" << batch << " index " << i;
        }
        EXPECT_EQ(to_f32(mul_mem), snap) << "batch=" << batch << ": user input buffer was written by the FC";
        EXPECT_FALSE(engine.is_the_same_buffer(*net.get_primitive(fc_id)->output_memory_ptr(), *net.get_primitive("mul_in")->output_memory_ptr()))
            << "batch=" << batch;
    }
}

// T4: dynamic, dep is a view (reshape) of O and O has a later reader. A runtime-skipped / optimized view makes the dep buffer O's buffer.
TEST(inplace_alias_conflict_f17, T4_dynamic_dep_is_view_of_o_with_later_reader) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(true, 0);
    auto wO = make_weights(engine, 32, 64, 1);
    auto wW = make_weights(engine, 32, 64, 0);
    auto make_topo = [&]() {
        return topology(input_layout("in1", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        input_layout("in3", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        data("wO", wO),
                        data("wW", wW),
                        fully_connected("O", input_info("in3"), "wO", "", 2),
                        reshape("V", input_info("O"), true, std::vector<int64_t>{0, 32}, ov::PartialShape{c.batch_dim, 32}),
                        fully_connected("W", input_info("in1"), "wW", "", 2),
                        eltwise("P", input_info("W"), input_info("V"), eltwise_mode::prod),
                        softmax("Z", input_info("P"), 1),
                        eltwise("Q", input_info("Z"), input_info("O"), eltwise_mode::max),
                        reorder("out", input_info("Q"), format::bfyx, data_types::f32));
    };
    ov::intel_gpu::ImplForcingMap forced{{"W", impl_desc(impl_types::onednn)}, {"O", impl_desc(impl_types::ocl)}};
    network net_ref(engine, make_topo(), accept_config(engine, false, forced));
    auto prog = program::build_program(engine, make_topo(), accept_config(engine, true, forced));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    const auto* desc = fused_prod(W);
    ASSERT_NE(desc, nullptr) << flags;
    ASSERT_EQ(W.get_dependency(desc->outer_dep_start_idx).id(), "V") << flags;
    auto& V = W.get_dependency(desc->outer_dep_start_idx);
    ASSERT_TRUE(V.can_be_optimized() || V.is_runtime_skippable()) << flags;
    EXPECT_FALSE(frozen_has_mul_inplace(W)) << flags;
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1) << flags;

    for (const auto batch : std::vector<int64_t>{16, 32, 8}) {
        auto in1 = make_f16(engine, ov::PartialShape{batch, 64}, 1);
        auto in3 = make_f16(engine, ov::PartialShape{batch, 64}, 2);
        net_ref.set_input_data("in1", in1);
        net_ref.set_input_data("in3", in3);
        auto ref = to_f32(net_ref.execute().at("out").get_memory());
        net.set_input_data("in1", in1);
        net.set_input_data("in3", in3);
        auto out = to_f32(net.execute().at("out").get_memory());
        trim(ref, batch * 32);
        trim(out, batch * 32);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(ref[i], out[i], 2e-2f + 1e-2f * std::fabs(ref[i])) << "batch=" << batch << " index " << i;
        }
        EXPECT_FALSE(engine.is_the_same_buffer(*net.get_primitive("W")->output_memory_ptr(), *net.get_primitive("O")->output_memory_ptr()))
            << "batch=" << batch << ": W dst is O's buffer";
    }
}

// T13: two dynamic FCs with identical parameters. W_b (dep = input_layout) rejects the in-place mul and runs first, W_a (dep = private
// activation) accepts it. If W_a's impl comes out of the static impls cache compiled for W_b, it carries a plain binary_mul and must
// not alias dst. Invariant checked each round: alias <=> the actual impl is compiled with binary_mul_inplace; outputs are exact.
// Observation on the dev machine: W_a ends up with a plain-mul impl and does not alias in any round, which is consistent with a static
// impls cache hit on W_b's impl (the cache key does not cover oneDNN post-op structure). This test cannot prove the cache was hit; the
// opposite order (W_a compiled first and aliasing, existence of an aliased round) is T13b below.
TEST(inplace_alias_conflict_f17, T13_dynamic_shared_impl_cache_alias_implies_inplace_impl) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(true, 0);
    auto wb = make_weights(engine, 32, 64, 0);
    auto wa = make_weights(engine, 32, 64, 0);  // identical content: same parameters
    auto make_topo = [&]() {
        return topology(input_layout("in_b", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        input_layout("mul_b", layout{ov::PartialShape{c.batch_dim, 32}, data_types::f16, format::bfyx}),
                        input_layout("in_a", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        input_layout("mul_a_in", layout{ov::PartialShape{c.batch_dim, 32}, data_types::f16, format::bfyx}),
                        data("wb", wb),
                        data("wa", wa),
                        activation("mul_a", input_info("mul_a_in"), activation_func::abs),
                        fully_connected("W_b", input_info("in_b"), "wb", "", 2),
                        eltwise("Pb", input_info("W_b"), input_info("mul_b"), eltwise_mode::prod),
                        softmax("Zb", input_info("Pb"), 1),
                        fully_connected("W_a", input_info("in_a"), "wa", "", 2),
                        eltwise("Pa", input_info("W_a"), input_info("mul_a"), eltwise_mode::prod),
                        softmax("Za", input_info("Pa"), 1),
                        eltwise("Q", input_info("Zb"), input_info("Za"), eltwise_mode::sum),
                        reorder("out", input_info("Q"), format::bfyx, data_types::f32));
    };
    ov::intel_gpu::ImplForcingMap forced{{"W_a", impl_desc(impl_types::onednn)}, {"W_b", impl_desc(impl_types::onednn)}};
    network net_ref(engine, make_topo(), accept_config(engine, false, forced));
    auto prog = program::build_program(engine, make_topo(), accept_config(engine, true, forced));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    auto& Wb = prog->get_node("W_b");
    auto& Wa = prog->get_node("W_a");
    ASSERT_NE(fused_prod(Wb), nullptr) << flags;
    ASSERT_NE(fused_prod(Wa), nullptr) << flags;
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(Wb), -1) << flags;
    const auto idx_a = fusing::get_reused_eltwmem_idx(Wa);
    ASSERT_NE(idx_a, -1) << "W_a is expected to accept the in-place mul\n" << flags;

    for (const auto batch : std::vector<int64_t>{16, 32, 8, 16}) {
        memmap ins{{"in_b", make_f16(engine, ov::PartialShape{batch, 64}, 1)},
                   {"mul_b", make_f16(engine, ov::PartialShape{batch, 32}, 2)},
                   {"in_a", make_f16(engine, ov::PartialShape{batch, 64}, 3)},
                   {"mul_a_in", make_f16(engine, ov::PartialShape{batch, 32}, 4)}};
        auto ref = run_net(net_ref, ins, {"out"});
        auto got = run_net(net, ins, {"out"});
        trim_map(ref, batch * 32);
        trim_map(got, batch * 32);
        expect_match(got, ref, "T13/batch" + std::to_string(batch));
        auto wa_inst = net.get_primitive("W_a");
        auto dep_inst = net.get_primitive("mul_a");
        const bool aliased = engine.is_the_same_buffer(*wa_inst->output_memory_ptr(), *dep_inst->output_memory_ptr());
        const bool allowed = wa_inst->reused_eltw_alias_allowed(idx_a);
        // ground truth, independent of reused_eltw_alias_allowed: W_a carries exactly one post-op (the prod), a binary at slot 0
        ASSERT_NE(wa_inst->get_impl(), nullptr);
        const bool impl_inplace = wa_inst->get_impl()->is_binary_mul_inplace_post_op({post_op_kind::binary}, 0);
        std::cout << "[T13] batch=" << batch << " aliased=" << aliased << " allowed=" << allowed << " impl_compiled_inplace=" << impl_inplace
                  << std::endl;
        EXPECT_EQ(wa_inst->get_fused_primitives_onednn().size(), 1u) << "the {binary},0 ground-truth call assumes W_a has exactly one post-op";
        if (aliased) {
            EXPECT_TRUE(impl_inplace) << "W_a dst aliases the dependency while its actual impl was not compiled with binary_mul_inplace";
        }
        if (impl_inplace) {
            EXPECT_TRUE(aliased) << "W_a's impl is compiled in place but dst is private (alias opportunity lost)";
        }
        // W_b rejects the alias, so it must never alias, whatever impl it holds
        EXPECT_FALSE(same_buf(engine, *net.get_primitive("W_b"), *net.get_primitive("mul_b")));
    }
}

// ----------------------------------------------------------------------------------------------------------------------------
// C1 / C2 unit tests and M8
// ----------------------------------------------------------------------------------------------------------------------------

// C1: reused_eltw_alias_allowed is fail-safe on a structural mismatch between the node-side post-op list and the compiled attrs.
TEST(inplace_alias_conflict_f17, C1_structural_mismatch_is_not_allowed) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(false, 16);
    auto weights_mem = make_weights(engine, 32, 64, 0);
    network net(program::build_program(engine, make_c1_topology(c, weights_mem), accept_config(engine, true, fc_onednn())), 0);
    auto fc = net.get_primitive("fc");
    const auto idx = fusing::get_reused_eltwmem_idx(net.get_program()->get_node("fc"));
    ASSERT_NE(idx, -1);
    ASSERT_TRUE(fc->reused_eltw_alias_allowed(idx));

    auto& list = fc->get_fused_primitives_onednn();
    const auto saved = list;
    ASSERT_FALSE(list.empty());

    // (a) one more entry than compiled slots
    list.push_back(list.front());
    EXPECT_FALSE(fc->reused_eltw_alias_allowed(idx)) << "slot count mismatch must not be trusted";
    list = saved;
    // (b) an entry that is not a compiled slot (optimized) hides nothing: still allowed
    list.push_back(list.front());
    list.back().op_type = onednn_post_op_type::optimized;
    EXPECT_TRUE(fc->reused_eltw_alias_allowed(idx));
    list = saved;
    // (c) kind mismatch: the list says eltwise where the compiled slot is binary
    for (auto& e : list) {
        if (e.op_type == onednn_post_op_type::binary_mul_inplace) {
            e.op_type = onednn_post_op_type::eltwise_act;
        }
    }
    EXPECT_FALSE(fc->reused_eltw_alias_allowed(idx)) << "kind mismatch must not be trusted";
    list = saved;
    // (d) the list says binary_mul (not in place) for the slot
    for (auto& e : list) {
        if (e.op_type == onednn_post_op_type::binary_mul_inplace) {
            e.op_type = onednn_post_op_type::binary_mul;
        }
    }
    EXPECT_FALSE(fc->reused_eltw_alias_allowed(idx));
    list = saved;
    EXPECT_TRUE(fc->reused_eltw_alias_allowed(idx));
}

// C2: when the alias becomes illegal in a later round (here: the live predicate turns false because the dependency is now an output),
// dst must be handed a private buffer instead of keeping the stale alias.
TEST(inplace_alias_conflict_f17, C2_dynamic_stale_alias_is_dropped) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(true, 0);
    auto weights_mem = make_weights(engine, 32, 64, 0);
    network net_ref(engine, make_c1_topology(c, weights_mem), accept_config(engine, false, fc_onednn()));
    auto prog = program::build_program(engine, make_c1_topology(c, weights_mem), accept_config(engine, true, fc_onednn()));
    network net(prog, 0);
    auto& fc_node = prog->get_node("fc");
    const auto idx = fusing::get_reused_eltwmem_idx(fc_node);
    ASSERT_NE(idx, -1);

    int64_t prev_batch = -1;
    // The alias is only re-evaluated by realloc_outputs, which runs when the shape changes (or the impl changes). Every round that
    // must drop the alias therefore uses a batch different from the previous one; this is asserted, not assumed.
    auto run_round = [&](int64_t batch, bool expect_alias, bool must_realloc) {
        if (must_realloc) {
            ASSERT_NE(batch, prev_batch) << "this round must change the shape, otherwise realloc_outputs (and the de-alias branch) is not reached";
        }
        prev_batch = batch;
        auto in_mem = make_f16(engine, ov::PartialShape{batch, 64}, 1);
        auto mul_mem = make_f16(engine, ov::PartialShape{batch, 32}, 2);
        net_ref.set_input_data("input", in_mem);
        net_ref.set_input_data("mul_in", mul_mem);
        auto ref = to_f32(net_ref.execute().at("out").get_memory());
        net.set_input_data("input", in_mem);
        net.set_input_data("mul_in", mul_mem);
        auto out = to_f32(net.execute().at("out").get_memory());
        trim(ref, batch * 32);
        trim(out, batch * 32);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(ref[i], out[i], 1e-2f) << "batch=" << batch << " index " << i << " of " << ref.size();
        }
        auto fc = net.get_primitive("fc");
        EXPECT_EQ(fc->get_output_layout().get_shape()[0], static_cast<size_t>(batch));
        EXPECT_EQ(same_buf(engine, *fc, *net.get_primitive("mul_data")), expect_alias) << "batch=" << batch;
        EXPECT_EQ(fc->output_aliases_reused_eltw(), expect_alias) << "batch=" << batch << ": alias flag";
    };
    run_round(16, true, false);
    run_round(16, true, false);
    // flip the live predicate (M3: dep becomes a network output); the frozen attrs and the impl are unchanged
    const std::string dep_id = fc_node.get_dependency(idx).id();
    prog->get_node(dep_id).set_output(true);
    ASSERT_EQ(fusing::get_reused_eltwmem_idx(fc_node), -1);
    // First round after the flip is a SHRINK: without the de-alias branch can_reuse_buffer would be true and the stale alias kept.
    run_round(8, false, true);
    run_round(32, false, true);
    run_round(16, false, true);
}

// T8 / M8: frozen binary_mul, live "in place". FC fused [prod(A), prod(B)]: the freeze gives A the in-place slot and B a plain
// binary_mul. If A is rejected live afterwards, B passes the live predicate but must not be aliased (oneDNN was never told).
TEST(inplace_alias_conflict_f17, T8_frozen_binary_mul_live_inplace_is_not_aliased) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto make_graph = [&](mul_graph& g) {
        auto& t = g.t;
        const ov::PartialShape sh{32, 16};
        t.add(input_layout("in1", layout{sh, data_types::f16, format::bfyx}));
        t.add(input_layout("ia", layout{sh, data_types::f16, format::bfyx}));
        t.add(input_layout("ib", layout{sh, data_types::f16, format::bfyx}));
        g.inputs = {{"in1", sh}, {"ia", sh}, {"ib", sh}};
        t.add(activation("A", input_info("ia"), activation_func::abs));
        t.add(activation("B", input_info("ib"), activation_func::abs));
        t.add(data("wW", make_weights(engine, 16, 16, 0)));
        t.add(fully_connected("W", input_info("in1"), "wW"));
        t.add(eltwise("M1", {input_info("W"), input_info("A")}, eltwise_mode::prod));
        t.add(eltwise("M2", {input_info("M1"), input_info("B")}, eltwise_mode::prod));
        t.add(data("wZ", make_weights(engine, 16, 16, 2)));
        t.add(fully_connected("Z", input_info("M2"), "wZ"));
        t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
        g.forced.emplace("W", impl_desc(impl_types::onednn));
        g.forced.emplace("Z", impl_desc(impl_types::ocl));
        g.outs = {"out_Z"};
    };
    mul_graph g;
    make_graph(g);
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, {}));
    const std::string flags = dump_flags(*prog);
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    int prods = 0;
    for (const auto& fd : W.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::prod) {
            prods++;
        }
    }
    ASSERT_EQ(prods, 2) << "both prods are expected to be fused into W\n" << flags;
    int inplace_frozen = 0, plain_frozen = 0;
    for (const auto& e : W.get_fused_primitives_onednn()) {
        inplace_frozen += e.op_type == onednn_post_op_type::binary_mul_inplace;
        plain_frozen += e.op_type == onednn_post_op_type::binary_mul;
    }
    ASSERT_EQ(inplace_frozen, 1) << flags;
    ASSERT_EQ(plain_frozen, 1) << flags;
    const auto idx_before = fusing::get_reused_eltwmem_idx(W);
    ASSERT_NE(idx_before, -1);
    const std::string first_dep = W.get_dependency(idx_before).id();
    const std::string second_dep = first_dep == "A" ? "B" : "A";

    // live predicate flips for the in-place slot owner only
    prog->get_node(first_dep).set_output(true);
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1) << "the plain-mul slot of " << second_dep << " must not be aliased";

    network net(prog, 0);
    EXPECT_FALSE(engine.is_the_same_buffer(*net.get_primitive("W")->output_memory_ptr(), *net.get_primitive(second_dep)->output_memory_ptr()));
    EXPECT_FALSE(engine.is_the_same_buffer(*net.get_primitive("W")->output_memory_ptr(), *net.get_primitive(first_dep)->output_memory_ptr()));
    EXPECT_EQ(net.get_primitive("W")->get_reused_eltw_dep_idx(), -1);

    memmap ins;
    int v = 1;
    for (const auto& in : g.inputs) {
        ins[in.first] = make_f16(engine, in.second, v++);
    }
    auto got = run_net(net, ins, g.outs);
    mul_graph gref;
    make_graph(gref);
    network net_ref(engine, gref.t, ref_config(engine, {}));
    expect_match(got, run_net(net_ref, ins, gref.outs), "T8");
}

// T14: cascades. [sum(S) rejected by F6, mul(M) accepted] and [mul(M) accepted, sum(S) rejected]: the in-place slot is the mul and the
// returned index must belong to it. [sum accepted, mul accepted]: the sum takes the slot, the mul is a plain binary_mul (not aliased).
enum class cascade { sum_rejected_then_mul, mul_then_sum_rejected, sum_then_mul_both_ok };

void run_cascade(cascade kind, const std::string& tag) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    const ov::PartialShape sh{32, 16};
    auto make_graph = [&](mul_graph& g) {
        auto& t = g.t;
        t.add(input_layout("in1", layout{sh, data_types::f16, format::bfyx}));
        t.add(input_layout("im", layout{sh, data_types::f16, format::bfyx}));
        g.inputs = {{"in1", sh}, {"im", sh}};
        t.add(activation("M", input_info("im"), activation_func::abs));
        std::string s_id = "S";
        if (kind == cascade::sum_then_mul_both_ok) {
            t.add(input_layout("is", layout{sh, data_types::f16, format::bfyx}));
            g.inputs.push_back({"is", sh});
            t.add(activation("S", input_info("is"), activation_func::abs));
        } else {
            t.add(input_layout("S", layout{sh, data_types::f16, format::bfyx}));  // external: F6 rejects the sum
            g.inputs.push_back({"S", sh});
        }
        t.add(data("wW", make_weights(engine, 16, 16, 0)));
        t.add(fully_connected("W", input_info("in1"), "wW"));
        if (kind == cascade::mul_then_sum_rejected) {
            t.add(eltwise("E1", {input_info("W"), input_info("M")}, eltwise_mode::prod));
            t.add(eltwise("E2", {input_info("E1"), input_info("S")}, eltwise_mode::sum));
        } else {
            t.add(eltwise("E1", {input_info("W"), input_info("S")}, eltwise_mode::sum));
            t.add(eltwise("E2", {input_info("E1"), input_info("M")}, eltwise_mode::prod));
        }
        t.add(data("wZ", make_weights(engine, 16, 16, 2)));
        t.add(fully_connected("Z", input_info("E2"), "wZ"));
        t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
        g.forced.emplace("W", impl_desc(impl_types::onednn));
        g.forced.emplace("Z", impl_desc(impl_types::ocl));
        g.outs = {"out_Z"};
    };
    mul_graph g;
    make_graph(g);
    auto prog = program::build_program(engine, g.t, opt_config(engine, g.forced, {}));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    auto& W = prog->get_node("W");
    ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn) << flags;
    ASSERT_EQ(W.get_fused_primitives().size() >= 2, true) << "both eltwise ops are expected to be fused into W\n" << flags;
    const auto idx = fusing::get_reused_eltwmem_idx(W);
    ASSERT_NE(idx, -1) << flags;
    const std::string expected_dep = kind == cascade::sum_then_mul_both_ok ? "S" : "M";
    EXPECT_EQ(W.get_dependency(idx).id(), expected_dep) << flags;
    auto w = net.get_primitive("W");
    EXPECT_TRUE(w->reused_eltw_alias_allowed(idx));
    EXPECT_TRUE(same_buf(engine, *w, *net.get_primitive(expected_dep)));
    const std::string other = expected_dep == "S" ? "M" : "S";
    EXPECT_FALSE(same_buf(engine, *w, *net.get_primitive(other)));
    // C4: F19 (sum only) must never record a prod writer; here it may record the sum case only
    if (kind != cascade::sum_then_mul_both_ok) {
        EXPECT_EQ(w->get_reused_eltw_dep_idx(), -1);
    }

    memmap ins;
    int v = 1;
    for (const auto& in : g.inputs) {
        ins[in.first] = make_f16(engine, in.second, v++);
    }
    auto got = run_net(net, ins, g.outs);
    mul_graph gref;
    make_graph(gref);
    network net_ref(engine, gref.t, ref_config(engine, {}));
    expect_match(got, run_net(net_ref, ins, gref.outs), tag);
}

TEST(inplace_alias_conflict_f17, T14a_cascade_sum_rejected_then_mul) {
    run_cascade(cascade::sum_rejected_then_mul, "T14a");
}
TEST(inplace_alias_conflict_f17, T14b_cascade_mul_then_sum_rejected) {
    run_cascade(cascade::mul_then_sum_rejected, "T14b");
}
TEST(inplace_alias_conflict_f17, T14c_cascade_sum_then_mul_mul_stays_plain) {
    run_cascade(cascade::sum_then_mul_both_ok, "T14c");
}

// T13b: the same shared-parameter pair in the opposite order. W_b consumes W_a's result, so W_a (dep = private activation) runs and is
// compiled first; W_b (dep = input_layout) rejects the alias afterwards. Existence check: W_a MUST alias in every round and its impl must
// be compiled in place; W_b must not alias (an in-place impl taken from the cache is the harmless direction for W_b).
TEST(inplace_alias_conflict_f17, T13b_dynamic_first_writer_aliases_second_rejects) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(true, 0);
    auto wa = make_weights(engine, 64, 64, 0);
    auto wb = make_weights(engine, 64, 64, 0);
    auto make_topo = [&]() {
        return topology(input_layout("in_a", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        input_layout("mul_a_in", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        input_layout("mul_b", layout{ov::PartialShape{c.batch_dim, 64}, data_types::f16, format::bfyx}),
                        data("wa", wa),
                        data("wb", wb),
                        activation("mul_a", input_info("mul_a_in"), activation_func::abs),
                        fully_connected("W_a", input_info("in_a"), "wa", "", 2),
                        eltwise("Pa", input_info("W_a"), input_info("mul_a"), eltwise_mode::prod),
                        softmax("Za", input_info("Pa"), 1),
                        fully_connected("W_b", input_info("Za"), "wb", "", 2),
                        eltwise("Pb", input_info("W_b"), input_info("mul_b"), eltwise_mode::prod),
                        softmax("Zb", input_info("Pb"), 1),
                        reorder("out", input_info("Zb"), format::bfyx, data_types::f32));
    };
    ov::intel_gpu::ImplForcingMap forced{{"W_a", impl_desc(impl_types::onednn)}, {"W_b", impl_desc(impl_types::onednn)}};
    network net_ref(engine, make_topo(), accept_config(engine, false, forced));
    auto prog = program::build_program(engine, make_topo(), accept_config(engine, true, forced));
    network net(prog, 0);
    const std::string flags = dump_flags(*prog);
    auto& Wa = prog->get_node("W_a");
    auto& Wb = prog->get_node("W_b");
    ASSERT_NE(fused_prod(Wa), nullptr) << flags;
    ASSERT_NE(fused_prod(Wb), nullptr) << flags;
    const auto idx_a = fusing::get_reused_eltwmem_idx(Wa);
    ASSERT_NE(idx_a, -1) << flags;
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(Wb), -1) << flags;
    for (const auto batch : std::vector<int64_t>{16, 32, 8, 16}) {
        memmap ins{{"in_a", make_f16(engine, ov::PartialShape{batch, 64}, 1)},
                   {"mul_a_in", make_f16(engine, ov::PartialShape{batch, 64}, 2)},
                   {"mul_b", make_f16(engine, ov::PartialShape{batch, 64}, 3)}};
        auto ref = run_net(net_ref, ins, {"out"});
        auto got = run_net(net, ins, {"out"});
        trim_map(ref, batch * 64);
        trim_map(got, batch * 64);
        expect_match(got, ref, "T13b/batch" + std::to_string(batch));
        auto wa_inst = net.get_primitive("W_a");
        EXPECT_EQ(wa_inst->get_fused_primitives_onednn().size(), 1u);
        ASSERT_NE(wa_inst->get_impl(), nullptr);
        EXPECT_TRUE(wa_inst->get_impl()->is_binary_mul_inplace_post_op({post_op_kind::binary}, 0)) << "batch=" << batch;
        EXPECT_TRUE(same_buf(engine, *wa_inst, *net.get_primitive("mul_a"))) << "batch=" << batch << ": W_a must alias its dependency";
        EXPECT_FALSE(same_buf(engine, *net.get_primitive("W_b"), *net.get_primitive("mul_b"))) << "batch=" << batch;
    }
}

// Pool block -> alias transition (Step 6 Medium-1): the first round runs with the alias rejected (dep flagged as output while the
// network is built, so W's dst is a pool/private block), then the dep stops being an output and the next shape binds the alias. The
// downstream dynamic users (softmax, sum) read W every round; outputs must stay correct and the executor must see the new dst.
TEST(inplace_alias_conflict_f17, T17_dynamic_alias_established_after_first_round) {
    F17_SKIP_IF_NO_IMMAD();
    auto& engine = get_test_engine();
    auto c = make_ctx(true, 0);
    auto weights_mem = make_weights(engine, 32, 64, 0);
    network net_ref(engine, make_c1_topology(c, weights_mem), accept_config(engine, false, fc_onednn()));
    auto prog = program::build_program(engine, make_c1_topology(c, weights_mem), accept_config(engine, true, fc_onednn()));
    auto& fc_node = prog->get_node("fc");
    const auto idx = fusing::get_reused_eltwmem_idx(fc_node);
    ASSERT_NE(idx, -1);
    const std::string dep_id = fc_node.get_dependency(idx).id();
    prog->get_node(dep_id).set_output(true);  // live predicate rejects while the first round(s) run
    ASSERT_EQ(fusing::get_reused_eltwmem_idx(fc_node), -1);
    network net(prog, 0);

    auto run_round = [&](int64_t batch, bool expect_alias) {
        auto in_mem = make_f16(engine, ov::PartialShape{batch, 64}, 1);
        auto mul_mem = make_f16(engine, ov::PartialShape{batch, 32}, 2);
        net_ref.set_input_data("input", in_mem);
        net_ref.set_input_data("mul_in", mul_mem);
        auto ref = to_f32(net_ref.execute().at("out").get_memory());
        net.set_input_data("input", in_mem);
        net.set_input_data("mul_in", mul_mem);
        auto out = to_f32(net.execute().at("out").get_memory());
        trim(ref, batch * 32);
        trim(out, batch * 32);
        ASSERT_EQ(ref.size(), out.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(ref[i], out[i], 1e-2f) << "batch=" << batch << " index " << i;
        }
        auto fc = net.get_primitive("fc");
        EXPECT_EQ(fc->output_aliases_reused_eltw(), expect_alias) << "batch=" << batch;
        EXPECT_EQ(same_buf(engine, *fc, *net.get_primitive("mul_data")), expect_alias) << "batch=" << batch;
    };
    run_round(16, false);
    run_round(16, false);
    prog->get_node(dep_id).set_output(false);
    ASSERT_NE(fusing::get_reused_eltwmem_idx(fc_node), -1);
    run_round(32, true);   // pool block -> alias
    run_round(8, true);
    run_round(32, true);
}

// Not constructed (reasons):
//  * M3c (W is a slot of an in-place concat): tried a static graph (W onednn FC, batch 1, feature-axis concat with an activation sibling,
//    oneDNN 64-byte offset satisfied); prepare_buffer_fusing still refuses to make the concat in-place (a fully_connected output does not
//    pass the predecessor padding/availability checks), so the vacuity guard "concat is in place" fails. M3c is therefore a defensive rule
//    for runtime-skippable/dynamic concats and has no test.
//  * T3 (static resample view): resample needs a 4D/5D tensor while an FC writer produces 2D/3D, so no resample-viewed dependency can
//    feed the prod of an FC. The same view-chain predicate is exercised through the strided_slice view (T3b) and reshape view (T2b).
//    Source-level coverage of the resample rebind guard is the P6 change in resample.cpp (reused_eltw_alias_allowed on the user).
//  * T9 (writer reads the dependency through another input or output port, M5/C6): with the dependency as the FC's own input the
//    prod is not fused at all (prepare_primitive_fusing refuses an eltwise whose other operand is the FC's ancestor), and the
//    other-OUTPUT-PORT variant needs a multi-output producer with an f16 [32,16] port 1, which no unit-graph primitive provides.
//  * T10 (dependency is port > 0 of a multi-output node): same reason (no f16 [32,16] port-1 producer available in a unit graph).
//  * T12 (dependency is a CPU impl or in a shape_of subgraph): no CPU impl exists for f16 tensors that can feed an FC prod, and a
//    shape_of subgraph yields i32 shape data, which cannot be the prod operand of an FC output.
//  * T15 (dynamic W whose forward view chain reaches an output through a runtime-skippable reorder): the output of a skippable
//    reorder cannot be made reproducibly skippable from a unit topology (the flag is set per round by shape).

}  // namespace
