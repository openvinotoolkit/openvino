// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Pre-0 F5 (structural part): the oneDNN in-place sum has to be consistent on four sides at the end of post_optimize_graph and stay
// so through a cache export/import:
//   frozen list entry == sum  <=>  compiled impl slot is a sum  <=>  get_reused_eltwmem_idx == dep  <=>  dst is bound to the addend.
// The new pass refreeze_onednn_inplace_post_ops (last statement of post_optimize_graph) downgrades a frozen sum that the live graph does
// not support any more (or whose impl cannot honour it) to a binary_add, in place, and rebuilds the oneDNN impl. It never upgrades.
//
// How the divergences are made: program::build_program() is monolithic, so the tests drive the same stages through program_wrapper
// (init_graph, pre_optimize_graph incl. the freeze, run_graph_compilation, post_optimize_graph incl. the pass, prepare_memory_dependencies,
// build_implementations) and mutate the graph in a hook between the freeze and post_optimize_graph, exactly where the real post-freeze
// passes (post_input_reorder, remove_redundant_reorders, mutable_data output propagation) act.
//
// Every test that needs a oneDNN sum is GTEST_SKIPped without supports_immad (a SKIP counts as "not covered").

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

#include "pass_manager.h"
#include "program_helpers.h"
#include "program_node.h"
#include "primitive_inst.h"
#include "program_wrapper.h"
#include "registry/implementation_manager.hpp"
#include "primitive_type.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

#ifdef ENABLE_ONEDNN_FOR_GPU

namespace {

using fusing = onednn_eltwise_fusing_helpers;
using reason_t = onednn_eltwise_fusing_helpers::sum_downgrade_reason;

const ov::PartialShape kShape{32, 16};

#define F5_SKIP_IF_NO_IMMAD()                                                       \
    do {                                                                            \
        if (!get_test_engine().get_device_info().supports_immad) {                  \
            GTEST_SKIP() << "the oneDNN in-place sum needs a oneDNN device (supports_immad)"; \
        }                                                                           \
    } while (0)

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

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, int variant, format fmt = format::bfyx) {
    layout l{shape, data_types::f16, fmt};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(static_cast<float>(((i * (3 + variant)) % 13)) * 0.0625f + 0.125f * variant);
    }
    set_values(mem, v);
    return mem;
}

memory::ptr make_weights(engine& eng, const ov::PartialShape& shape, int variant) {
    layout l{shape, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t k = 0; k < v.size(); k++) {
        v[k] = ov::float16(static_cast<float>((k * (5 + variant)) % 7) * 0.03125f - 0.0625f);
    }
    set_values(mem, v);
    return mem;
}

memory::ptr make_fc_weights(engine& eng, int64_t o, int64_t i, int variant) {
    return make_weights(eng, ov::PartialShape{o, i}, variant);
}

ov::intel_gpu::ImplementationDesc impl_desc(impl_types t, format f = format::bfyx) {
    return ov::intel_gpu::ImplementationDesc{f, "", t};
}

ExecutionConfig opt_config(engine& eng, const ov::intel_gpu::ImplForcingMap& forced, const std::vector<std::string>& custom_outputs,
                           QueueTypes q = QueueTypes::in_order, bool new_shape_infer = true) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(new_shape_infer));
    config.set_property(ov::intel_gpu::force_implementations(forced));
    config.set_property(ov::intel_gpu::queue_type(q));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

ExecutionConfig ref_config(engine& eng, const std::vector<std::string>& custom_outputs, bool new_shape_infer = true) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(new_shape_infer));
    if (!custom_outputs.empty()) {
        config.set_property(ov::intel_gpu::custom_outputs(custom_outputs));
    }
    return config;
}

struct f5_graph {
    topology t;
    ov::intel_gpu::ImplForcingMap forced;
    std::vector<std::string> outs;            // outputs compared with the reference
    std::vector<std::string> custom_outputs;  // empty: the sink reorder is the only output
    std::vector<std::pair<std::string, layout>> inputs;
    bool new_shape_infer = true;
    std::vector<std::string> extra_probe;     // node ids whose output buffer is compared after execute() (not network outputs)
};

using hook_t = std::function<void(program&)>;

// Same stages as program::build_program(), with a hook between the oneDNN attribute freeze + impl selection and post_optimize_graph.
program::ptr build_staged(engine& eng, const f5_graph& g, const hook_t& hook, QueueTypes q = QueueTypes::in_order) {
    auto cfg = opt_config(eng, g.forced, g.custom_outputs, q, g.new_shape_infer);
    auto prog = program::build_program(eng, g.t, cfg, /*is_internal=*/false, /*no_optimizations=*/true);
    program_wrapper::pre_optimize_graph(*prog, false);
    program_wrapper::run_graph_compilation(*prog);
    if (hook) {
        hook(*prog);
    }
    program_wrapper::post_optimize_graph(*prog, false);
    program_wrapper::prepare_memory_dependencies(*prog);
    program_wrapper::apply_opt_pass<build_implementations>(*prog);
    if (eng.get_device_info().has_separate_cache) {
        program_wrapper::transfer_memory_to_device(*prog);
    }
    return prog;
}

using result_map = std::map<std::string, std::vector<float>>;

result_map run_net(network& net, engine& eng, const f5_graph& g, const std::vector<std::string>& probes = {}) {
    int variant = 1;
    for (const auto& in : g.inputs) {
        net.set_input_data(in.first, make_f16(eng, in.second.get_partial_shape(), variant++, in.second.format));
    }
    auto outs = net.execute();
    result_map r;
    for (const auto& id : g.outs) {
        r[id] = to_f32(outs.at(id).get_memory());
    }
    for (const auto& id : probes) {
        r["probe:" + id] = to_f32(net.get_primitive(id)->output_memory_ptr());
    }
    return r;
}

result_map run_reference(engine& eng, const f5_graph& g, const std::vector<std::string>& probes = {}) {
    std::vector<std::string> outputs = g.custom_outputs;
    for (const auto& p : probes) {
        if (std::find(outputs.begin(), outputs.end(), p) == outputs.end()) {
            outputs.push_back(p);
        }
    }
    if (outputs.empty()) {
        outputs = g.outs;
    } else {
        for (const auto& o : g.outs) {
            if (std::find(outputs.begin(), outputs.end(), o) == outputs.end()) {
                outputs.push_back(o);
            }
        }
    }
    network net_ref(eng, g.t, ref_config(eng, outputs, g.new_shape_infer));
    return run_net(net_ref, eng, g, probes);
}

void expect_match(const result_map& got, const result_map& ref, const std::string& tag) {
    for (const auto& kv : ref) {
        const auto it = got.find(kv.first);
        ASSERT_TRUE(it != got.end()) << tag << ": missing " << kv.first;
        const auto& g = it->second;
        ASSERT_EQ(g.size(), kv.second.size()) << kv.first;
        size_t bad = 0;
        float max_diff = 0.f;
        for (size_t i = 0; i < g.size(); i++) {
            const float d = std::fabs(g[i] - kv.second[i]);
            max_diff = std::max(max_diff, d);
            if (!(d <= 0.02f + 0.01f * std::fabs(kv.second[i]))) {
                bad++;
            }
        }
        std::cout << "[CMP " << tag << "] " << kv.first << " max_diff=" << max_diff << " mismatches=" << bad << "/" << g.size() << std::endl;
        EXPECT_EQ(bad, 0u) << tag << ": output " << kv.first << " does not match the reference";
    }
}

// ---- the five-item consistency check -----------------------------------------------------------------------------------------------

const fused_primitive_desc* fused_sum_op(const program_node& p) {
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == eltwise_mode::sum && fd.deps.size() == 1 && fd.has_outer_dep()) {
            return &fd;
        }
    }
    return nullptr;
}

const fused_primitive_desc* fused_op_of_mode(const program_node& p, eltwise_mode mode, size_t nth = 0) {
    size_t seen = 0;
    for (const auto& fd : p.get_fused_primitives()) {
        if (fd.is_type<eltwise>() && fd.typed_desc<eltwise>()->mode == mode && fd.deps.size() == 1 && fd.has_outer_dep()) {
            if (seen++ == nth) {
                return &fd;
            }
        }
    }
    return nullptr;
}

bool impl_has_sum_at(const program_node& n, int32_t dep) {
    const auto* impl = n.get_selected_impl();
    if (!impl || !impl->is_onednn()) {
        return false;
    }
    const auto slots = get_compiled_post_op_slots(n.get_fused_primitives_onednn(),
                                                  static_cast<size_t>(dep),
                                                  {onednn_post_op_type::sum, onednn_post_op_type::optimized_sum});
    return slots.slot && impl->is_sum_post_op(slots.kinds, *slots.slot);
}

size_t count_inplace_entries(const program_node& n) {
    size_t c = 0;
    for (const auto& e : n.get_fused_primitives_onednn()) {
        if (e.op_type == onednn_post_op_type::sum || e.op_type == onednn_post_op_type::optimized_sum ||
            e.op_type == onednn_post_op_type::binary_mul_inplace) {
            c++;
        }
    }
    return c;
}

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " can_be_optimized=" << n->can_be_optimized()
           << " is_output=" << n->is_output() << " fused=" << n->get_fused_primitives().size() << " users=" << n->get_users().size()
           << " can_share_buffer=" << n->can_share_buffer() << " impl=" << static_cast<int>(n->get_preferred_impl_type()) << "\n";
    }
    return os.str();
}

// The invariant I-F5-1 on node `w` for the fused sum on dependency `dep`: frozen <=> impl sum <=> get_reused == dep <=> can_share_buffer
// false (the writer is excluded from the pool) <=> (once a network exists) dst bound to the addend. `expect_inplace` is what the scenario
// demands; `net` may be null (program-level checks only). `check_can_share` is for graphs where nothing else clears the flag.
void expect_inplace_consistent(program& prog, network* net, const std::string& w_id, int32_t dep, bool expect_inplace, bool check_can_share = true) {
    auto& w = prog.get_node(w_id);
    const std::string flags = dump_flags(prog);
    // 1. frozen list
    EXPECT_EQ(fusing::is_frozen_inplace_post_op(w, dep, eltwise_mode::sum), expect_inplace) << "frozen entry\n" << flags;
    // 2. compiled impl slot
    EXPECT_EQ(impl_has_sum_at(w, dep), expect_inplace) << "impl slot\n" << flags;
    // 3. get_reused
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(w) == dep, expect_inplace) << "get_reused_eltwmem_idx\n" << flags;
    // 4. can_share_buffer
    if (check_can_share) {
        EXPECT_EQ(!w.can_share_buffer(), expect_inplace) << "can_share_buffer\n" << flags;
    }
    // 5. same buffer after construction
    if (net != nullptr) {
        auto w_inst = net->get_primitive(w_id);
        auto dep_inst = net->get_primitive(w.get_dependency(static_cast<size_t>(dep)).id());
        auto mw = w_inst->output_memory_ptr();
        auto md = dep_inst->output_memory_ptr();
        const bool same = mw && md && prog.get_engine().is_the_same_buffer(*mw, *md);
        EXPECT_EQ(same, expect_inplace) << "same_buffer after construction\n" << flags;
    }
    // the shared decision function agrees with the frozen state
    if (expect_inplace) {
        EXPECT_TRUE(fusing::frozen_sum_downgrade_reason(w, dep) == reason_t::none) << flags;
    }
}

// ---- graphs -----------------------------------------------------------------------------------------------------------------------------

// X = FC(in3) ocl (single user); W = FC(in1) onednn; S = sum(W, X) [fused into W]; A = relu(S) [fused]; Z = FC(A) ocl; out_Z.
f5_graph build_base(engine& eng, bool with_act = true, bool output_probe_x = false) {
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{kShape, data_types::f16, format::bfyx}}, {"in3", layout{kShape, data_types::f16, format::bfyx}}};
    t.add(data("wX", make_fc_weights(eng, 16, 16, 1)));
    t.add(fully_connected("X", input_info("in3"), "wX"));
    t.add(data("wW", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
    std::string z_in = "S";
    if (with_act) {
        t.add(activation("A", input_info("S"), activation_func::relu));
        z_in = "A";
    }
    t.add(data("wZ", make_fc_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info(z_in), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("X", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};
    if (output_probe_x) {
        g.extra_probe = {"X"};
    }
    return g;
}

// Builds, runs and compares one scenario; returns the program for further inspection.
program::ptr run_and_compare(engine& eng, const f5_graph& g, const hook_t& hook, const std::string& tag, QueueTypes q = QueueTypes::in_order,
                             const std::vector<std::string>& probes = {}, std::shared_ptr<network>* net_out = nullptr) {
    auto prog = build_staged(eng, g, hook, q);
    auto net = std::make_shared<network>(prog, 0);
    auto res = run_net(*net, eng, g, probes);
    auto ref = run_reference(eng, g, probes);
    // the reference has the probes as plain outputs under the same key
    result_map ref_keyed;
    for (const auto& kv : ref) {
        ref_keyed[kv.first] = kv.second;
    }
    for (const auto& p : probes) {
        auto it = ref.find(p);
        if (it != ref.end()) {
            ref_keyed["probe:" + p] = it->second;
            ref_keyed.erase(p);
        }
    }
    expect_match(res, ref_keyed, tag);
    if (net_out) {
        *net_out = net;
    }
    return prog;
}

// dnnl post-op chain of the node attributes: kind per slot.
std::vector<dnnl::primitive::kind> attr_kinds(program_node& n) {
    std::vector<dnnl::primitive::kind> r;
    const auto ops = n.get_onednn_primitive_attributes()->get_post_ops();
    for (int i = 0; i < ops.len(); i++) {
        r.push_back(ops.kind(i));
    }
    return r;
}

}  // namespace

// =============================================================================================================================
// T0. White-box mechanics of one downgrade (X becomes a network output after the freeze: dst of the sum would overwrite it).
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T0_downgrade_mechanics) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);

    size_t offset_before = 0, dep_before = 0;
    size_t entries_before = 0;
    std::vector<dnnl::primitive::kind> kinds_before;
    const primitive_impl* impl_before = nullptr;
    int32_t dep = -1;
    size_t slot = 0;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        const auto* op = fused_sum_op(W);
        ASSERT_NE(op, nullptr);
        dep = op->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum)) << "vacuity: the freeze must have chosen sum";
        ASSERT_TRUE(impl_has_sum_at(W, dep));
        const auto& list = W.get_fused_primitives_onednn();
        entries_before = list.size();
        for (size_t i = 0; i < list.size(); i++) {
            if (list[i].op_type == onednn_post_op_type::sum) {
                slot = i;
                offset_before = list[i].mem_offset;
                dep_before = list[i].mem_dep;
            }
        }
        kinds_before = attr_kinds(W);
        impl_before = W.get_selected_impl();
        // the post-freeze divergence: X becomes user-visible
        p.get_node("X").set_output(true);
        ASSERT_TRUE(fusing::get_add_fusing_type(W, *op) != add_fusing_type::sum) << "vacuity: the live predicate must now reject the sum";
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, "T0", QueueTypes::in_order, {"X"}, &net);

    auto& W = prog->get_node("W");
    const auto& list = W.get_fused_primitives_onednn();
    ASSERT_EQ(list.size(), entries_before);
    EXPECT_EQ(list[slot].op_type, onednn_post_op_type::binary_add);
    EXPECT_EQ(list[slot].mem_offset, offset_before);
    EXPECT_EQ(list[slot].mem_dep, dep_before);
    EXPECT_FALSE(list[slot].flatten);
    // all fields written back (MAJOR in the audit): a cache import rebuilds the md from them
    EXPECT_FALSE(list[slot].dims.empty());
    EXPECT_NE(list[slot].tag, dnnl::memory::format_tag::undef);
    EXPECT_NE(list[slot].dt, dnnl::memory::data_type::undef);
    const auto expected = W.make_onednn_binary_post_op(W.get_input_layout(dep), W.get_output_layout());
    EXPECT_EQ(list[slot].dims, expected.dims);
    EXPECT_EQ(list[slot].tag, expected.tag);
    EXPECT_EQ(list[slot].dt, expected.dt);
    // attrs: same chain length, only that slot differs
    const auto kinds_after = attr_kinds(W);
    ASSERT_EQ(kinds_after.size(), kinds_before.size());
    for (size_t i = 0; i < kinds_after.size(); i++) {
        if (i == slot) {
            EXPECT_TRUE(kinds_before[i] == dnnl::primitive::kind::sum);
            EXPECT_TRUE(kinds_after[i] == dnnl::primitive::kind::binary);
            dnnl::algorithm alg;
            dnnl::memory::desc md;
            W.get_onednn_primitive_attributes()->get_post_ops().get_params_binary(static_cast<int>(i), alg, md);
            EXPECT_EQ(alg, dnnl::algorithm::binary_add);
        } else {
            EXPECT_TRUE(kinds_after[i] == kinds_before[i]) << "slot " << i;
        }
    }
    // the impl was rebuilt around the new attrs
    EXPECT_NE(W.get_selected_impl(), impl_before);
    EXPECT_TRUE(W.get_selected_impl()->is_onednn());
    EXPECT_EQ(count_inplace_entries(W), 0u);
    expect_inplace_consistent(*prog, net.get(), "W", dep, /*expect_inplace=*/false);
}

// =============================================================================================================================
// T1. post_input_reorder-like hop: one more dep(0) hop after the freeze moves the sum beyond max_ancestor_walk_depth.
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T1_extra_dep0_hop_after_freeze) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    // in -> x_proj -> fc_1 .. fc_5 -> p ; res = sum(p, x_proj): 6 dep(0) hops from p to x_proj (the last distance that is a sum)
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in", layout{kShape, data_types::f16, format::bfyx}}};
    auto add_fc = [&](const std::string& id, const std::string& in, int variant) {
        t.add(data(id + "_w", make_fc_weights(eng, 16, 16, variant)));
        t.add(fully_connected(id, input_info(in), id + "_w"));
        g.forced.emplace(id, impl_desc(impl_types::ocl));
    };
    add_fc("x_proj", "in", 1);
    std::string cur = "x_proj";
    for (int i = 1; i <= 5; i++) {
        add_fc("fc_" + std::to_string(i), cur, 1 + i);
        cur = "fc_" + std::to_string(i);
    }
    t.add(data("p_w", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("p", input_info(cur), "p_w"));
    g.forced["p"] = impl_desc(impl_types::onednn);
    t.add(eltwise("res", {input_info("p"), input_info("x_proj")}, eltwise_mode::sum));
    t.add(data("z_w", make_fc_weights(eng, 16, 16, 9)));
    t.add(fully_connected("Z", input_info("res"), "z_w"));
    g.forced["Z"] = impl_desc(impl_types::ocl);
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.outs = {"out_Z"};

    int32_t dep = -1;
    auto hook = [&](program& p) {
        auto& P = p.get_node("p");
        const auto* op = fused_sum_op(P);
        ASSERT_NE(op, nullptr);
        dep = op->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(P, dep, eltwise_mode::sum)) << "vacuity: 6 hops is a sum at the freeze";
        // an identity activation on p's input: what post_input_reorder does with a reorder, one more dep(0) hop
        auto act = std::make_shared<activation>("hop", input_info("fc_5"), activation_func::linear, activation_additional_params{1.f, 0.f});
        auto& act_node = p.get_or_create(act);
        p.add_intermediate(act_node, P, 0);
        act_node.get_output_layout(false);
        act_node.set_selected_impl(act_node.type()->create_impl(act_node));
        if (auto* impl = act_node.get_selected_impl()) {
            auto params = act_node.get_kernel_impl_params();
            p.get_kernels_cache().add_kernels_source(*params, impl->get_kernels_source());
        }
        ASSERT_TRUE(fusing::get_add_fusing_type(P, *op) != add_fusing_type::sum) << "vacuity: 7 hops is not a sum";
    };
    auto prog = run_and_compare(eng, g, hook, "T1");
    auto& P = prog->get_node("p");
    EXPECT_EQ(count_inplace_entries(P), 0u);
    EXPECT_TRUE(P.can_share_buffer());
    expect_inplace_consistent(*prog, nullptr, "p", dep, false);
}

// =============================================================================================================================
// T2. mutable_data output marking after the freeze (single level and cascaded): the sum would write into a user-visible buffer.
// =============================================================================================================================
namespace {
void run_t2(bool cascaded) {
    auto& eng = get_test_engine();
    // The mutable_data hangs off the fused eltwise S: after the fusion the writer W inherits it as a user, and the output marker of the
    // mutable_data (set by the hook after the freeze) makes W a user-visible buffer.
    auto g = build_base(eng, /*with_act=*/false);
    g.t.add(mutable_data("md1", {input_info("S")}, make_f16(eng, kShape, 3)));
    std::string last = "md1";
    if (cascaded) {
        g.t.add(mutable_data("md2", {input_info("md1")}, make_f16(eng, kShape, 4)));
        last = "md2";
    }
    // a consumer keeps the mutable_data from being an output at the freeze; the hook flips the marker afterwards
    g.t.add(reorder("md_sink", input_info(last), format::bfyx, data_types::f32));
    g.outs = {"out_Z", "md_sink"};  // md_sink carries the value of S (a mutable_data copies its dependency): the visible result of W
    int32_t dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        const auto* op = fused_sum_op(W);
        ASSERT_NE(op, nullptr);
        dep = op->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum));
        p.get_node(last).set_output(true);
        ASSERT_TRUE(program_helpers::is_or_becomes_output(W));
        ASSERT_TRUE(fusing::get_add_fusing_type(W, *op) != add_fusing_type::sum);
    };
    auto prog = build_staged(eng, g, hook);
    network net(prog, 0);
    EXPECT_EQ(count_inplace_entries(prog->get_node("W")), 0u);
    expect_inplace_consistent(*prog, &net, "W", dep, false);
    auto res = run_net(net, eng, g);
    auto ref = run_reference(eng, g);
    const auto& mdv = ref.at("md_sink");
    EXPECT_GT(std::count_if(mdv.begin(), mdv.end(), [](float v) { return v != 0.f; }), 0) << "vacuous reference";
    result_map keyed = ref;
    expect_match(res, keyed, cascaded ? "T2_cascaded" : "T2_single");
}
}  // namespace

TEST(inplace_alias_conflict_f5, T2_mutable_data_output_after_freeze_single) {
    F5_SKIP_IF_NO_IMMAD();
    run_t2(false);
}

TEST(inplace_alias_conflict_f5, T2_mutable_data_output_after_freeze_cascaded) {
    F5_SKIP_IF_NO_IMMAD();
    run_t2(true);
}

// =============================================================================================================================
// T3. A dependency that only becomes a view after the freeze (static no-op strided_slice of O): freeze and live resolve identically,
//     the sum stands (vacuity guard for the post-phase view candidates of resolve_buffer_owner).
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T3_post_freeze_view_dependency_stays_sum) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{kShape, data_types::f16, format::bfyx}}, {"in3", layout{kShape, data_types::f16, format::bfyx}}};
    t.add(data("wO", make_fc_weights(eng, 16, 16, 1)));
    t.add(fully_connected("O", input_info("in3"), "wO"));
    t.add(strided_slice("V", input_info("O"), {0, 0}, {32, 16}, {1, 1}, {}, {1, 1}, {}, {}, {}, ov::Shape{32, 16}));
    t.add(data("wW", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("V")}, eltwise_mode::sum));
    t.add(data("wZ", make_fc_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};

    int32_t dep = -1;
    auto hook = [&](program& p) {
        dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(p.get_node("W"), dep, eltwise_mode::sum));
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, "T3", QueueTypes::in_order, {}, &net);
    expect_inplace_consistent(*prog, net.get(), "W", dep, /*expect_inplace=*/true, /*check_can_share=*/false);
}

// =============================================================================================================================
// T4/T4b. The output reorder behind a writer with a frozen sum must survive (the sum overwrites the addend; an output without a
//         reorder would hand the user the addend buffer): directly, and behind a reshape chain.
// =============================================================================================================================
namespace {
void run_t4(bool through_reshapes) {
    auto& eng = get_test_engine();
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{kShape, data_types::f16, format::bfyx}}, {"in3", layout{kShape, data_types::f16, format::bfyx}}};
    t.add(data("wX", make_fc_weights(eng, 16, 16, 1)));
    t.add(fully_connected("X", input_info("in3"), "wX"));
    t.add(data("wW", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
    std::string last = "S";
    if (through_reshapes) {
        t.add(reshape("R1", input_info("S"), false, std::vector<int64_t>{1, 32, 16}, ov::PartialShape{1, 32, 16}));
        t.add(reshape("R2", input_info("R1"), false, std::vector<int64_t>{32, 16}, kShape));
        last = "R2";
    }
    t.add(reorder("out_W", input_info(last), format::bfyx, data_types::f16));
    g.forced.emplace("X", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.outs = {"out_W"};

    int32_t dep = -1;
    auto hook = [&](program& p) {
        dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(p.get_node("W"), dep, eltwise_mode::sum));
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, through_reshapes ? "T4b" : "T4", QueueTypes::in_order, {}, &net);
    auto& W = prog->get_node("W");
    EXPECT_FALSE(W.is_output()) << "W must not have become the output: the output reorder must be kept\n" << dump_flags(*prog);
    EXPECT_TRUE(prog->get_node("out_W").is_output());
    expect_inplace_consistent(*prog, net.get(), "W", dep, /*expect_inplace=*/true, /*check_can_share=*/false);
}
}  // namespace

TEST(inplace_alias_conflict_f5, T4_output_reorder_kept_behind_frozen_sum) {
    F5_SKIP_IF_NO_IMMAD();
    run_t4(false);
}

TEST(inplace_alias_conflict_f5, T4b_output_reorder_kept_behind_reshape_chain) {
    F5_SKIP_IF_NO_IMMAD();
    run_t4(true);
}

// =============================================================================================================================
// T5. Never upgrade, and the runtime gate.
// =============================================================================================================================
// [sum, sum]: only the first one was frozen as in-place. When it is downgraded, the second (live: a sum) must stay a binary_add.
TEST(inplace_alias_conflict_f5, T5_no_upgrade_of_second_sum) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in4", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{kShape, data_types::f16, format::bfyx}},
                {"in3", layout{kShape, data_types::f16, format::bfyx}},
                {"in4", layout{kShape, data_types::f16, format::bfyx}}};
    t.add(data("wX1", make_fc_weights(eng, 16, 16, 1)));
    t.add(fully_connected("X1", input_info("in3"), "wX1"));
    t.add(data("wX2", make_fc_weights(eng, 16, 16, 3)));
    t.add(fully_connected("X2", input_info("in4"), "wX2"));
    t.add(data("wW", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S1", {input_info("W"), input_info("X1")}, eltwise_mode::sum));
    t.add(eltwise("S2", {input_info("S1"), input_info("X2")}, eltwise_mode::sum));
    t.add(data("wZ", make_fc_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info("S2"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("X1", impl_desc(impl_types::ocl));
    g.forced.emplace("X2", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};

    int32_t dep1 = -1, dep2 = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        const auto* s1 = fused_op_of_mode(W, eltwise_mode::sum, 0);
        const auto* s2 = fused_op_of_mode(W, eltwise_mode::sum, 1);
        ASSERT_NE(s1, nullptr);
        ASSERT_NE(s2, nullptr);
        dep1 = s1->outer_dep_start_idx;
        dep2 = s2->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep1, eltwise_mode::sum)) << "the first sum is frozen in-place";
        ASSERT_FALSE(fusing::is_frozen_inplace_post_op(W, dep2, eltwise_mode::sum)) << "the second one is a frozen binary_add";
        p.get_node(p.get_node("W").get_dependency(static_cast<size_t>(dep1)).id()).set_output(true);
        ASSERT_TRUE(fusing::get_add_fusing_type(W, *s2) == add_fusing_type::sum) << "vacuity: the live predicate still says sum for the second";
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, "T5", QueueTypes::in_order, {"X1"}, &net);
    auto& W = prog->get_node("W");
    EXPECT_EQ(count_inplace_entries(W), 0u) << "downgrading the first sum must not promote the second";
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1);
    EXPECT_FALSE(impl_has_sum_at(W, dep2));
    expect_inplace_consistent(*prog, net.get(), "W", dep1, false);
    expect_inplace_consistent(*prog, net.get(), "W", dep2, false);
}

// Runtime gate (static node that re-creates its attributes, e.g. a shape_of subgraph node going through update_impl): after a
// downgrade for a reason the live predicate does not see (here reason 2), the re-created attributes must not bring the sum back.
TEST(inplace_alias_conflict_f5, T5b_runtime_gate_keeps_downgrade) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    int32_t dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        dep = fused_sum_op(W)->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum));
        W.set_in_shape_of_subgraph(true);
        ASSERT_TRUE(fusing::get_add_fusing_type(W, *fused_sum_op(W)) == add_fusing_type::sum) << "vacuity: only reason 2 applies";
    };
    auto prog = build_staged(eng, g, hook);
    auto& W = prog->get_node("W");
    EXPECT_EQ(count_inplace_entries(W), 0u);
    ASSERT_TRUE(fusing::get_add_fusing_type(W, *fused_sum_op(W)) == add_fusing_type::sum) << "the live predicate still says sum";

    auto params = W.get_kernel_impl_params();
    std::shared_ptr<dnnl::primitive_attr> attrs = std::make_shared<dnnl::primitive_attr>();
    std::vector<fused_primitive_desc_onednn> list;
    W.create_onednn_primitive_attributes(W.get_fused_primitives(), attrs, list, params.get());
    for (const auto& e : list) {
        EXPECT_NE(e.op_type, onednn_post_op_type::sum) << "the gate must keep the downgraded sum a binary_add";
        EXPECT_NE(e.op_type, onednn_post_op_type::optimized_sum);
    }
    EXPECT_EQ(attrs->get_post_ops().kind(0) == dnnl::primitive::kind::sum, false);
}

// =============================================================================================================================
// T6. Cache export / import.
// =============================================================================================================================
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

void expect_same_frozen_lists(program& a, program& b) {
    for (auto* n : a.get_processing_order()) {
        if (!b.has_node(n->id())) {
            continue;
        }
        const auto& la = n->get_fused_primitives_onednn();
        const auto& lb = b.get_node(n->id()).get_fused_primitives_onednn();
        ASSERT_EQ(la.size(), lb.size()) << n->id();
        for (size_t i = 0; i < la.size(); i++) {
            EXPECT_EQ(la[i].op_type, lb[i].op_type) << n->id() << " #" << i;
            EXPECT_EQ(la[i].mem_offset, lb[i].mem_offset) << n->id() << " #" << i;
            EXPECT_EQ(la[i].mem_dep, lb[i].mem_dep) << n->id() << " #" << i;
            EXPECT_EQ(la[i].tag, lb[i].tag) << n->id() << " #" << i;
            EXPECT_EQ(la[i].flatten, lb[i].flatten) << n->id() << " #" << i;
            EXPECT_EQ(la[i].dims, lb[i].dims) << n->id() << " #" << i;
            EXPECT_EQ(la[i].dt, lb[i].dt) << n->id() << " #" << i;
        }
    }
}
}  // namespace

// A legitimate round trip must not throw (C5: a false positive would silently invalidate the cache), keeps the in-place sum and
// does not recompute the frozen list.
TEST(inplace_alias_conflict_f5, T6a_cache_roundtrip_consistent_sum) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    auto cfg = opt_config(eng, g.forced, g.custom_outputs);
    int32_t dep = -1;
    auto prog = build_staged(eng, g, [&](program& p) { dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx; });
    ASSERT_TRUE(fusing::is_frozen_inplace_post_op(prog->get_node("W"), dep, eltwise_mode::sum));
    program::ptr imported;
    ASSERT_NO_THROW(imported = roundtrip(eng, *prog, cfg));
    ASSERT_NE(imported, nullptr);
    expect_same_frozen_lists(*prog, *imported);
    // import must not have run the lazy re-init: a non-null placeholder is in place
    EXPECT_NE(imported->get_node("W").get_onednn_primitive_attributes(), nullptr);
    network net(imported);
    expect_inplace_consistent(*imported, &net, "W", dep, true, /*check_can_share=*/false);
    auto res = run_net(net, eng, g);
    expect_match(res, run_reference(eng, g), "T6a");
}

// A downgraded node: the imported list carries the full binary_add entry (tag, dims, dt), the compiled impl is a binary_add.
TEST(inplace_alias_conflict_f5, T6b_cache_roundtrip_after_downgrade) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    auto cfg = opt_config(eng, g.forced, g.custom_outputs);
    int32_t dep = -1;
    auto prog = build_staged(eng, g, [&](program& p) {
        dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx;
        p.get_node("X").set_output(true);
    });
    ASSERT_EQ(count_inplace_entries(prog->get_node("W")), 0u);
    program::ptr imported;
    ASSERT_NO_THROW(imported = roundtrip(eng, *prog, cfg));
    expect_same_frozen_lists(*prog, *imported);
    network net(imported);
    expect_inplace_consistent(*imported, &net, "W", dep, false, /*check_can_share=*/false);
    auto res = run_net(net, eng, g);
    expect_match(res, run_reference(eng, g), "T6b");
}

// Inconsistent blobs are rejected at import: frozen entry says sum but the impl is a binary_add, and the reverse.
TEST(inplace_alias_conflict_f5, T6c_cache_import_rejects_inconsistent_state) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    auto cfg = opt_config(eng, g.forced, g.custom_outputs);
    {
        // frozen sum, impl compiled as binary_add
        int32_t dep = -1;
        auto prog = build_staged(eng, g, [&](program& p) {
            dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx;
            p.get_node("X").set_output(true);
        });
        ASSERT_EQ(count_inplace_entries(prog->get_node("W")), 0u);
        for (auto& e : prog->get_node("W").get_fused_primitives_onednn()) {
            if (e.mem_dep == static_cast<size_t>(dep)) {
                e.op_type = onednn_post_op_type::sum;
            }
        }
        EXPECT_THROW(roundtrip(eng, *prog, cfg), ov::Exception) << "frozen sum over a binary_add impl";
    }
    {
        // frozen binary_add, impl compiled as sum
        int32_t dep = -1;
        auto prog = build_staged(eng, g, [&](program& p) { dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx; });
        ASSERT_EQ(count_inplace_entries(prog->get_node("W")), 1u);
        for (auto& e : prog->get_node("W").get_fused_primitives_onednn()) {
            if (e.mem_dep == static_cast<size_t>(dep)) {
                e.op_type = onednn_post_op_type::binary_add;
            }
        }
        EXPECT_THROW(roundtrip(eng, *prog, cfg), ov::Exception) << "frozen binary_add over a sum impl";
    }
    {
        // a frozen sum that the graph no longer supports (live predicate rejects it) although the impl is a sum
        int32_t dep = -1;
        auto prog = build_staged(eng, g, [&](program& p) { dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx; });
        prog->get_node("X").set_output(true);
        ASSERT_TRUE(fusing::frozen_sum_downgrade_reason(prog->get_node("W"), dep) != reason_t::none);
        EXPECT_THROW(roundtrip(eng, *prog, cfg), ov::Exception) << "frozen sum that the live graph rejects";
    }
}

// =============================================================================================================================
// T7. [mul, sum] / [sum, mul] cascades (B3: a live sum behind a frozen in-place mul is no pool-exempt writer).
// =============================================================================================================================
namespace {
f5_graph build_cascade(engine& eng, bool mul_first) {
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in4", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{kShape, data_types::f16, format::bfyx}},
                {"in3", layout{kShape, data_types::f16, format::bfyx}},
                {"in4", layout{kShape, data_types::f16, format::bfyx}}};
    t.add(data("wA", make_fc_weights(eng, 16, 16, 1)));
    t.add(fully_connected("A", input_info("in3"), "wA"));
    t.add(data("wB", make_fc_weights(eng, 16, 16, 3)));
    t.add(fully_connected("B", input_info("in4"), "wB"));
    t.add(data("wW", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    if (mul_first) {
        t.add(eltwise("P", {input_info("W"), input_info("A")}, eltwise_mode::prod));
        t.add(eltwise("S", {input_info("P"), input_info("B")}, eltwise_mode::sum));
    } else {
        t.add(eltwise("S", {input_info("W"), input_info("A")}, eltwise_mode::sum));
        t.add(eltwise("P", {input_info("S"), input_info("B")}, eltwise_mode::prod));
    }
    t.add(data("wZ", make_fc_weights(eng, 16, 16, 2)));
    t.add(fully_connected("Z", input_info(mul_first ? "S" : "P"), "wZ"));
    t.add(reorder("out_Z", input_info("Z"), format::bfyx, data_types::f32));
    g.forced.emplace("A", impl_desc(impl_types::ocl));
    g.forced.emplace("B", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("Z", impl_desc(impl_types::ocl));
    g.outs = {"out_Z"};
    return g;
}
}  // namespace

TEST(inplace_alias_conflict_f5, T7a_mul_first_then_live_sum_is_not_a_pool_exempt_writer) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_cascade(eng, /*mul_first=*/true);
    int32_t mul_dep = -1, sum_dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        const auto* m = fused_op_of_mode(W, eltwise_mode::prod);
        const auto* s = fused_op_of_mode(W, eltwise_mode::sum);
        ASSERT_NE(m, nullptr) << "the mul did not fuse: topology failure\n" << dump_flags(p);
        ASSERT_NE(s, nullptr) << "the sum did not fuse: topology failure\n" << dump_flags(p);
        mul_dep = m->outer_dep_start_idx;
        sum_dep = s->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, mul_dep, eltwise_mode::prod)) << "the first in-place candidate takes the slot";
        ASSERT_FALSE(fusing::is_frozen_inplace_post_op(W, sum_dep, eltwise_mode::sum)) << "the sum is a frozen binary_add";
        ASSERT_TRUE(fusing::get_add_fusing_type(W, *s) == add_fusing_type::sum) << "vacuity: the live predicate says sum";
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, "T7a", QueueTypes::in_order, {}, &net);
    auto& W = prog->get_node("W");
    // B3: the live sum behind the frozen mul must not take W (and the sum addend's owner) out of the pool
    EXPECT_TRUE(W.can_share_buffer()) << dump_flags(*prog);
    EXPECT_TRUE(prog->get_node("B").can_share_buffer()) << dump_flags(*prog);
    EXPECT_FALSE(impl_has_sum_at(W, sum_dep));
    expect_inplace_consistent(*prog, net.get(), "W", sum_dep, false);
}

TEST(inplace_alias_conflict_f5, T7b_sum_first_then_mul_stays_consistent) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_cascade(eng, /*mul_first=*/false);
    int32_t sum_dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        const auto* s = fused_op_of_mode(W, eltwise_mode::sum);
        ASSERT_NE(s, nullptr) << dump_flags(p);
        sum_dep = s->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, sum_dep, eltwise_mode::sum));
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, "T7b", QueueTypes::in_order, {}, &net);
    auto& W = prog->get_node("W");
    EXPECT_EQ(count_inplace_entries(W), 1u);
    expect_inplace_consistent(*prog, net.get(), "W", sum_dep, true, /*check_can_share=*/false);
}

// =============================================================================================================================
// T9. preferred impl is oneDNN but the selected impl is an OCL one: the OCL kernel does not know the oneDNN sum, no alias.
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T9_preferred_onednn_selected_ocl) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    int32_t dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        dep = fused_sum_op(W)->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum));
        auto params = W.get_kernel_impl_params();
        std::unique_ptr<primitive_impl> ocl_impl;
        for (const auto& m : W.type()->get_all_implementations()) {
            if (m->get_impl_type() == impl_types::ocl) {
                ocl_impl = m->create(W, *params);
                break;
            }
        }
        ASSERT_NE(ocl_impl, nullptr) << "no OCL fully_connected impl for the node";
        W.set_selected_impl(std::move(ocl_impl));
        auto new_params = W.get_kernel_impl_params();
        p.get_kernels_cache().add_kernels_source(*new_params, W.get_selected_impl()->get_kernels_source());
        ASSERT_EQ(W.get_preferred_impl_type(), impl_types::onednn);
        ASSERT_FALSE(W.get_selected_impl()->is_onednn());
    };
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, hook, "T9", QueueTypes::in_order, {}, &net);
    auto& W = prog->get_node("W");
    EXPECT_EQ(count_inplace_entries(W), 0u);
    EXPECT_FALSE(W.get_selected_impl()->is_onednn()) << "an OCL impl is not rebuilt as oneDNN";
    EXPECT_EQ(fusing::get_reused_eltwmem_idx(W), -1);
    EXPECT_FALSE(net->get_primitive("W")->reused_eltw_alias_allowed(dep));
    expect_inplace_consistent(*prog, net.get(), "W", dep, false);
}

// =============================================================================================================================
// T9b (B1). A node whose preferred impl is oneDNN but whose selected impl is not (OCL here, same construction as T9; the lazy
// get_kernel_impl_params() freeze gives such nodes a frozen sum too) must only have the frozen entry relabelled by the F5 pass: building a
// oneDNN md for an addend dtype oneDNN cannot convert (i64) throws. Pre-fix the pass went through downgrade_onednn_sum_to_binary_add ->
// make_onednn_binary_post_op -> convert_data_type and threw. An i64 addend does not fuse as a sum in a real graph (an OCL-forced FC does
// not freeze a sum at all), so the addend layout dtype is patched in the hook (white box) and the network is not executed with it;
// T9 / T9c cover execution with a correct output for the same state with a convertible dtype.
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T9b_non_onednn_impl_frozen_sum_unconvertible_dtype_does_not_throw) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    bool had_frozen_sum = false;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        for (const auto& e : W.get_fused_primitives_onednn()) {
            had_frozen_sum |= (e.op_type == onednn_post_op_type::sum || e.op_type == onednn_post_op_type::optimized_sum);
        }
        ASSERT_TRUE(had_frozen_sum) << "vacuity: " << dump_flags(p);
        auto params = W.get_kernel_impl_params();
        std::unique_ptr<primitive_impl> ocl_impl;
        for (const auto& m : W.type()->get_all_implementations()) {
            if (m->get_impl_type() == impl_types::ocl) {
                ocl_impl = m->create(W, *params);
                break;
            }
        }
        ASSERT_NE(ocl_impl, nullptr);
        W.set_selected_impl(std::move(ocl_impl));
        auto new_params = W.get_kernel_impl_params();
        p.get_kernels_cache().add_kernels_source(*new_params, W.get_selected_impl()->get_kernels_source());
        auto& X = p.get_node("X");
        auto l = X.get_output_layout();
        l.data_type = data_types::i64;
        X.set_output_layout(l, false);
    };
    program::ptr prog;
    ASSERT_NO_THROW(prog = build_staged(eng, g, hook));
    auto& W = prog->get_node("W");
    ASSERT_FALSE(W.get_selected_impl()->is_onednn());
    EXPECT_EQ(count_inplace_entries(W), 0u) << "no frozen sum may survive on a non-oneDNN node";
}

TEST(inplace_alias_conflict_f5, T9c_ocl_forced_node_runs_correctly) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    g.forced["W"] = impl_desc(impl_types::ocl);
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, nullptr, "T9c", QueueTypes::in_order, {}, &net);
    EXPECT_EQ(count_inplace_entries(prog->get_node("W")), 0u);
}

// =============================================================================================================================
// T10. Out-of-order queue variant (the pass is queue independent) and the plain control group.
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T10a_control_group_in_order) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    int32_t dep = -1;
    std::shared_ptr<network> net;
    auto prog = run_and_compare(eng, g, [&](program& p) { dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx; }, "T10a",
                                QueueTypes::in_order, {}, &net);
    expect_inplace_consistent(*prog, net.get(), "W", dep, true);
}

TEST(inplace_alias_conflict_f5, T10b_out_of_order_control_and_downgrade) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    int32_t dep = -1;
    {
        std::shared_ptr<network> net;
        auto prog = run_and_compare(eng, g, [&](program& p) { dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx; }, "T10b_ctl",
                                    QueueTypes::out_of_order, {}, &net);
        expect_inplace_consistent(*prog, net.get(), "W", dep, true);
    }
    {
        std::shared_ptr<network> net;
        auto prog = run_and_compare(eng, g, [&](program& p) {
            dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx;
            p.get_node("X").set_output(true);
        }, "T10b_downgrade", QueueTypes::out_of_order, {"X"}, &net);
        expect_inplace_consistent(*prog, net.get(), "W", dep, false);
    }
}

// =============================================================================================================================
// T11. F21 characterization (DISABLED): compile_graph lazily initializes the attributes of a dynamic oneDNN node as empty; if
//      propagate_constants later makes the node static and the impl is re-selected, the static oneDNN impl would miss every post-op.
//      Not reproduced yet: the body states the property that should hold once F21 is fixed.
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, DISABLED_T11_F21_dynamic_node_turned_static_keeps_post_ops) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    auto prog = build_staged(eng, g, nullptr);
    auto& W = prog->get_node("W");
    const auto kinds = attr_kinds(W);
    EXPECT_EQ(kinds.size(), get_compiled_post_op_slots(W.get_fused_primitives_onednn(), 0, {}).kinds.size());
}

// =============================================================================================================================
// T12. A static sum in a shape_of subgraph is downgraded (such nodes recompute their attributes at runtime).
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T12_static_sum_in_shape_of_subgraph) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    int32_t dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        dep = fused_sum_op(W)->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum));
        W.set_in_shape_of_subgraph(true);
        ASSERT_TRUE(fusing::frozen_sum_downgrade_reason(W, dep) == reason_t::in_shape_of_subgraph);
    };
    // The shape_of marker is injected by hand (no real shape_of consumer exists), so the network is constructed but not executed: the
    // runtime of a shape_of subgraph node expects a dependant shape_of instance.
    auto prog = build_staged(eng, g, hook);
    network net(prog, 0);
    EXPECT_EQ(count_inplace_entries(prog->get_node("W")), 0u);
    expect_inplace_consistent(*prog, &net, "W", dep, false, /*check_can_share=*/false);
}

// =============================================================================================================================
// T13. Convolution (weights format any in the pd): the downgrade keeps the weights md the weights dependency was reordered to.
//      Blocked fsv16 (a plain bfyx conv does not freeze a sum: the oneDNN conv output is blocked), also through a cache round trip (the binary post-op md is rebuilt from the list entry).
// =============================================================================================================================
namespace {
f5_graph build_conv(engine& eng, format fmt) {
    f5_graph g;
    auto& t = g.t;
    const ov::PartialShape in_shape{1, 16, 8, 8};
    t.add(input_layout("in1", layout{in_shape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{in_shape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{in_shape, data_types::f16, format::bfyx}}, {"in3", layout{in_shape, data_types::f16, format::bfyx}}};
    // X is not a convolution: a convolution peer is treated as the LoRA pattern by prepare_primitive_fusing (fusion into the peer).
    t.add(activation("X", input_info("in3"), activation_func::abs));
    t.add(data("wW", make_weights(eng, ov::PartialShape{16, 16, 1, 1}, 0)));
    t.add(convolution("W", input_info("in1"), "wW", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
    t.add(reorder("out", input_info("S"), format::bfyx, data_types::f32));
    g.forced.emplace("X", impl_desc(impl_types::ocl, fmt));
    g.forced.emplace("W", impl_desc(impl_types::onednn, fmt));
    g.outs = {"out"};
    g.new_shape_infer = false;
    return g;
}

void run_t13(format fmt, bool roundtrip_cache) {
    auto& eng = get_test_engine();
    auto g = build_conv(eng, fmt);
    auto cfg = opt_config(eng, g.forced, g.custom_outputs, QueueTypes::in_order, g.new_shape_infer);
    int32_t dep = -1;
    std::shared_ptr<dnnl::memory::desc> md_before;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        ASSERT_NE(fused_sum_op(W), nullptr) << "the sum did not fuse into the conv\n" << dump_flags(p);
        dep = fused_sum_op(W)->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum)) << dump_flags(p);
        auto wrp = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(W.get_selected_impl()->get_weights_reorder_params());
        if (wrp) {
            md_before = std::make_shared<dnnl::memory::desc>(*wrp->_out_desc);
        }
        p.get_node("X").set_output(true);
    };
    auto prog = build_staged(eng, g, hook);
    auto& W = prog->get_node("W");
    EXPECT_EQ(count_inplace_entries(W), 0u);
    auto wrp = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(W.get_selected_impl()->get_weights_reorder_params());
    if (md_before) {
        ASSERT_NE(wrp, nullptr);
        EXPECT_TRUE(*wrp->_out_desc == *md_before) << "the rebuilt impl expects another weights md";
    }
    const auto expected = W.make_onednn_binary_post_op(W.get_input_layout(dep), W.get_output_layout());
    for (const auto& e : W.get_fused_primitives_onednn()) {
        if (e.op_type == onednn_post_op_type::binary_add && e.mem_dep == static_cast<size_t>(dep)) {
            EXPECT_EQ(e.dims, expected.dims);
            EXPECT_EQ(e.tag, expected.tag);
            EXPECT_EQ(e.dt, expected.dt);
            EXPECT_FALSE(e.flatten);
        }
    }
    program::ptr run_prog = prog;
    if (roundtrip_cache) {
        ASSERT_NO_THROW(run_prog = roundtrip(eng, *prog, cfg));
        expect_same_frozen_lists(*prog, *run_prog);
    }
    network net(run_prog);
    expect_inplace_consistent(*run_prog, &net, "W", dep, false, /*check_can_share=*/false);
    auto res = run_net(net, eng, g);
    expect_match(res, run_reference(eng, g), roundtrip_cache ? "T13_cache" : "T13");
}
}  // namespace

TEST(inplace_alias_conflict_f5, T13a_conv_downgrade_keeps_weights_md_fsv16) {
    F5_SKIP_IF_NO_IMMAD();
    run_t13(format::b_fs_yx_fsv16, false);
}

TEST(inplace_alias_conflict_f5, T13b_conv_downgrade_cache_roundtrip_fsv16_blocked_md) {
    F5_SKIP_IF_NO_IMMAD();
    run_t13(format::b_fs_yx_fsv16, true);
}

// =============================================================================================================================
// T14. Construction-time assertion (I-F5-3, Release): an impl compiled with an in-place sum whose dst is not bound to the addend.
// =============================================================================================================================
TEST(inplace_alias_conflict_f5, T14_construction_asserts_unbound_sum_dst) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_base(eng);
    int32_t dep = -1;
    auto prog = build_staged(eng, g, [&](program& p) { dep = fused_sum_op(p.get_node("W"))->outer_dep_start_idx; });
    ASSERT_TRUE(impl_has_sum_at(prog->get_node("W"), dep));
    // control: consistent state constructs
    ASSERT_NO_THROW(network(prog, 0));
    // the live graph stops supporting the sum after the single re-freeze point (what F5 must make impossible): no alias, but the impl sums
    prog->get_node("X").set_output(true);
    ASSERT_EQ(fusing::get_reused_eltwmem_idx(prog->get_node("W")), -1);
    try {
        network net(prog, 0);
        FAIL() << "the network must not be constructed over an in-place sum impl with an unbound dst";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("F5"), std::string::npos) << e.what();
    }
}

// =============================================================================================================================
// T14b / reason 6. The writer is a slot of an in-place concat (do_allocate_memory returns false: no alias block exists at all).
// =============================================================================================================================
namespace {
f5_graph build_concat_slot(engine& eng) {
    f5_graph g;
    auto& t = g.t;
    t.add(input_layout("in1", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in3", layout{kShape, data_types::f16, format::bfyx}));
    t.add(input_layout("in4", layout{kShape, data_types::f16, format::bfyx}));
    g.inputs = {{"in1", layout{kShape, data_types::f16, format::bfyx}},
                {"in3", layout{kShape, data_types::f16, format::bfyx}},
                {"in4", layout{kShape, data_types::f16, format::bfyx}}};
    t.add(data("wX", make_fc_weights(eng, 16, 16, 1)));
    t.add(fully_connected("X", input_info("in3"), "wX"));
    t.add(data("wW", make_fc_weights(eng, 16, 16, 0)));
    t.add(fully_connected("W", input_info("in1"), "wW"));
    t.add(eltwise("S", {input_info("W"), input_info("X")}, eltwise_mode::sum));
    t.add(data("wO", make_fc_weights(eng, 16, 16, 4)));
    t.add(fully_connected("O", input_info("in4"), "wO"));
    t.add(concatenation("C", {input_info("S"), input_info("O")}, 0));
    t.add(reorder("out_C", input_info("C"), format::bfyx, data_types::f32));
    g.forced.emplace("X", impl_desc(impl_types::ocl));
    g.forced.emplace("W", impl_desc(impl_types::onednn));
    g.forced.emplace("O", impl_desc(impl_types::ocl));
    g.forced.emplace("C", impl_desc(impl_types::ocl));
    g.outs = {"out_C"};
    return g;
}
}  // namespace

TEST(inplace_alias_conflict_f5, T14b_writer_is_concat_slot_downgraded_reason6) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_concat_slot(eng);
    int32_t dep = -1;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        ASSERT_NE(fused_sum_op(W), nullptr) << dump_flags(p);
        dep = fused_sum_op(W)->outer_dep_start_idx;
        ASSERT_TRUE(fusing::is_frozen_inplace_post_op(W, dep, eltwise_mode::sum)) << "vacuity: " << dump_flags(p);
        // the concat becomes an in-place concat after the freeze (prepare_buffer_fusing refuses it before, because W may sum)
        p.get_node("C").can_be_optimized(true);
        ASSERT_TRUE(fusing::frozen_sum_downgrade_reason(W, dep) == reason_t::writer_is_concat_slot_or_visible ||
                    fusing::frozen_sum_downgrade_reason(W, dep) == reason_t::live_not_sum);
    };
    auto prog = build_staged(eng, g, hook);
    auto& W = prog->get_node("W");
    EXPECT_EQ(count_inplace_entries(W), 0u) << "no frozen sum may survive on a concat slot";
    EXPECT_FALSE(impl_has_sum_at(W, dep));
    // the writer skips allocation (single user is an optimized concat); the assertion has to hold on this path as well
    ASSERT_NO_THROW(network(prog, 0));
    expect_inplace_consistent(*prog, nullptr, "W", dep, false, /*check_can_share=*/false);

    // Step 6 #5: this is also the "frozen sum on a node whose allocate_memory is false (can_be_optimized)" case for the Release ctor
    // assertion in primitive_inst: the writer skips allocation, F5 downgrades its sum (reason 6), and construction above did not throw.
    // The converse (a frozen sum that survives on a non-allocating node) cannot be built on purpose: reason 6 downgrades it, which is
    // the invariant under test, so there is nothing left for the assertion to fire on wrongly.
}

// Reason 6 in isolation: the live predicate still says sum, only the concat-slot rule applies.
TEST(inplace_alias_conflict_f5, T14c_reason6_isolated_writer_concat_slot) {
    F5_SKIP_IF_NO_IMMAD();
    auto& eng = get_test_engine();
    auto g = build_concat_slot(eng);
    int32_t dep = -1;
    reason_t seen = reason_t::none;
    auto hook = [&](program& p) {
        auto& W = p.get_node("W");
        dep = fused_sum_op(W)->outer_dep_start_idx;
        p.get_node("C").can_be_optimized(true);
        seen = fusing::frozen_sum_downgrade_reason(W, dep);
    };
    auto prog = build_staged(eng, g, hook);
    EXPECT_TRUE(seen == reason_t::writer_is_concat_slot_or_visible) << "reason " << static_cast<int>(seen);
    EXPECT_EQ(count_inplace_entries(prog->get_node("W")), 0u);
}

// =============================================================================================================================
// T15. The addend is not output port 0 of its producer: the sum dst cannot be bound to a port > 0 (reason 4).
//      Needs a multi-output primitive whose port-1 buffer has the writer's byte size; see the end of the file.
// =============================================================================================================================
// Not constructed here: building a sum whose addend is a port >0 output needs a multi-output primitive (topk / unique) with a matching
// byte size, which prepare_primitive_fusing does not fuse as an outer dependency on the available primitives. The decision function
// covers it (frozen_sum_downgrade_reason, reason 4) and the Debug assertion in primitive_inst::rebind_reused_eltw_output makes the case
// loud if it ever becomes reachable.

#endif  // ENABLE_ONEDNN_FOR_GPU
