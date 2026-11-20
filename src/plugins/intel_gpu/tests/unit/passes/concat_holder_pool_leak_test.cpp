// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P1-B.0 (G1 / F26): stale memory-pool user records ("ghost users").
//
// P1-L1 (G1): a node that obtained a pool block in an earlier iteration and later switches to a direct (non-pool) allocation
//             (dynamic in-place concat holder: rejected -> fused with growth) must drop its pool user record.
// P1-L2 (F26): the forward probe in primitive_inst::realloc_outputs replaces _outputs[0] by an external output block; the previous
//              pool block of the node must not keep the node's user record.
// Design: /home/ceciliapeng/repos/reports/memman/layer1/11_p1_design_r2.zh-CN.md section 1 (P1-B.0).

#include <algorithm>
#include <iostream>
#include <vector>

#include "concatenation_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/plugin/output_memory_block.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
#include "primitive_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

ExecutionConfig leak_cfg(engine& eng) {
    ExecutionConfig config = get_test_default_config(eng);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    return config;
}

size_t pool_records_of(network& net, const primitive_id& id) {
    const auto uid = net.get_primitive(id)->get_node().get_unique_id();
    return net.get_memory_pool().count_records_with_user(uid, net.get_id());
}

std::vector<float> make_vals(const ov::Shape& s, int seed) {
    std::vector<float> v(ov::shape_size(s));
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = static_cast<float>((static_cast<int>(i) * 7 + seed * 13) % 16);  // small integers are exact in f16
    }
    return v;
}

memory::ptr make_mem(engine& eng, const ov::Shape& s, data_types dt, const std::vector<float>& v) {
    auto m = eng.allocate_memory(layout{ov::PartialShape(s), dt, format::bfyx});
    if (!v.empty()) {
        if (dt == data_types::f32) {
            set_values<float>(m, v);
        } else {
            std::vector<ov::float16> h(v.begin(), v.end());
            set_values<ov::float16>(m, h);
        }
    }
    return m;
}

// Reads `count` elements of an output memory (the buffer may be larger than the logical output).
template <class T>
std::vector<float> read_vals(memory::ptr m, size_t count) {
    std::vector<float> r(count);
    if (count == 0) {
        return r;
    }
    mem_lock<T> lock(m, get_test_stream());
    for (size_t i = 0; i < count; i++) {
        r[i] = static_cast<float>(lock[i]);
    }
    return r;
}

}  // namespace

// B0-T1. in1 -> P1 (reorder f16, fsv16) --+
//                                         C (axis 1) -> output (reorder f32, bfyx)
//        in2 -> P2 (reorder f16, fsv16) --+
// Round 1: feature(P1) = 3 is not a multiple of 16, so the runtime match fails: C is not optimized and its holder buffer comes from
// the pool (C is a pool user). Round 2: both features are 16 and the spatial size grows, so C is fused in place and its holder is
// reallocated outside the pool; the stale pool record of C must be gone.
TEST(concat_holder_pool_leak, b0_t1_rejected_then_fused_with_growth) {
    auto& eng = get_test_engine();
    const auto dyn = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology t;
    t.add(input_layout("in1", dyn));
    t.add(input_layout("in2", dyn));
    t.add(reorder("P1", input_info("in1"), format::b_fs_yx_fsv16, data_types::f16));
    t.add(reorder("P2", input_info("in2"), format::b_fs_yx_fsv16, data_types::f16));
    t.add(concatenation("C", {input_info("P1"), input_info("P2")}, 1));
    t.add(reorder("output", input_info("C"), format::bfyx, data_types::f32));

    auto net = get_network(eng, t, leak_cfg(eng), get_test_stream_ptr(), false);
    auto c = net->get_primitive("C");
    ASSERT_TRUE(c->get_node().is_type<concatenation>());
    ASSERT_TRUE(c->get_node().can_be_optimized()) << "build-time in-place concat expected (dynamic runtime fusion candidate)";

    // batch is 1, so the reference of a concat along the feature axis is the plain concatenation of the two inputs
    auto run = [&](const ov::Shape& s1, const ov::Shape& s2, int seed) {
        const auto d1 = make_vals(s1, seed);
        const auto d2 = make_vals(s2, seed + 1);
        net->set_input_data("in1", make_mem(eng, s1, data_types::f32, d1));
        net->set_input_data("in2", make_mem(eng, s2, data_types::f32, d2));
        auto outs = net->execute();
        std::vector<float> ref = d1;
        ref.insert(ref.end(), d2.begin(), d2.end());
        const auto got = read_vals<float>(outs.at("output").get_memory(), net->get_output_layout("output").count());
        ASSERT_EQ(got.size(), ref.size());
        EXPECT_EQ(got, ref) << "concat output differs from the reference for shapes " << s1.to_string() << " / " << s2.to_string();
    };

    ASSERT_NO_FATAL_FAILURE(run({1, 3, 4, 4}, {1, 16, 4, 4}, 1));
    ASSERT_FALSE(c->can_be_optimized()) << "round 1 must reject the runtime in-place concat";
    const auto after_round1 = pool_records_of(*net, "C");
    std::cout << "[B0-T1] pool records of C after rejected round: " << after_round1 << std::endl;
    ASSERT_EQ(after_round1, 1u) << "precondition: the rejected holder must be a pool user";

    ASSERT_NO_FATAL_FAILURE(run({1, 16, 32, 32}, {1, 16, 32, 32}, 2));
    ASSERT_TRUE(c->can_be_optimized()) << "round 2 must fuse the concat in place";
    const auto after_round2 = pool_records_of(*net, "C");
    std::cout << "[B0-T1] pool records of C after fused+grown round: " << after_round2 << std::endl;
    EXPECT_LE(after_round2, 1u);
    EXPECT_EQ(after_round2, 0u) << "ghost pool user record of the in-place holder";
}

// B0-T2. in -> S (permute, runtime skippable) -> output (reorder, optimized out, ext_block registered)
// Round 1 (no ext_block): S executes (not skipped) into a pool block. The block is registered afterwards; network::
// invalidate_ext_block_compute_nodes stops before the runtime-skippable S, so S keeps its pooled buffer. Round 2 grows the shape:
// S reallocates, the forward probe substitutes the ext_block for S's output, and S's user record on the old block must be dropped.
TEST(concat_holder_pool_leak, b0_t2_forward_probe_replaces_pooled_output) {
    auto& eng = get_test_engine();
    if (!eng.get_device_info().supports_usm) {
        GTEST_SKIP() << "USM not supported: OutputMemoryBlock needs usm_host";
    }
    topology t;
    t.add(input_layout("input", layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx}));
    t.add(permute("S", input_info("input"), std::vector<uint16_t>{0, 2, 1, 3}));
    t.add(reorder("output", input_info("S"), format::bfyx, data_types::f16));

    auto net = get_network(eng, t, leak_cfg(eng), get_test_stream_ptr(), false);
    auto s = net->get_primitive("S");
    auto out = net->get_primitive("output");
    ASSERT_TRUE(s->get_node().is_runtime_skippable());
    ASSERT_TRUE(out->can_be_optimized()) << "output reorder must be optimized out";

    // reference: permute order {0, 2, 1, 3} of a bfyx f16 tensor
    auto run = [&](const ov::Shape& sh, int seed) {
        const auto d = make_vals(sh, seed);
        net->set_input_data("input", make_mem(eng, sh, data_types::f16, d));
        auto outs = net->execute();
        std::vector<float> ref(d.size());
        const size_t B = sh[0], F = sh[1], Y = sh[2], X = sh[3];
        for (size_t b = 0; b < B; b++)
            for (size_t f = 0; f < F; f++)
                for (size_t y = 0; y < Y; y++)
                    for (size_t x = 0; x < X; x++)
                        ref[((b * Y + y) * F + f) * X + x] = d[((b * F + f) * Y + y) * X + x];
        const auto got = read_vals<ov::float16>(outs.at("output").get_memory(), net->get_output_layout("output").count());
        ASSERT_EQ(got.size(), ref.size());
        EXPECT_EQ(got, ref) << "permute output differs from the reference for shape " << sh.to_string();
    };

    ASSERT_NO_FATAL_FAILURE(run({12, 3, 2, 14}, 1));
    ASSERT_FALSE(s->can_be_optimized()) << "S must execute (not skipped) in round 1";
    const auto before = pool_records_of(*net, "S");
    std::cout << "[B0-T2] pool records of S before ext_block: " << before << std::endl;
    ASSERT_EQ(before, 1u) << "precondition: S output is pool memory";

    ov::intel_gpu::OutputMemoryBlock block(eng);
    // Declared after `block`, so it is destroyed first (also on an early ASSERT exit): the network must not keep a pointer to a dead block.
    struct block_registration_guard {
        network& n;
        ~block_registration_guard() {
            n.clear_output_memory_blocks();
        }
    } guard{*net};
    net->register_output_memory_block("output", &block);
    ASSERT_NO_FATAL_FAILURE(run({12, 6, 8, 14}, 2));
    ASSERT_NE(block.memory(), nullptr);
    ASSERT_EQ(block.memory()->buffer_ptr(), s->output_memory_ptr()->buffer_ptr()) << "S must write into the ext_block via the forward probe";
    const auto after = pool_records_of(*net, "S");
    std::cout << "[B0-T2] pool records of S after forward probe: " << after << std::endl;
    EXPECT_EQ(after, 0u) << "ghost pool user record of the node whose output was replaced by the ext_block";
}
