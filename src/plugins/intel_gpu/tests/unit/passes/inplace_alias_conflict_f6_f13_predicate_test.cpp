// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Helper-level tests of the F6/F13 production helpers. They need NO oneDNN device and never skip: is_or_becomes_output() and
// is_external_buffer_root() are pure functions of node flags, users and node types (no fused sum is required).
//
// The predicate tests that need a fused eltwise sum (A3 before/after basic_memory_dependencies, view through a static
// strided_slice, the DISABLED A4 case) live in inplace_alias_conflict_f6_f13_test.cpp: prepare_primitive_fusing only fuses
// a sum with an outer dependency when oneDNN impls are enabled, so they cannot be built on a device without supports_immad.

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

#include "program_node.h"
#include "program_helpers.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <iostream>
#include <sstream>

using namespace cldnn;
using namespace ::tests;

namespace {

const ov::PartialShape kShape{1, 16, 32, 32};

std::string dump_flags(program& prog) {
    std::ostringstream os;
    for (auto* n : prog.get_processing_order()) {
        os << "[FLAGS] #" << prog.get_processing_order().get_processing_number(n) << " " << n->id()
           << " type=" << n->get_primitive()->type_string() << " is_output=" << n->is_output()
           << " users=" << n->get_users().size() << "\n";
    }
    return os.str();
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, float base) {
    layout l{shape, data_types::f16, format::bfyx};
    auto mem = eng.allocate_memory(l);
    std::vector<ov::float16> v(l.count());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = ov::float16(base + static_cast<float>(i % 7) * 0.03125f);
    }
    set_values(mem, v);
    return mem;
}

}  // namespace

// No immad and no optimizations needed: the helper only reads flags and users.
TEST(inplace_alias_conflict_f6_f13, F6_md_output_precompute) {
    auto& engine = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{kShape, data_types::f16, format::bfyx}));
    t.add(activation("D", input_info("in"), activation_func::relu));
    t.add(activation("X", input_info("D"), activation_func::abs));
    t.add(mutable_data("md", {input_info("D")}, make_f16(engine, kShape, 0.f)));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"X", "md"}));
    auto prog = program::build_program(engine, t, config, false, /*no_optimizations=*/true);
    auto& D = prog->get_node("D");
    auto& X = prog->get_node("X");
    auto& md = prog->get_node("md");
    ASSERT_TRUE(md.is_output()) << dump_flags(*prog);
    ASSERT_TRUE(X.is_output());
    EXPECT_FALSE(D.is_output());
    EXPECT_TRUE(program_helpers::is_or_becomes_output(D));
    EXPECT_TRUE(program_helpers::is_or_becomes_output(X));   // already an output
    EXPECT_TRUE(program_helpers::is_or_becomes_output(md));  // already an output
    EXPECT_FALSE(program_helpers::is_or_becomes_output(prog->get_node("in")));

    program_wrapper::apply_opt_pass<basic_memory_dependencies>(*prog);
    EXPECT_TRUE(D.is_output()) << "the precomputation must agree with what basic_memory_dependencies propagates";
}

// md -> md chain: the helper is recursive over mutable_data users.
TEST(inplace_alias_conflict_f6_f13, F6_md_output_precompute_cascaded) {
    auto& engine = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{kShape, data_types::f16, format::bfyx}));
    t.add(activation("D", input_info("in"), activation_func::relu));
    t.add(mutable_data("md1", {input_info("D")}, make_f16(engine, kShape, 0.f)));
    t.add(mutable_data("md2", {input_info("md1")}, make_f16(engine, kShape, 0.f)));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"md2"}));
    auto prog = program::build_program(engine, t, config, false, /*no_optimizations=*/true);
    auto& D = prog->get_node("D");
    auto& md1 = prog->get_node("md1");
    ASSERT_TRUE(prog->get_node("md2").is_output()) << dump_flags(*prog);
    EXPECT_FALSE(D.is_output());
    EXPECT_FALSE(md1.is_output());
    EXPECT_TRUE(program_helpers::is_or_becomes_output(md1));
    EXPECT_TRUE(program_helpers::is_or_becomes_output(D));
}

TEST(inplace_alias_conflict_f6_f13, F6_external_buffer_root_types) {
    auto& engine = get_test_engine();
    topology t;
    t.add(input_layout("in", layout{kShape, data_types::f16, format::bfyx}));
    t.add(data("wd", make_f16(engine, ov::PartialShape{16, 16, 1, 1}, 0.f)));
    t.add(mutable_data("m", make_f16(engine, kShape, 0.f)));
    t.add(convolution("fc", input_info("in"), "wd", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    t.add(eltwise("e", {input_info("fc"), input_info("m")}, eltwise_mode::sum));
    t.add(read_value{"rv", {input_info("in")}, "v0", {layout{kShape, data_types::f16, format::bfyx}}});
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, t, config, false, /*no_optimizations=*/true);
    EXPECT_TRUE(program_helpers::is_external_buffer_root(prog->get_node("in")));
    EXPECT_TRUE(program_helpers::is_external_buffer_root(prog->get_node("wd")));
    EXPECT_TRUE(program_helpers::is_external_buffer_root(prog->get_node("m")));
    EXPECT_TRUE(program_helpers::is_external_buffer_root(prog->get_node("rv")));
    // condition/loop are not asserted here: building one needs branch programs (see F6i in the f6_f13 test file)
    EXPECT_FALSE(program_helpers::is_external_buffer_root(prog->get_node("fc")));
    EXPECT_FALSE(program_helpers::is_external_buffer_root(prog->get_node("e")));
}


// F5 T8: a dynamic node is never a sum (the oneDNN attributes of dynamic nodes are rebuilt per inference from this predicate, so the
// frozen list of a dynamic node is empty and the re-freeze pass has nothing to downgrade). Needs no oneDNN device: the fused eltwise
// descriptor is built by hand, only the predicate is exercised, for both query kinds.
TEST(inplace_alias_conflict_f6_f13, F5_T8_dynamic_node_is_never_sum) {
    auto& engine = get_test_engine();
    const ov::PartialShape dyn_shape{ov::Dimension::dynamic(), 16};
    topology t;
    t.add(input_layout("in", layout{dyn_shape, data_types::f16, format::bfyx}));
    t.add(input_layout("add", layout{dyn_shape, data_types::f16, format::bfyx}));
    t.add(data("w", make_f16(engine, ov::PartialShape{16, 16}, 0.f)));
    t.add(fully_connected("fc", input_info("in"), "w"));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, t, config, false, /*no_optimizations=*/true);
    auto& fc = prog->get_node("fc");
    ASSERT_TRUE(fc.is_dynamic());

    auto make_desc = [](int32_t outer_dep_idx) {
        fused_primitive_desc fd(std::make_shared<eltwise>("e", std::vector<input_info>{input_info("fc"), input_info("add")}, eltwise_mode::sum));
        fd.deps = {{"add", 0}};
        fd.outer_dep_start_idx = outer_dep_idx;
        return fd;
    };
    // outer dependency 1 is the (static) weights, 0 is the dynamic input: both cases must be rejected because the writer is dynamic
    for (int32_t idx : {0, 1}) {
        const auto fd = make_desc(idx);
        EXPECT_EQ(onednn_eltwise_fusing_helpers::get_add_fusing_type(fc, fd, onednn_eltwise_fusing_helpers::add_fusing_query::exact),
                  add_fusing_type::not_supported)
            << "dep " << idx;
        EXPECT_EQ(onednn_eltwise_fusing_helpers::get_add_fusing_type(fc, fd, onednn_eltwise_fusing_helpers::add_fusing_query::guard_superset),
                  add_fusing_type::not_supported)
            << "dep " << idx;
    }
    // the shared decision function: a dynamic node has no frozen sum, nothing can be reported as "none" for a frozen entry
    EXPECT_FALSE(onednn_eltwise_fusing_helpers::is_frozen_inplace_post_op(fc, 0, eltwise_mode::sum));
}
