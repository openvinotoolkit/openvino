// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"
#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "convolution_inst.h"
#include "permute_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

TEST(test_select_preferred_formats, setting_target_conv_format) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 32, 64, 64 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 32, 32, 3, 3 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), format::b_fs_yx_fsv16, data_types::f16));
    topology.add(convolution("conv1", input_info("reorder"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::onednn };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv1", impl} }));

    auto prog = program::build_program(engine, topology, config, false, true);
    if (engine.get_device_info().supports_immad) {
        prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();
    }

    // It initializes output_layout.
    // It's necessary because this test runs select_preferred_formats pass alone.
    prog->get_node("conv1").get_output_layouts(false);
    program_wrapper::apply_opt_pass<select_preferred_formats>(*prog);

    ASSERT_NE(prog, nullptr);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<convolution>())
            continue;

        auto& node = node_ptr->as<convolution>();
        auto input_fmt = node.get_preferred_input_fmt(0);
        auto output_fmt = node.get_preferred_output_fmt(0);
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(input_fmt, format::b_fs_yx_fsv16);
            ASSERT_EQ(output_fmt, format::b_fs_yx_fsv16);
        } else {
            ASSERT_EQ(input_fmt, format::any);
            ASSERT_EQ(output_fmt, format::any);
        }
    }
}

TEST(test_select_preferred_formats, fsv2_fallback_to_byxf) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 96, 3002 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 32, 3, 3, 1 } });

    // find_data_format() returns b_fs_yx_fsv2 for convolution input in the below topology.
    // The b_fs_yx_fsv2 should fallback to byxf in current WA.
    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), format::bfyx, data_types::f16));
    topology.add(convolution("conv1", input_info("reorder"), "weights", "", 2, {1, 1}, {1, 1}, {2, 1}, {0, 1}, true));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::onednn };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv1", impl} }));

    auto prog = program::build_program(engine, topology, config, false, true);
    if (engine.get_device_info().supports_immad) {
        prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();
    }

    // It initializes output_layout.
    // It's necessary because this test runs select_preferred_formats pass alone.
    prog->get_node("conv1").get_output_layouts(false);
    program_wrapper::apply_opt_pass<select_preferred_formats>(*prog);

    ASSERT_NE(prog, nullptr);

    auto itr = prog->get_processing_order().begin();
    while (itr != prog->get_processing_order().end()) {
        auto node_ptr = *itr++;
        if (!node_ptr->is_type<convolution>())
            continue;

        auto& node = node_ptr->as<convolution>();
        auto input_fmt = node.get_preferred_input_fmt(0);
        auto output_fmt = node.get_preferred_output_fmt(0);
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(input_fmt, format::byxf);
            ASSERT_EQ(output_fmt, format::b_fs_yx_fsv16);
        } else {
            ASSERT_EQ(input_fmt, format::any);
            ASSERT_EQ(output_fmt, format::any);
        }
    }
}

TEST(test_select_preferred_formats, permute_conv_incompatible_format) {
    // Negative tests for is_compatible_format in optimize_permute_conv:
    //   1) node_fmt = b_fs_yx_fsv16 (not byxf) -> optimization does NOT trigger
    //   2) upstream preferred_output_fmt = incompatible (b_fs_yx_fsv16) -> optimization does NOT trigger
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto build_permute_conv_program = [&engine](const std::vector<uint16_t>& permute_order, tensor input_size, tensor weight_size, format forced_fmt) {
        auto input = engine.allocate_memory({data_types::f16, format::bfyx, input_size});
        auto weights = engine.allocate_memory({data_types::f16, format::bfyx, weight_size});

        topology topo;
        topo.add(data("weights", weights));
        topo.add(input_layout("input", input->get_layout()));
        topo.add(permute("perm", input_info("input"), permute_order));
        topo.add(convolution("conv1", input_info("perm"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topo.add(reorder("output_reorder", input_info("conv1"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        ov::intel_gpu::ImplementationDesc impl = {forced_fmt, std::string(""), impl_types::onednn};
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv1", impl}}));

        auto prog = program::build_program(engine, topo, config, false, true);
        prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();
        prog->get_node("conv1").get_output_layouts(false);
        return prog;
    };

    // Sub-case 1: conv preferred_output_fmt not compatible -> optimization should NOT trigger
    {
        auto prog = build_permute_conv_program({0, 3, 1, 2}, {1, 32, 16, 8}, {32, 16, 3, 3}, format::b_fs_yx_fsv16);
        program_wrapper::apply_opt_pass<select_preferred_formats>(*prog);

        auto& conv_node = prog->get_node("conv1");
        auto& perm_node = prog->get_node("perm");
        ASSERT_NE(perm_node.get_preferred_output_fmt(0), format::byxf);
        ASSERT_NE(conv_node.get_preferred_output_fmt(0), format::byxf);
        ASSERT_FALSE(perm_node.can_be_optimized());
    }

    // Sub-case 2: upstream preferred_output_fmt = incompatible format -> optimization should NOT trigger
    {
        auto prog = build_permute_conv_program({0, 3, 1, 2}, {1, 32, 16, 8}, {32, 16, 3, 3}, format::byxf);
        prog->get_node("input").set_preferred_output_fmt(0, format::b_fs_yx_fsv16);
        program_wrapper::apply_opt_pass<select_preferred_formats>(*prog);

        auto& perm_node = prog->get_node("perm");
        ASSERT_NE(perm_node.get_preferred_output_fmt(0), format::byxf);
        ASSERT_FALSE(perm_node.can_be_optimized());
    }
}

TEST(test_select_preferred_formats, permute_conv_accuracy) {
    // Verifies that optimize_permute_conv produces numerically correct output
    // by comparing optimized path against reference, both using oneDNN conv.
    //
    // opt_topo: input -> permute -> conv(oneDNN, byxf) -> output_reorder
    //           conv has 1 user -> optimize_permute_conv fires
    //
    // ref_topo: same as opt_topo + extra leaf reorder ("ref_tap") from conv
    //           conv has 2 users -> node.get_users().size() != 1
    //           blocks optimize_permute_conv, rest of the pipeline is identical
    //
    // Two sub-cases:
    //   1) Unfused permute: can_be_optimized = true
    //   2) Fused eltwise(prod) into permute: can_be_optimized = false,
    //      but formats still set. Uses position-dependent scale values so any
    //      layout misalignment produces detectably wrong numerical results.
    // IMMAD-only: optimize_permute_conv only runs when oneDNN impl is selected.
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {1, 2, 2, 16}});
    auto weights = engine.allocate_memory({data_types::f16, format::bfyx, {2, 2, 1, 1}});

    const int num_elements = 64;  // 1*2*2*16
    std::vector<ov::float16> input_vals(num_elements);
    for (int i = 0; i < num_elements; ++i)
        input_vals[i] = ov::float16(static_cast<float>(i + 1));
    set_values<ov::float16>(input, input_vals);

    std::vector<ov::float16> weight_vals(2 * 2, ov::float16(0.f));
    for (int i = 0; i < 2; ++i)
        weight_vals[i * 2 + i] = ov::float16(1.f);
    set_values<ov::float16>(weights, weight_vals);

    // Run ref and opt networks (both oneDNN conv), verify numerical match.
    // Returns the opt network's program for state checks.
    auto run_and_verify = [&](const topology& opt_topo, const topology& ref_topo) -> std::shared_ptr<program> {
        ov::intel_gpu::ImplementationDesc conv_impl = {format::byxf, std::string(""), impl_types::onednn};

        ExecutionConfig ref_config = get_test_default_config(engine);
        ref_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        ref_config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv1", conv_impl}}));
        network ref_net(engine, ref_topo, ref_config);
        ref_net.set_input_data("input", input);
        auto ref_outputs = ref_net.execute();
        auto ref_mem = ref_outputs.at("output_reorder").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> ref_ptr(ref_mem, get_test_stream());

        ExecutionConfig opt_config = get_test_default_config(engine);
        opt_config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        opt_config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv1", conv_impl}}));
        network opt_net(engine, opt_topo, opt_config);
        opt_net.set_input_data("input", input);
        auto opt_outputs = opt_net.execute();
        auto opt_mem = opt_outputs.at("output_reorder").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> opt_ptr(opt_mem, get_test_stream());

        EXPECT_EQ(ref_mem->get_layout().count(), opt_mem->get_layout().count());
        for (size_t i = 0; i < ref_mem->get_layout().count(); ++i) {
            EXPECT_NEAR(ref_ptr[i], opt_ptr[i], 1e-3f) << "Mismatch at index " << i;
        }
        return opt_net.get_program();
    };

    // Sub-case 1: unfused permute -> can_be_optimized = true
    {
        topology opt_topo;
        opt_topo.add(data("weights", weights));
        opt_topo.add(input_layout("input", input->get_layout()));
        opt_topo.add(permute("perm", input_info("input"), {0, 3, 1, 2}));
        opt_topo.add(convolution("conv1", input_info("perm"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        opt_topo.add(reorder("output_reorder", input_info("conv1"), format::bfyx, data_types::f32));

        // ref_topo = opt_topo + extra consumer on conv1 to block optimization
        topology ref_topo(opt_topo);
        ref_topo.add(reorder("ref_tap", input_info("conv1"), format::bfyx, data_types::f16));

        auto prog = run_and_verify(opt_topo, ref_topo);
        auto& perm_node = prog->get_node("perm");
        ASSERT_EQ(perm_node.get_preferred_output_fmt(0), format::byxf);
        ASSERT_FALSE(perm_node.has_fused_primitives());
        ASSERT_TRUE(perm_node.can_be_optimized());
    }

    // Sub-case 2: fused eltwise(prod) into permute -> can_be_optimized = false
    {
        auto scale_data = engine.allocate_memory({data_types::f16, format::bfyx, {1, 2, 16, 2}});
        std::vector<ov::float16> scale_vals(num_elements);
        for (int i = 0; i < num_elements; ++i)
            scale_vals[i] = ov::float16(static_cast<float>(i) * 0.1f + 0.5f);
        set_values<ov::float16>(scale_data, scale_vals);

        topology opt_topo;
        opt_topo.add(data("weights", weights));
        opt_topo.add(data("scale_data", scale_data));
        opt_topo.add(input_layout("input", input->get_layout()));
        opt_topo.add(permute("perm", input_info("input"), {0, 3, 1, 2}));
        opt_topo.add(eltwise("scale", {input_info("perm"), input_info("scale_data")}, eltwise_mode::prod));
        opt_topo.add(convolution("conv1", input_info("scale"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        opt_topo.add(reorder("output_reorder", input_info("conv1"), format::bfyx, data_types::f32));

        // ref_topo = opt_topo + extra consumer on conv1 to block optimization
        topology ref_topo(opt_topo);
        ref_topo.add(reorder("ref_tap", input_info("conv1"), format::bfyx, data_types::f16));

        auto prog = run_and_verify(opt_topo, ref_topo);
        auto& perm_node = prog->get_node("perm");
        ASSERT_EQ(perm_node.get_preferred_output_fmt(0), format::byxf);
        ASSERT_TRUE(perm_node.has_fused_primitives());
        ASSERT_FALSE(perm_node.can_be_optimized());
    }
}