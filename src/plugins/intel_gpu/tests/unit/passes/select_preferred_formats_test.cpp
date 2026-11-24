// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "convolution_inst.h"
#include "data_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/reduce.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "layout_optimizer.h"
#include "pass_manager.h"
#include "permute_inst.h"
#include "primitive_inst.h"
#include "program_wrapper.h"
#include "test_utils.h"
#include "to_string_utils.h"

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

TEST(test_select_preferred_formats, conv_reshape_reorder_planar_output) {
    // optimize_conv_reshape_reorder: a rank-changing ov::Reshape is lowered to
    // "reorder(planar) -> reshape" at plugin build time, so the reorder is a full copy of the
    // convolution output. Letting the oneDNN convolution emit that planar format itself makes
    // the reorder a buffer reinterpret that remove_redundant_reorders can drop.
    //
    //   conv [1,32,16,16] -> flatten_reorder (bfyx [1,32,256,1]) -> flatten (reshape) -> out
    //
    // The negative case drops the reshape so the pattern must not trigger, and the convolution
    // keeps the forced blocked output format. Like optimize_conv_permute, the pattern
    // optimization takes precedence over an explicitly forced output format.
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto build = [&engine](bool with_reshape) {
        auto input = engine.allocate_memory({data_types::f16, format::bfyx, {1, 32, 16, 16}});
        auto weights = engine.allocate_memory({data_types::f16, format::bfyx, {32, 32, 1, 1}});

        // Same element count as the convolution output, flattened over the spatial axes.
        const auto flat_layout = layout{data_types::f16, format::bfyx, tensor{1, 32, 1, 256}};

        topology topo;
        topo.add(data("weights", weights));
        topo.add(input_layout("input", input->get_layout()));
        topo.add(convolution("conv1", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topo.add(reorder("flatten_reorder", input_info("conv1"), flat_layout));
        if (with_reshape) {
            topo.add(reshape("flatten", input_info("flatten_reorder"), tensor{1, 32, 1, 256}));
            topo.add(reorder("out", input_info("flatten"), format::bfyx, data_types::f32));
        } else {
            topo.add(reorder("out", input_info("flatten_reorder"), format::bfyx, data_types::f32));
        }

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        ov::intel_gpu::ImplementationDesc impl = {format::b_fs_yx_fsv16, std::string(""), impl_types::onednn};
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv1", impl}}));

        auto prog = program::build_program(engine, topo, config, false, true);
        prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

        // Initializes output_layout, necessary because this test runs the pass alone.
        prog->get_node("conv1").get_output_layouts(false);
        prog->get_node("flatten_reorder").get_output_layouts(false);
        program_wrapper::apply_opt_pass<select_preferred_formats>(*prog);
        return prog;
    };

    ASSERT_EQ(build(true)->get_node("conv1").get_preferred_output_fmt(0), format::bfyx);
    ASSERT_EQ(build(false)->get_node("conv1").get_preferred_output_fmt(0), format::b_fs_yx_fsv16);
}

TEST(test_select_preferred_formats, dynamic_compressed_fc_keeps_fbyx_scale_and_zp) {
    auto& engine = get_test_engine();

    const int64_t ofm = 7680;
    const int64_t ifm = 5120;
    const int64_t groups = 80;

    auto weights_mem = engine.allocate_memory({ov::PartialShape{ofm, ifm}, data_types::u2, format::bfyx});
    auto scale_mem = engine.allocate_memory({ov::PartialShape{ofm, groups}, data_types::f16, format::bfyx});
    auto zp_mem = engine.allocate_memory({ov::PartialShape{ofm, groups, 1}, data_types::u8, format::bfyx});

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape{1, -1, ifm}, data_types::f16, format::bfyx}));
    topology.add(data("weights", weights_mem));
    topology.add(data("scale", scale_mem));
    topology.add(data("zp", zp_mem));
    topology.add(reorder("scale_reorder", input_info("scale"),
                         layout{ov::PartialShape{ofm, groups}, data_types::f16, format::fbyx}));
    topology.add(reshape("zp_reshape", input_info("zp"), false, {ofm, groups}, ov::PartialShape{ofm, groups}));
    topology.add(reorder("zp_reorder", input_info("zp_reshape"),
                         layout{ov::PartialShape{ofm, groups}, data_types::u8, format::fbyx}));
    topology.add(fully_connected("fc", input_info("input"), "weights", "", "scale_reorder", "zp_reorder",
                                 data_types::f16, 3, 2));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    // oneDNN FC is static-shape only, so forcing it on a dynamic FC leaves no factory.
    ov::intel_gpu::ImplementationDesc fc_impl = {format::any, "", impl_types::onednn};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", fc_impl}}));

    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    program_wrapper::apply_opt_pass<select_preferred_formats>(*prog);

    auto& fc_node = prog->get_node("fc");
    ASSERT_TRUE(fc_node.is_dynamic());
    ASSERT_EQ(fc_node.get_dependency(2).id(), "scale_reorder");
    ASSERT_EQ(fc_node.get_dependency(3).id(), "zp_reorder");
    ASSERT_EQ(fc_node.get_preferred_input_fmt(2), format::fbyx);
    ASSERT_EQ(fc_node.get_preferred_input_fmt(3), format::fbyx);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(test_select_preferred_formats, onednn_dyn_conv_fp16_preferred_format) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN convolutions are only used on devices with immad support";
    }

    struct ConvFmtCase {
        int64_t ic;
        int64_t oc;
        format::type expected_in_fmt;
        format::type expected_out_fmt;
    };

    const std::vector<ConvFmtCase> cases = {
        {3, 32, format::bfyx, format::b_fs_yx_fsv16},
        {8, 32, format::bfyx, format::b_fs_yx_fsv16},
        {8, 64, format::bfyx, format::b_fs_yx_fsv16},
        {9, 32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16},
        {32, 16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16},
        {16, 16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16},
        {16, 96, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16},
        {32, 8, format::b_fs_yx_fsv16, format::bfyx},
        {32, 9, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16},
        {32, 4, format::b_fs_yx_fsv16, format::bfyx},
        {4, 16, format::bfyx, format::b_fs_yx_fsv16},
    };

    for (const auto& tc : cases) {
        auto weights = engine.allocate_memory({ov::PartialShape{tc.oc, tc.ic, 3, 3}, data_types::f16, format::bfyx});
        set_values(weights, std::vector<ov::float16>(weights->count(), ov::float16(0.01F)));
        topology topo;
        topo.add(input_layout("input", layout{ov::PartialShape{1, tc.ic, -1, -1}, data_types::f16, format::bfyx}));
        topo.add(data("weights", weights));
        topo.add(convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
        topo.add(reorder("out", input_info("conv"), format::bfyx, data_types::f32));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network net(engine, topo, config);
        ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);

        const auto& conv_node = net.get_primitive("conv")->get_node();
        ASSERT_TRUE(conv_node.is_dynamic());
        ASSERT_EQ(conv_node.get_preferred_impl_type(), impl_types::onednn);
        EXPECT_EQ(conv_node.get_preferred_input_fmt(0), tc.expected_in_fmt) << "ic=" << tc.ic << ", oc=" << tc.oc;
        EXPECT_EQ(conv_node.get_preferred_output_fmt(0), tc.expected_out_fmt) << "ic=" << tc.ic << ", oc=" << tc.oc;
        EXPECT_EQ(conv_node.get_input_layout(0).format, tc.expected_in_fmt) << "ic=" << tc.ic << ", oc=" << tc.oc;
        EXPECT_EQ(conv_node.get_output_layout(0).format, tc.expected_out_fmt) << "ic=" << tc.ic << ", oc=" << tc.oc;

        auto input_mem = engine.allocate_memory({ov::PartialShape{1, tc.ic, 16, 16}, data_types::f16, format::bfyx});
        set_values(input_mem, std::vector<ov::float16>(input_mem->count(), ov::float16(1.0F)));
        net.set_input_data("input", input_mem);
        auto outputs = net.execute();
        ASSERT_EQ(outputs.count("out"), 1U);

        const auto* conv_impl = net.get_primitive("conv")->get_impl();
        ASSERT_NE(conv_impl, nullptr);
        EXPECT_TRUE(conv_impl->is_onednn()) << "ic=" << tc.ic << ", oc=" << tc.oc;
        EXPECT_EQ(conv_impl->get_kernel_name().find("ref"), std::string::npos)
            << "ic=" << tc.ic << ", oc=" << tc.oc << ", kernel=" << conv_impl->get_kernel_name();
    }
}

TEST(test_select_preferred_formats, onednn_dyn_conv_16ch_reduce_mean_uses_onednn) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "oneDNN convolutions and reductions are only used on devices with immad support";
    }

    auto weights1 = engine.allocate_memory({ov::PartialShape{16, 16, 3, 3}, data_types::f16, format::bfyx});
    auto weights2 = engine.allocate_memory({ov::PartialShape{4, 16, 1, 1}, data_types::f16, format::bfyx});
    set_values(weights1, std::vector<ov::float16>(weights1->count(), ov::float16(0.01F)));
    set_values(weights2, std::vector<ov::float16>(weights2->count(), ov::float16(0.01F)));

    // Squeeze-and-excitation block pattern: 16ch conv -> ReduceMean(H, W) -> 1x1 conv (16 -> 4).
    // With 16ch kept in b_fs_yx_fsv16, ReduceMean keeps blocked layout and uses oneDNN reduction at runtime.
    topology topo;
    topo.add(input_layout("input", layout{ov::PartialShape{1, 16, -1, -1}, data_types::f16, format::bfyx}));
    topo.add(data("weights1", weights1));
    topo.add(data("weights2", weights2));
    topo.add(convolution("conv1", input_info("input"), "weights1", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    topo.add(reduce("reduce_mean", input_info("conv1"), reduce_mode::mean, {2, 3}, true));
    topo.add(convolution("conv2", input_info("reduce_mean"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topo.add(reorder("out", input_info("conv2"), format::bfyx, data_types::f32));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, topo, config);
    const auto& conv1_node = net.get_primitive("conv1")->get_node();
    const auto& reduce_node = net.get_primitive("reduce_mean")->get_node();
    ASSERT_EQ(conv1_node.get_output_layout(0).format, format::b_fs_yx_fsv16);
    ASSERT_EQ(reduce_node.get_input_layout(0).format, format::b_fs_yx_fsv16);
    ASSERT_EQ(reduce_node.get_output_layout(0).format, format::b_fs_yx_fsv16);

    auto input_mem = engine.allocate_memory({ov::PartialShape{1, 16, 32, 32}, data_types::f16, format::bfyx});
    set_values(input_mem, std::vector<ov::float16>(input_mem->count(), ov::float16(1.0F)));
    net.set_input_data("input", input_mem);
    auto outputs = net.execute();
    ASSERT_EQ(outputs.count("out"), 1U);

    const auto* reduce_impl = net.get_primitive("reduce_mean")->get_impl();
    ASSERT_NE(reduce_impl, nullptr);
    EXPECT_TRUE(reduce_impl->is_onednn()) << "Expected oneDNN reduction, got: " << reduce_impl->get_kernel_name();
}
#endif  // ENABLE_ONEDNN_FOR_GPU

TEST(test_select_preferred_formats, get_convolution_channel_count_grouped_dynamic_channels) {
    auto& engine = get_test_engine();

    struct ChannelCase {
        uint32_t groups;
        bool grouped_weights;
        bool allow_new_shape_infer;
        ov::PartialShape input_shape;
        ov::PartialShape weights_shape;
        format::type weights_fmt;
        int64_t expected_ic;
        int64_t expected_oc;
    };

    const std::vector<ChannelCase> cases = {
        {560, true, true, ov::PartialShape{1, -1, -1, -1}, ov::PartialShape{560, 1, 1, 3, 3}, format::bfzyx, 560, 560},
        {4, true, true, ov::PartialShape{1, -1, -1, -1}, ov::PartialShape{4, 8, 6, 3, 3}, format::bfzyx, 24, 32},
        {1, true, true, ov::PartialShape{1, -1, -1, -1}, ov::PartialShape{1, 32, 16, 3, 3}, format::bfzyx, 16, 32},
        {4, false, false, ov::PartialShape{1, 24, 8, 8}, ov::PartialShape{32, 6, 3, 3}, format::bfyx, 24, 32},
        {1, false, true, ov::PartialShape{1, -1, -1, -1}, ov::PartialShape{32, 3, 3, 3}, format::bfyx, 3, 32},
    };

    const layout dyn_ch_layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx};
    for (const auto& tc : cases) {
        auto weights = engine.allocate_memory({tc.weights_shape, data_types::f16, tc.weights_fmt});
        topology topo;
        topo.add(input_layout("input", layout{tc.input_shape, data_types::f16, format::bfyx}));
        topo.add(data("weights", weights));
        topo.add(convolution("conv", input_info("input"), "weights", "", tc.groups, {1, 1}, {1, 1}, {1, 1}, {1, 1}, tc.grouped_weights));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(tc.allow_new_shape_infer));

        auto prog = program::build_program(engine, topo, config, false, true);
        const auto& conv_node = prog->get_node("conv").as<convolution>();
        EXPECT_EQ(get_convolution_channel_count(conv_node, dyn_ch_layout, true), tc.expected_ic)
            << "groups=" << tc.groups << ", grouped_weights=" << tc.grouped_weights;
        EXPECT_EQ(get_convolution_channel_count(conv_node, dyn_ch_layout, false), tc.expected_oc)
            << "groups=" << tc.groups << ", grouped_weights=" << tc.grouped_weights;
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    if (engine.get_device_info().supports_immad) {
        auto weights = engine.allocate_memory({ov::PartialShape{64, 1, 1, 3, 3}, data_types::f16, format::bfzyx});
        topology topo;
        topo.add(input_layout("input", layout{ov::PartialShape{1, -1, -1, -1}, data_types::f16, format::bfyx}));
        topo.add(data("weights", weights));
        topo.add(convolution("conv", input_info("input"), "weights", "", 64, {1, 1}, {1, 1}, {1, 1}, {1, 1}, true));
        topo.add(reorder("out", input_info("conv"), format::bfyx, data_types::f32));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network net(engine, topo, config);
        ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);
        const auto& conv_node = net.get_primitive("conv")->get_node();
        ASSERT_EQ(conv_node.get_preferred_impl_type(), impl_types::onednn);
        EXPECT_EQ(conv_node.get_preferred_input_fmt(0), format::b_fs_yx_fsv16);
        EXPECT_EQ(conv_node.get_preferred_output_fmt(0), format::b_fs_yx_fsv16);
        EXPECT_EQ(conv_node.get_input_layout(0).format, format::b_fs_yx_fsv16);
        EXPECT_EQ(conv_node.get_output_layout(0).format, format::b_fs_yx_fsv16);
    }
#endif  // ENABLE_ONEDNN_FOR_GPU
}