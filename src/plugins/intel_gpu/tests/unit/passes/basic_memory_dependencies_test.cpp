// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "convolution_inst.h"
#include "reshape_inst.h"
#include "reorder_inst.h"
#include "resample_inst.h"
#include "program_helpers.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

// Verify that end-to-end inference produces correct results when a fused
// eltwise-sum residual passes through an optimized-out reshape (in-place chain).
// Compares GPU output with memory pool enabled vs disabled.
TEST(basic_memory_dependencies, inplace_chain_eltwise_sum_correctness) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight_layout = layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);

    tests::random_generator rg;
    rg.set_seed("basic_memory_dependencies_correctness");
    {
        auto rnd = rg.generate_random_1d<ov::float16>(weight_layout.count(), -1, 1);
        set_values(weight_mem, rnd);
    }

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight_mem));
    topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reshape("reshape", input_info("conv1"), tensor(1, 16, 32, 32)));
    topology.add(convolution("conv2", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv3", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("conv3"), input_info("reshape"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));

    auto input_mem = engine.allocate_memory(in_layout);
    {
        auto rnd = rg.generate_random_1d<ov::float16>(in_layout.count(), -1, 1);
        set_values(input_mem, rnd);
    }

    // Reference: optimized but with memory pool disabled
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(true));
    config_ref.set_property(ov::intel_gpu::enable_memory_pool(false));
    network net_ref(engine, topology, config_ref);
    net_ref.set_input_data("input", input_mem);
    auto outputs_ref = net_ref.execute();

    // Test: optimized with memory pool enabled
    ExecutionConfig config_opt = get_test_default_config(engine);
    config_opt.set_property(ov::intel_gpu::optimize_data(true));
    config_opt.set_property(ov::intel_gpu::enable_memory_pool(true));
    network net_opt(engine, topology, config_opt);
    net_opt.set_input_data("input", input_mem);
    auto outputs_opt = net_opt.execute();

    auto out_ref = outputs_ref.at("reorder").get_memory();
    auto out_opt = outputs_opt.at("reorder").get_memory();

    ASSERT_NE(out_ref, nullptr);
    ASSERT_NE(out_opt, nullptr);
    ASSERT_EQ(out_ref->count(), out_opt->count());

    cldnn::mem_lock<float> ref_ptr(out_ref, get_test_stream());
    cldnn::mem_lock<float> opt_ptr(out_opt, get_test_stream());

    const float tolerance = 1e-3f;
    for (size_t i = 0; i < out_ref->count(); i++) {
        ASSERT_NEAR(ref_ptr[i], opt_ptr[i], tolerance)
            << "Mismatch at index " << i
            << ": ref=" << ref_ptr[i] << " opt=" << opt_ptr[i]
            << "\nThis may indicate memory pool corruption from buffer reuse "
               "through an in-place chain";
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
// regression coverage: optimized-out resample feeds oneDNN conv with fused sum.
TEST(basic_memory_dependencies, optimized_resample_to_onednn_sum_reuse_correctness) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight_layout = layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);

    tests::random_generator rg;
    rg.set_seed("optimized_resample_to_onednn_sum_reuse_correctness");
    {
        auto rnd = rg.generate_random_1d<ov::float16>(weight_layout.count(), -0.25f, 0.25f);
        set_values(weight_mem, rnd);
    }

    auto add_test_topology = [&](topology& topology) {
        topology.add(input_layout("input", in_layout));
        topology.add(data("weight", weight_mem));
        topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(convolution("conv2", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    // Extra conv on conv3 branch makes conv3's processing number > resample's,
    // so the fusion swap heuristic in prepare_primitive_fusing keeps eltwise(sum)
    // fused INTO conv3 (the oneDNN conv) instead of moving it into resample.
    // This is required for oneDNN sum-reuse: conv3 writes its sum DST into the
    // buffer that gets rebind from the optimized-out resample's output.
        topology.add(convolution("conv2b", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(convolution("conv3", input_info("conv2b"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
        topology.add(resample("resample", input_info("conv1"), tensor(1, 16, 32, 32), 1,
                              ov::op::v4::Interpolate::InterpolateMode::NEAREST));
        topology.add(eltwise("eltwise", input_info("conv3"), input_info("resample"), eltwise_mode::sum));
        topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));
    };

    topology test_topology;
    add_test_topology(test_topology);

    topology ref_topology;
    add_test_topology(ref_topology);

    auto input_mem = engine.allocate_memory(in_layout);
    {
        auto rnd = rg.generate_random_1d<ov::float16>(in_layout.count(), -0.25f, 0.25f);
        set_values(input_mem, rnd);
    }

    ov::intel_gpu::ImplementationDesc conv1_impl = {format::b_fs_yx_fsv16, "", impl_types::onednn};
    ov::intel_gpu::ImplementationDesc conv2_impl = {format::b_fs_yx_fsv16, "", impl_types::onednn};
    ov::intel_gpu::ImplementationDesc conv2b_impl = {format::b_fs_yx_fsv16, "", impl_types::onednn};
    ov::intel_gpu::ImplementationDesc resample_impl = {format::b_fs_yx_fsv16, "", impl_types::ocl};
    ov::intel_gpu::ImplementationDesc conv3_impl = {format::b_fs_yx_fsv16, "", impl_types::onednn};
    ov::intel_gpu::ImplementationDesc eltwise_impl = {format::b_fs_yx_fsv16, "", impl_types::ocl};

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv1", conv1_impl},
                                                                                          {"conv2", conv2_impl},
                                                                                          {"conv2b", conv2b_impl},
                                                                                          {"resample", resample_impl},
                                                                                          {"conv3", conv3_impl},
                                                                                          {"eltwise", eltwise_impl}}));

    // Reference: disable optimize_data so resample is executed as a normal node
    // and the optimized-out alias/rebind path is bypassed.
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));

    network net_ref(engine, ref_topology, config_ref);
    net_ref.set_input_data("input", input_mem);
    auto outputs_ref = net_ref.execute();

    auto prog = program::build_program(engine, test_topology, config);
    network net(prog, 0);

    auto resample_inst = net.get_primitive("resample");
    ASSERT_TRUE(resample_inst->can_be_optimized());
    auto& conv3_node = net.get_program()->get_node("conv3");
    ASSERT_EQ(conv3_node.get_preferred_impl_type(), impl_types::onednn);
    auto& resample_node = net.get_program()->get_node("resample");
    ASSERT_EQ(resample_node.get_output_layout().format, format::b_fs_yx_fsv16);
    ASSERT_EQ(conv3_node.get_output_layout().format, format::b_fs_yx_fsv16);

    // eltwise(sum) must be fused INTO conv3 (oneDNN), with resample as the outer dependency.
    bool has_sum_fusion_in_conv3 = false;
    for (const auto& fused_desc : conv3_node.get_fused_primitives()) {
        if (!fused_desc.is_type<eltwise>() || !fused_desc.has_outer_dep())
            continue;
        const auto add_fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(conv3_node, fused_desc);
        if (add_fusing_type != onednn_add_fusing_helpers::add_fusing_type::sum)
            continue;
        const auto outer_dep_idx = static_cast<size_t>(fused_desc.outer_dep_start_idx);
        if (outer_dep_idx >= conv3_node.get_dependencies().size())
            continue;
        if (conv3_node.get_dependencies()[outer_dep_idx].first->id() == resample_node.id()) {
            has_sum_fusion_in_conv3 = true;
            break;
        }
    }
    ASSERT_TRUE(has_sum_fusion_in_conv3)
        << "ICNet pattern requires eltwise(sum) fused INTO conv3 with resample as outer dep";

    // conv3 should expose a valid reused eltwise mem idx pointing to resample.
    int conv3_reused_idx = onednn_add_fusing_helpers::get_reused_eltwmem_idx(conv3_node);
    ASSERT_GE(conv3_reused_idx, 0);
    auto conv3_inst = net.get_primitive("conv3");
    ASSERT_LT(static_cast<size_t>(conv3_reused_idx), conv3_inst->dependencies().size());
    ASSERT_EQ(conv3_inst->dependencies()[conv3_reused_idx].first->id(), std::string("resample"));

    net.set_input_data("input", input_mem);
    auto outputs = net.execute();

    // After execute, rebind_onednn_reuse_optimized_dst_if_needed should bind
    // resample's output buffer to conv3's reused sum DST -- they must alias.
    auto resample_mem = resample_inst->output_memory_ptr();
    auto conv3_mem = conv3_inst->output_memory_ptr();
    ASSERT_TRUE(engine.is_the_same_buffer(*resample_mem, *conv3_mem))
        << "oneDNN sum-reuse rebind failed: optimized-out resample output buffer "
        << "should alias conv3's reused sum destination buffer.";

    auto out_ref = outputs_ref.at("reorder").get_memory();
    auto out_mem = outputs.at("reorder").get_memory();
    ASSERT_NE(out_ref, nullptr);
    ASSERT_NE(out_mem, nullptr);
    ASSERT_EQ(out_ref->count(), out_mem->count());

    cldnn::mem_lock<float> ref_ptr(out_ref, get_test_stream());
    cldnn::mem_lock<float> opt_ptr(out_mem, get_test_stream());

    const float rel_tolerance = 5e-3f;
    const float abs_tolerance = 3e-2f;
    const float abs_tolerance_limit = 5e-2f;
    for (size_t i = 0; i < out_ref->count(); i++) {
        ASSERT_TRUE(are_equal(ref_ptr[i], opt_ptr[i], rel_tolerance, abs_tolerance, abs_tolerance_limit))
            << "Mismatch at index " << i
            << ": ref=" << ref_ptr[i] << " opt=" << opt_ptr[i]
            << " rel_tolerance=" << rel_tolerance
            << " abs_tolerance=" << abs_tolerance;
    }
}
#endif  // ENABLE_ONEDNN_FOR_GPU
