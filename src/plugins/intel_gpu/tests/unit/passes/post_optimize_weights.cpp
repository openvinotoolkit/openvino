// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#include "gru_seq_inst.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#endif

using namespace cldnn;
using namespace ::tests;

TEST(post_optimize_weights, dynamic_gru_weights_are_permuted) {
    auto& engine = get_test_engine();

    auto hidden = engine.allocate_memory({ov::PartialShape{1, 1, 3}, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({ov::PartialShape{1, 9, 4}, data_types::f16, format::bfyx});
    auto recurrent = engine.allocate_memory({ov::PartialShape{1, 9, 3}, data_types::f16, format::bfyx});
    auto bias = engine.allocate_memory({ov::PartialShape{1, 12}, data_types::f16, format::bfyx});
    auto sequence_lengths = engine.allocate_memory({ov::PartialShape{1}, data_types::i32, format::bfyx});

    gru_seq gru("gru",
                input_info("input"),
                input_info("hidden"),
                input_info(),
                input_info("weights"),
                input_info("recurrent"),
                input_info("bias"),
                input_info("sequence_lengths"),
                0.f,
                false,
                {activation_func::logistic, activation_func::hyperbolic_tan},
                {},
                lstm_weights_order::iofz,
                ov::op::RecurrentSequenceDirection::FORWARD,
                2);
    gru.linear_before_reset = true;

    topology topology(input_layout("input", {ov::PartialShape{1, -1, 4}, data_types::f16, format::bfyx}),
                      data("hidden", hidden),
                      data("weights", weights),
                      data("recurrent", recurrent),
                      data("bias", bias),
                      data("sequence_lengths", sequence_lengths),
                      gru);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    auto& gru_node = prog->get_node("gru");
    EXPECT_EQ(gru_node.get_dependency(2).get_output_layout().get_shape(), ov::Shape({1, 4, 9}));
    EXPECT_EQ(gru_node.get_dependency(3).get_output_layout().get_shape(), ov::Shape({1, 3, 9}));
}

TEST(post_optimize_weights, fuse_reorder_to_weights_reorder_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 2, 32 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    ASSERT_TRUE(format::is_weights_format(prog->get_node("reorder_dt").get_output_layout().format));
}

TEST(post_optimize_weights, fuse_reorder_to_weights_reorder_test_dynamic) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;

    auto weights = engine.allocate_memory({{ 2, 32 }, data_types::f32, format::bfyx });

    auto in_layout = layout{ ov::PartialShape{ov::Dimension(1), ov::Dimension(-1), ov::Dimension(32)}, data_types::f16, format::bfyx };

    topology topology(
        input_layout("input", in_layout),
        input_layout("weights", weights->get_layout()),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16, 3)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    ASSERT_NE(prog->get_node("fc").get_selected_impl(), nullptr);
    ASSERT_TRUE(format::is_weights_format(prog->get_node("reorder_dt").get_output_layout().format));
}

TEST(post_optimize_weights, weights_reorder_constant_folding_test) {
    auto& engine = get_test_engine();

    ov::Shape pshape = { 4, 16 };
    auto input = engine.allocate_memory({ pshape, data_types::f32, format::bfyx });
    auto weights = engine.allocate_memory({ pshape, data_types::f32, format::bfyx });

    std::vector<float> weights_data(pshape[0] * pshape[1]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        fully_connected("fc", input_info("input"), { "weights" })
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    ASSERT_TRUE(has_node(*prog, "weights_weights_reorder_0"));
    auto& weights_node = prog->get_node("weights_weights_reorder_0");
    ASSERT_TRUE(weights_node.is_type<data>());

    size_t align = 16; // os_iyx_osv16 format
    size_t aligned_b_size = pshape[0] % align == 0 ? pshape[0]
                                                   : pshape[0] - pshape[0] % align + align;
    std::vector<float> expected(aligned_b_size * pshape[1], 0.f);
    size_t input_idx = 0;
    for (size_t i = 0; i < pshape[0]; ++i) {
        for (size_t j = 0; j < pshape[1]; ++j) {
            expected[j * align + i] = weights_data[input_idx++];
        }
    }

    auto weights_mem_ptr = weights_node.as<data>().get_attached_memory_ptr();
    cldnn::mem_lock<float, mem_lock_type::read> weights_mem(weights_mem_ptr, get_test_stream());

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(weights_mem[i], expected[i]);
    }
}

TEST(post_optimize_weights, weights_reorder_constant_folding_test_dynamic) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;
    ov::Shape pshape = { 4, 32 };
    auto in_layout = layout{ ov::PartialShape{ov::Dimension(1), ov::Dimension(-1), ov::Dimension(32)}, data_types::f16, format::bfyx };
    auto weights = engine.allocate_memory({pshape, data_types::f32, format::bfyx });

    std::vector<float> weights_data(pshape[0] * pshape[1]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", in_layout),
        data("weights", weights),
        fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f16, 3)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    ASSERT_TRUE(has_node(*prog, "weights_weights_reorder_0"));
    auto& weights_node = prog->get_node("weights_weights_reorder_0");
    ASSERT_TRUE(weights_node.is_type<data>());

    size_t align = 16; // os_iyx_osv16 format
    size_t aligned_b_size = pshape[0] % align == 0 ? pshape[0]
                                                   : pshape[0] - pshape[0] % align + align;
    std::vector<float> expected(aligned_b_size * pshape[1], 0.f);
    size_t input_idx = 0;
    for (size_t i = 0; i < pshape[0]; ++i) {
        for (size_t j = 0; j < pshape[1]; ++j) {
            expected[j * align + i] = weights_data[input_idx++];
        }
    }

    auto weights_mem_ptr = weights_node.as<data>().get_attached_memory_ptr();
    cldnn::mem_lock<float, mem_lock_type::read> weights_mem(weights_mem_ptr, get_test_stream());

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(weights_mem[i], expected[i]);
    }
}

TEST(post_optimize_weights, fuse_only_with_supported_weights_layout) {
    // This test case checks that the reorder node is not fused with unsupported weights layout such as b_fs_yx_fsv16.
    //    - Previous : When the weight layout which is a reorder node, is b_fs_yx_fsv16, fusing was attempted but the program terminates.
    //    - Current : If the layout format is not supported, do not fuse and the program won't terminate.
    //
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 2, 32 }, data_types::f32, format::b_fs_yx_fsv16 });

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        reorder("reorder", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder"));
}

TEST(post_optimize_weights, onednn_group_conv_weights_reorder_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    ov::Shape pshape = { 1, 512, 2, 332 };
    ov::Shape weight_pshape = { 64, 1, 8, 5, 5 };
    auto input = engine.allocate_memory({ pshape, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({ weight_pshape, data_types::f16, format::bfzyx });

    std::vector<ov::float16> weights_data(weight_pshape[0] * weight_pshape[1] * weight_pshape[2] * weight_pshape[3] * weight_pshape[4]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", "", 64, {1, 1}, {1, 1}, {2, 2}, {2, 2}, true)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", conv_impl} }));
    }

    auto prog = program::build_program(engine, topology, config);

    auto& conv_node = prog->get_node("conv");
    auto weights_param = conv_node.as<convolution>().get_selected_impl()->get_weights_reorder_params();
    ASSERT_TRUE(format::is_weights_format(prog->get_node("weights_weights_reorder_0").get_output_layout().format));

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Check onednn_weights_params->_out_desc.get_size() with reorder output_layout bytes_count
    auto onednn_weights_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(weights_param);
    ASSERT_TRUE(onednn_weights_params->_out_desc->get_size() == prog->get_node("weights_weights_reorder_0").get_output_layout().bytes_count());
#endif
}

template <typename T>
static void check_folded_fc_weights(program_node& node, const std::vector<T>& src, int32_t ofm, int32_t ifm) {
    const auto& out_layout = node.get_output_layout();
    ASSERT_EQ(out_layout.data_type, data_types::f16);
    cldnn::mem_lock<ov::float16, mem_lock_type::read> mem(node.as<data>().get_attached_memory_ptr(), get_test_stream());
    for (int32_t o = 0; o < ofm; ++o) {
        for (int32_t i = 0; i < ifm; ++i) {
            const auto offset = out_layout.get_linear_offset(tensor(batch(o), feature(i), spatial(0, 0, 0, 0)));
            ASSERT_EQ(static_cast<float>(mem[offset]), static_cast<float>(ov::float16(src[static_cast<size_t>(o * ifm + i)])))
                << "o=" << o << " i=" << i << " format=" << out_layout.format.to_string();
        }
    }
}

TEST(post_optimize_weights, fc_5d_weights_reorder_constant_folding_test) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape{1, 2, 2, 4, 16}, data_types::f16, format::bfzyx};
    ov::Shape weight_shape = {1, 1, 1, 32, 16};
    auto weights = engine.allocate_memory({ov::PartialShape(weight_shape), data_types::f16, format::bfzyx});

    std::vector<ov::float16> weights_data(ov::shape_size(weight_shape));
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", in_layout),
        data("weights", weights),
        fully_connected("fc", input_info("input"), {"weights"}, "", data_types::f16, 5, 5)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc fc_impl = {format::bfzyx, "", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", fc_impl}}));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);
    ASSERT_NO_THROW(program_wrapper::apply_opt_pass<propagate_constants>(*prog));

    ASSERT_TRUE(has_node(*prog, "weights_weights_reorder_0"));
    auto& weights_node = prog->get_node("weights_weights_reorder_0");
    ASSERT_TRUE(weights_node.is_type<data>());
    EXPECT_TRUE(format::is_weights_format(weights_node.get_output_layout().format));
    EXPECT_EQ(weights_node.get_output_layout().get_partial_shape(), ov::PartialShape({32, 16}));
    EXPECT_GT(weights_node.get_output_layout().bytes_count(), 0u);
    check_folded_fc_weights(weights_node, weights_data, 32, 16);
}

TEST(post_optimize_weights, fc_5d_fuse_reorder_to_weights_reorder_test) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape{1, 2, 2, 4, 16}, data_types::f16, format::bfzyx};
    ov::Shape weight_shape = {1, 1, 1, 32, 16};
    auto weights = engine.allocate_memory({ov::PartialShape(weight_shape), data_types::f32, format::bfzyx});

    std::vector<float> weights_data(ov::shape_size(weight_shape));
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", in_layout),
        data("weights", weights),
        reorder("reorder_dt", input_info("weights"), format::any, data_types::f16),
        fully_connected("fc", input_info("input"), {"reorder_dt"}, "", data_types::f16, 5, 5)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc fc_impl = {format::bfzyx, "", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", fc_impl}}));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    auto& reorder_before_prop = prog->get_node("reorder_dt");
    ASSERT_TRUE(reorder_before_prop.is_type<reorder>());
    auto weights_reorder_params = reorder_before_prop.as<reorder>().get_primitive()->weights_reorder_params;
    ASSERT_NE(weights_reorder_params, nullptr);
    EXPECT_EQ(weights_reorder_params->get_input_layout().format, format::oiyx);
    EXPECT_EQ(weights_reorder_params->get_output_layout().get_partial_shape(), ov::PartialShape({32, 16}));

    ASSERT_NO_THROW(program_wrapper::apply_opt_pass<propagate_constants>(*prog));

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    auto& folded_node = prog->get_node("reorder_dt");
    ASSERT_TRUE(folded_node.is_type<data>());
    EXPECT_TRUE(format::is_weights_format(folded_node.get_output_layout().format));
    EXPECT_EQ(folded_node.get_output_layout().get_partial_shape(), ov::PartialShape({32, 16}));
    EXPECT_GT(folded_node.get_output_layout().bytes_count(), 0u);
    check_folded_fc_weights(folded_node, weights_data, 32, 16);
}

TEST(post_optimize_weights, ocl_group_conv_fuse_reorder_preserves_grouped_format_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({{1, 64, 11, 11}, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({{64, 1, 9, 9}, data_types::f32, format::bfyx});

    std::vector<ov::float16> input_data(input->count(), ov::float16(1.0F));
    set_values(input, input_data);

    std::vector<float> weights_data(weights->count(), 0.0F);
    for (size_t g = 0; g < 64; ++g) {
        weights_data[g * 81 + g] = static_cast<float>(g + 1);
    }
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("input_weights", weights->get_layout()),
        reorder("reorder_weights", input_info("input_weights"), format::bfzyx, data_types::f16),
        convolution("conv", input_info("input"), "reorder_weights", "", 64, {1, 1}, {1, 1}, {0, 0}, {0, 0}, true)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = {format::any, "", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", conv_impl}}));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder_weights"));
    auto& reorder_node = prog->get_node("reorder_weights");
    ASSERT_TRUE(reorder_node.is_type<reorder>());
    auto weights_reorder_params = reorder_node.as<reorder>().get_primitive()->weights_reorder_params;
    ASSERT_NE(weights_reorder_params, nullptr);
    EXPECT_EQ(weights_reorder_params->get_input_layout().format, format::goiyx);

    network net(engine, topology, config);
    net.set_input_data("input", input);
    net.set_input_data("input_weights", weights);
    auto outputs = net.execute();
    auto out_mem = outputs.at("conv").get_memory();
    const auto out_layout = out_mem->get_layout();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(out_mem, get_test_stream());

    for (int32_t f = 0; f < 64; ++f) {
        const auto offset = out_layout.get_linear_offset(tensor(batch(0), feature(f), spatial(0, 0, 0, 0)));
        ASSERT_FLOAT_EQ(static_cast<float>(out_ptr[offset]), static_cast<float>(f + 1)) << "f=" << f;
    }
}
