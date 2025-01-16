// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#endif

using namespace cldnn;
using namespace ::tests;

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

TEST(post_optimize_weights, fuse_reorder_to_onednn_weights_reorder_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    ov::Shape pshape = { 4, 16 };
    auto input = engine.allocate_memory({ pshape, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({ pshape, data_types::f32, format::bfyx });

    std::vector<float> weights_data(pshape[0] * pshape[1]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_impl} }));
    }

    auto prog = program::build_program(engine, topology, config);

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    auto& fc_node = prog->get_node("fc");
    auto weights_param = fc_node.as<fully_connected>().get_selected_impl()->get_weights_reorder_params();
    ASSERT_TRUE(format::is_weights_format(prog->get_node("reorder_dt").get_output_layout().format));
    ASSERT_TRUE(prog->get_node("reorder_dt").get_input_layout().data_type == data_types::f32);
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Check onednn_weights_params->_in_desc data_type is properly updated
    auto onednn_weights_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(weights_param);
    ASSERT_TRUE(onednn_weights_params->_in_desc.get_data_type() == onednn::convert_data_type(data_types::f32));
#endif
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
    ASSERT_TRUE(onednn_weights_params->_out_desc.get_size() == prog->get_node("weights_weights_reorder_0").get_output_layout().bytes_count());
#endif
}
