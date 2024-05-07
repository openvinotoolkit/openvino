// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "convolution_inst.h"
#include "gemm_inst.h"
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

    layout_optimizer lo(true);
    auto prog = program::build_program(engine, topology, config, false, true);

    // It initializes output_layout.
    // It's necessary because this test runs select_preferred_formats pass alone.
    prog->get_node("conv1").get_output_layouts(false);
    program_wrapper::apply_opt_pass<select_preferred_formats>(*prog, lo);

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


TEST(test_select_preferred_formats, setting_target_gemm_transpose_input_format) {
    auto& engine = get_test_engine();
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 128, 64, 12 } });
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 128, 64, 12 } });

    topology topology;
    topology.add(input_layout("input0", input0->get_layout()));
    topology.add(permute("permute0", input_info("input0"), {0, 2, 1, 3}));
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(permute("permute1", input_info("input1"), {0, 2, 1, 3}));
    topology.add(gemm("gemm", { input_info("permute0"), input_info("permute1") }, data_types::f16, false, true));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc impl = { format::bfyx, std::string(""), impl_types::onednn };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm", impl} }));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);
    lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, true);

    // It initializes output_layout.
    // It's necessary because this test runs select_preferred_formats pass alone.
    prog->get_node("gemm").get_output_layouts(false);
    program_wrapper::apply_opt_pass<select_preferred_formats>(*prog, lo);

    ASSERT_NE(prog, nullptr);

    // "permute1" is transpose_b input of gemm, so it shouldn't have preferred out fmt during select_preferred_formats pass.
    auto permute_node_ptr = prog->get_node_ptr("permute1");
    auto& permute_node = permute_node_ptr->as<permute>();
    auto output_fmt = permute_node.get_preferred_output_fmt(0);
    ASSERT_EQ(output_fmt, format::any);
}