// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "activation_inst.h"
#include "reorder_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"
#include "program_helpers.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(add_onednn_optimization_attributes, init_attribute_for_fused_onednn_primitive) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape({-1, 3, 112, 112}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 3, 112, 112}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({128, 3, 3, 3}), data_types::f16, format::bfyx});
    auto const1 = engine.allocate_memory(layout{ov::PartialShape({1, 128, 1, 1}), data_types::f16, format::bfyx});
    auto const2 = engine.allocate_memory(layout{ov::PartialShape({1, 128, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(data("const1", const1));
    topology.add(data("const2", const2));
    topology.add(convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("convolution"), input_info("const1"), eltwise_mode::sum));
    topology.add(activation("prelu", input_info("eltwise"), "const2", activation_func::relu_negative_slope));
    topology.add(reorder("reorder", input_info("prelu"), format::bfyx, data_types::f32));


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node(*prog, "eltwise"));
    ASSERT_FALSE(has_node(*prog, "prelu"));
}

TEST(add_onednn_optimization_attributes, sum_post_op_for_residual_connection) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv2", input_info("conv1"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv3", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("conv1"), input_info("conv3"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    auto &conv3 = prog->get_node("conv3");
    auto &cldnn_post_ops = conv3.get_fused_primitives();
    ASSERT_EQ(cldnn_post_ops.size(), 1);
    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(conv3, cldnn_post_ops[0]);

    // Check whether fusing_type is properly selected as sum for residual connection pattern
    ASSERT_EQ(fusing_type, add_fusing_type::sum);

}
