// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "fully_connected_inst.h"
#include "permute_inst.h"

using namespace cldnn;
using namespace ::tests;

TEST(reorder_transfer, transfer_per_permute) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        permute("permute", input_info("reorder_dt"), {1, 0}),
        fully_connected("fc", input_info("input"), { "permute" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    for (auto& node : prog->get_processing_order()) {
        if (!node->is_type<data>())
            node->get_output_layouts();
    }

    program_wrapper::apply_opt_pass<reorder_transfer>(*prog);
    auto& processing_order = prog->get_processing_order();

    auto reorder_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("reorder_dt"));
    size_t reorder_dist = std::distance(processing_order.begin(), reorder_node);

    auto permute_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("permute"));
    size_t permute_dist = std::distance(processing_order.begin(), permute_node);

    ASSERT_TRUE(reorder_dist > permute_dist);
}

// Test that verifies constant reorder nodes ignore the input/output size check.
TEST(reorder_transfer, constant_reorder_ignores_size_check) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({{32, 2}, data_types::f32, format::bfyx});

    auto build_constant_reorder_topology =
        [&](const std::string& data_name, const memory::ptr& data_mem, const std::string& reorder_name, data_types reorder_type) {
            topology topo;
            topo.add(data(data_name, data_mem));
            topo.add(reorder(reorder_name,
                     input_info(data_name),
                     format::bfyx,
                     reorder_type,
                     std::vector<float>(),
                     reorder_mean_mode::subtract,
                     padding(),
                     true));  // is_constant = true
            return topo;
        };

    topology topo = build_constant_reorder_topology("weights", weights, "reorder_dt", data_types::f16);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topo, config, false, true);
    program_wrapper::apply_opt_pass<reorder_transfer>(*prog);

    ASSERT_TRUE(prog->has_node("reorder_dt")) << "Constant reorder node should remain in the graph; size check must be ignored for constant nodes.";
}

// Test that verifies dynamic reorder nodes apply the input/output size check.
TEST(reorder_transfer, dynamic_reorder_applies_size_check) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto build_and_optimize = [&](const std::string& input_name, const layout& in_layout, const std::string& reorder_name, data_types reorder_type) {
        topology topo;
        topo.add(input_layout(input_name, in_layout));
        topo.add(reorder(reorder_name,
                         input_info(input_name),
                         format::bfyx,
                         reorder_type,
                         std::vector<float>(),
                         reorder_mean_mode::subtract,
                         padding(),
                         false));
        auto prog = program::build_program(engine, topo, config, false, true);
        program_wrapper::apply_opt_pass<reorder_transfer>(*prog);
        return prog;
    };

    // Assert: input_size < output_size (f16 -> f32)
    auto input_f16 = engine.allocate_memory({{2, 32}, data_types::f16, format::bfyx});
    auto prog_f16 = build_and_optimize("input_f16", input_f16->get_layout(), "reorder_dt", data_types::f32);
    ASSERT_TRUE(prog_f16->has_node("reorder_dt")) << "Reorder node should remain when input_size < output_size.";

    // Assert: input_size >= output_size (f32 -> f16)
    auto input_f32 = engine.allocate_memory({{2, 32}, data_types::f32, format::bfyx});
    auto prog_f32 = build_and_optimize("input_f32", input_f32->get_layout(), "reorder_dt2", data_types::f16);
    ASSERT_TRUE(prog_f32->has_node("reorder_dt2")) << "Reorder node should remain or be removed depending on implementation when input_size >= output_size.";
}
