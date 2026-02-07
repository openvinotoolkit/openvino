// Copyright (C) 2018-2026 Intel Corporation
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

TEST(reorder_transfer, constant_reorder_ignores_size_check) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({{32, 2}, data_types::f32, format::bfyx});

    topology topo;
    topo.add(data("weights", weights));
    topo.add(reorder("reorder_dt",
                     input_info("weights"),
                     format::bfyx,
                     data_types::f16,
                     std::vector<float>(),
                     reorder_mean_mode::subtract,
                     padding(),
                     true));  // constant = true
    topo.add(permute("permute", input_info("reorder_dt"), {1, 0}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topo, config, false, true);
    program_wrapper::apply_opt_pass<reorder_transfer>(*prog);

    // Check the reorder node exists
    ASSERT_TRUE(prog->has_node("reorder_dt")) << "Constant reorder node should remain in the graph.";

    // Check the reorder node is before permute node in processing_order
    auto& processing_order = prog->get_processing_order();
    auto reorder_it = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("reorder_dt"));
    auto permute_it = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("permute"));
    size_t reorder_dist = std::distance(processing_order.begin(), reorder_it);
    size_t permute_dist = std::distance(processing_order.begin(), permute_it);
    ASSERT_TRUE(reorder_dist < permute_dist) << "Constant reorder node should remain before permute node after optimization.";
}

TEST(reorder_transfer, transfer_per_permute_datatype_check) {
    auto& engine = get_test_engine();

    auto run_case = [&](cldnn::data_types reorder_input_type, cldnn::data_types reorder_output_type,
                        const std::string& case_desc, bool reorder_should_move_after_permute, bool apply_optimization) {
        auto input = engine.allocate_memory({{2, 32}, reorder_input_type, format::bfyx});
        auto weights = engine.allocate_memory({{32, 2}, reorder_output_type, format::bfyx});

        topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            reorder("reorder_dt", input_info("input"), format::bfyx, reorder_output_type, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
            permute("permute", input_info("reorder_dt"), {1, 0}),
            fully_connected("fc", input_info("permute"), {"weights"}, "", reorder_output_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto prog = program::build_program(engine, topology, config, false, true);

        for (auto& node : prog->get_processing_order()) {
            if (!node->is_type<data>())
                node->get_output_layouts();
        }

        if (apply_optimization) {
            program_wrapper::apply_opt_pass<reorder_transfer>(*prog);
        }

        auto& processing_order = prog->get_processing_order();

        auto reorder_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("reorder_dt"));
        auto permute_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("permute"));

        ASSERT_NE(reorder_node, processing_order.end()) << "[" << case_desc << "] reorder node not found!";
        ASSERT_NE(permute_node, processing_order.end()) << "[" << case_desc << "] permute node not found!";

        size_t reorder_dist = std::distance(processing_order.begin(), reorder_node);
        size_t permute_dist = std::distance(processing_order.begin(), permute_node);

        if (apply_optimization) {
            if (reorder_should_move_after_permute) {
                ASSERT_TRUE(reorder_dist > permute_dist) << "[" << case_desc << "][optimized] reorder should be after permute (lower to higher precision).";
            } else {
                ASSERT_TRUE(reorder_dist < permute_dist) << "[" << case_desc << "][optimized] reorder should be before permute (higher to lower precision).";
            }
        } else {
            ASSERT_TRUE(reorder_dist < permute_dist) << "[" << case_desc << "][no optimization] reorder should be before permute (topology order).";
        }
    };

    // Case 1: fp16 -> fp32 (reorder should be after permute with optimization)
    run_case(data_types::f16, data_types::f32, "f16->f32", true, true);   // with optimizer
    run_case(data_types::f16, data_types::f32, "f16->f32", true, false);  // without optimizer

    // Case 2: fp32 -> fp16 (reorder should be before permute with optimization)
    run_case(data_types::f32, data_types::f16, "f32->f16", false, true);   // with optimizer
    run_case(data_types::f32, data_types::f16, "f32->f16", false, false);  // without optimizer
}
