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
