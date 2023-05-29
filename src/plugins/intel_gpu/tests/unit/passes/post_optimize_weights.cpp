// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "fully_connected_inst.h"

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
