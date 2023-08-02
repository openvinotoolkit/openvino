// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "fully_connected_inst.h"
#include "permute_inst.h"

using namespace cldnn;
using namespace ::tests;

TEST(fuse_constant_transposes, transpose_removal_check) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        permute("permute", input_info("weights"), {1, 0}),
        reorder("reorder_dt", input_info("permute"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<fuse_constant_transposes>(*prog);

    ASSERT_TRUE(!has_node(*prog, "permute"));
    ASSERT_EQ(prog->get_node("weights").get_output_layout().format, format::ioyx);
}

TEST(fuse_constant_transposes, accuracy_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16),
        permute("permute", input_info("reorder_dt"), {1, 0}),
        fully_connected("fc", input_info("input"), { "permute" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto output = outputs.at("fc").get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr(output, get_test_stream());

    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));
    config_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network_ref(engine, topology, config_ref);
    network_ref.set_input_data("input", input);

    auto outputs_ref = network_ref.execute();
    auto output_ref = outputs_ref.at("fc").get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr_ref(output_ref, get_test_stream());

    for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
        ASSERT_EQ(output_ptr[i], output_ptr_ref[i]);
    }
}
