// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "pass_manager.h"

using namespace cldnn;
using namespace ::tests;

TEST(propagate_constants, select_impl_after_dynamic_to_static_transition) {
    auto& engine = get_test_engine();

    auto dynamic_layout = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};

    topology topology(
        input_layout("input", dynamic_layout),
        reorder("reorder", input_info("input"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<compile_graph>(*prog);

    auto& input_node = prog->get_node("input");
    auto static_layout = layout{{1, 3, 4, 4}, data_types::f32, format::bfyx};
    input_node.set_output_layout(static_layout, true);

    auto& reorder_node = prog->get_node("reorder");
    ASSERT_FALSE(reorder_node.is_valid_output_layout());

    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    ASSERT_NE(reorder_node.get_selected_impl(), nullptr);
}
