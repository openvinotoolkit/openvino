// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "reshape_inst.h"
#include "fully_connected_inst.h"
#include "permute_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_buffer_fusing, optimize_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto pattern_layout = layout{ov::PartialShape::dynamic(4), data_types::i64, format::bfyx};
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("pattern", pattern_layout));
    topology.add(permute("permute1", input_info("input"), {0, 2, 3, 1}));
    topology.add(reshape("reshape", input_info("permute1"), input_info("pattern"), false, ov::PartialShape::dynamic(4)));
    topology.add(permute("permute2", input_info("reshape"), {0, 3, 2, 1}));
    topology.add(reorder("reorder", input_info("permute2"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 2, 4}, data_types::f16, format::bfyx });
    auto pattern_memory = engine.allocate_memory(layout{ ov::PartialShape{4}, data_types::i64, format::bfyx });
    set_values<float>(input_memory, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7, 4.8, 19.2, -10.1, 8.1, 10.2, 1.3, 1.44, 1.5});
    set_values<int64_t>(pattern_memory, {1, 4, 1, -1});

    net.set_input_data("input", input_memory);
    net.set_input_data("pattern", pattern_memory);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("reorder");
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);
    ASSERT_EQ(out_mem->count(), 16);
}

TEST(prepare_buffer_fusing, static_node_after_optimized_out_dyn_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape{1, 2, -1}, data_types::f32, format::bfyx };
    auto weights_layout = layout{ov::PartialShape{2, 4}, data_types::f32, format::bfyx};
    auto weights_memory = engine.allocate_memory(weights_layout);
    set_values<float>(weights_memory, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights_memory));
    topology.add(permute("permute1", input_info("input"), {0, 2, 1}));
    topology.add(reshape("reshape", input_info("permute1"), false, {2, 4}, ov::PartialShape{2, 4}));
    topology.add(fully_connected("fc", input_info("reshape"), "weights", "", {}, 2));
    topology.add(reorder("reorder", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    prog->get_node("reorder").get_output_layout(true);
    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*prog);
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    ASSERT_NO_THROW(prog->get_node("reshape"));
    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());
    program_wrapper::apply_opt_pass<build_implementations>(*prog);

    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 4}, data_types::f32, format::bfyx });
    set_values<float>(input_memory, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7});

    net.set_input_data("input", input_memory);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    ASSERT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("reorder");
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);
    ov::PartialShape expected_shape = {2, 2};
    ASSERT_EQ(out_mem->count(), 4);
    ASSERT_EQ(out_mem->get_layout().get_partial_shape(), expected_shape);
}
