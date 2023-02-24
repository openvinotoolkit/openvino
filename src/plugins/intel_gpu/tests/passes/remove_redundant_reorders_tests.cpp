// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "convolution_inst.h"
#include "reorder_inst.h"
#include "softmax_inst.h"

#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(remove_redundant_reorders, remove_dep_dynamic) {
    // Topology:
    // convolution -> reorder -> softmax
    //
    // Expectation:
    // The preferred format of convolution should be selected as b_fs_yx_fsv16 (reorder_inputs)
    // A new reorder that converts to bfyx should be inserted after convolution (reorder_inputs)
    // In reorders, output format of dependency reorder should be saved as output_format of orginial reorder (remove_redundant_reorders)

    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 224, 224 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 64, 3, 7, 7 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(convolution("conv", input_info("input"), { "weights" }));
    topology.add(reorder("reorder", input_info("conv"), format::any, data_types::f32));
    topology.add(softmax("softmax", input_info("reorder"), 1));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    network.execute();

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);
    auto& softmax_node = prog->get_node("softmax");
    auto softmax_layout = softmax_node.get_output_layout();

    ASSERT_EQ(softmax_layout.format.value, format::bfyx);
}
