// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(handle_reshape, dont_remove_reshape_that_changes_rank) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{}, data_types::f16, format::bfyx });
    auto data1_layout = engine.allocate_memory({ ov::PartialShape{1}, data_types::f16, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(0), data_types::f16, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("data0", data0_layout));
    topology.add(data("data1", data1_layout));
    topology.add(eltwise("e1", input_info("input"), input_info("data0"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("e1"), false, {1}, {1}));
    topology.add(eltwise("e2", input_info("reshape"), input_info("data1"), eltwise_mode::sum));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);
    program_wrapper::apply_opt_pass<handle_reshape>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());
}
