// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_primitive_fusing, fuse_to_fc_dyn) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ ov::PartialShape{ 16, 32 }, data_types::u8, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(2), data_types::u8, format::bfyx };

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", in_layout));
    topology.add(fully_connected("fc", "input", { "weights" }));
    topology.add(activation("act", "fc", activation_func::relu));
    topology.add(reorder("reorder", "act", format::bfyx, data_types::f32));

    build_options build_opts;
    auto prog = program::build_program(engine, topology, build_opts, false, true);

    layout_optimizer lo(true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog, lo);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node_with_type<activation>(*prog));
}
