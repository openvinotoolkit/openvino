// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "quantize_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_quantization, program_replace_check_num_of_nodes) {
    auto& engine = get_test_engine();
    auto data0_layout = engine.allocate_memory({ ov::PartialShape{1}, data_types::f32, format::bfyx });
    auto data1_layout = engine.allocate_memory({ ov::PartialShape{1}, data_types::f32, format::bfyx });
    auto in_layout = layout{ ov::PartialShape::dynamic(0), data_types::f32, format::bfyx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("input_low", data0_layout));
    topology.add(data("input_high", data1_layout));
    topology.add(quantize("quantize", input_info("input"), input_info("input_low"), input_info("input_high"), input_info("input_low"), input_info("input_high"), 256, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->get_node("quantize").get_dependencies().size() == 5);

    program_wrapper::apply_opt_pass<prepare_quantization>(*prog);

    ASSERT_TRUE(prog->get_node("quantize").get_dependencies().size() == 9);
}
