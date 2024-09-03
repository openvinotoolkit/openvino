// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "activation_inst.h"
#include "reorder_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(add_onednn_optimization_attributes, init_attribute_for_fused_onednn_primitive) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape({-1, 3, 112, 112}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 3, 112, 112}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({128, 3, 3, 3}), data_types::f16, format::bfyx});
    auto const1 = engine.allocate_memory(layout{ov::PartialShape({1, 128, 1, 1}), data_types::f16, format::bfyx});
    auto const2 = engine.allocate_memory(layout{ov::PartialShape({1, 128, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(data("const1", const1));
    topology.add(data("const2", const2));
    topology.add(convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("convolution"), input_info("const1"), eltwise_mode::sum));
    topology.add(activation("prelu", input_info("eltwise"), "const2", activation_func::relu_negative_slope));
    topology.add(reorder("reorder", input_info("prelu"), format::bfyx, data_types::f32));


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    prog->get_layout_optimizer().set_optimization_attribute(layout_optimizer::optimization_attributes_type::use_onednn_impls, true);

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node(*prog, "eltwise"));
    ASSERT_FALSE(has_node(*prog, "prelu"));
}
