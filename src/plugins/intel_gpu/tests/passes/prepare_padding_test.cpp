// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_padding, groupconv_with_output) {
    auto& engine = get_test_engine();
    auto in_layout = layout{data_types::f16, format::bfyx, tensor{1, 18, 135, 76}};
    auto weight_layout = layout{data_types::f16, format::bfzyx, tensor{1, 18, 3, 3, 18}};
    auto weights_data = generate_random_5d<FLOAT16>(1, 18, 18, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ data_types::f16, format::bfzyx, weight_layout.get_tensor()});
    set_values(weights_mem, weights_data);

    auto output_size = tensor{1, 18, 135, 76};
    ov::CoordinateDiff pad = {0, 0};
    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("input"), { "weight" }, {}, 1, {1, 1}, {0, 0}, {1, 1}, output_size, data_types::f16, true));
    topo.add(reorder("reorder", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topo, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<prepare_padding>(*prog, true);
    const auto& node = prog->get_node("reorder_input_conv");
    auto params = node.get_kernel_impl_params();
    ASSERT_EQ(params->get_output_layout().data_padding.upper_size().spatial[2], 0);
}
