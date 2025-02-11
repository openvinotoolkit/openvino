// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "mvn_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_padding, groupconv_with_output) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{{1, 18, 76, 135}, data_types::f16, format::bfyx};
    auto weights_data = rg.generate_random_5d<ov::float16>(1, 18, 1, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ {18, 1, 1, 3, 3}, data_types::f16, format::bfzyx});
    set_values(weights_mem, weights_data);

    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {2, 2}, true));
    topo.add(reorder("reorder", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topo, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<prepare_padding>(*prog, true);
    const auto& node = prog->get_node("reorder_input_conv");
    auto params = node.get_kernel_impl_params();
    ASSERT_EQ(params->get_output_layout().data_padding._upper_size[2 + 0], 2);
}

TEST(prepare_padding, mvn_conv) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{{1, 3, 512, 512}, data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(in_layout);
    auto weights_data = rg.generate_random_4d<ov::float16>(3, 3, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ {3, 3, 3, 3}, data_types::f16, format::bfyx});
    set_values(weights_mem, weights_data);

    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(mvn("mvn", input_info("input"), true, 1e-10f, true, { 2 }));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("mvn"), "weight", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    topo.add(reorder("reorder_output", input_info("conv"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topo, config);
    network.set_input_data("input", input);
    EXPECT_NO_THROW(network.execute());

    for (auto& item : network.get_executed_primitives()) {
        auto prim = network.get_primitive(item.first);
        if (prim->get_node().is_type<convolution>() && !prim->get_impl()->is_onednn()) {
            ASSERT_TRUE(has_node(*network.get_program(), "conv_padding_reorder_for_mvn"));
        }
    }
}
