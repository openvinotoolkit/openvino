// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "activation_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "reshape_inst.h"
#include "pass_manager.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

/* Basic test to show how the program can be build and run within internal tests
   in similar way as it is done in tests utilizing clDNN API */
TEST(basic, test1) {
    auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { FLOAT16(1.1f), FLOAT16(1.2f), FLOAT16(1.3f), FLOAT16(1.4f) });
    set_values(weights1, { FLOAT16(2.1f), FLOAT16(3.1f) });
    set_values(weights2, { 1.1f, 0.1f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(reshape("reshape1", "weights1", tensor(spatial(1, 2))));
    topology.add(reorder("reorder2", "input", layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(reorder("reorder1", "reshape1", layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(concatenation("concat", { "reorder1", "weights2" }, 3));
    topology.add(convolution("conv2", { "reorder2" }, { "concat" }));

    program::ptr prog = program::build_program(engine, topology, build_opt, false);
    network::ptr network = network::allocate_network(engine, prog);
    network->set_input_data("input", input);

    auto outputs = network->execute();

    float epsilon = 1e-2f;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        EXPECT_NEAR(7.8f, output[0], epsilon);
    }
}

// This test creates a program without optimization passes, even the compilation is being run manualy.
// Thus, a single method from program like add_intermediate might be tested separately.
TEST(add_intermediate_gpu, test1)
{
    build_options build_opt;
    topology topology;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {2, 2, 2, 2} });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, {2, 2, 2, 2} });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 1, 1 } });

    set_values(input, { (1.1f), (1.2f), (1.3f), (1.4f),
                        (2.1f), (2.2f), (2.3f), (2.4f),
                        (3.1f), (3.2f), (3.3f), (3.4f),
                        (4.1f), (4.2f), (4.3f), (4.4f) });
    set_values(weights, { (1.5f), (1.6f), (1.7f), (1.8f),
                          (2.5f), (2.6f), (2.7f), (2.8f),
                          (3.5f), (3.6f), (3.7f), (3.8f),
                          (4.5f), (4.6f), (4.7f), (4.8f) });

    set_values(weights2, { (5.5f), (5.6f), (5.7f), (5.8f) });
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("weights2", weights2));
    topology.add(cldnn::convolution("conv1a", { "input" }, { "weights" }));
    topology.add(cldnn::convolution("conv1b", { "input" }, { "weights" }));
    topology.add(cldnn::convolution("conv2a", { "conv1a" }, { "weights2" }));
    auto new_reorder = std::make_shared<reorder>("reorder","nothing", input->get_layout());
    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);
    prog->add_intermediate(new_reorder, prog->get_node("conv1a"), 0);
    prog->dump_program("custom_dump", true);

    program_wrapper::build(*prog);

    network::ptr network = network::allocate_network(engine, prog);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    std::vector<float> expected_output_vec = {
        32.2f, 60.2f, 66.6f, 126.6f,
        514.22f, 532.7f, 1075.26f, 1113.9f
    };

    uint32_t output_size = 4;
    uint32_t output_index = 0;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (uint32_t x = 0; x < output_size; x++)
        {
            EXPECT_FLOAT_EQ(expected_output_vec[x+output_size*output_index], output[x]);
        }
        output_index++;
    }
}

/* This test shows how to use private members (here: add_connection) of program using program_wraper */
// Disabled for now as it produces wrong results
TEST(add_intermediate_gpu, test2)
{
    build_options build_opt;
    topology topology;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 1, 1 } });

    set_values(input, { (1.1f), (1.2f), (1.3f), (1.4f),
        (2.1f), (2.2f), (2.3f), (2.4f),
        (3.1f), (3.2f), (3.3f), (3.4f),
        (4.1f), (4.2f), (4.3f), (4.4f) });
    set_values(weights, { (1.5f), (1.6f), (1.7f), (1.8f),
        (2.5f), (2.6f), (2.7f), (2.8f),
        (3.5f), (3.6f), (3.7f), (3.8f),
        (4.5f), (4.6f), (4.7f), (4.8f) });

    set_values(weights2, { (5.5f), (5.6f), (5.7f), (5.8f) });

    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights2", weights2));

    topology.add(cldnn::convolution("conv2a", { "input" }, { "weights2" }));
    topology.add(cldnn::convolution("conv2b", { "input" }, { "weights2" }));

    std::vector<primitive_id> w_vec;
    w_vec.push_back("weights");
    auto new_conv = std::make_shared<convolution>("conv1a", "input", w_vec);
    auto weights_node = std::make_shared<data>("weights", weights);
    program::ptr prog = program::build_program(engine, topology, build_opt, false, true);

    prog->add_intermediate(new_conv, prog->get_node("conv2a"), 0, true, true);
    program_wrapper::add_connection(*prog, prog->get_or_create(weights_node), prog->get_or_create(new_conv));
    prog->dump_program("custom_dump", true);

    program_wrapper::build(*prog);

    network::ptr network = network::allocate_network(engine, prog);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    std::vector<float> expected_output_vec = {
        514.22f, 532.7f, 1075.26f, 1113.9f
    };

    uint32_t output_size = 4;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (uint32_t x = 0; x < output_size; x++)
        {
            EXPECT_FLOAT_EQ(expected_output_vec[x], output[x]);
        }
    }
}
