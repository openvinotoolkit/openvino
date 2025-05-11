// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

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
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { ov::float16(1.1f), ov::float16(1.2f), ov::float16(1.3f), ov::float16(1.4f) });
    set_values(weights1, { ov::float16(2.1f), ov::float16(3.1f) });
    set_values(weights2, { 1.1f, 0.1f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(reshape("reshape1", input_info("weights1"), tensor(spatial(1, 2))));
    topology.add(reorder("reorder2", input_info("input"), layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(reorder("reorder1", input_info("reshape1"), layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(concatenation("concat", { input_info("reorder1"), input_info("weights2") }, 3));
    topology.add(convolution("conv2", input_info("reorder2"), "concat", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    program::ptr prog = program::build_program(engine, topology, config, false);
    network::ptr network = network::allocate_network(engine, prog);
    network->set_input_data("input", input);

    auto outputs = network->execute();

    float epsilon = 1e-2f;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        ASSERT_NEAR(7.8f, output[0], epsilon);
    }
}

// This test creates a program without optimization passes, even the compilation is being run manualy.
// Thus, a single method from program like add_intermediate might be tested separately.
TEST(add_intermediate_gpu, test1)
{
    topology topology;
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);

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
    topology.add(cldnn::convolution("conv1a", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(cldnn::convolution("conv1b", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(cldnn::convolution("conv2a", input_info("conv1a"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    auto new_reorder = std::make_shared<reorder>("reorder", input_info("nothing"), input->get_layout());
    program::ptr prog = program::build_program(engine, topology, config, false, true);
    prog->add_intermediate(new_reorder, prog->get_node("conv1a"), 0);

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
            ASSERT_FLOAT_EQ(expected_output_vec[x+output_size*output_index], output[x]);
        }
        output_index++;
    }
}

/* This test shows how to use private members (here: add_connection) of program using program_wraper */
// Disabled for now as it produces wrong results
TEST(add_intermediate_gpu, test2)
{
    topology topology;
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);

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

    topology.add(cldnn::convolution("conv2a", input_info("input"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(cldnn::convolution("conv2b", input_info("input"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    auto new_conv = std::make_shared<convolution>("conv1a", input_info("input"), "weights", "", 1, ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, false);
    auto weights_node = std::make_shared<data>("weights", weights);
    program::ptr prog = program::build_program(engine, topology, config, false, true);

    prog->add_intermediate(new_conv, prog->get_node("conv2a"), 0, true, true);
    program_wrapper::add_connection(*prog, prog->get_or_create(weights_node), prog->get_or_create(new_conv));

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
            ASSERT_FLOAT_EQ(expected_output_vec[x], output[x]);
        }
    }
}

TEST(processing_order, bfs_order_restoring) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    if (config.get_queue_type() != QueueTypes::out_of_order)
        GTEST_SKIP();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 8, 8 } });
    auto eltwise_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto input_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });
    auto input_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });
    auto output_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto output_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(eltwise_mem,  rg.generate_random_1d<float>(1, 0, 10));
    set_values(input_low_mem,  rg.generate_random_1d<float>(3, -10, 0));
    set_values(input_high_mem, rg.generate_random_1d<float>(3, 1, 10));
    set_values(output_low_mem, { -127.0f });
    set_values(output_high_mem, { 127.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        activation("act", input_info("input"), activation_func::relu),
        data("eltwise_data", eltwise_mem),
        data("in_low", input_low_mem),
        data("in_high", input_high_mem),
        data("out_low", output_low_mem),
        data("out_high", output_high_mem),
        eltwise("eltwise", { input_info("act"), input_info("eltwise_data") }, eltwise_mode::prod, data_types::f32),
        activation("act2", input_info("eltwise"), activation_func::pow),
        quantize("quant", input_info("act2"), input_info("in_low"), input_info("in_high"),
                 input_info("out_low"), input_info("out_high"), 256, data_types::u8),
        activation("act3", input_info("quant"), activation_func::relu),
        reorder("reorder_bfyx1", input_info("act3"), format::bfyx, data_types::f32),
        activation("act4", input_info("quant"), activation_func::floor),
        reorder("reorder_bfyx2", input_info("act4"), format::bfyx, data_types::f32)
    );

    auto prog = program::build_program(engine, topology, config);
    const auto& processing_order = prog->get_processing_order();

    auto act3_processing_number = processing_order.get_processing_number(prog->get_node_ptr("act3").get());
    auto act4_processing_number = processing_order.get_processing_number(prog->get_node_ptr("act4").get());

    // Make sure activations (act3 and act4) on parallel branches execute one straight after another
    auto processing_number_diff = std::abs(act3_processing_number - act4_processing_number);

    ASSERT_EQ(processing_number_diff, 1);
}
