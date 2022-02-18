// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/concatenation.hpp"
#include <intel_gpu/primitives/data.hpp>

using namespace cldnn;
using namespace ::tests;

/*
    This set of tests has been designed to check the correctness of trim_to_outputs optimization pass
*/

/*
   In this test we check if the convolution conv2 will be eliminated from the network. This is expected to be done in trim_to_outputs optimization pass

   Network structure:  input  -> conv1 (output)
                           \
                            ---> conv2 (to be eliminated)
*/
TEST(trim_to_outputs, one_node_to_eliminate_case1) {
    auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(cldnn::build_option::outputs({ "conv1" }));
    build_opt.set_option(build_option::optimize_data(false));             // to avoid adding reorders

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 1, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto bias = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.1f });
    set_values(weights, { 2.1f });
    set_values(bias, { 1.6f });

    std::vector<float> out_data = { 3.91f };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights", weights));
    topology.add(data("bias", bias));
    topology.add(cldnn::convolution("conv1", { "input" }, { "weights" }, { "bias" }));
    topology.add(cldnn::convolution("conv2", { "input" }, { "weights" }, { "bias" }));

    network network(engine, topology, build_opt);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), (size_t)1); // there is only one output
    EXPECT_EQ(network.get_executed_primitives().size(), (size_t)2);   // input and conv1 where executed
    EXPECT_EQ(network.get_all_primitive_ids().size(), (size_t)4);     // also bias and weights still exist

    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output_ptr(it.second.get_memory(), get_test_stream());
        for (size_t cntr = 0; cntr < out_data.size(); cntr++)
        {
            EXPECT_NEAR(output_ptr[cntr], out_data[cntr], 1e-4);
        }
        EXPECT_EQ(it.first, "conv1");
    }
}

/*
in this test we check if the convolution conv2 will be eliminated from the network. This is expected to be done in trim_to_outputs optimization pass

Network structure:  input  -> conv1 (output)
                        \
                         ---> conv2 (to be eliminated along with its weights and bias)
*/
TEST(trim_to_outputs, one_node_to_eliminate_case2) {
    auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(cldnn::build_option::outputs({ "conv1" }));
    build_opt.set_option(build_option::optimize_data(false));             // to avoid adding reorders

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 1, 1 } });
    auto weights1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto bias1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto bias2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.1f });
    set_values(weights1, { 2.1f });
    set_values(bias1, { 1.6f });
    set_values(weights2, { 0.3f });
    set_values(bias2, { 0.2f });

    std::vector<float> out_data = { 3.91f };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("bias1", bias1));
    topology.add(cldnn::convolution("conv1", { "input" }, { "weights1" }, { "bias1" }));
    topology.add(data("weights2", weights2));
    topology.add(data("bias2", bias2));
    topology.add(cldnn::convolution("conv2", { "input" }, { "weights2" }, { "bias2" }));

    network network(engine, topology, build_opt);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), (size_t)1); // there is only one output
    EXPECT_EQ(network.get_executed_primitives().size(), (size_t)2);   // input and conv1 where executed
    EXPECT_EQ(network.get_all_primitive_ids().size(), (size_t)4);     // also bias1 and weights1 still exist

    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output_ptr(it.second.get_memory(), get_test_stream());

        for (size_t cntr = 0; cntr < out_data.size(); cntr++)
        {
            EXPECT_NEAR(output_ptr[cntr], out_data[cntr], 1e-4);
        }
        EXPECT_EQ(it.first, "conv1");
    }
}

/*
in this test we check if the convolution conv2 will be eliminated from the network. This is expected to be done in trim_to_outputs optimization pass

Network structure:  input ---> conv1 --- ---> conv4 (output)
                        \
                         --->  conv2  ---> conv3
Convolutions conv2, conv3 should be optimized out along with weights23 shered by conv2 and conv3.
*/
TEST(trim_to_outputs, two_nodes_to_eliminate_case1) {
    auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(cldnn::build_option::outputs({ "conv4" }));
    build_opt.set_option(build_option::optimize_data(false));             // to avoid adding reorders

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 1, 1 } });
    auto weights1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto weights23 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto weights4 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto bias = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.1f });
    set_values(weights1, { 2.1f });
    set_values(weights23, { 3.0f });
    set_values(weights4, { 2.0f });
    set_values(bias, { 1.6f });

    std::vector<float> out_data = { 9.42f };

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("bias", bias));
    topology.add(cldnn::convolution("conv1", { "input" }, { "weights1" }, { "bias" }));
    topology.add(data("weights23", weights23));
    topology.add(cldnn::convolution("conv2", { "input" }, { "weights23" }, { "bias" }));
    topology.add(cldnn::convolution("conv3", { "conv2" }, { "weights23" }, { "bias" }));
    topology.add(data("weights4", weights4));
    topology.add(cldnn::convolution("conv4", { "conv1" }, { "weights4" }, { "bias" }));

    network network(engine, topology, build_opt);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), (size_t)1); // there is only one output
    EXPECT_EQ(network.get_executed_primitives().size(), (size_t)3);   // input, conv1 and conv4  where executed
    EXPECT_EQ(network.get_all_primitive_ids().size(), (size_t)6);     // also bias weights1 and weights4 still exist

    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output_ptr(it.second.get_memory(), get_test_stream());

        for (size_t cntr = 0; cntr < out_data.size(); cntr++)
        {
            EXPECT_NEAR(output_ptr[cntr], out_data[cntr], 1e-4);
        }
        EXPECT_EQ(it.first, "conv4");
    }
}
