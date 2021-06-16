// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "cldnn/runtime/engine.hpp"

#include "program_impl.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "network_impl.h"
#include "pass_manager.h"

#include "program_impl_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;


std::map<primitive_id, network_output> test_prepare_conv_eltw_fusing(bool eltw1, bool eltw2)
{
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(true));

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.1f, 1.2f, 1.3f, 1.4f });
    set_values(weights1, { 2.1f});
    set_values(weights2, { -1.5f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(convolution("conv1", { "input" }, { "weights1" }));
    topology.add(convolution("conv2", { "input" }, { "weights2" }));
    if (eltw1)
    {
        topology.add(eltwise("eltw1_no_relu", "conv1", "conv2", cldnn::eltwise_mode::sum));
        topology.add(activation("eltw1", "eltw1_no_relu", activation_func::relu));
    }
    if (eltw2)
    {
        topology.add(eltwise("eltw2_no_relu", "conv2", "conv1", cldnn::eltwise_mode::sum));
        topology.add(activation("eltw2", "eltw2_no_relu", activation_func::relu));
    }
    if (eltw1 && eltw2)
    {
        topology.add(eltwise("eltw3", "eltw1", "eltw2", cldnn::eltwise_mode::prod));
    }
    else if (eltw1)
    {
        topology.add(eltwise("eltw3", "eltw1", "conv2", cldnn::eltwise_mode::prod));
    }
    else if (eltw2)
    {
        topology.add(eltwise("eltw3", "eltw2", "conv2", cldnn::eltwise_mode::prod));
    }
    else
    {
        topology.add(eltwise("eltw3", "conv1", "conv2", cldnn::eltwise_mode::sum));
    }
    program_impl::ptr prog = program_impl::build_program(engine, *topology.get(), build_opt, false, true);

    layout_optimizer lo;
    program_impl_wrapper::apply_opt_pass<prepare_conv_eltw_fusing>(*prog, lo);

    program_impl_wrapper::run_graph_compilation(*prog);
    program_impl_wrapper::prepare_memory_dependencies(*prog);
    program_impl_wrapper::compile(*prog);
    program_impl_wrapper::init_kernels(*prog);
    std::shared_ptr<cldnn::network_impl> net = network_impl::allocate_network(engine, prog);
    network network(net);
    network.set_input_data("input", input);

    return network.execute();
}

/*
Create a network with three eltwise nodes:
     /-> Convolution -> \ / -> Eltwise  -\
Input                    X                 Eltwise
     \-> Convolution -> / \ -> Eltwise  -/
*/
TEST(prepare_conv_eltw_fusing, testlp)
{
    auto outputs = test_prepare_conv_eltw_fusing(true, true);
    float ref_out[] = { 0.4356f, 0.5184f, 0.6084f, 0.7056f };
    float epsilon = 1e-3f;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (int i = 0; i < 4; i++)
            EXPECT_NEAR(ref_out[i], output[i], epsilon);
    }
    return;
}

/*
Create a network with two eltwise nodes:
     /-> Convolution -> \
Input                    \
     \-> Convolution ---> \ -> Eltwise  ->  Eltwise
*/
TEST(prepare_conv_eltw_fusing, testl)
{
    auto outputs = test_prepare_conv_eltw_fusing(true, false);
    float ref_out[] = { -1.089f, -1.296f, -1.521f, -1.764f };
    float epsilon = 1e-3f;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (int i = 0; i < 4; i++)
            EXPECT_NEAR(ref_out[i], output[i], epsilon);
    }
    return;
}

/*
Create a network with two eltwise nodes
(symetric to the previous one but the second conv should be merged with eltwise):
     /-> Convolution ---> / -> Eltwise  ->  Eltwise
Input                    /
     \-> Convolution -> /
*/
TEST(prepare_conv_eltw_fusing, testp)
{
    auto outputs = test_prepare_conv_eltw_fusing(false, true);
    float ref_out[] = { -1.089f, -1.296f, -1.521f, -1.764f };
    float epsilon = 1e-3f;
    for (auto& it : outputs)
    {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (int i = 0; i < 4; i++)
            EXPECT_NEAR(ref_out[i], output[i], epsilon);
    }
    return;
}
