// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "cldnn/runtime/engine.hpp"

#include "data_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "pass_manager.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;


std::map<primitive_id, network_output> test_conv_reorder() {
    build_options build_opt;

    implementation_desc conv1_impl = { format::bfyx, "convolution_gpu_bfyx_os_iyx_osv16" };
    implementation_desc conv2_impl = { format::fs_b_yx_fsv32, "" };
    implementation_desc reor1_impl = { format::bfwzyx, "reorder_data" };

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 32, 8, 8 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::bfyx,{ 16, 32, 1, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f16, format::bfyx,{ 16, 16, 3, 3 } });

    std::vector<FLOAT16> input_data(input->count(), FLOAT16(1.0f));
    set_values(input, input_data);

    std::vector<FLOAT16> weights1_data(weights1->count(), FLOAT16(1.0f));
    set_values(weights1, weights1_data);

    std::vector<FLOAT16> weights2_data(weights2->count(), FLOAT16(1.0f));
    set_values(weights2, weights2_data);

    build_opt.set_option(build_option::force_implementations({ {"conv1", conv1_impl}, {"conv2", conv2_impl}, {"reorder1", reor1_impl}}));
    build_opt.set_option(build_option::optimize_data(true));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(convolution("conv1", "input", { "weights1" }));

    topology.add(reorder("reorder1", "conv1",  {data_types::f16, format::bfwzyx,{ 2,16,8,8,1,1 } }));
    topology.add(reshape("reshape1", "reorder1", tensor(batch(2), feature(16), spatial(2, 2, 4, 4))));
    topology.add(activation("act", "reshape1", activation_func::relu));

    topology.add(convolution("conv2", "conv1", { "weights2" }, {1, 1, 1, 1}, {0, 0, -1, -1, 0, 0}));
    topology.add(reorder("reorder2", "conv2",  {data_types::f16, format::bfwzyx,{ 2,16,8,8,1,1 } }));
    topology.add(reshape("reshape2", "reorder2", tensor(batch(2), feature(16), spatial(2, 2, 4, 4))));
    network network(engine, topology, build_opt);

    network.set_input_data("input", input);

    return network.execute();
}

/*
Create a network:
                      /-> Convolution_2 -> Reorder_2 -> Reshape_2
Input -> Convolution_1                           
                      \-> Reorder_1 -> Reshape_1 -> Activate -> Validate
*/
TEST(test_conv_reorder, testlp) {
    auto outputs = test_conv_reorder();
    float epsilon = 1e-3f;
    cldnn::mem_lock<uint16_t> output(outputs.begin()->second.get_memory(), get_test_stream());
    for (int i = 0; i < 2*16*64; i++){
        ASSERT_NEAR(FLOAT16(32.0f), float16_to_float32(output[i]), epsilon);
    }
}
