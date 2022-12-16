// Copyright (C) 2022-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <vector>
#include <iostream>

using namespace cldnn;
using namespace ::tests;


TEST(multistream_gpu, basic) {
    const int num_streams = 2;
    auto config = InferenceEngine::CPUStreamsExecutor::Config();
    config._streams = num_streams;
    auto task_executor = std::make_shared<InferenceEngine::CPUStreamsExecutor>(config);
    auto& engine = get_test_engine();

    build_options bo;
    bo.set_option(build_option::allow_new_shape_infer(true));

    auto input1_dyn_layout = layout{ ov::PartialShape::dynamic(3), data_types::f16,format::bfyx };
    auto input2_dyn_layout = layout{ ov::PartialShape::dynamic(3), data_types::f16,format::bfyx };
    auto weights    = engine.allocate_memory({ {512, 512}, data_types::f32, format::bfyx});

    topology topology;
    topology.add(input_layout("input1", input1_dyn_layout));
    topology.add(input_layout("input2", input2_dyn_layout));
    topology.add(data("weights", weights));
    topology.add(eltwise("eltwise", input_info("input1"), input_info("input2"), eltwise_mode::sum));
    topology.add(fully_connected("fc", input_info("eltwise"), "weights"));
    topology.add(shape_of("shape_of", input_info("fc"), 3, data_types::i32));

    auto prog_ptr = program::build_program(engine, topology, bo);
    std::vector<network::ptr> networks;
    for (size_t i = 0; i < num_streams; i++) {
        networks.push_back(network::allocate_network(engine, prog_ptr));
    }

    std::vector<InferenceEngine::Task> tasks;
    for (size_t i = 0; i < num_streams; i++) {
        tasks.push_back([&networks, i, &engine] {
            auto net = networks[i];
            std::vector<int> various_size = {32, 128, 16, 64};
            for (size_t iter = 0; iter < 8; iter++) {
                int len = various_size[iter % various_size.size()];
                auto input1_mem = engine.allocate_memory({ ov::PartialShape{1,len,512}, data_types::f16,format::bfyx });
                auto input2_mem = engine.allocate_memory({ ov::PartialShape{1,len,512}, data_types::f16,format::bfyx });
                net->set_input_data("input1", input1_mem);
                net->set_input_data("input2", input2_mem);

                auto outputs = net->execute();

                auto output = outputs.at("shape_of").get_memory();
                cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

                std::vector<int32_t> expected_results = {1, len, 512};

                for (size_t out_idx = 0; out_idx < expected_results.size(); ++out_idx) {
                    ASSERT_TRUE(are_equal(expected_results[out_idx], output_ptr[out_idx]));
                }
            }
        });
    }

    task_executor->runAndWait(tasks);
    tasks.clear();
    networks.clear();
}
