// Copyright (C) 2022-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <vector>
#include <iostream>

#include "primitive_inst.h"

using namespace cldnn;
using namespace ::tests;


TEST(multistream_gpu, basic) {
    const int num_streams = 2;
    auto task_config = ov::threading::IStreamsExecutor::Config();
    task_config._streams = num_streams;
    auto task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(task_config);
    auto& engine = get_test_engine();

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

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

    auto prog_ptr = program::build_program(engine, topology, config);
    auto &node = prog_ptr->get_node("shape_of");
    auto strm = node.get_kernel_impl_params()->get_stream_ptr();
    ASSERT_EQ(prog_ptr->get_stream_ptr(), strm);

    std::vector<network::ptr> networks;
    std::vector<stream::ptr> streams;
    for (size_t i = 0; i < num_streams; i++) {
        networks.push_back(network::allocate_network(engine, prog_ptr));
        streams.push_back(networks[i]->get_stream_ptr());
    }

    std::vector<ov::threading::Task> tasks;
    for (size_t i = 0; i < num_streams; i++) {
        tasks.push_back([&networks, &streams, i, &engine] {
            auto cfg = get_test_default_config(engine);
            auto stream = engine.create_stream(cfg);
            auto net = networks[i];
            std::vector<int> various_size = {32, 128, 16, 64};
            for (size_t iter = 0; iter < 8; iter++) {
                int len = various_size[iter % various_size.size()];
                auto input1_mem = engine.allocate_memory({ ov::PartialShape{1,len,512}, data_types::f16,format::bfyx });
                auto input2_mem = engine.allocate_memory({ ov::PartialShape{1,len,512}, data_types::f16,format::bfyx });
                net->set_input_data("input1", input1_mem);
                net->set_input_data("input2", input2_mem);

                auto outputs = net->execute();

                auto inst = net->get_primitive("shape_of");
                auto strm = inst->get_impl_params()->get_stream_ptr();
                ASSERT_EQ(streams[i], strm);

                auto output = outputs.at("shape_of").get_memory();
                cldnn::mem_lock<int32_t> output_ptr(output, *stream);

                std::vector<int32_t> expected_results = {1, len, 512};

                for (size_t out_idx = 0; out_idx < expected_results.size(); ++out_idx) {
                    ASSERT_TRUE(are_equal(expected_results[out_idx], output_ptr[out_idx]));
                }
            }
        });
    }

    task_executor->run_and_wait(tasks);
    tasks.clear();
    networks.clear();
}
