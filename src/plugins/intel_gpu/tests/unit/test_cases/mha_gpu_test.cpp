// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mha.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

void test_simple_input(bool is_caching_test);

void test_simple_input(bool is_caching_test) {
    // # N = 2
    // # b = 1
    // # d = 3
    // q = np.array([5, 2, 1,
    //               1, 2, 5]).reshape([1, 1, 1, 2, 3]) # b x N x d

    // k = np.array([1, 5,
    //               2, 2,
    //               5, 1]).reshape([1, 1, 1, 3, 2]) # b x d x N

    // v = np.array([5, 0, 5,
    //               0, 5, 5]).reshape([1, 1, 1, 2, 3]) # b x N x d
    // return (q, k, v)

    // =simple r:
    // [[[[[5.62675810e-07 4.99999944e+00 5.00000000e+00]
    //     [4.99999944e+00 5.62675810e-07 5.00000000e+00]]]]]
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 1, 3, 2 } }); // query
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 1, 2, 3 } }); // key
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 1, 3, 2 } }); // value

    set_values(input1, {
        FLOAT16(5.0f), FLOAT16(2.0f), FLOAT16(1.0f),
        FLOAT16(3.0f), FLOAT16(7.0f), FLOAT16(9.0f),
    });

    set_values(input2, {
        FLOAT16(2.0f), FLOAT16(13.0f),
        FLOAT16(4.0f), FLOAT16(6.0f),
        FLOAT16(10.0f), FLOAT16(3.0f),
    });

    set_values(input3, {
        FLOAT16(30.0f), FLOAT16(0.0f), FLOAT16(35.0f),
        FLOAT16(0.0f), FLOAT16(45.0f), FLOAT16(55.0f),
    });

    topology topology;
    topology.add(input_layout("Query", input1->get_layout()));
    topology.add(input_layout("Key", input2->get_layout()));
    topology.add(input_layout("Value", input3->get_layout()));
    topology.add(
        mha("mha", input_info("Query"), input_info("Key"), input_info("Value"))
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Query", input1);
    network->set_input_data("Key", input2);
    network->set_input_data("Value", input3);

    auto outputs = network->execute();

    auto output = outputs.at("mha").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 45.f, 55.f,
        30.f, 0.f, 35.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_NEAR(expected_results[i], half_to_float(output_ptr[i]), 1e-3);
    }
}

TEST(mha_gpu_fp16, simple_input) {
    test_simple_input(false);
}

/* FIXME: add caching test*/