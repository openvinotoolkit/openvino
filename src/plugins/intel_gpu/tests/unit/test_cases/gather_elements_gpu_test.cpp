// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include "gather_elements_inst.h"

#include <cstddef>
#include <gtest/gtest.h>

using namespace cldnn;
using namespace ::tests;

inline void DoTest(engine& engine,
    const cldnn::memory::ptr& input0, // data
    const cldnn::memory::ptr& input1, // indices
    const std::vector<float>& expected_results,
    const tensor& output_tensor,
    const int64_t axis,
    bool is_caching_test=false) {
    topology topology;
    topology.add(input_layout("InputData", input0->get_layout()));
    topology.add(input_layout("InputIndices", input1->get_layout()));
    topology.add(
        gather_elements("gather_elements", input_info("InputData"), input_info("InputIndices"), input1->get_layout().format, output_tensor, axis)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("InputData", input0);
    network->set_input_data("InputIndices", input1);
    auto outputs = network->execute();
    auto output = outputs.at("gather_elements").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(gather_elements_gpu_fp16, d3283_i2283_a0) {
    auto& engine = get_test_engine();

    auto axis = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 8, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 8, 3 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8), ov::float16(5), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(5),
        ov::float16(7), ov::float16(0), ov::float16(4), ov::float16(0), ov::float16(4), ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(7), ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0), ov::float16(8), ov::float16(3), ov::float16(6), ov::float16(8), ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(10), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8), ov::float16(5), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(10), ov::float16(0), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(3),
        ov::float16(10), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(10), ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(10), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(0), ov::float16(8), ov::float16(8),
        ov::float16(9), ov::float16(1), ov::float16(0), ov::float16(7), ov::float16(9), ov::float16(6), ov::float16(8), ov::float16(7),
        ov::float16(10), ov::float16(9), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(5), ov::float16(6), ov::float16(9),
        ov::float16(4), ov::float16(9), ov::float16(2), ov::float16(4), ov::float16(5), ov::float16(5), ov::float16(3), ov::float16(1),
        ov::float16(1), ov::float16(6), ov::float16(8), ov::float16(0), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(8),
        ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(9), ov::float16(1), ov::float16(2), ov::float16(7), ov::float16(1),
        ov::float16(1), ov::float16(3), ov::float16(0), ov::float16(4), ov::float16(0), ov::float16(7), ov::float16(10), ov::float16(2),
        ov::float16(1), ov::float16(3), ov::float16(9), ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(1), ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(10), ov::float16(6), ov::float16(1),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(2),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(4), ov::float16(2), ov::float16(4), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(1), ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(9), ov::float16(0), ov::float16(5), ov::float16(3),
        ov::float16(6), ov::float16(5), ov::float16(6), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(6), ov::float16(1),
        ov::float16(3), ov::float16(5), ov::float16(5), ov::float16(4), ov::float16(4), ov::float16(7), ov::float16(8), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(7), ov::float16(9), ov::float16(8), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(1), ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(0), ov::float16(6), ov::float16(1),
        ov::float16(2), ov::float16(9), ov::float16(2), ov::float16(4), ov::float16(5), ov::float16(2), ov::float16(3), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(4), ov::float16(0), ov::float16(5), ov::float16(0), ov::float16(5), ov::float16(3),
        ov::float16(6), ov::float16(9), ov::float16(2), ov::float16(0), ov::float16(4), ov::float16(2), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(3), ov::float16(0), ov::float16(4), ov::float16(10), ov::float16(7), ov::float16(10), ov::float16(2),
        ov::float16(9), ov::float16(3), ov::float16(0), ov::float16(7), ov::float16(6), ov::float16(8), ov::float16(8), ov::float16(4),
        ov::float16(2), ov::float16(10), ov::float16(7), ov::float16(3), ov::float16(3), ov::float16(10), ov::float16(6), ov::float16(1),
    };

    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 8, 3), axis);
}

TEST(gather_elements_gpu_fp16, d2235_i2235_a3) {
    auto& engine = get_test_engine();

    auto axis = 3;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 3, 5 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 3, 5 } }); // indices
    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(5), ov::float16(5), ov::float16(2),
        ov::float16(0), ov::float16(7), ov::float16(7),
        ov::float16(10), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(0),
        ov::float16(5), ov::float16(7), ov::float16(0),
        ov::float16(4), ov::float16(0), ov::float16(4),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(4), ov::float16(7),
        ov::float16(10), ov::float16(8), ov::float16(2),
        ov::float16(0), ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8), ov::float16(10),
        ov::float16(4), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(7),
        ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8),
        ov::float16(5), ov::float16(2), ov::float16(3),
        ov::float16(3), ov::float16(1), ov::float16(5),
        ov::float16(9), ov::float16(10), ov::float16(0),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(2), ov::float16(2), ov::float16(5),
        ov::float16(0), ov::float16(0), ov::float16(7),
        ov::float16(10), ov::float16(10), ov::float16(10),
        ov::float16(0), ov::float16(9), ov::float16(0),
        ov::float16(7), ov::float16(0), ov::float16(7),
        ov::float16(4), ov::float16(0), ov::float16(4),
        ov::float16(6), ov::float16(7), ov::float16(10),
        ov::float16(5), ov::float16(9), ov::float16(5),
        ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(8), ov::float16(2), ov::float16(2),
        ov::float16(8), ov::float16(8), ov::float16(8),
        ov::float16(8), ov::float16(6), ov::float16(10),
        ov::float16(4), ov::float16(10), ov::float16(10),
        ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(0), ov::float16(0), ov::float16(9),
        ov::float16(4), ov::float16(8), ov::float16(8),
        ov::float16(3), ov::float16(3), ov::float16(5),
        ov::float16(5), ov::float16(3), ov::float16(3),
        ov::float16(9), ov::float16(9), ov::float16(0),
    };

    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 3, 5), axis);
}

TEST(gather_elements_gpu_fp16, d1329_i1359_an1) {
    auto& engine = get_test_engine();

    auto axis = 3;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 3, 2, 9 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 3, 5, 9 } }); // indices
    set_values(input0, {
        ov::float16(0), ov::float16(1),
        ov::float16(8), ov::float16(5),
        ov::float16(5), ov::float16(2),
        ov::float16(0), ov::float16(7),
        ov::float16(7), ov::float16(10),
        ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0),
        ov::float16(0), ov::float16(5),
        ov::float16(7), ov::float16(0),
        ov::float16(4), ov::float16(0),
        ov::float16(4), ov::float16(7),
        ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5),
        ov::float16(1), ov::float16(7),
        ov::float16(4), ov::float16(7),
        ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0),
        ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8),
        ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8),
        ov::float16(7), ov::float16(0),
        ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4),
        ov::float16(8), ov::float16(5),
        ov::float16(2), ov::float16(3),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(5), ov::float16(8),
        ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(5),
        ov::float16(0), ov::float16(7), ov::float16(0), ov::float16(7), ov::float16(7),
        ov::float16(10), ov::float16(7), ov::float16(7), ov::float16(10), ov::float16(10),
        ov::float16(4), ov::float16(4), ov::float16(5), ov::float16(4), ov::float16(4),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(9), ov::float16(9),
        ov::float16(5), ov::float16(0), ov::float16(0), ov::float16(5), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(4), ov::float16(4), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(4), ov::float16(7),
        ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(10),
        ov::float16(5), ov::float16(9), ov::float16(5), ov::float16(9), ov::float16(5),
        ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(1), ov::float16(7),
        ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(7), ov::float16(7),
        ov::float16(8), ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(8),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(8), ov::float16(8), ov::float16(3), ov::float16(8), ov::float16(8),
        ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(8), ov::float16(6),
        ov::float16(10), ov::float16(4), ov::float16(10), ov::float16(10), ov::float16(10),
        ov::float16(10), ov::float16(2), ov::float16(2), ov::float16(10), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(0), ov::float16(7), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(6), ov::float16(9), ov::float16(9), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(2), ov::float16(4), ov::float16(2), ov::float16(4),
        ov::float16(5), ov::float16(8), ov::float16(8), ov::float16(5), ov::float16(8),
        ov::float16(3), ov::float16(3), ov::float16(2), ov::float16(3), ov::float16(3),
    };

    DoTest(engine, input0, input1, expected_results, tensor(1, 3, 5, 9), axis);
}

TEST(gather_elements_gpu_fp16, d12853_i12923_a3) {
    auto& engine = get_test_engine();

    auto axis = 3;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 2, 8, 5, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 2, 8, 2, 3 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8), ov::float16(5), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(5),
        ov::float16(7), ov::float16(0), ov::float16(4), ov::float16(0), ov::float16(4), ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(7), ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0), ov::float16(8), ov::float16(3), ov::float16(6), ov::float16(8), ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(10), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8), ov::float16(5), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(10), ov::float16(0), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(3),
        ov::float16(10), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(10), ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(10), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(0), ov::float16(8), ov::float16(8),
        ov::float16(9), ov::float16(1), ov::float16(0), ov::float16(7), ov::float16(9), ov::float16(6), ov::float16(8), ov::float16(7),
        ov::float16(10), ov::float16(9), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(5), ov::float16(6), ov::float16(9),
        ov::float16(4), ov::float16(9), ov::float16(2), ov::float16(4), ov::float16(5), ov::float16(5), ov::float16(3), ov::float16(1),
        ov::float16(1), ov::float16(6), ov::float16(8), ov::float16(0), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(8),
        ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(9), ov::float16(1), ov::float16(2), ov::float16(7), ov::float16(1),
        ov::float16(1), ov::float16(3), ov::float16(0), ov::float16(4), ov::float16(0), ov::float16(7), ov::float16(10), ov::float16(2),
        ov::float16(1), ov::float16(3), ov::float16(9), ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(1), ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(10), ov::float16(6), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1), ov::float16(6), ov::float16(2), ov::float16(5), ov::float16(5), ov::float16(10),
        ov::float16(1), ov::float16(2), ov::float16(3), ov::float16(6), ov::float16(1), ov::float16(7), ov::float16(6), ov::float16(8),
        ov::float16(2), ov::float16(5), ov::float16(4), ov::float16(2), ov::float16(0), ov::float16(9), ov::float16(4), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1), ov::float16(9), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(4),
        ov::float16(2), ov::float16(1), ov::float16(8), ov::float16(5), ov::float16(3), ov::float16(4), ov::float16(8), ov::float16(10),
        ov::float16(7), ov::float16(2), ov::float16(7), ov::float16(9), ov::float16(2), ov::float16(9), ov::float16(5), ov::float16(5),
        ov::float16(6), ov::float16(8), ov::float16(8), ov::float16(5), ov::float16(10), ov::float16(6), ov::float16(4), ov::float16(9),
        ov::float16(7), ov::float16(7), ov::float16(10), ov::float16(10), ov::float16(9), ov::float16(3), ov::float16(5), ov::float16(5),
        ov::float16(1), ov::float16(4), ov::float16(6), ov::float16(9), ov::float16(4), ov::float16(8), ov::float16(9), ov::float16(7),
        ov::float16(8), ov::float16(7), ov::float16(8), ov::float16(0), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(0),
        ov::float16(7), ov::float16(5), ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(10), ov::float16(9), ov::float16(9),
        ov::float16(5), ov::float16(1), ov::float16(4), ov::float16(10), ov::float16(2), ov::float16(4), ov::float16(3), ov::float16(5),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(2), ov::float16(4), ov::float16(3), ov::float16(4), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(4), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(3), ov::float16(1), ov::float16(4), ov::float16(2), ov::float16(4), ov::float16(2), ov::float16(1), ov::float16(3),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(4), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(3),
        ov::float16(4), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(4), ov::float16(0),
        ov::float16(3), ov::float16(4), ov::float16(3), ov::float16(4), ov::float16(4), ov::float16(1), ov::float16(0), ov::float16(3),
        ov::float16(2), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(0), ov::float16(4), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(4), ov::float16(3), ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(3), ov::float16(4), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(3), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(3), ov::float16(3), ov::float16(4), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(3),
        ov::float16(3), ov::float16(4), ov::float16(3), ov::float16(3), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(3),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(4), ov::float16(0), ov::float16(4),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(0), ov::float16(8), ov::float16(7), ov::float16(6), ov::float16(2), ov::float16(0), ov::float16(5),
        ov::float16(2), ov::float16(1), ov::float16(4), ov::float16(5), ov::float16(9), ov::float16(2), ov::float16(0), ov::float16(5),
        ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(0), ov::float16(10), ov::float16(5), ov::float16(3), ov::float16(4),
        ov::float16(5), ov::float16(4), ov::float16(10), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(6), ov::float16(9), ov::float16(2), ov::float16(4), ov::float16(5), ov::float16(6), ov::float16(7), ov::float16(7),
        ov::float16(1), ov::float16(9), ov::float16(8), ov::float16(9), ov::float16(1), ov::float16(5), ov::float16(8), ov::float16(8),
        ov::float16(5), ov::float16(2), ov::float16(3), ov::float16(6), ov::float16(1), ov::float16(7), ov::float16(6), ov::float16(2),
        ov::float16(1), ov::float16(3), ov::float16(0), ov::float16(6), ov::float16(2), ov::float16(7), ov::float16(6), ov::float16(1),
        ov::float16(7), ov::float16(8), ov::float16(8), ov::float16(5), ov::float16(0), ov::float16(9), ov::float16(0), ov::float16(4),
        ov::float16(2), ov::float16(2), ov::float16(7), ov::float16(5), ov::float16(3), ov::float16(9), ov::float16(4), ov::float16(5),
        ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(7), ov::float16(4), ov::float16(8), ov::float16(5), ov::float16(9),
        ov::float16(1), ov::float16(7), ov::float16(10), ov::float16(0), ov::float16(9), ov::float16(4), ov::float16(5), ov::float16(5),
    };

    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 8, 2, 3), axis);
}

TEST(gather_elements_gpu_fp16, d25441_i22441_an4) {
    auto& engine = get_test_engine();

    auto axis = 1;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 5, 4, 4, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 4, 4, 1 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8), ov::float16(5),
        ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(5),
        ov::float16(7), ov::float16(0), ov::float16(4), ov::float16(0),
        ov::float16(4), ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1), ov::float16(7),
        ov::float16(4), ov::float16(7), ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0), ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8), ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(10), ov::float16(7), ov::float16(8),
        ov::float16(7), ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8), ov::float16(5),
        ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(10), ov::float16(0),
        ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(3),
        ov::float16(10), ov::float16(5), ov::float16(2), ov::float16(0),
        ov::float16(10), ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(10), ov::float16(5), ov::float16(5),
        ov::float16(10), ov::float16(0), ov::float16(8), ov::float16(8),
        ov::float16(9), ov::float16(1), ov::float16(0), ov::float16(7),
        ov::float16(9), ov::float16(6), ov::float16(8), ov::float16(7),
        ov::float16(10), ov::float16(9), ov::float16(2), ov::float16(3),
        ov::float16(3), ov::float16(5), ov::float16(6), ov::float16(9),
        ov::float16(4), ov::float16(9), ov::float16(2), ov::float16(4),
        ov::float16(5), ov::float16(5), ov::float16(3), ov::float16(1),
        ov::float16(1), ov::float16(6), ov::float16(8), ov::float16(0),
        ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(8),
        ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(9),
        ov::float16(1), ov::float16(2), ov::float16(7), ov::float16(1),
        ov::float16(1), ov::float16(3), ov::float16(0), ov::float16(4),
        ov::float16(0), ov::float16(7), ov::float16(10), ov::float16(2),
        ov::float16(1), ov::float16(3), ov::float16(9), ov::float16(7),
        ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(1), ov::float16(6), ov::float16(9),
        ov::float16(6), ov::float16(10), ov::float16(6), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1), ov::float16(6),
        ov::float16(2), ov::float16(5), ov::float16(5), ov::float16(10),
        ov::float16(1), ov::float16(2), ov::float16(3), ov::float16(6),
        ov::float16(1), ov::float16(7), ov::float16(6), ov::float16(8),

    });

    set_values(input1, {
        ov::float16(0), ov::float16(2), ov::float16(4), ov::float16(3),
        ov::float16(4), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(4), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(3), ov::float16(1), ov::float16(4), ov::float16(2),
        ov::float16(4), ov::float16(2), ov::float16(1), ov::float16(3),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(4),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(3),
        ov::float16(4), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(4), ov::float16(0),
        ov::float16(3), ov::float16(4), ov::float16(3), ov::float16(4),
        ov::float16(4), ov::float16(1), ov::float16(0), ov::float16(3),
        ov::float16(2), ov::float16(4), ov::float16(4), ov::float16(4),
        ov::float16(4), ov::float16(0), ov::float16(4), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(4),
        ov::float16(3), ov::float16(0), ov::float16(2), ov::float16(4),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(5),
        ov::float16(10), ov::float16(2), ov::float16(0), ov::float16(10),
        ov::float16(3), ov::float16(10), ov::float16(1), ov::float16(5),
        ov::float16(4), ov::float16(0), ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(3),
        ov::float16(10), ov::float16(8), ov::float16(6), ov::float16(1),
        ov::float16(2), ov::float16(5), ov::float16(7), ov::float16(5),
        ov::float16(4), ov::float16(0), ov::float16(6), ov::float16(3),
        ov::float16(10), ov::float16(9), ov::float16(6), ov::float16(9),
        ov::float16(1), ov::float16(6), ov::float16(5), ov::float16(7),
        ov::float16(5), ov::float16(2), ov::float16(6), ov::float16(6),
        ov::float16(1), ov::float16(5), ov::float16(6), ov::float16(1),
        ov::float16(6), ov::float16(4), ov::float16(1), ov::float16(6),
        ov::float16(2), ov::float16(6), ov::float16(5), ov::float16(7),
        ov::float16(1), ov::float16(9), ov::float16(2), ov::float16(6),
        ov::float16(6), ov::float16(5), ov::float16(10), ov::float16(8),
    };

    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 4, 4, 1), axis);
}

TEST(gather_elements_gpu_fp16, d32843_i12843_a0) {
    auto& engine = get_test_engine();

    auto axis = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 8, 4, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 2, 8, 4, 3 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8), ov::float16(5), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(5),
        ov::float16(7), ov::float16(0), ov::float16(4), ov::float16(0), ov::float16(4), ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(7), ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0), ov::float16(8), ov::float16(3), ov::float16(6), ov::float16(8), ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(10), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8), ov::float16(5), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(10), ov::float16(0), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(3),
        ov::float16(10), ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(10), ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(10), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(0), ov::float16(8), ov::float16(8),
        ov::float16(9), ov::float16(1), ov::float16(0), ov::float16(7), ov::float16(9), ov::float16(6), ov::float16(8), ov::float16(7),
        ov::float16(10), ov::float16(9), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(5), ov::float16(6), ov::float16(9),
        ov::float16(4), ov::float16(9), ov::float16(2), ov::float16(4), ov::float16(5), ov::float16(5), ov::float16(3), ov::float16(1),
        ov::float16(1), ov::float16(6), ov::float16(8), ov::float16(0), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(8),
        ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(9), ov::float16(1), ov::float16(2), ov::float16(7), ov::float16(1),
        ov::float16(1), ov::float16(3), ov::float16(0), ov::float16(4), ov::float16(0), ov::float16(7), ov::float16(10), ov::float16(2),
        ov::float16(1), ov::float16(3), ov::float16(9), ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(1), ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(10), ov::float16(6), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1), ov::float16(6), ov::float16(2), ov::float16(5), ov::float16(5), ov::float16(10),
        ov::float16(1), ov::float16(2), ov::float16(3), ov::float16(6), ov::float16(1), ov::float16(7), ov::float16(6), ov::float16(8),
        ov::float16(2), ov::float16(5), ov::float16(4), ov::float16(2), ov::float16(0), ov::float16(9), ov::float16(4), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1), ov::float16(9), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(4),
        ov::float16(2), ov::float16(1), ov::float16(8), ov::float16(5), ov::float16(3), ov::float16(4), ov::float16(8), ov::float16(10),
        ov::float16(7), ov::float16(2), ov::float16(7), ov::float16(9), ov::float16(2), ov::float16(9), ov::float16(5), ov::float16(5),
        ov::float16(6), ov::float16(8), ov::float16(8), ov::float16(5), ov::float16(10), ov::float16(6), ov::float16(4), ov::float16(9),
        ov::float16(7), ov::float16(7), ov::float16(10), ov::float16(10), ov::float16(9), ov::float16(3), ov::float16(5), ov::float16(5),
        ov::float16(1), ov::float16(4), ov::float16(6), ov::float16(9), ov::float16(4), ov::float16(8), ov::float16(9), ov::float16(7),
        ov::float16(8), ov::float16(7), ov::float16(8), ov::float16(0), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(0),
        ov::float16(7), ov::float16(5), ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(10), ov::float16(9), ov::float16(9),
        ov::float16(5), ov::float16(1), ov::float16(4), ov::float16(10), ov::float16(2), ov::float16(4), ov::float16(3), ov::float16(5),
        ov::float16(9), ov::float16(4), ov::float16(5), ov::float16(8), ov::float16(4), ov::float16(2), ov::float16(10), ov::float16(1),
        ov::float16(6), ov::float16(6), ov::float16(0), ov::float16(0), ov::float16(8), ov::float16(8), ov::float16(3), ov::float16(4),
        ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(9), ov::float16(7), ov::float16(9), ov::float16(1), ov::float16(0),
        ov::float16(8), ov::float16(6), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(4), ov::float16(10), ov::float16(10),
        ov::float16(4), ov::float16(2), ov::float16(7), ov::float16(3), ov::float16(8), ov::float16(8), ov::float16(4), ov::float16(3),
        ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(10), ov::float16(2), ov::float16(9), ov::float16(1), ov::float16(4),
        ov::float16(6), ov::float16(1), ov::float16(9), ov::float16(1), ov::float16(10), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(6), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(6),
        ov::float16(0), ov::float16(6), ov::float16(2), ov::float16(3), ov::float16(7), ov::float16(1), ov::float16(8), ov::float16(5),
        ov::float16(6), ov::float16(6), ov::float16(3), ov::float16(7), ov::float16(1), ov::float16(1), ov::float16(5), ov::float16(9),
        ov::float16(8), ov::float16(6), ov::float16(8), ov::float16(3), ov::float16(1), ov::float16(5), ov::float16(3), ov::float16(6),
        ov::float16(5), ov::float16(4), ov::float16(2), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(0), ov::float16(4), ov::float16(2), ov::float16(7), ov::float16(7), ov::float16(5), ov::float16(8),
        ov::float16(7), ov::float16(10), ov::float16(5), ov::float16(10), ov::float16(3), ov::float16(5), ov::float16(5), ov::float16(7),
        ov::float16(4), ov::float16(6), ov::float16(10), ov::float16(1), ov::float16(7), ov::float16(3), ov::float16(5), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(3), ov::float16(7), ov::float16(6), ov::float16(10), ov::float16(2), ov::float16(10),
        ov::float16(2), ov::float16(9), ov::float16(7), ov::float16(5), ov::float16(8), ov::float16(0), ov::float16(1), ov::float16(7),
        ov::float16(7), ov::float16(4), ov::float16(6), ov::float16(8), ov::float16(10), ov::float16(7), ov::float16(3), ov::float16(8),
        ov::float16(1), ov::float16(0), ov::float16(5), ov::float16(0), ov::float16(1), ov::float16(9), ov::float16(8), ov::float16(8),
        ov::float16(4), ov::float16(0), ov::float16(6), ov::float16(5), ov::float16(0), ov::float16(5), ov::float16(4), ov::float16(2),
        ov::float16(4), ov::float16(6), ov::float16(7), ov::float16(7), ov::float16(5), ov::float16(3), ov::float16(8), ov::float16(4),
        ov::float16(7), ov::float16(3), ov::float16(0), ov::float16(1), ov::float16(5), ov::float16(8), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(7), ov::float16(3), ov::float16(0), ov::float16(5), ov::float16(5), ov::float16(5),
        ov::float16(4), ov::float16(1), ov::float16(3), ov::float16(9), ov::float16(7), ov::float16(6), ov::float16(7), ov::float16(3),
        ov::float16(0), ov::float16(10), ov::float16(5), ov::float16(0), ov::float16(9), ov::float16(0), ov::float16(4), ov::float16(5),
        ov::float16(6), ov::float16(8), ov::float16(7), ov::float16(5), ov::float16(0), ov::float16(1), ov::float16(10), ov::float16(2),
        ov::float16(3), ov::float16(6), ov::float16(6), ov::float16(1), ov::float16(6), ov::float16(10), ov::float16(3), ov::float16(9),
        ov::float16(10), ov::float16(2), ov::float16(2), ov::float16(4), ov::float16(8), ov::float16(9), ov::float16(2), ov::float16(8),
        ov::float16(7), ov::float16(4), ov::float16(2), ov::float16(7), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(6),
        ov::float16(0), ov::float16(1), ov::float16(6), ov::float16(4), ov::float16(0), ov::float16(7), ov::float16(4), ov::float16(9),
        ov::float16(1), ov::float16(10), ov::float16(0), ov::float16(0), ov::float16(5), ov::float16(8), ov::float16(10), ov::float16(2),
        ov::float16(3), ov::float16(8), ov::float16(5), ov::float16(8), ov::float16(7), ov::float16(7), ov::float16(8), ov::float16(0),
        ov::float16(2), ov::float16(2), ov::float16(6), ov::float16(7), ov::float16(6), ov::float16(4), ov::float16(2), ov::float16(2),
        ov::float16(7), ov::float16(1), ov::float16(8), ov::float16(1), ov::float16(0), ov::float16(7), ov::float16(1), ov::float16(10),
        ov::float16(5), ov::float16(6), ov::float16(10), ov::float16(0), ov::float16(6), ov::float16(7), ov::float16(5), ov::float16(0),
        ov::float16(4), ov::float16(5), ov::float16(8), ov::float16(0), ov::float16(4), ov::float16(10), ov::float16(5), ov::float16(3),
        ov::float16(4), ov::float16(8), ov::float16(2), ov::float16(1), ov::float16(4), ov::float16(10), ov::float16(10), ov::float16(2),
        ov::float16(0), ov::float16(1), ov::float16(5), ov::float16(1), ov::float16(5), ov::float16(1), ov::float16(9), ov::float16(4),
        ov::float16(4), ov::float16(3), ov::float16(7), ov::float16(6), ov::float16(9), ov::float16(8), ov::float16(9), ov::float16(7),
        ov::float16(4), ov::float16(10), ov::float16(6), ov::float16(3), ov::float16(5), ov::float16(5), ov::float16(4), ov::float16(2),
        ov::float16(0), ov::float16(4), ov::float16(5), ov::float16(3), ov::float16(1), ov::float16(2), ov::float16(8), ov::float16(5),
        ov::float16(7), ov::float16(9), ov::float16(2), ov::float16(7), ov::float16(2), ov::float16(4), ov::float16(0), ov::float16(5),

    });

    set_values(input1, {
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(8), ov::float16(5), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(4), ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(9), ov::float16(0), ov::float16(5), ov::float16(5),
        ov::float16(4), ov::float16(4), ov::float16(7), ov::float16(9), ov::float16(5), ov::float16(8), ov::float16(6), ov::float16(4),
        ov::float16(8), ov::float16(5), ov::float16(8), ov::float16(1), ov::float16(4), ov::float16(7), ov::float16(5), ov::float16(0),
        ov::float16(0), ov::float16(5), ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(8), ov::float16(5), ov::float16(4),
        ov::float16(4), ov::float16(1), ov::float16(3), ov::float16(9), ov::float16(7), ov::float16(0), ov::float16(6), ov::float16(3),
        ov::float16(9), ov::float16(10), ov::float16(5), ov::float16(0), ov::float16(9), ov::float16(3), ov::float16(4), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(10), ov::float16(5), ov::float16(0), ov::float16(5), ov::float16(3), ov::float16(4),
        ov::float16(3), ov::float16(6), ov::float16(2), ov::float16(9), ov::float16(10), ov::float16(10), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(2), ov::float16(2), ov::float16(4), ov::float16(0), ov::float16(0), ov::float16(8), ov::float16(8),
        ov::float16(4), ov::float16(4), ov::float16(7), ov::float16(7), ov::float16(9), ov::float16(6), ov::float16(4), ov::float16(6),
        ov::float16(10), ov::float16(9), ov::float16(2), ov::float16(10), ov::float16(2), ov::float16(7), ov::float16(6), ov::float16(9),
        ov::float16(1), ov::float16(9), ov::float16(2), ov::float16(4), ov::float16(10), ov::float16(5), ov::float16(3), ov::float16(2),
        ov::float16(3), ov::float16(6), ov::float16(5), ov::float16(0), ov::float16(5), ov::float16(8), ov::float16(7), ov::float16(8),
        ov::float16(0), ov::float16(6), ov::float16(2), ov::float16(9), ov::float16(6), ov::float16(1), ov::float16(7), ov::float16(2),
        ov::float16(1), ov::float16(3), ov::float16(3), ov::float16(7), ov::float16(0), ov::float16(7), ov::float16(5), ov::float16(9),
        ov::float16(8), ov::float16(3), ov::float16(10), ov::float16(3), ov::float16(1), ov::float16(5), ov::float16(4), ov::float16(6),
        ov::float16(4), ov::float16(5), ov::float16(6), ov::float16(4), ov::float16(4), ov::float16(10), ov::float16(5), ov::float16(1),
        ov::float16(3), ov::float16(4), ov::float16(2), ov::float16(1), ov::float16(7), ov::float16(7), ov::float16(5), ov::float16(10),
        ov::float16(7), ov::float16(1), ov::float16(5), ov::float16(10), ov::float16(3), ov::float16(1), ov::float16(5), ov::float16(4),
        ov::float16(2), ov::float16(3), ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(8), ov::float16(5), ov::float16(5),
        ov::float16(4), ov::float16(4), ov::float16(3), ov::float16(3), ov::float16(5), ov::float16(10), ov::float16(4), ov::float16(2),
        ov::float16(2), ov::float16(9), ov::float16(7), ov::float16(5), ov::float16(3), ov::float16(4), ov::float16(8), ov::float16(5),
        ov::float16(7), ov::float16(4), ov::float16(6), ov::float16(8), ov::float16(2), ov::float16(7), ov::float16(3), ov::float16(5),
    };

    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 8, 4, 3), axis);
}

TEST(gather_elements_gpu_fp16, d223442_i226442_a5) {
    auto& engine = get_test_engine();

    auto axis = 5;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 3, 4, 4, 2 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 6, 4, 4, 2 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(5), ov::float16(5), ov::float16(2),
        ov::float16(0), ov::float16(7), ov::float16(7),
        ov::float16(10), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(0),
        ov::float16(5), ov::float16(7), ov::float16(0),
        ov::float16(4), ov::float16(0), ov::float16(4),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(4), ov::float16(7),
        ov::float16(10), ov::float16(8), ov::float16(2),
        ov::float16(0), ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8), ov::float16(10),
        ov::float16(4), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(7),
        ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8),
        ov::float16(5), ov::float16(2), ov::float16(3),
        ov::float16(3), ov::float16(1), ov::float16(5),
        ov::float16(9), ov::float16(10), ov::float16(0),
        ov::float16(9), ov::float16(5), ov::float16(5),
        ov::float16(3), ov::float16(10), ov::float16(5),
        ov::float16(2), ov::float16(0), ov::float16(10),
        ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(10), ov::float16(5),
        ov::float16(5), ov::float16(10), ov::float16(0),
        ov::float16(8), ov::float16(8), ov::float16(9),
        ov::float16(1), ov::float16(0), ov::float16(7),
        ov::float16(9), ov::float16(6), ov::float16(8),
        ov::float16(7), ov::float16(10), ov::float16(9),
        ov::float16(2), ov::float16(3), ov::float16(3),
        ov::float16(5), ov::float16(6), ov::float16(9),
        ov::float16(4), ov::float16(9), ov::float16(2),
        ov::float16(4), ov::float16(5), ov::float16(5),
        ov::float16(3), ov::float16(1), ov::float16(1),
        ov::float16(6), ov::float16(8), ov::float16(0),
        ov::float16(5), ov::float16(5), ov::float16(10),
        ov::float16(8), ov::float16(6), ov::float16(9),
        ov::float16(6), ov::float16(9), ov::float16(1),
        ov::float16(2), ov::float16(7), ov::float16(1),
        ov::float16(1), ov::float16(3), ov::float16(0),
        ov::float16(4), ov::float16(0), ov::float16(7),
        ov::float16(10), ov::float16(2), ov::float16(1),
        ov::float16(3), ov::float16(9), ov::float16(7),
        ov::float16(1), ov::float16(7), ov::float16(4),
        ov::float16(4), ov::float16(5), ov::float16(1),
        ov::float16(6), ov::float16(9), ov::float16(6),
        ov::float16(10), ov::float16(6), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1),
        ov::float16(6), ov::float16(2), ov::float16(5),
        ov::float16(5), ov::float16(10), ov::float16(1),
        ov::float16(2), ov::float16(3), ov::float16(6),
        ov::float16(1), ov::float16(7), ov::float16(6),
        ov::float16(8), ov::float16(2), ov::float16(5),
        ov::float16(4), ov::float16(2), ov::float16(0),
        ov::float16(9), ov::float16(4), ov::float16(1),
        ov::float16(10), ov::float16(4), ov::float16(1),
        ov::float16(9), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(4), ov::float16(2),
        ov::float16(1), ov::float16(8), ov::float16(5),
        ov::float16(3), ov::float16(4), ov::float16(8),
        ov::float16(10), ov::float16(7), ov::float16(2),
        ov::float16(7), ov::float16(9), ov::float16(2),
        ov::float16(9), ov::float16(5), ov::float16(5),
        ov::float16(6), ov::float16(8), ov::float16(8),
        ov::float16(5), ov::float16(10), ov::float16(6),
        ov::float16(4), ov::float16(9), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(10),
        ov::float16(9), ov::float16(3), ov::float16(5),
        ov::float16(5), ov::float16(1), ov::float16(4),
        ov::float16(6), ov::float16(9), ov::float16(4),
        ov::float16(8), ov::float16(9), ov::float16(7),
        ov::float16(8), ov::float16(7), ov::float16(8),
        ov::float16(0), ov::float16(9), ov::float16(5),
        ov::float16(5), ov::float16(0), ov::float16(7),
        ov::float16(5), ov::float16(7), ov::float16(7),
        ov::float16(2), ov::float16(10), ov::float16(9),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(4), ov::float16(10), ov::float16(2),
        ov::float16(4), ov::float16(3), ov::float16(5),
        ov::float16(9), ov::float16(4), ov::float16(5),
        ov::float16(8), ov::float16(4), ov::float16(2),
        ov::float16(10), ov::float16(1), ov::float16(6),
        ov::float16(6), ov::float16(0), ov::float16(0),
        ov::float16(8), ov::float16(8), ov::float16(3),
        ov::float16(4), ov::float16(7), ov::float16(7),
        ov::float16(2), ov::float16(9), ov::float16(7),
        ov::float16(9), ov::float16(1), ov::float16(0),
        ov::float16(8), ov::float16(6), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(4),
        ov::float16(10), ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(7), ov::float16(3),
        ov::float16(8), ov::float16(8), ov::float16(4),
        ov::float16(3), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(10), ov::float16(2),
        ov::float16(9), ov::float16(1), ov::float16(4),
        ov::float16(6), ov::float16(1), ov::float16(9),
        ov::float16(1), ov::float16(10), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(6), ov::float16(7), ov::float16(8),
        ov::float16(7), ov::float16(8), ov::float16(7),
        ov::float16(6), ov::float16(0), ov::float16(6),
        ov::float16(2), ov::float16(3), ov::float16(7),
        ov::float16(1), ov::float16(8), ov::float16(5),
        ov::float16(6), ov::float16(6), ov::float16(3),
        ov::float16(7), ov::float16(1), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(8),
        ov::float16(6), ov::float16(8), ov::float16(3),
        ov::float16(1), ov::float16(5), ov::float16(3),
        ov::float16(6), ov::float16(5), ov::float16(4),
        ov::float16(2), ov::float16(4), ov::float16(4),
        ov::float16(4), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(0), ov::float16(4),
        ov::float16(2), ov::float16(7), ov::float16(7),
        ov::float16(5), ov::float16(8), ov::float16(7),
        ov::float16(10), ov::float16(5), ov::float16(10),
        ov::float16(3), ov::float16(5), ov::float16(5),
        ov::float16(7), ov::float16(4), ov::float16(6),
        ov::float16(10), ov::float16(1), ov::float16(7),
        ov::float16(3), ov::float16(5), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(3),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(2), ov::float16(10), ov::float16(2),
        ov::float16(9), ov::float16(7), ov::float16(5),
        ov::float16(8), ov::float16(0), ov::float16(1),
        ov::float16(7), ov::float16(7), ov::float16(4),
        ov::float16(6), ov::float16(8), ov::float16(10),
        ov::float16(7), ov::float16(3), ov::float16(8),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(1), ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(1), ov::float16(2),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(1), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(0),
        ov::float16(5), ov::float16(5), ov::float16(2), ov::float16(5), ov::float16(5), ov::float16(5),
        ov::float16(7), ov::float16(0), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(5), ov::float16(4), ov::float16(5), ov::float16(4), ov::float16(10), ov::float16(5),
        ov::float16(0), ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(9), ov::float16(9),
        ov::float16(7), ov::float16(0), ov::float16(0), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(0), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(4),
        ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(7), ov::float16(7), ov::float16(10),
        ov::float16(5), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(9),
        ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(2), ov::float16(10), ov::float16(8), ov::float16(8), ov::float16(2), ov::float16(2),
        ov::float16(8), ov::float16(8), ov::float16(0), ov::float16(3), ov::float16(0), ov::float16(0),
        ov::float16(6), ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(8), ov::float16(6),
        ov::float16(4), ov::float16(10), ov::float16(2), ov::float16(10), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(0), ov::float16(6), ov::float16(6), ov::float16(9), ov::float16(0), ov::float16(0),
        ov::float16(8), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(4), ov::float16(2),
        ov::float16(5), ov::float16(3), ov::float16(3), ov::float16(5), ov::float16(3), ov::float16(5),
        ov::float16(3), ov::float16(1), ov::float16(1), ov::float16(3), ov::float16(1), ov::float16(1),
        ov::float16(10), ov::float16(9), ov::float16(0), ov::float16(10), ov::float16(9), ov::float16(0),
        ov::float16(9), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(5),
        ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(3), ov::float16(5), ov::float16(10),
        ov::float16(2), ov::float16(0), ov::float16(2), ov::float16(0), ov::float16(10), ov::float16(10),
        ov::float16(0), ov::float16(5), ov::float16(4), ov::float16(4), ov::float16(5), ov::float16(0),
        ov::float16(10), ov::float16(3), ov::float16(5), ov::float16(5), ov::float16(10), ov::float16(10),
        ov::float16(10), ov::float16(5), ov::float16(10), ov::float16(0), ov::float16(10), ov::float16(10),
        ov::float16(8), ov::float16(9), ov::float16(8), ov::float16(9), ov::float16(8), ov::float16(9),
        ov::float16(7), ov::float16(0), ov::float16(0), ov::float16(7), ov::float16(0), ov::float16(0),
        ov::float16(8), ov::float16(9), ov::float16(6), ov::float16(8), ov::float16(8), ov::float16(6),
        ov::float16(9), ov::float16(9), ov::float16(7), ov::float16(10), ov::float16(10), ov::float16(10),
        ov::float16(2), ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(2), ov::float16(3),
        ov::float16(6), ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(6), ov::float16(5),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(9), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(5), ov::float16(4), ov::float16(4), ov::float16(5), ov::float16(5),
        ov::float16(3), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(8), ov::float16(8), ov::float16(6), ov::float16(6), ov::float16(0),
        ov::float16(10), ov::float16(10), ov::float16(5), ov::float16(10), ov::float16(5), ov::float16(5),
        ov::float16(6), ov::float16(8), ov::float16(9), ov::float16(9), ov::float16(8), ov::float16(9),
        ov::float16(9), ov::float16(6), ov::float16(6), ov::float16(1), ov::float16(9), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(1), ov::float16(7), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(3), ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(0), ov::float16(4), ov::float16(4), ov::float16(7), ov::float16(4), ov::float16(0),
        ov::float16(10), ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(10), ov::float16(10),
        ov::float16(3), ov::float16(3), ov::float16(3), ov::float16(9), ov::float16(9), ov::float16(7),
        ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(1), ov::float16(4), ov::float16(4),
        ov::float16(4), ov::float16(4), ov::float16(1), ov::float16(1), ov::float16(5), ov::float16(5),
        ov::float16(9), ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(9), ov::float16(9),
        ov::float16(6), ov::float16(10), ov::float16(6), ov::float16(10), ov::float16(10), ov::float16(10),
        ov::float16(1), ov::float16(10), ov::float16(1), ov::float16(10), ov::float16(1), ov::float16(10),
        ov::float16(2), ov::float16(5), ov::float16(6), ov::float16(2), ov::float16(2), ov::float16(6),
        ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(1), ov::float16(10), ov::float16(10),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(6),
        ov::float16(1), ov::float16(1), ov::float16(6), ov::float16(7), ov::float16(7), ov::float16(6),
        ov::float16(8), ov::float16(5), ov::float16(5), ov::float16(8), ov::float16(8), ov::float16(2),
        ov::float16(4), ov::float16(2), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(4),
        ov::float16(1), ov::float16(9), ov::float16(4), ov::float16(9), ov::float16(9), ov::float16(4),
        ov::float16(1), ov::float16(4), ov::float16(1), ov::float16(4), ov::float16(4), ov::float16(10),
        ov::float16(1), ov::float16(1), ov::float16(9), ov::float16(1), ov::float16(9), ov::float16(1),
        ov::float16(4), ov::float16(4), ov::float16(0), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(8), ov::float16(1), ov::float16(1), ov::float16(1), ov::float16(5), ov::float16(8),
        ov::float16(3), ov::float16(4), ov::float16(3), ov::float16(3), ov::float16(3), ov::float16(8),
        ov::float16(10), ov::float16(10), ov::float16(7), ov::float16(10), ov::float16(10), ov::float16(2),
        ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(9), ov::float16(9), ov::float16(9),
        ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(9), ov::float16(9), ov::float16(9),
        ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(8),
        ov::float16(5), ov::float16(6), ov::float16(6), ov::float16(5), ov::float16(10), ov::float16(5),
        ov::float16(7), ov::float16(9), ov::float16(7), ov::float16(7), ov::float16(9), ov::float16(7),
        ov::float16(10), ov::float16(10), ov::float16(7), ov::float16(10), ov::float16(7), ov::float16(10),
        ov::float16(5), ov::float16(3), ov::float16(9), ov::float16(3), ov::float16(9), ov::float16(3),
        ov::float16(5), ov::float16(1), ov::float16(1), ov::float16(4), ov::float16(4), ov::float16(4),
        ov::float16(9), ov::float16(9), ov::float16(9), ov::float16(4), ov::float16(6), ov::float16(6),
        ov::float16(9), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(7), ov::float16(9),
        ov::float16(8), ov::float16(8), ov::float16(7), ov::float16(8), ov::float16(8), ov::float16(8),
        ov::float16(9), ov::float16(0), ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(5), ov::float16(7), ov::float16(7), ov::float16(0), ov::float16(0),
        ov::float16(5), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(2), ov::float16(9), ov::float16(2), ov::float16(9), ov::float16(9), ov::float16(10),
        ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(1), ov::float16(5), ov::float16(9),
        ov::float16(4), ov::float16(10), ov::float16(2), ov::float16(10), ov::float16(4), ov::float16(4),
        ov::float16(5), ov::float16(3), ov::float16(4), ov::float16(3), ov::float16(4), ov::float16(5),
        ov::float16(5), ov::float16(9), ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(4),
        ov::float16(4), ov::float16(8), ov::float16(8), ov::float16(2), ov::float16(4), ov::float16(4),
        ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(1), ov::float16(10), ov::float16(6),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(6), ov::float16(0), ov::float16(0),
        ov::float16(3), ov::float16(8), ov::float16(8), ov::float16(3), ov::float16(8), ov::float16(8),
        ov::float16(4), ov::float16(7), ov::float16(4), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(9), ov::float16(2), ov::float16(7), ov::float16(9), ov::float16(7), ov::float16(7),
        ov::float16(9), ov::float16(0), ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(2), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(0), ov::float16(2), ov::float16(0),
        ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(10),
        ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(3), ov::float16(7), ov::float16(3),
        ov::float16(4), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(8), ov::float16(8),
        ov::float16(3), ov::float16(0), ov::float16(3), ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(2), ov::float16(10), ov::float16(10), ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(9), ov::float16(4), ov::float16(1), ov::float16(1), ov::float16(4), ov::float16(4),
        ov::float16(6), ov::float16(1), ov::float16(6), ov::float16(9), ov::float16(6), ov::float16(1),
        ov::float16(10), ov::float16(2), ov::float16(1), ov::float16(10), ov::float16(1), ov::float16(10),
        ov::float16(2), ov::float16(1), ov::float16(1), ov::float16(2), ov::float16(2), ov::float16(1),
        ov::float16(8), ov::float16(6), ov::float16(6), ov::float16(8), ov::float16(6), ov::float16(6),
        ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(0), ov::float16(0), ov::float16(6), ov::float16(6), ov::float16(0), ov::float16(0),
        ov::float16(7), ov::float16(3), ov::float16(3), ov::float16(2), ov::float16(7), ov::float16(3),
        ov::float16(5), ov::float16(1), ov::float16(1), ov::float16(5), ov::float16(8), ov::float16(5),
        ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(6),
        ov::float16(1), ov::float16(1), ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(7),
        ov::float16(9), ov::float16(5), ov::float16(8), ov::float16(8), ov::float16(5), ov::float16(9),
        ov::float16(6), ov::float16(8), ov::float16(8), ov::float16(6), ov::float16(6), ov::float16(6),
        ov::float16(3), ov::float16(5), ov::float16(3), ov::float16(5), ov::float16(1), ov::float16(1),
        ov::float16(6), ov::float16(5), ov::float16(4), ov::float16(5), ov::float16(6), ov::float16(5),
        ov::float16(4), ov::float16(2), ov::float16(4), ov::float16(4), ov::float16(2), ov::float16(2),
        ov::float16(4), ov::float16(5), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(4),
        ov::float16(3), ov::float16(3), ov::float16(0), ov::float16(4), ov::float16(3), ov::float16(4),
        ov::float16(7), ov::float16(7), ov::float16(2), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(5), ov::float16(7), ov::float16(8), ov::float16(7), ov::float16(5), ov::float16(5),
        ov::float16(10), ov::float16(5), ov::float16(10), ov::float16(10), ov::float16(10), ov::float16(5),
        ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(3), ov::float16(5), ov::float16(5),
        ov::float16(6), ov::float16(6), ov::float16(7), ov::float16(7), ov::float16(7), ov::float16(7),
        ov::float16(10), ov::float16(1), ov::float16(7), ov::float16(1), ov::float16(7), ov::float16(7),
        ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(5), ov::float16(3), ov::float16(5),
        ov::float16(0), ov::float16(9), ov::float16(3), ov::float16(9), ov::float16(0), ov::float16(3),
        ov::float16(6), ov::float16(6), ov::float16(6), ov::float16(10), ov::float16(10), ov::float16(6),
        ov::float16(2), ov::float16(2), ov::float16(2), ov::float16(10), ov::float16(10), ov::float16(10),
        ov::float16(5), ov::float16(9), ov::float16(7), ov::float16(7), ov::float16(5), ov::float16(9),
        ov::float16(0), ov::float16(8), ov::float16(0), ov::float16(1), ov::float16(1), ov::float16(8),
        ov::float16(7), ov::float16(7), ov::float16(4), ov::float16(4), ov::float16(4), ov::float16(4),
        ov::float16(8), ov::float16(10), ov::float16(8), ov::float16(6), ov::float16(10), ov::float16(8),
        ov::float16(3), ov::float16(3), ov::float16(7), ov::float16(8), ov::float16(3), ov::float16(8),
    };

    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 6, 4, 4, 2), axis);
}

TEST(gather_elements_gpu_fp16, d124251_i124221_an3) {
    auto& engine = get_test_engine();

    auto axis = 3;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 4, 2, 5, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 4, 2, 2, 1 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8), ov::float16(5),
        ov::float16(5), ov::float16(2), ov::float16(0), ov::float16(7),
        ov::float16(7), ov::float16(10), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(0), ov::float16(5),
        ov::float16(7), ov::float16(0), ov::float16(4), ov::float16(0),
        ov::float16(4), ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1), ov::float16(7),
        ov::float16(4), ov::float16(7), ov::float16(10), ov::float16(8),
        ov::float16(2), ov::float16(0), ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8), ov::float16(10), ov::float16(4),
        ov::float16(2), ov::float16(10), ov::float16(7), ov::float16(8),
        ov::float16(7), ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8), ov::float16(5),
        ov::float16(2), ov::float16(3), ov::float16(3), ov::float16(1),
        ov::float16(5), ov::float16(9), ov::float16(10), ov::float16(0),
        ov::float16(9), ov::float16(5), ov::float16(5), ov::float16(3),
        ov::float16(10), ov::float16(5), ov::float16(2), ov::float16(0),
        ov::float16(10), ov::float16(0), ov::float16(5), ov::float16(4),
        ov::float16(3), ov::float16(10), ov::float16(5), ov::float16(5),
        ov::float16(10), ov::float16(0), ov::float16(8), ov::float16(8),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(2), ov::float16(4), ov::float16(3),
        ov::float16(4), ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(4), ov::float16(0), ov::float16(1), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1), ov::float16(1),
        ov::float16(3), ov::float16(1), ov::float16(4), ov::float16(2),
        ov::float16(4), ov::float16(2), ov::float16(1), ov::float16(3),
        ov::float16(2), ov::float16(1), ov::float16(2), ov::float16(4),
        ov::float16(1), ov::float16(0), ov::float16(2), ov::float16(4),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(0), ov::float16(8), ov::float16(7),
        ov::float16(6), ov::float16(2), ov::float16(0), ov::float16(5),
        ov::float16(2), ov::float16(1), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(2), ov::float16(0), ov::float16(5),
        ov::float16(10), ov::float16(4), ov::float16(5), ov::float16(0),
        ov::float16(10), ov::float16(5), ov::float16(3), ov::float16(4),
        ov::float16(5), ov::float16(4), ov::float16(10), ov::float16(5),
        ov::float16(2), ov::float16(0), ov::float16(5), ov::float16(8),
    };

    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 4, 2, 2, 1), axis);
}

TEST(gather_elements_gpu_fp16, d233113_i233115_a2) {
    auto& engine = get_test_engine();

    auto axis = 2;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 3, 1, 1, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 3, 1, 1, 5 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(5), ov::float16(5), ov::float16(2),
        ov::float16(0), ov::float16(7), ov::float16(7),
        ov::float16(10), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(0),
        ov::float16(5), ov::float16(7), ov::float16(0),
        ov::float16(4), ov::float16(0), ov::float16(4),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(4), ov::float16(7),
        ov::float16(10), ov::float16(8), ov::float16(2),
        ov::float16(0), ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8), ov::float16(10),
        ov::float16(4), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(7),
        ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8),
        ov::float16(5), ov::float16(2), ov::float16(3),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(2),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(5), ov::float16(7),
        ov::float16(0), ov::float16(7), ov::float16(8),
        ov::float16(0), ov::float16(1), ov::float16(7),
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(5), ov::float16(1), ov::float16(2),
        ov::float16(9), ov::float16(7), ov::float16(0),
        ov::float16(5), ov::float16(0), ov::float16(0),
        ov::float16(9), ov::float16(4), ov::float16(0),
        ov::float16(9), ov::float16(4), ov::float16(0),
        ov::float16(5), ov::float16(4), ov::float16(5),
        ov::float16(7), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(7), ov::float16(0), ov::float16(1),
        ov::float16(4), ov::float16(5), ov::float16(1),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(4), ov::float16(3),
        ov::float16(10), ov::float16(8), ov::float16(3),
        ov::float16(0), ov::float16(8), ov::float16(7),
        ov::float16(0), ov::float16(4), ov::float16(7),
        ov::float16(7), ov::float16(4), ov::float16(3),
        ov::float16(7), ov::float16(8), ov::float16(10),
        ov::float16(4), ov::float16(8), ov::float16(7),
        ov::float16(4), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(10),
        ov::float16(6), ov::float16(8), ov::float16(7),
        ov::float16(5), ov::float16(4), ov::float16(9),
        ov::float16(0), ov::float16(2), ov::float16(8),
        ov::float16(5), ov::float16(4), ov::float16(3),
        ov::float16(0), ov::float16(6), ov::float16(8),
        ov::float16(5), ov::float16(6), ov::float16(3),
    };

    DoTest(engine, input0, input1, expected_results, tensor(2, 3, 3, 1, 1, 5), axis, false);
}

TEST(gather_elements_gpu_fp16, export_import) {
    auto& engine = get_test_engine();

    auto axis = 2;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 3, 1, 1, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 3, 1, 1, 5 } }); // indices

    set_values(input0, {
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(5), ov::float16(5), ov::float16(2),
        ov::float16(0), ov::float16(7), ov::float16(7),
        ov::float16(10), ov::float16(4), ov::float16(5),
        ov::float16(9), ov::float16(0), ov::float16(0),
        ov::float16(5), ov::float16(7), ov::float16(0),
        ov::float16(4), ov::float16(0), ov::float16(4),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(4), ov::float16(7),
        ov::float16(10), ov::float16(8), ov::float16(2),
        ov::float16(0), ov::float16(8), ov::float16(3),
        ov::float16(6), ov::float16(8), ov::float16(10),
        ov::float16(4), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(7),
        ov::float16(0), ov::float16(6), ov::float16(9),
        ov::float16(2), ov::float16(4), ov::float16(8),
        ov::float16(5), ov::float16(2), ov::float16(3),
    });

    set_values(input1, {
        ov::float16(0), ov::float16(1), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(1),
        ov::float16(1), ov::float16(0), ov::float16(2),
        ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(2), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(0), ov::float16(2),
        ov::float16(2), ov::float16(0), ov::float16(1),
        ov::float16(1), ov::float16(2), ov::float16(2),
        ov::float16(1), ov::float16(1), ov::float16(0),
        ov::float16(2), ov::float16(0), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(2),
        ov::float16(2), ov::float16(1), ov::float16(0),
        ov::float16(0), ov::float16(2), ov::float16(1),
        ov::float16(2), ov::float16(1), ov::float16(2),
        ov::float16(0), ov::float16(0), ov::float16(1),
        ov::float16(2), ov::float16(0), ov::float16(2),
    });

    std::vector<float> expected_results = {
        ov::float16(0), ov::float16(5), ov::float16(7),
        ov::float16(0), ov::float16(7), ov::float16(8),
        ov::float16(0), ov::float16(1), ov::float16(7),
        ov::float16(0), ov::float16(1), ov::float16(8),
        ov::float16(5), ov::float16(1), ov::float16(2),
        ov::float16(9), ov::float16(7), ov::float16(0),
        ov::float16(5), ov::float16(0), ov::float16(0),
        ov::float16(9), ov::float16(4), ov::float16(0),
        ov::float16(9), ov::float16(4), ov::float16(0),
        ov::float16(5), ov::float16(4), ov::float16(5),
        ov::float16(7), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(6), ov::float16(10),
        ov::float16(7), ov::float16(0), ov::float16(1),
        ov::float16(4), ov::float16(5), ov::float16(1),
        ov::float16(9), ov::float16(5), ov::float16(1),
        ov::float16(7), ov::float16(4), ov::float16(3),
        ov::float16(10), ov::float16(8), ov::float16(3),
        ov::float16(0), ov::float16(8), ov::float16(7),
        ov::float16(0), ov::float16(4), ov::float16(7),
        ov::float16(7), ov::float16(4), ov::float16(3),
        ov::float16(7), ov::float16(8), ov::float16(10),
        ov::float16(4), ov::float16(8), ov::float16(7),
        ov::float16(4), ov::float16(2), ov::float16(10),
        ov::float16(7), ov::float16(8), ov::float16(10),
        ov::float16(6), ov::float16(8), ov::float16(7),
        ov::float16(5), ov::float16(4), ov::float16(9),
        ov::float16(0), ov::float16(2), ov::float16(8),
        ov::float16(5), ov::float16(4), ov::float16(3),
        ov::float16(0), ov::float16(6), ov::float16(8),
        ov::float16(5), ov::float16(6), ov::float16(3),
    };

    DoTest(engine, input0, input1, expected_results, tensor(2, 3, 3, 1, 1, 5), axis, true);
}

TEST(gather_elements_gpu, dynamic) {
    auto& engine = get_test_engine();

    auto axis = 3;

    ov::Shape in0_shape = { 1, 2, 1, 5, 2, 4 };
    ov::Shape in1_shape = { 1, 2, 1, 2, 2, 4 };

    auto in0_dyn_layout = layout{ov::PartialShape::dynamic(in0_shape.size()), data_types::u8, format::bfwzyx};
    auto in1_dyn_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::u8, format::bfwzyx};

    auto input0 = engine.allocate_memory({in0_shape, data_types::u8, format::bfwzyx}); // data
    auto input1 = engine.allocate_memory({in1_shape, data_types::u8, format::bfwzyx}); // indices

    set_values<uint8_t>(input0, {
        0,  1,  8,  5,  5,  2,  0,  7,
        7,  10, 4,  5,  9,  0,  0,  5,
        7,  0,  4,  0,  4,  7,  6,  10,
        9,  5,  1,  7,  4,  7,  10, 8,
        2,  0,  8,  3,  6,  8,  10, 4,
        2,  10, 7,  8,  7,  0,  6,  9,
        2,  4,  8,  5,  2,  3,  3,  1,
        5,  9,  10, 0,  9,  5,  5,  3,
        10, 5,  2,  0,  10, 0,  5,  4,
        3,  10, 5,  5,  10, 0,  8,  8
    });

    set_values<uint8_t>(input1, {
        0, 2, 4, 3,
        4, 0, 0, 1,
        4, 0, 1, 0,
        1, 0, 1, 1,
        3, 1, 4, 2,
        4, 2, 1, 3,
        2, 1, 2, 4,
        1, 0, 2, 4
    });

    std::vector<uint8_t> expected_results = {
        0,  0,  8,  7,
        6,  2,  0,  5,
        2,  1,  4,  5,
        9,  2,  0,  5,
        10, 4,  5,  0,
        10, 5,  3,  4,
        5,  4,  10, 5,
        2,  0,  5,  8
    };

    topology topology;
    topology.add(input_layout("InputData", in0_dyn_layout));
    topology.add(input_layout("InputIndices", in1_dyn_layout));
    topology.add(gather_elements("gather_elements", input_info("InputData"), input_info("InputIndices"), axis));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("InputData", input0);
    network.set_input_data("InputIndices", input1);

    auto inst = network.get_primitive("gather_elements");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gather_elements").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}
