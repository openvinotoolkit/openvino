// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>
#include <intel_gpu/primitives/reorder.hpp>
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
    bool is_caching_test=false,
    format blocked_format = format::any) {
    topology topology;
    topology.add(input_layout("InputData", input0->get_layout()));
    topology.add(input_layout("InputIndices", input1->get_layout()));

    input_info data_input("InputData");
    input_info indices_input("InputIndices");
    format gather_format = input1->get_layout().format;

    if (blocked_format != format::any) {
        topology.add(reorder("InputData_reordered", data_input, blocked_format, input0->get_layout().data_type));
        topology.add(reorder("InputIndices_reordered", indices_input, blocked_format, input1->get_layout().data_type));
        data_input = input_info("InputData_reordered");
        indices_input = input_info("InputIndices_reordered");
        gather_format = blocked_format;
    }

    topology.add(
        gather_elements("gather_elements", data_input, indices_input, gather_format, output_tensor, axis)
    );

    primitive_id output_id = "gather_elements";
    if (blocked_format != format::any) {
        topology.add(reorder("gather_elements_to_bfyx", input_info("gather_elements"), format::bfyx, input0->get_layout().data_type));
        output_id = "gather_elements_to_bfyx";
    }

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("InputData", input0);
    network->set_input_data("InputIndices", input1);
    auto outputs = network->execute();

    if (blocked_format != format::any) {
        ASSERT_EQ(network->get_primitive("gather_elements")->get_impl_params()->get_output_layout().format,
                  format::b_fs_yx_fsv16);
    }

    auto output = outputs.at(output_id).get_memory();
    cldnn::mem_lock<uint16_t, mem_lock_type::read> output_ptr(output, get_test_stream());

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

TEST(gather_elements_gpu_fp16, d13212_i13212_a1_b_fs_yx_fsv16) {
    auto& engine = get_test_engine();

    auto axis = 1;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 32, 1, 2 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 32, 1, 2 } }); // indices
    set_values(input0, {
        ov::float16(53), ov::float16(29), ov::float16(39), ov::float16(24), ov::float16(13), ov::float16(46), ov::float16(27), ov::float16(26),
        ov::float16(36), ov::float16(21), ov::float16(37), ov::float16(14), ov::float16(11), ov::float16(4), ov::float16(34), ov::float16(58),
        ov::float16(35), ov::float16(55), ov::float16(25), ov::float16(41), ov::float16(9), ov::float16(47), ov::float16(19), ov::float16(17),
        ov::float16(28), ov::float16(56), ov::float16(20), ov::float16(64), ov::float16(15), ov::float16(18), ov::float16(22), ov::float16(5),
        ov::float16(60), ov::float16(32), ov::float16(33), ov::float16(62), ov::float16(63), ov::float16(12), ov::float16(54), ov::float16(10),
        ov::float16(30), ov::float16(44), ov::float16(31), ov::float16(40), ov::float16(42), ov::float16(51), ov::float16(48), ov::float16(49),
        ov::float16(2), ov::float16(16), ov::float16(23), ov::float16(50), ov::float16(7), ov::float16(61), ov::float16(45), ov::float16(43),
        ov::float16(3), ov::float16(38), ov::float16(52), ov::float16(59), ov::float16(6), ov::float16(1), ov::float16(8), ov::float16(57),
    });

    set_values(input1, {
        ov::float16(10), ov::float16(3), ov::float16(4), ov::float16(31), ov::float16(10), ov::float16(4), ov::float16(13), ov::float16(9),
        ov::float16(1), ov::float16(22), ov::float16(7), ov::float16(9), ov::float16(19), ov::float16(29), ov::float16(2), ov::float16(26),
        ov::float16(17), ov::float16(2), ov::float16(11), ov::float16(17), ov::float16(9), ov::float16(19), ov::float16(9), ov::float16(29),
        ov::float16(2), ov::float16(24), ov::float16(22), ov::float16(31), ov::float16(21), ov::float16(29), ov::float16(30), ov::float16(2),
        ov::float16(30), ov::float16(10), ov::float16(28), ov::float16(28), ov::float16(18), ov::float16(17), ov::float16(7), ov::float16(11),
        ov::float16(26), ov::float16(27), ov::float16(7), ov::float16(6), ov::float16(31), ov::float16(24), ov::float16(16), ov::float16(12),
        ov::float16(22), ov::float16(6), ov::float16(10), ov::float16(17), ov::float16(9), ov::float16(20), ov::float16(28), ov::float16(4),
        ov::float16(5), ov::float16(25), ov::float16(13), ov::float16(3), ov::float16(18), ov::float16(2), ov::float16(14), ov::float16(5),
    });

    std::vector<float> expected_results = {
        ov::float16(9), ov::float16(26), ov::float16(36), ov::float16(57), ov::float16(9), ov::float16(21), ov::float16(20), ov::float16(41),
        ov::float16(39), ov::float16(51), ov::float16(34), ov::float16(41), ov::float16(54), ov::float16(59), ov::float16(13), ov::float16(61),
        ov::float16(33), ov::float16(46), ov::float16(19), ov::float16(62), ov::float16(25), ov::float16(10), ov::float16(25), ov::float16(59),
        ov::float16(13), ov::float16(16), ov::float16(42), ov::float16(57), ov::float16(31), ov::float16(59), ov::float16(6), ov::float16(46),
        ov::float16(6), ov::float16(47), ov::float16(3), ov::float16(38), ov::float16(63), ov::float16(62), ov::float16(34), ov::float16(17),
        ov::float16(7), ov::float16(43), ov::float16(34), ov::float16(4), ov::float16(8), ov::float16(16), ov::float16(60), ov::float16(56),
        ov::float16(42), ov::float16(4), ov::float16(9), ov::float16(62), ov::float16(25), ov::float16(44), ov::float16(3), ov::float16(21),
        ov::float16(37), ov::float16(50), ov::float16(20), ov::float16(26), ov::float16(63), ov::float16(46), ov::float16(15), ov::float16(14),
    };

    DoTest(engine, input0, input1, expected_results, tensor(1, 32, 1, 2), axis, false, format::b_fs_yx_fsv16);
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
    cldnn::mem_lock<uint8_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}
