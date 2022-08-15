// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

inline void do_count_non_zero_test(engine& engine,
                                   const cldnn::memory::ptr& input_data,
                                   const std::vector<int32_t>& expected_results)
{
    topology topology;
    topology.add(input_layout("InputData", input_data->get_layout()));
    topology.add(
        count_nonzero("count_nonzero", "InputData")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_data);
    auto outputs = network.execute();
    auto output = outputs.at("count_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

inline void do_gather_non_zero_test(engine& engine,
                                    const cldnn::memory::ptr& input_data,
                                    const cldnn::memory::ptr& output_shape,
                                    const std::vector<int32_t>& expected_results)
{
    topology topology;
    topology.add(input_layout("InputData", input_data->get_layout()));
    topology.add(input_layout("OutputShape", output_shape->get_layout()));
    topology.add(
        gather_nonzero("gather_nonzero", "InputData", "OutputShape")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_data);
    network.set_input_data("OutputShape", output_shape);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    cldnn::mem_lock<int32_t> shape_ptr(output_shape, get_test_stream());

    int num_ranks = shape_ptr[0];
    int num_nonzero = shape_ptr[1];

    for (int i = 0; i < num_nonzero; i++) {
        bool found = false;
        for (int j = 0; j < num_nonzero; j++) {
            for (int k = 0; k < num_ranks; k++) {
                if (output_ptr[i+num_nonzero*k] != expected_results[j+num_nonzero*k])
                    break;

                if (k == (num_ranks - 1)) {
                    found = true;
                }
            }
            if (found)
                break;
        }

        EXPECT_TRUE(found);
    }
}

inline void do_non_zero_test(engine& engine,
                          const cldnn::memory::ptr& input_data,
                          const std::vector<int32_t>& expected_shape,
                          const std::vector<int32_t>& expected_results)
{
    topology topology;
    topology.add(input_layout("InputData", input_data->get_layout()));
    topology.add(
        count_nonzero("count_nonzero", "InputData")
    );
    topology.add(
        gather_nonzero("gather_nonzero", "InputData", "count_nonzero")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_data);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    std::vector<int32_t> output_list = std::vector<int32_t>(output_ptr.begin(), output_ptr.end());

    int num_ranks = expected_shape[0];
    int num_nonzero = expected_shape[1];

    EXPECT_EQ(num_ranks*num_nonzero, output_list.size());

    for (int i = 0; i < num_nonzero; i++) {
        bool found = false;
        for (int j = 0; j < num_nonzero; j++) {
            for (int k = 0; k < num_ranks; k++) {
                if (output_list[i+num_nonzero*k] != expected_results[j+num_nonzero*k])
                    break;

                if (k == (num_ranks - 1)) {
                    found = true;
                }
            }
            if (found)
                break;
        }

        EXPECT_TRUE(found);
    }
}

TEST(count_nonzero_gpu_fp16, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(2),
        FLOAT16(7), FLOAT16(10), FLOAT16(4),
    });

    std::vector<int32_t> expected_results = {
        4,  8,  1,  1,
    };

    do_count_non_zero_test(engine, input, expected_results);
}

TEST(gather_nonzero_gpu_fp16, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });
    auto output_shape = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 4, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(2),
        FLOAT16(7), FLOAT16(10), FLOAT16(4),
    });

    set_values(output_shape, {
        4,  8,  1,  1,
    });

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  1,  2,  2,  2,
        0,  0,  0,  0,  0,  0,  0,  0,
        1,  2,  0,  1,  2,  0,  1,  2,
    };

    do_gather_non_zero_test(engine, input, output_shape, expected_results);
}

TEST(gather_nonzero_gpu_fp16, test2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });
    auto output_shape = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 4, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(0),
        FLOAT16(7), FLOAT16(0), FLOAT16(4),
    });

    set_values(output_shape, {
        4,  6,  1,  1,
    });

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  2,  2,
        0,  0,  0,  0,  0,  0,
        1,  2,  0,  1,  0,  2,
    };

    do_gather_non_zero_test(engine, input, output_shape, expected_results);
}

TEST(nonzero_gpu_fp16, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(0),
        FLOAT16(7), FLOAT16(0), FLOAT16(4),
    });

    std::vector<int32_t> expected_shape = {
        4,  6,  1, 1,
    };

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  2,  2,
        0,  0,  0,  0,  0,  0,
        1,  2,  0,  1,  0,  2,
    };

    do_non_zero_test(engine, input, expected_shape, expected_results);
}

TEST(nonzero_gpu_fp16, test2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 1, 3, 3, 3, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(7), FLOAT16(10), FLOAT16(4),
        FLOAT16(7), FLOAT16(0), FLOAT16(4),
        FLOAT16(9), FLOAT16(5), FLOAT16(1),
        FLOAT16(2), FLOAT16(0), FLOAT16(8),
        FLOAT16(2), FLOAT16(10), FLOAT16(7),
        FLOAT16(2), FLOAT16(4), FLOAT16(8),
        FLOAT16(5), FLOAT16(9), FLOAT16(10),
        FLOAT16(10), FLOAT16(5), FLOAT16(2),
    });

    std::vector<int32_t> expected_shape = {
        5,  24,  1, 1,
    };

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  1,  2,  2,  0,  0,  0,  1,  1,  2,  2,  2,  0,  0,  0,  1,  1,  1,  2,  2,  2,
        1,  2,  0,  1,  2,  0,  2,  0,  1,  2,  0,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,
    };

    do_non_zero_test(engine, input, expected_shape, expected_results);
}

TEST(nonzero_gpu_fp16, test3) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 1, 3, 3, 3, 3, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1),  FLOAT16(8), FLOAT16(7), FLOAT16(10), FLOAT16(4),  FLOAT16(6),  FLOAT16(5), FLOAT16(4),
        FLOAT16(7), FLOAT16(0),  FLOAT16(4), FLOAT16(9), FLOAT16(5),  FLOAT16(1),  FLOAT16(2),  FLOAT16(2), FLOAT16(0),
        FLOAT16(2), FLOAT16(0),  FLOAT16(8), FLOAT16(2), FLOAT16(10), FLOAT16(7),  FLOAT16(7),  FLOAT16(0), FLOAT16(6),
        FLOAT16(2), FLOAT16(4),  FLOAT16(8), FLOAT16(5), FLOAT16(9),  FLOAT16(10), FLOAT16(10), FLOAT16(5), FLOAT16(2),
        FLOAT16(4), FLOAT16(8),  FLOAT16(2), FLOAT16(1), FLOAT16(4),  FLOAT16(10), FLOAT16(10), FLOAT16(2), FLOAT16(21),
        FLOAT16(0), FLOAT16(1),  FLOAT16(5), FLOAT16(1), FLOAT16(5),  FLOAT16(1),  FLOAT16(9),  FLOAT16(4), FLOAT16(22),
        FLOAT16(4), FLOAT16(3),  FLOAT16(7), FLOAT16(6), FLOAT16(9),  FLOAT16(8),  FLOAT16(9),  FLOAT16(7), FLOAT16(23),
        FLOAT16(4), FLOAT16(10), FLOAT16(6), FLOAT16(3), FLOAT16(5),  FLOAT16(5),  FLOAT16(4),  FLOAT16(2), FLOAT16(23),
        FLOAT16(0), FLOAT16(4),  FLOAT16(5), FLOAT16(3), FLOAT16(1),  FLOAT16(2),  FLOAT16(8),  FLOAT16(5), FLOAT16(0),
    });

    std::vector<int32_t> expected_shape = {
        6,  73,  1, 1,
    };

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
        0,  0,  1,  1,  1,  2,  2,  2,  0,  0,  1,  1,  1,  2,  2,  0,  0,  1,  1,  1,  2,  2,  0,  0,  0,  1,  1,  1,  2,  2,  2,  0,  0,  0,  1,  1,  1,  2,  2,  2,  0,  0,  1,  1,  1,  2,  2,  2,  0,  0,  0,  1,  1,  1,  2,  2,  2,  0,  0,  0,  1,  1,  1,  2,  2,  2,  0,  0,  1,  1,  1,  2,  2,
        1,  2,  0,  1,  2,  0,  1,  2,  0,  2,  0,  1,  2,  0,  1,  0,  2,  0,  1,  2,  0,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  2,  1,  2,  0,  1,  2,  0,  1,
    };

    do_non_zero_test(engine, input, expected_shape, expected_results);
}

TEST(nonzero_gpu_fp32, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 3, 5, 4 } });

    set_values(input, {
        6, 6, 0, 3, 0,  4, 1, 0, 8, 4,  8, 5, 8, 6, 0,  2, 0, 9, 6, 9,
        1, 2, 4, 9, 0,  8, 5, 7, 4, 6,  8, 0, 6, 2, 3,  5, 0, 9, 8, 7,
        3, 6, 5, 3, 8,  4, 7, 5, 7, 8,  5, 2, 1, 8, 9,  2, 1, 4, 3, 3,

        7, 3, 9, 9, 0,  2, 4, 0, 4, 9,  5, 9, 4, 5, 8,  1, 2, 9, 7, 6,
        7, 9, 6, 7, 2,  9, 2, 7, 8, 3,  1, 2, 7, 4, 6,  2, 3, 7, 0, 5,
        2, 3, 7, 7, 0,  3, 4, 0, 9, 0,  9, 0, 2, 7, 7,  8, 6, 6, 0, 8,
    });

    std::vector<int32_t> expected_shape = {
        4, 104, 1, 1,
    };

    std::vector<int32_t> expected_results = {
        0,  0,      0,      0,  0,      0,  0,  0,  0,  0,  0,      0,      0,  0,  0,
        0,  0,  0,  0,      0,  0,  0,  0,  0,  0,      0,  0,  0,  0,      0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        1,  1,  1,  1,      1,  1,      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      1,
        1,  1,  1,  1,      1,  1,      1,      1,      1,  1,  1,  1,  1,  1,      1,


        0,  0,      0,      0,  0,      0,  0,  0,  0,  0,  0,      0,      0,  0,  0,
        1,  1,  1,  1,      1,  1,  1,  1,  1,  1,      1,  1,  1,  1,      1,  1,  1,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        0,  0,  0,  0,      0,  0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      1,
        2,  2,  2,  2,      2,  2,      2,      2,      2,  2,  2,  2,  2,  2,      2,


        0,  0,      0,      1,  1,      1,  1,  2,  2,  2,  2,      3,      3,  3,  3,
        0,  0,  0,  0,      1,  1,  1,  1,  1,  2,      2,  2,  2,  3,      3,  3,  3,
        0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,
        0,  0,  0,  0,      1,  1,      1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,
        0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,      3,
        0,  0,  0,  0,      1,  1,      1,      2,      2,  2,  2,  3,  3,  3,      3,


        0,  1,      3,      0,  1,      3,  4,  0,  1,  2,  3,      0,      2,  3,  4,
        0,  1,  2,  3,      0,  1,  2,  3,  4,  0,      2,  3,  4,  0,      2,  3,  4,
        0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,
        0,  1,  2,  3,      0,  1,      3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,
        0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,      4,
        0,  1,  2,  3,      0,  1,      3,      0,      2,  3,  4,  0,  1,  2,      4,
    };

    do_non_zero_test(engine, input, expected_shape, expected_results);
}
