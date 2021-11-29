// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/eltwise.hpp"
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/data.hpp>
#include <api/loop.hpp>
#include <api/mutable_data.hpp>
#include <api/data.hpp>

#include <cassert>
#include <cmath>
#include <gmock/gmock.h>
#include <limits>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(loop_gpu, basic_no_concat)
{
    const auto& engine = get_test_engine();

    auto input_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto operand_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto trip_count_mem = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto initial_condition_mem = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto num_iteration_mem = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, 1, 1 } });

    std::vector<float> input_data{
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    };
    set_values(input_mem, input_data);

    std::vector<float> eltwise_operand {
        1.f, -2.f, 3.f, -4.f, 3.0f, -2.0f, 1.f, -2.f, 3.0f, -4.0f,
        3.f, -2.f, 1.f, -2.f, 3.5f, -4.5f, 5.f, -4.f, 3.5f, -2.2f
    };
    set_values(operand_mem, eltwise_operand);

    int trip_count = 8;
    set_values(trip_count_mem, {trip_count});

    int initial_condition = 1;
    set_values(initial_condition_mem, {initial_condition});

    topology body(
        data("eltwise_operand", operand_mem),
        eltwise("eltwise", "input", "eltwise_operand", eltwise_mode::sum)
    );

    std::vector<loop::io_primitive_map> input_primitive_maps { loop::io_primitive_map("input", "input") };
    std::vector<loop::io_primitive_map> output_primitive_maps { loop::io_primitive_map("loop", "eltwise") };

    std::vector<loop::backedge_mapping> back_edges {
        loop::backedge_mapping("eltwise", "input")
    };

    topology topology(
        input_layout("input", input_mem.get_layout()),
        input_layout("trip_count", trip_count_mem.get_layout()),
        input_layout("initial_condition", initial_condition_mem.get_layout()),
        mutable_data("num_iteration", num_iteration_mem),
        loop("loop", {"input"}, body,
             "trip_count", "initial_condition", "num_iteration",
             input_primitive_maps, output_primitive_maps, back_edges, 8)
    );

    network network(engine, topology);
    network.set_input_data("input", input_mem);
    network.set_input_data("trip_count", trip_count_mem);
    network.set_input_data("initial_condition", initial_condition_mem);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.size.batch[0], 1);
    EXPECT_EQ(output_layout.size.feature[0], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 4);
    EXPECT_EQ(output_layout.size.spatial[1], 5);

    auto ptr = num_iteration_mem.pointer<int32_t>();
    EXPECT_EQ(ptr[0], trip_count);

    // value check
    auto output_ptr = output.pointer<float>();
    EXPECT_EQ(output_ptr.size(), input_data.size());
    for (size_t i=0, iend = input_data.size(); i<iend; ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], input_data[i] + eltwise_operand[i] * trip_count);
    }
}

TEST(loop_gpu, basic_concat)
{
    const auto& engine = get_test_engine();

    auto input_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } }); // b,f,x,y
    auto operand_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 1 } }); // b,f,x,y
    auto trip_count_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto initial_condition_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto num_iteration_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });

    std::vector<float> input_data{
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    };
    set_values(input_mem, input_data);

    std::vector<float> eltwise_operand {
        1.f, -2.f, 3.f, -4.f
    };
    set_values(operand_mem, eltwise_operand);

    size_t trip_count = input_data.size()/eltwise_operand.size();
    set_values(trip_count_mem, {trip_count});

    int initial_condition = 1;
    set_values(initial_condition_mem, {initial_condition});

    topology body(
        input_layout("input", operand_mem.get_layout()),
        data("eltwise_operand", operand_mem),
        eltwise("eltwise", "input", "eltwise_operand", eltwise_mode::sum)
    );

    std::vector<loop::io_primitive_map> input_primitive_maps { loop::io_primitive_map("input", "input", 2) };
    std::vector<loop::io_primitive_map> output_primitive_maps { loop::io_primitive_map("loop", "eltwise", 2) };

    std::vector<loop::backedge_mapping> back_edges {};

    topology topology(
        input_layout("input", input_mem.get_layout()),
        input_layout("trip_count", trip_count_mem.get_layout()),
        input_layout("initial_condition", initial_condition_mem.get_layout()),
        mutable_data("num_iteration", num_iteration_mem),
        loop("loop", {"input"}, body,
             "trip_count", "initial_condition", "num_iteration",
             input_primitive_maps, output_primitive_maps, back_edges, trip_count)
    );

    network network(engine, topology);
    network.set_input_data("input", input_mem);
    network.set_input_data("trip_count", trip_count_mem);
    network.set_input_data("initial_condition", initial_condition_mem);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output.get_layout();

    EXPECT_EQ(output_layout.size.batch[0], 1);
    EXPECT_EQ(output_layout.size.feature[0], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 4);
    EXPECT_EQ(output_layout.size.spatial[1], 5);

    auto ptr = num_iteration_mem.pointer<int32_t>();
    const int32_t actual_iterations = ptr[0];
    EXPECT_EQ(actual_iterations, trip_count);

    // value check
    auto output_ptr = output.pointer<float>();
    for (size_t i=0, iend = input_data.size(); i<iend; ++i) {
        const size_t j = i % eltwise_operand.size();
        float expected = input_data[i] + eltwise_operand[j];
        EXPECT_FLOAT_EQ(output_ptr[i], expected);
    }
}

TEST(loop_gpu, basic_concat_nested)
{
    const auto& engine = get_test_engine();

    auto input_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } }); // b,f,x,y
    auto trip_count_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto initial_condition_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto num_iteration_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_trip_count_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_initial_condition_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_num_iteration_mem = memory::allocate(engine, { data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_operand_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 1 } }); // b,f,x,y

    /////////////////////////////////
    // set data
    /////////////////////////////////
    std::vector<float> input_data{
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    };
    set_values(input_mem, input_data);

    std::vector<float> inner_eltwise_operand {
        1.f, -2.f, 3.f, -4.f
    };
    set_values(inner_operand_mem, inner_eltwise_operand);

    size_t inner_trip_count = input_data.size() / inner_eltwise_operand.size();
    set_values(inner_trip_count_mem, {inner_trip_count});

    int inner_initial_condition = 1;
    set_values(inner_initial_condition_mem, {inner_initial_condition});

    int outer_trip_count = 8;
    set_values(trip_count_mem, {outer_trip_count});

    int outer_initial_condition = 1;
    set_values(initial_condition_mem, {outer_initial_condition});


    /////////////////////////////////
    // set inner loop body
    /////////////////////////////////
    topology inner_loop_body(
        input_layout("inner_input", input_mem.get_layout()),
        data("inner_eltwise_operand", inner_operand_mem),
        eltwise("inner_eltwise", "inner_input", "inner_eltwise_operand", eltwise_mode::sum)
    );
    std::vector<loop::io_primitive_map> inner_input_primitive_maps { loop::io_primitive_map("inner_input", "inner_input", 2) };
    std::vector<loop::io_primitive_map> inner_output_primitive_maps { loop::io_primitive_map("inner_loop", "inner_eltwise", 2) };
    std::vector<loop::backedge_mapping> inner_back_edges {};

    /////////////////////////////////
    // set outer loop body
    /////////////////////////////////
    topology outer_loop_body(
        input_layout("inner_input", input_mem.get_layout()),
        input_layout("trip_count", inner_trip_count_mem.get_layout()),
        input_layout("initial_condition", inner_initial_condition_mem.get_layout()),
        mutable_data("inner_num_iteration", inner_num_iteration_mem),
        loop("inner_loop", {"inner_input", "trip_count", "initial_condition"},
            inner_loop_body, "trip_count", "initial_condition", "inner_num_iteration",
            inner_input_primitive_maps, inner_output_primitive_maps, inner_back_edges, inner_trip_count)
    );
    std::vector<loop::io_primitive_map> outer_input_primitive_maps {
        loop::io_primitive_map("input", "inner_input"),
        loop::io_primitive_map("inner_trip_count", "trip_count"),
        loop::io_primitive_map("inner_initial_condition", "initial_condition"),
    };
    std::vector<loop::io_primitive_map> outer_output_primitive_maps {
        loop::io_primitive_map("loop", "inner_loop"),
    };
    std::vector<loop::backedge_mapping> outer_back_edges { {"inner_loop", "inner_input"} };

    /////////////////////////////////
    // set main topology
    /////////////////////////////////
    topology main_topology(
        input_layout("input", input_mem.get_layout()),
        input_layout("trip_count", trip_count_mem.get_layout()),
        input_layout("initial_condition", initial_condition_mem.get_layout()),
        mutable_data("num_iteration", num_iteration_mem),
        input_layout("inner_trip_count", inner_trip_count_mem.get_layout()),
        input_layout("inner_initial_condition", inner_initial_condition_mem.get_layout()),
        loop("loop", {"input", "inner_trip_count", "inner_initial_condition"},
            outer_loop_body, "trip_count", "initial_condition", "num_iteration",
            outer_input_primitive_maps, outer_output_primitive_maps, outer_back_edges, outer_trip_count)
    );

    /////////////////////////////////
    // network execution
    /////////////////////////////////
    network network(engine, main_topology);
    network.set_input_data("input", input_mem);
    network.set_input_data("trip_count", trip_count_mem);
    network.set_input_data("initial_condition", initial_condition_mem);
    network.set_input_data("inner_trip_count", inner_trip_count_mem);
    network.set_input_data("inner_initial_condition", inner_initial_condition_mem);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output.get_layout();

    /////////////////////////////////
    // calculate expected output
    /////////////////////////////////
    std::vector<float> input_data2(input_data);
    std::vector<float> expected(input_data2.size());
    for (int i=0 ; i<outer_trip_count ; ++i) {
        for (size_t i=0, iend = input_data2.size(); i<iend; ++i) {
            // eltwise sum in inner loop
            const size_t j = i % inner_eltwise_operand.size();
            expected[i] = input_data2[i] + inner_eltwise_operand [j];
        }
        input_data2 = expected; // backedge
    }

    /////////////////////////////////
    // compare
    /////////////////////////////////
    EXPECT_EQ(output_layout.size.batch[0], 1);
    EXPECT_EQ(output_layout.size.feature[0], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 4);
    EXPECT_EQ(output_layout.size.spatial[1], 5);

    // check trip count = actual iteration
    auto inner_num_iteration_ptr = inner_num_iteration_mem.pointer<int64_t>();
    int64_t inner_actual_iterations = inner_num_iteration_ptr[0];
    EXPECT_EQ(inner_actual_iterations, inner_trip_count);
    auto num_iteration_ptr = num_iteration_mem.pointer<int64_t>();
    int64_t actual_iterations = num_iteration_ptr[0];
    EXPECT_EQ(actual_iterations, outer_trip_count);

    // check output values
    EXPECT_EQ(output_layout.count(), expected.size());
    auto output_ptr = output.pointer<float>();
    for (size_t i=0 ;i<output_layout.count(); ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], expected.at(i));
    }
}
