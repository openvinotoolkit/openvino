// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"

#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/eltwise.hpp"
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/loop.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/graph/program.hpp>

#include <cassert>
#include <cmath>
#include <gmock/gmock.h>
#include <limits>

using namespace cldnn;
using namespace tests;
using namespace testing;

static program::ptr build_program(engine& engine,
                                    topology& body_topology,
                                    primitive_id execution_condition_id,
                                    std::vector<loop::io_primitive_map> output_primitive_maps,
                                    std::vector<loop::backedge_mapping> back_edges) {
    std::vector<cldnn::primitive_id> output_names_vec;
    for (auto out_map : output_primitive_maps) {
        output_names_vec.push_back(out_map.internal_id.pid);
    }

    // setup outputs for backedges
    for (auto& back_edge : back_edges) {
        output_names_vec.push_back(back_edge.from);
    }

    // if execution_condition_id is specified, we need to add the id in build_option::outputs
    if (!execution_condition_id.empty()) {
        output_names_vec.push_back(execution_condition_id);
    }

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(output_names_vec));
    config.set_property(ov::intel_gpu::max_dynamic_batch(1));

    return program::build_program(engine, body_topology, config, false, false, true);
}

template <typename T>
void test_loop_gpu_basic_no_concat(bool is_caching_test)
{
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto operand_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto trip_count_mem = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto initial_condition_mem = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto num_iteration_mem = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });

    std::vector<T> input_data{
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    };
    std::vector<T> eltwise_operand {
        1.f, -2.f, 3.f, -4.f, 3.0f, -2.0f, 1.f, -2.f, 3.0f, -4.0f,
        3.f, -2.f, 1.f, -2.f, 3.5f, -4.5f, 5.f, -4.f, 3.5f, -2.2f
    };
    int trip_count = 8;
    int initial_condition = 1;

    // initialize input buffers
    set_values(input_mem, input_data);
    set_values(operand_mem, eltwise_operand);
    set_values(trip_count_mem, { trip_count });
    set_values(initial_condition_mem, {initial_condition});

    topology body(
        input_layout("input", input_mem->get_layout()),
        data("eltwise_operand", operand_mem),
        eltwise("eltwise", input_info("input"), input_info("eltwise_operand"), eltwise_mode::sum)
    );

    std::vector<loop::io_primitive_map> input_primitive_maps { loop::io_primitive_map("input", "input") };
    std::vector<loop::io_primitive_map> output_primitive_maps { loop::io_primitive_map("loop", "eltwise") };
    std::vector<loop::backedge_mapping> back_edges { loop::backedge_mapping("eltwise", "input") };

    auto body_program = build_program(engine, body, "", output_primitive_maps, back_edges);

    topology topology(
        input_layout("input", input_mem->get_layout()),
        input_layout("trip_count", trip_count_mem->get_layout()),
        input_layout("initial_condition", initial_condition_mem->get_layout()),
        mutable_data("num_iteration", num_iteration_mem),
        loop("loop", { input_info("num_iteration"), input_info("input") }, body_program,
             "trip_count", "initial_condition", "num_iteration",
             input_primitive_maps, output_primitive_maps, back_edges, 8)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input_mem);
    network->set_input_data("trip_count", trip_count_mem);
    network->set_input_data("initial_condition", initial_condition_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output->get_layout();

    ASSERT_EQ(output_layout.batch(), 1);
    ASSERT_EQ(output_layout.feature(), 1);
    ASSERT_EQ(output_layout.spatial(0), 4);
    ASSERT_EQ(output_layout.spatial(1), 5);

    // value check
    {
        mem_lock<T> output_ptr{ output, get_test_stream() };
        ASSERT_EQ(output_ptr.size(), input_data.size());
        for (size_t i = 0, iend = input_data.size(); i < iend; ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], input_data[i] + eltwise_operand[i] * trip_count);
        }
    }

    // allocate new output memory
    layout loop_l = network->get_output_memory("loop")->get_layout();
    auto output_mem = engine.allocate_memory(loop_l);
    network->set_output_memory("loop", output_mem);

    //one more execute
    set_values(input_mem, input_data);
    set_values(operand_mem, eltwise_operand);
    set_values(trip_count_mem, { trip_count });
    set_values(initial_condition_mem, { initial_condition });
    outputs = network->execute();

    // check everything once again
    ASSERT_EQ(outputs.size(), 1);
    auto output2 = outputs.begin()->second.get_memory();
    {
        mem_lock<T> output_ptr2{ output2, get_test_stream() };
        ASSERT_EQ(output_ptr2.size(), input_data.size());
        for (size_t i = 0, iend = input_data.size(); i < iend; ++i) {
            ASSERT_FLOAT_EQ(output_ptr2[i], input_data[i] + eltwise_operand[i] * trip_count);
        }
    }
}

TEST(loop_gpu, basic_no_concat) {
    test_loop_gpu_basic_no_concat<float>(false);
}

template <typename T>
void test_loop_gpu_basic_concat(bool is_caching_test)
{
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } }); // b,f,x,y
    auto operand_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 1 } }); // b,f,x,y
    auto trip_count_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto initial_condition_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto num_iteration_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });

    std::vector<T> input_data{
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    };
    std::vector<T> eltwise_operand {
        1.f, -2.f, 3.f, -4.f
    };
    size_t trip_count = input_data.size()/eltwise_operand.size();
    int initial_condition = 1;

    // initialize input buffers
    set_values(input_mem, input_data);
    set_values(operand_mem, eltwise_operand);
    set_values(trip_count_mem, {trip_count});
    set_values(initial_condition_mem, {initial_condition});

    topology body(
        input_layout("input", operand_mem->get_layout()),
        data("eltwise_operand", operand_mem),
        eltwise("eltwise", input_info("input"), input_info("eltwise_operand"), eltwise_mode::sum)
    );

    std::vector<loop::io_primitive_map> input_primitive_maps { loop::io_primitive_map("input", "input", 2) };
    std::vector<loop::io_primitive_map> output_primitive_maps { loop::io_primitive_map("loop", "eltwise", 2) };
    std::vector<loop::backedge_mapping> back_edges {};

    auto body_program = build_program(engine, body, "", output_primitive_maps, back_edges);

    topology topology(
        input_layout("input", input_mem->get_layout()),
        input_layout("trip_count", trip_count_mem->get_layout()),
        input_layout("initial_condition", initial_condition_mem->get_layout()),
        mutable_data("num_iteration", num_iteration_mem),
        loop("loop", { input_info("num_iteration"), input_info("input") }, body_program,
             "trip_count", "initial_condition", "num_iteration",
             input_primitive_maps, output_primitive_maps, back_edges, trip_count)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input_mem);
    network->set_input_data("trip_count", trip_count_mem);
    network->set_input_data("initial_condition", initial_condition_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output->get_layout();

    ASSERT_EQ(output_layout.batch(), 1);
    ASSERT_EQ(output_layout.feature(), 1);
    ASSERT_EQ(output_layout.spatial(0), 4);
    ASSERT_EQ(output_layout.spatial(1), 5);

    // value check
    {
        mem_lock<T> output_ptr{ output, get_test_stream() };
        for (size_t i = 0, iend = input_data.size(); i < iend; ++i) {
            const size_t j = i % eltwise_operand.size();
            float expected = input_data[i] + eltwise_operand[j];
            ASSERT_FLOAT_EQ(output_ptr[i], expected);
        }
    }

    // allocate new output memory
    layout loop_l = network->get_output_memory("loop")->get_layout();
    auto output_mem = engine.allocate_memory(loop_l);
    network->set_output_memory("loop", output_mem);

    set_values(input_mem, input_data);
    set_values(operand_mem, eltwise_operand);
    set_values(trip_count_mem, { trip_count });
    set_values(initial_condition_mem, { initial_condition });
    outputs = network->execute();
    auto output2 = outputs.begin()->second.get_memory();
    {
        mem_lock<T> output_ptr2{ output2, get_test_stream() };
        for (size_t i = 0, iend = input_data.size(); i < iend; ++i) {
            const size_t j = i % eltwise_operand.size();
            float expected = input_data[i] + eltwise_operand[j];
            ASSERT_FLOAT_EQ(output_ptr2[i], expected);
        }
    }
}

TEST(loop_gpu, basic_concat) {
    test_loop_gpu_basic_concat<float>(false);
}

template <typename T>
void test_loop_gpu_basic_concat_nested(bool is_caching_test)
{
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } }); // b,f,x,y
    auto trip_count_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto initial_condition_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto num_iteration_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_trip_count_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_initial_condition_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_num_iteration_mem = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 1, 1 } });
    auto inner_operand_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 1 } }); // b,f,x,y

    /////////////////////////////////
    // set data
    /////////////////////////////////
    std::vector<T> input_data{
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    };

    std::vector<T> inner_eltwise_operand {
        1.f, -2.f, 3.f, -4.f
    };

    size_t inner_trip_count = input_data.size() / inner_eltwise_operand.size();
    int inner_initial_condition = 1;
    int outer_trip_count = 8;
    int outer_initial_condition = 1;

    set_values(input_mem, input_data);
    set_values(inner_operand_mem, inner_eltwise_operand);
    set_values(inner_trip_count_mem, { inner_trip_count });
    set_values(inner_initial_condition_mem, { inner_initial_condition });
    set_values(trip_count_mem, { outer_trip_count });
    set_values(initial_condition_mem, { outer_initial_condition });

    /////////////////////////////////
    // set inner loop body
    /////////////////////////////////
    topology inner_loop_body(
        input_layout("inner_input", input_mem->get_layout()),
        data("inner_eltwise_operand", inner_operand_mem),
        eltwise("inner_eltwise", input_info("inner_input"), input_info("inner_eltwise_operand"), eltwise_mode::sum)
    );
    std::vector<loop::io_primitive_map> inner_input_primitive_maps { loop::io_primitive_map("inner_input", "inner_input", 2) };
    std::vector<loop::io_primitive_map> inner_output_primitive_maps { loop::io_primitive_map("inner_loop", "inner_eltwise", 2) };
    std::vector<loop::backedge_mapping> inner_back_edges {};

    auto inner_body_program = build_program(engine, inner_loop_body, "", inner_output_primitive_maps, inner_back_edges);

    /////////////////////////////////
    // set outer loop body
    /////////////////////////////////
    topology outer_loop_body(
        input_layout("inner_input", input_mem->get_layout()),
        input_layout("trip_count", inner_trip_count_mem->get_layout()),
        input_layout("initial_condition", inner_initial_condition_mem->get_layout()),
        mutable_data("inner_num_iteration", inner_num_iteration_mem),
        loop("inner_loop", { input_info("inner_num_iteration"), input_info("inner_input"), input_info("trip_count"), input_info("initial_condition") },
            inner_body_program, "trip_count", "initial_condition", "inner_num_iteration",
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

    auto outer_body_program = build_program(engine, outer_loop_body, "", outer_output_primitive_maps, outer_back_edges);

    /////////////////////////////////
    // set main topology
    /////////////////////////////////
    topology main_topology(
        input_layout("input", input_mem->get_layout()),
        input_layout("trip_count", trip_count_mem->get_layout()),
        input_layout("initial_condition", initial_condition_mem->get_layout()),
        mutable_data("num_iteration", num_iteration_mem),
        input_layout("inner_trip_count", inner_trip_count_mem->get_layout()),
        input_layout("inner_initial_condition", inner_initial_condition_mem->get_layout()),
        loop("loop", { input_info("num_iteration"), input_info("input"), input_info("inner_trip_count"), input_info("inner_initial_condition") },
            outer_body_program, "trip_count", "initial_condition", "num_iteration",
            outer_input_primitive_maps, outer_output_primitive_maps, outer_back_edges, outer_trip_count)
    );

    /////////////////////////////////
    // network execution
    /////////////////////////////////
    cldnn::network::ptr network = get_network(engine, main_topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input_mem);
    network->set_input_data("trip_count", trip_count_mem);
    network->set_input_data("initial_condition", initial_condition_mem);
    network->set_input_data("inner_trip_count", inner_trip_count_mem);
    network->set_input_data("inner_initial_condition", inner_initial_condition_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output->get_layout();

    /////////////////////////////////
    // calculate expected output
    /////////////////////////////////
    std::vector<T> input_data2(input_data);
    std::vector<T> expected(input_data2.size());
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
    ASSERT_EQ(output_layout.batch(), 1);
    ASSERT_EQ(output_layout.feature(), 1);
    ASSERT_EQ(output_layout.spatial(0), 4);
    ASSERT_EQ(output_layout.spatial(1), 5);

    // check output values
    ASSERT_EQ(output_layout.count(), expected.size());
    {
        mem_lock<T> output_ptr{ output, get_test_stream() };
        for (size_t i = 0; i < output_layout.count(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], expected.at(i));
        }
    }

    // allocate new output memory, run and test everything once again
    layout loop_l = network->get_output_memory("loop")->get_layout();
    auto output_mem = engine.allocate_memory(loop_l);
    network->set_output_memory("loop", output_mem);

    set_values(input_mem, input_data);
    set_values(inner_operand_mem, inner_eltwise_operand);
    set_values(inner_trip_count_mem, { inner_trip_count });
    set_values(inner_initial_condition_mem, { inner_initial_condition });
    set_values(trip_count_mem, { outer_trip_count });
    set_values(initial_condition_mem, { outer_initial_condition });

    outputs = network->execute();
    auto output2 = outputs.begin()->second.get_memory();
    {
        mem_lock<T> output_ptr{ output2, get_test_stream() };
        for (size_t i = 0; i < output_layout.count(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], expected.at(i));
        }
    }
}

TEST(loop_gpu, basic_concat_nested) {
    test_loop_gpu_basic_concat_nested<float>(false);
}
#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(loop_gpu, basic_no_concat_cached) {
    test_loop_gpu_basic_no_concat<float>(true);
}

TEST(loop_gpu, basic_concat_cached) {
    test_loop_gpu_basic_concat<float>(true);
}
#endif // RUN_ALL_MODEL_CACHING_TESTS
TEST(loop_gpu, basic_concat_nested_cached) {
    test_loop_gpu_basic_concat_nested<float>(true);
}
