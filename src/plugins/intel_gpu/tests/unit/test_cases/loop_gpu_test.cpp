// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"

#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/graph/network.hpp>
#include "intel_gpu/plugin/common_utils.hpp"
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/eltwise.hpp"
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/loop.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "intel_gpu/primitives/permute.hpp"
#include <intel_gpu/graph/program.hpp>

#include "program_wrapper.h"

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
                                    std::vector<loop::backedge_mapping> back_edges,
                                    bool allow_new_shape_infer = false) {
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
    config.set_property(ov::intel_gpu::allow_new_shape_infer(allow_new_shape_infer));

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
        loop("loop", { input_info("num_iteration"), input_info("trip_count"), input_info("initial_condition"), input_info("input") }, body_program,
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
    int64_t trip_count = input_data.size()/eltwise_operand.size();
    int initial_condition = 1;

    // initialize input buffers
    set_values(input_mem, input_data);
    set_values(operand_mem, eltwise_operand);
    set_values<int64_t>(trip_count_mem, {trip_count});
    set_values<int64_t>(initial_condition_mem, {initial_condition});

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
        loop("loop", { input_info("num_iteration"), input_info("trip_count"), input_info("initial_condition"), input_info("input") }, body_program,
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
        input_layout("inner_input", { { 1, 1, 1, 4 }, data_types::f32, format::bfyx }),
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
        loop("inner_loop", { input_info("inner_num_iteration"), input_info("trip_count"), input_info("initial_condition"), input_info("inner_input") },
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
        loop("loop", { input_info("num_iteration"), input_info("trip_count"), input_info("initial_condition"),
                        input_info("input"), input_info("inner_trip_count"), input_info("inner_initial_condition") },
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

static void test_loop_gpu_wo_trip_count(ov::PartialShape body_input_layout,
                                        ov::PartialShape whole_layout,
                                        std::vector<float> input_data,
                                        std::vector<float> expected_output_data,
                                        size_t axis,
                                        size_t exit_value,
                                        bool is_caching_test = false) {
    auto& engine = get_test_engine();

    auto e_input_layout = cldnn::layout{ whole_layout, data_types::f32, format::bfyx };
    auto b_input_layout = cldnn::layout{ body_input_layout, data_types::f32, format::bfyx };
    auto const_layout = cldnn::layout{ {}, data_types::i64, format::bfyx };

    auto e_input_mem = engine.allocate_memory(e_input_layout); // b,f,x,y
    auto e_initial_condition_mem = engine.allocate_memory(const_layout);
    auto e_num_iteration_mem = engine.allocate_memory(const_layout);
    auto b_exit_value_mem = engine.allocate_memory(const_layout);
    auto b_index_inc_mem = engine.allocate_memory(const_layout);

    auto expected_output_layout = whole_layout;

    // initialize input buffers
    set_values(e_input_mem, input_data);
    set_values(e_initial_condition_mem, {1});
    set_values(b_exit_value_mem, {exit_value});
    set_values(b_index_inc_mem, {1});

    primitive_id body_current_iteration_id = "b_index";
    primitive_id body_execution_condition_id = "b_cond_exit_value";

    cldnn::topology body(
        input_layout(body_current_iteration_id, const_layout),
        input_layout("b_add_data", b_input_layout),
        input_layout("b_mul_data", b_input_layout),
        data("b_exit_value", b_exit_value_mem),
        data("b_index_inc", b_index_inc_mem),
        eltwise("b_index_update", input_info(body_current_iteration_id), input_info("b_index_inc"), eltwise_mode::sum),
        reorder("b_index_cast", input_info("b_index_update"),
                    cldnn::format::any, data_types::f32, {}, cldnn::reorder_mean_mode::subtract, cldnn::padding(), true),
        eltwise(body_execution_condition_id, input_info("b_index"), input_info("b_exit_value"), eltwise_mode::lt),
        eltwise("b_add", input_info("b_add_data"), input_info("b_index_cast"), eltwise_mode::sum),
        eltwise("b_mul", input_info("b_mul_data"), input_info("b_index_cast"), eltwise_mode::prod)
    );

    primitive_id trip_count_id = "";
    primitive_id actual_iteration_count_id = "actual_iteration_count";
    primitive_id initial_condition_id = "initial_condition";
    int64_t num_iterations = -1;

    std::vector<loop::io_primitive_map> input_primitive_maps {
        loop::io_primitive_map("input", "b_add_data", axis),
        loop::io_primitive_map("input", "b_mul_data", axis),
        loop::io_primitive_map(actual_iteration_count_id, body_current_iteration_id) };
    std::vector<loop::io_primitive_map> output_primitive_maps {
        loop::io_primitive_map(cldnn::input_info("loop", 0), cldnn::input_info("b_add", 0), axis),
        loop::io_primitive_map(cldnn::input_info("loop", 1), cldnn::input_info("b_mul", 0), axis) };
    std::vector<loop::backedge_mapping> back_edges {
        loop::backedge_mapping("b_index_update", body_current_iteration_id) };

    auto body_program = build_program(engine, body, body_execution_condition_id, output_primitive_maps, back_edges, true);

    cldnn::topology topology(
        input_layout("input", e_input_layout),
        input_layout(initial_condition_id, e_initial_condition_mem->get_layout()),
        mutable_data(actual_iteration_count_id, e_num_iteration_mem),
        loop("loop", { input_info(actual_iteration_count_id), input_info(initial_condition_id), input_info("input") }, body_program,
             trip_count_id, initial_condition_id, actual_iteration_count_id,
             input_primitive_maps, output_primitive_maps, back_edges,
             num_iterations, body_current_iteration_id, body_execution_condition_id, 2),
        eltwise("out_sum", input_info("loop", 0), input_info("loop", 1), eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", e_input_mem);
    network->set_input_data(initial_condition_id, e_initial_condition_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), 1);

    auto expected_num_iterations = (exit_value + 1);
    expected_output_layout[axis] = expected_num_iterations;
    auto e_output_layout = cldnn::layout{ expected_output_layout, data_types::f32, format::bfyx };

    auto num_iter_mem = network->get_output_memory(actual_iteration_count_id);
    if (num_iter_mem != nullptr) {
        mem_lock<int64_t> num_iter_ptr{ num_iter_mem, get_test_stream() };
        ASSERT_EQ(num_iter_ptr.data()[0], expected_num_iterations);
    }

    std::vector<float> expected(input_data.size());
    if (expected_output_data.size() == 0) {
        for (size_t j = 0; j < input_data.size(); j++) {
            auto val = static_cast<size_t>(j / 4) + 1;
            expected[j] = static_cast<float>(input_data[j] + val) + static_cast<float>(input_data[j] * val);
        }
    } else {
        expected = expected_output_data;
    }

    auto output_mem = outputs.begin()->second.get_memory();
    auto output_layout = output_mem->get_layout();
    ASSERT_EQ(output_layout.batch(), e_output_layout.batch());
    ASSERT_EQ(output_layout.feature(), e_output_layout.feature());
    ASSERT_EQ(output_layout.spatial(0), e_output_layout.spatial(0));
    ASSERT_EQ(output_layout.spatial(1), e_output_layout.spatial(1));
    // value check
    {
        mem_lock<float> output_ptr{ output_mem, get_test_stream() };
        for (size_t i = 0, iend = output_layout.count(); i < iend; ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], expected.at(i));
        }
    }
}


std::vector<float> input_data_5_4{
    1.0f,  2.0f, -15.f,  3.0f,
    4.0f, -15.f, 5.0f,  6.0f,
    -15.f, 7.0f, -15.f, 0.0f,
    0.0f, -15.f, 0.5f, -0.5f,
    -15.f, 8.0f,  1.5f,  5.2f
};

TEST(loop_gpu, support_dynamic_tensoriterator) {
    test_loop_gpu_wo_trip_count({ 1, 1, 1, 4 }, { 1, 1, 5, 4 }, input_data_5_4, std::vector<float>(), 2, 3);
}

TEST(loop_gpu, support_loop_w_dynamic_body_input) {
    test_loop_gpu_wo_trip_count({ 1, -1, 1, 4 }, { 1, 1, 5, 4 }, input_data_5_4, std::vector<float>(), 2, 3);
}

TEST(loop_gpu, support_dynamic_tensoriterator_cached) {
    test_loop_gpu_wo_trip_count({ 1, 1, 1, 4 }, { 1, 1, 5, 4 }, input_data_5_4, std::vector<float>(), 2, 3, true);
}

TEST(loop_gpu, support_loop_w_dynamic_body_input_cached) {
    test_loop_gpu_wo_trip_count({ 1, -1, 1, 4 }, { 1, 1, 5, 4 }, input_data_5_4, std::vector<float>(), 2, 3, true);
}

TEST(loop_gpu, support_dynamic_tensoriterator_feature_iter_1) {
    test_loop_gpu_wo_trip_count({ 1, 1, 4, 1}, { 1, 5, 4, 1}, input_data_5_4, std::vector<float>(), 1, 3);
}


TEST(loop_gpu, support_dynamic_tensoriterator_feature_iter_2) {
    test_loop_gpu_wo_trip_count({ 1, 1, 2, 2}, { 1, 5, 2, 2}, input_data_5_4, std::vector<float>(), 1, 3);
}

TEST(loop_gpu, support_dynamic_tensoriterator_batch_axis) {
    test_loop_gpu_wo_trip_count({ 1, 2, 2, 1}, { 5, 2, 2, 1}, input_data_5_4, std::vector<float>(), 0, 3);
}

TEST(loop_gpu, support_dynamic_tensoriterator_outer_axis) {
    // Reference output data (generated by reference::split)
    std::vector<float> output_data_5_4{
         3.0f,  5.0f,  -43.f,  11.0f,
        19.0f, -57.f,  29.0f,  34.0f,
        -85.f, 47.0f, -29.0f,   1.0f,
        2.0f, -43.0f,   5.0f,   1.0f,
        -71.0f, 44.0f,  14.0f,  36.2f
    };

    test_loop_gpu_wo_trip_count({ 2, 1, 1, 2}, { 2, 5, 1, 2}, input_data_5_4, output_data_5_4, 1, 4);
}

static void test_loop_gpu_wo_trip_count_w_multiple_shapes(ov::PartialShape body_input_layout,
                                        std::vector<ov::PartialShape> whole_layouts,
                                        std::vector<std::vector<float>> input_data_list,
                                        std::vector<float> expected_output_data,
                                        size_t axis,
                                        size_t exit_value,
                                        bool is_caching_test = false) {
    auto& engine = get_test_engine();

    auto b_input_layout = cldnn::layout{ body_input_layout, data_types::f32, format::bfyx };

    ov::PartialShape sliced_input_shape = body_input_layout;
    sliced_input_shape[axis] = 1;
    auto sliced_input_layout = cldnn::layout{ sliced_input_shape, data_types::f32, format::bfyx };

    auto const_layout = cldnn::layout{ {}, data_types::i64, format::bfyx };

    auto e_initial_condition_mem = engine.allocate_memory(const_layout);
    auto e_num_iteration_mem = engine.allocate_memory(const_layout);
    auto b_exit_value_mem = engine.allocate_memory(const_layout);
    auto b_index_inc_mem = engine.allocate_memory(const_layout);

    // initialize input buffers
    set_values(e_initial_condition_mem, {1});
    set_values(b_exit_value_mem, {exit_value});
    set_values(b_index_inc_mem, {1});
    set_values(e_num_iteration_mem, {0});

    primitive_id body_current_iteration_id = "b_index";
    primitive_id body_execution_condition_id = "b_cond_exit_value";

    cldnn::topology body(
        input_layout(body_current_iteration_id, const_layout),
        input_layout("b_add_data", sliced_input_layout),
        input_layout("b_mul_data", sliced_input_layout),
        data("b_exit_value", b_exit_value_mem),
        data("b_index_inc", b_index_inc_mem),
        eltwise("b_index_update", input_info(body_current_iteration_id), input_info("b_index_inc"), eltwise_mode::sum),
        reorder("b_index_cast", input_info("b_index_update"),
                    cldnn::format::any, data_types::f32, {}, cldnn::reorder_mean_mode::subtract, cldnn::padding(), true),
        eltwise(body_execution_condition_id, input_info("b_index"), input_info("b_exit_value"), eltwise_mode::lt),
        eltwise("b_add", input_info("b_add_data"), input_info("b_index_cast"), eltwise_mode::sum),
        eltwise("b_mul", input_info("b_mul_data"), input_info("b_index_cast"), eltwise_mode::prod));

    primitive_id trip_count_id = "";
    primitive_id actual_iteration_count_id = "actual_iteration_count";
    primitive_id initial_condition_id = "initial_condition";
    int64_t num_iterations = -1;

    std::vector<loop::io_primitive_map> input_primitive_maps {
        loop::io_primitive_map("input", "b_add_data", axis),
        loop::io_primitive_map("input", "b_mul_data", axis),
        loop::io_primitive_map(actual_iteration_count_id, body_current_iteration_id) };
    std::vector<loop::io_primitive_map> output_primitive_maps {
        loop::io_primitive_map(cldnn::input_info("loop", 0), cldnn::input_info("b_add", 0), axis),
        loop::io_primitive_map(cldnn::input_info("loop", 1), cldnn::input_info("b_mul", 0), axis) };
    std::vector<loop::backedge_mapping> back_edges {
        loop::backedge_mapping("b_index_update", body_current_iteration_id) };

    auto body_program = build_program(engine, body, body_execution_condition_id, output_primitive_maps, back_edges, true);

    auto const_shape = engine.allocate_memory({ov::PartialShape{4}, data_types::i32, format::bfyx});
    std::vector<int32_t> body_input_layouts;
    for (size_t i = 0; i < body_input_layout.size(); i++) {
        if (body_input_layout[i].is_dynamic())
            body_input_layouts.push_back(-1);
        else
            body_input_layouts.push_back(body_input_layout[i].get_length());
    }
    set_values<int32_t>(const_shape, body_input_layouts);

    cldnn::topology topology(
        input_layout("input_origin", b_input_layout),
        input_layout(initial_condition_id, e_initial_condition_mem->get_layout()),
        mutable_data(actual_iteration_count_id, e_num_iteration_mem),

        shape_of("shape_of_input", input_info("input_origin"), data_types::i32),
        reduce("reduced_shape", input_info("shape_of_input"), reduce_mode::prod, {0}, true),
        reshape("reshape1", input_info("input_origin"), input_info("reduced_shape"), false, ov::PartialShape::dynamic(1)),
        data("const", const_shape),
        reshape("input", input_info("reshape1"), input_info("const"), false, ov::PartialShape::dynamic(4)),

        loop("loop", { input_info(actual_iteration_count_id), input_info(initial_condition_id), input_info("input") }, body_program,
             trip_count_id, initial_condition_id, actual_iteration_count_id,
             input_primitive_maps, output_primitive_maps, back_edges,
             num_iterations, body_current_iteration_id, body_execution_condition_id, 2),
        eltwise("out_sum", input_info("loop", 0), input_info("loop", 1), eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    for (size_t i = 0 ; i < whole_layouts.size(); i++) {
        auto whole_layout = whole_layouts[i];
        auto input_data = input_data_list[i];

        // initialize input buffers
        set_values(e_initial_condition_mem, {1});
        set_values(b_exit_value_mem, {exit_value});
        set_values(b_index_inc_mem, {1});
        set_values(e_num_iteration_mem, {0});

        auto e_input_layout = cldnn::layout{ whole_layout, data_types::f32, format::bfyx };
        auto e_input_mem = engine.allocate_memory(e_input_layout); // b,f,x,y
        auto expected_output_layout = whole_layout;
        set_values(e_input_mem, input_data);
        network->set_input_data("input_origin", e_input_mem);

        network->set_input_data(initial_condition_id, e_initial_condition_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), 1);

        auto expected_num_iterations = (exit_value + 1);
        expected_output_layout[axis] = expected_num_iterations;
        auto e_output_layout = cldnn::layout{ expected_output_layout, data_types::f32, format::bfyx };

        auto num_iter_mem = network->get_output_memory(actual_iteration_count_id);
        if (num_iter_mem != nullptr) {
            mem_lock<int64_t> num_iter_ptr{ num_iter_mem, get_test_stream() };
            ASSERT_EQ(num_iter_ptr.data()[0], expected_num_iterations);
        }

        std::vector<float> expected(input_data.size());
        if (expected_output_data.size() == 0) {
            size_t unit = 1;
            for (size_t k = axis; k < whole_layout.size(); k++) {
                unit *= whole_layout[k].get_length();
            }

            for (size_t j = 0; j < input_data.size(); j++) {
                auto val = static_cast<size_t>((j % unit) / 4) + 1;
                expected[j] = static_cast<float>(input_data[j] + val) + static_cast<float>(input_data[j] * val);
            }
        } else {
            expected = expected_output_data;
        }

        auto output_mem = outputs.begin()->second.get_memory();
        auto output_layout = output_mem->get_layout();
        ASSERT_EQ(output_layout.batch(), e_output_layout.batch());
        ASSERT_EQ(output_layout.feature(), e_output_layout.feature());
        ASSERT_EQ(output_layout.spatial(0), e_output_layout.spatial(0));
        ASSERT_EQ(output_layout.spatial(1), e_output_layout.spatial(1));
        // value check
        {
            mem_lock<float> output_ptr{ output_mem, get_test_stream() };
            for (size_t i = 0, iend = output_layout.count(); i < iend; ++i) {
                ASSERT_FLOAT_EQ(output_ptr[i], expected.at(i));
            }
        }
    }
}

static void test_loop_gpu_multiple_shapes(ov::PartialShape body_input_layout,
                                        std::vector<ov::PartialShape> whole_layouts,
                                        std::vector<std::vector<float>> input_data_list,
                                        std::vector<float> expected_output_data,
                                        int32_t axis,
                                        size_t exit_value,
                                        bool is_caching_test = false) {
    auto& engine = get_test_engine();

    auto b_input_layout = cldnn::layout{ body_input_layout, data_types::f32, format::bfyx };
    auto const_layout = cldnn::layout{ {}, data_types::i64, format::bfyx };

    auto e_initial_condition_mem = engine.allocate_memory(const_layout);
    auto e_num_iteration_mem = engine.allocate_memory(const_layout);
    auto b_exit_value_mem = engine.allocate_memory(const_layout);
    auto b_index_inc_mem = engine.allocate_memory(const_layout);

    // initialize input buffers
    set_values(e_initial_condition_mem, {1});
    set_values(b_exit_value_mem, {exit_value});
    set_values(b_index_inc_mem, {1});
    set_values(e_num_iteration_mem, {10});

    primitive_id body_current_iteration_id = "b_index";
    primitive_id body_execution_condition_id = "b_cond_exit_value";

    cldnn::topology body(
        input_layout(body_current_iteration_id, const_layout),
        input_layout("b_add_data", b_input_layout),
        input_layout("b_mul_data", b_input_layout),
        data("b_exit_value", b_exit_value_mem),
        data("b_index_inc", b_index_inc_mem),
        eltwise("b_index_update", input_info(body_current_iteration_id), input_info("b_index_inc"), eltwise_mode::sum),
        reorder("b_index_cast", input_info("b_index_update"),
                    cldnn::format::any, data_types::f32, {}, cldnn::reorder_mean_mode::subtract, cldnn::padding(), true),
        eltwise(body_execution_condition_id, input_info("b_index"), input_info("b_exit_value"), eltwise_mode::lt),
        eltwise("b_add", input_info("b_add_data"), input_info("b_index_cast"), eltwise_mode::sum),
        eltwise("b_mul", input_info("b_mul_data"), input_info("b_index_cast"), eltwise_mode::prod));

    primitive_id trip_count_id = "";
    primitive_id actual_iteration_count_id = "actual_iteration_count";
    primitive_id initial_condition_id = "initial_condition";
    int64_t num_iterations = -1;

    std::vector<loop::io_primitive_map> input_primitive_maps {
        loop::io_primitive_map("input1", "b_add_data", axis),
        loop::io_primitive_map("input2", "b_mul_data", axis),
        loop::io_primitive_map(actual_iteration_count_id, body_current_iteration_id) };
    std::vector<loop::io_primitive_map> output_primitive_maps {
        loop::io_primitive_map(cldnn::input_info("loop", 0), cldnn::input_info("b_add", 0), axis),
        loop::io_primitive_map(cldnn::input_info("loop", 1), cldnn::input_info("b_mul", 0), axis) };
    std::vector<loop::backedge_mapping> back_edges {
        loop::backedge_mapping("b_index_update", body_current_iteration_id) };

    auto body_program = build_program(engine, body, body_execution_condition_id, output_primitive_maps, back_edges, true);

    auto const_shape = engine.allocate_memory({ov::PartialShape{4}, data_types::i32, format::bfyx});
    std::vector<int32_t> body_input_layouts;
    for (size_t i = 0; i < body_input_layout.size(); i++) {
        if (body_input_layout[i].is_dynamic())
            body_input_layouts.push_back(-1);
        else
            body_input_layouts.push_back(body_input_layout[i].get_length());
    }
    set_values<int32_t>(const_shape, body_input_layouts);

    cldnn::topology topology(
        input_layout("input_origin", b_input_layout),
        input_layout(initial_condition_id, e_initial_condition_mem->get_layout()),
        mutable_data(actual_iteration_count_id, e_num_iteration_mem),
        permute("input2", input_info("input_origin"), {0, 1, 2, 3}),
        data("const", const_shape),
        permute("permute1", input_info("input_origin"), {0, 1, 2, 3}),
        concatenation("input1", {input_info("permute1"), input_info("input_origin")}, 0),
        loop("loop",
             {input_info(actual_iteration_count_id), input_info(initial_condition_id), input_info("input1"), input_info("input2")},
             body_program, trip_count_id, initial_condition_id, actual_iteration_count_id,
             input_primitive_maps, output_primitive_maps, back_edges,
             num_iterations, body_current_iteration_id, body_execution_condition_id, 2),
        eltwise("out_sum", input_info("loop", 0), input_info("loop", 1), eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    for (size_t i = 0 ; i < whole_layouts.size(); i++) {
        auto whole_layout = whole_layouts[i];
        auto input_data = input_data_list[i];

        set_values(e_initial_condition_mem, {1});
        set_values(b_exit_value_mem, {exit_value});
        set_values(b_index_inc_mem, {1});
        set_values(e_num_iteration_mem, {10});

        auto e_input_layout = cldnn::layout{ whole_layout, data_types::f32, format::bfyx };
        auto e_input_mem = engine.allocate_memory(e_input_layout); // b,f,x,y
        auto expected_output_layout = whole_layout;
        set_values(e_input_mem, input_data);
        network.set_input_data("input_origin", e_input_mem);

        network.set_input_data(initial_condition_id, e_initial_condition_mem);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), 1);
        auto output_layout = outputs.begin()->second.get_layout();
        auto input_layout = network.get_primitive("input1")->get_output_layout();

        ASSERT_EQ(output_layout.batch(), input_layout.batch());
        ASSERT_EQ(output_layout.feature(), input_layout.feature());
        ASSERT_EQ(output_layout.spatial(0), input_layout.spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input_layout.spatial(1));
    }
}

static void test_loop_gpu_multiple_shapes_single_shared(ov::PartialShape body_input_layout,
                                        std::vector<ov::PartialShape> whole_layouts,
                                        std::vector<std::vector<float>> input_data_list,
                                        std::vector<float> expected_output_data,
                                        int32_t axis,
                                        size_t exit_value,
                                        bool is_caching_test = false) {
    auto& engine = get_test_engine();

    auto b_input_layout = cldnn::layout{ body_input_layout, data_types::f32, format::bfyx };
    auto const_layout = cldnn::layout{ {}, data_types::i64, format::bfyx };

    auto e_initial_condition_mem = engine.allocate_memory(const_layout);
    auto e_num_iteration_mem = engine.allocate_memory(const_layout);
    auto b_exit_value_mem = engine.allocate_memory(const_layout);
    auto b_index_inc_mem = engine.allocate_memory(const_layout);

    // initialize input buffers
    set_values(e_initial_condition_mem, {1});
    set_values(b_exit_value_mem, {exit_value});
    set_values(b_index_inc_mem, {1});
    set_values(e_num_iteration_mem, {10});

    primitive_id body_current_iteration_id = "b_index";
    primitive_id body_execution_condition_id = "b_cond_exit_value";

    cldnn::topology body(
        input_layout(body_current_iteration_id, const_layout),
        input_layout("b_parameter", b_input_layout),
        data("b_exit_value", b_exit_value_mem),
        data("b_index_inc", b_index_inc_mem),
        eltwise("b_index_update", input_info(body_current_iteration_id), input_info("b_index_inc"), eltwise_mode::sum),
        eltwise("b_permute", input_info("b_parameter"), input_info("b_index_update"), eltwise_mode::sum),
        reorder("b_result", input_info("b_permute"), b_input_layout),
        eltwise(body_execution_condition_id, input_info(body_current_iteration_id), input_info("b_exit_value"), eltwise_mode::lt)
    );

    primitive_id trip_count_id = "";
    primitive_id actual_iteration_count_id = "actual_iteration_count";
    primitive_id initial_condition_id = "initial_condition";
    int64_t num_iterations = -1;

    std::vector<loop::io_primitive_map> input_primitive_maps {
        loop::io_primitive_map("input", "b_parameter", axis),
        loop::io_primitive_map(actual_iteration_count_id, body_current_iteration_id) };
    std::vector<loop::io_primitive_map> output_primitive_maps {
        loop::io_primitive_map(cldnn::input_info("loop"), cldnn::input_info("b_result"), axis) };
    std::vector<loop::backedge_mapping> back_edges {
        loop::backedge_mapping("b_result", "b_parameter"),
        loop::backedge_mapping("b_index_update", body_current_iteration_id) };

    auto body_program = build_program(engine, body, body_execution_condition_id, output_primitive_maps, back_edges, true);

    auto const_shape = engine.allocate_memory({ov::PartialShape{4}, data_types::i32, format::bfyx});
    std::vector<int32_t> body_input_layouts;
    for (size_t i = 0; i < body_input_layout.size(); i++) {
        if (body_input_layout[i].is_dynamic())
            body_input_layouts.push_back(-1);
        else
            body_input_layouts.push_back(body_input_layout[i].get_length());
    }
    set_values<int32_t>(const_shape, body_input_layouts);

    cldnn::topology topology(
        input_layout("input_origin", b_input_layout),
        input_layout(initial_condition_id, e_initial_condition_mem->get_layout()),
        mutable_data(actual_iteration_count_id, e_num_iteration_mem),
        permute("input2", input_info("input_origin"), {0, 1, 2, 3}),
        data("const", const_shape),
        permute("permute1", input_info("input_origin"), {0, 1, 2, 3}),
        concatenation("input", {input_info("permute1"), input_info("input_origin")}, 0),
        loop("loop",
             {input_info(actual_iteration_count_id), input_info(initial_condition_id), input_info("input")},
             body_program, trip_count_id, initial_condition_id, actual_iteration_count_id,
             input_primitive_maps, output_primitive_maps, back_edges,
             num_iterations, body_current_iteration_id, body_execution_condition_id, 1),
        permute("result", input_info("loop"), {0, 1, 2, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    for (size_t i = 0 ; i < whole_layouts.size(); i++) {
        auto whole_layout = whole_layouts[i];
        auto input_data = input_data_list[i];

        set_values(e_initial_condition_mem, {1});
        set_values(b_exit_value_mem, {exit_value});
        set_values(b_index_inc_mem, {1});
        set_values(e_num_iteration_mem, {10});

        auto e_input_layout = cldnn::layout{ whole_layout, data_types::f32, format::bfyx };
        auto e_input_mem = engine.allocate_memory(e_input_layout); // b,f,x,y
        auto expected_output_layout = whole_layout;
        set_values(e_input_mem, input_data);

        network.set_input_data("input_origin", e_input_mem);
        network.set_input_data(initial_condition_id, e_initial_condition_mem);

        auto outputs = network.execute();
        auto output_layout = outputs.begin()->second.get_layout();
        auto input_layout = network.get_primitive("input")->get_output_layout();

        ASSERT_EQ(output_layout.feature(), input_layout.feature());
        ASSERT_EQ(output_layout.spatial(0), input_layout.spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input_layout.spatial(1));
    }
}

std::vector<float> input_data_2_4{
    1.0f,  2.0f,
    4.0f, -15.f,
    -15.f, 7.0f,
    0.0f, -15.f,
};

std::vector<float> input_data_4_4{
    1.0f,  2.0f, -15.f,  3.0f,
    4.0f, -15.f, 5.0f,  6.0f,
    -15.f, 7.0f, -15.f, 0.0f,
    0.0f, -15.f, 0.5f, -0.5f,
};

std::vector<float> input_data_2_4_4{
    1.0f,  2.0f, -15.f,  3.0f,
    4.0f, -15.f, 5.0f,  6.0f,
    -15.f, 7.0f, -15.f, 0.0f,
    0.0f, -15.f, 0.5f, -0.5f,

    1.0f,  2.0f, -15.f,  3.0f,
    4.0f, -15.f, 5.0f,  6.0f,
    -15.f, 7.0f, -15.f, 0.0f,
    0.0f, -15.f, 0.5f, -0.5f,
};

TEST(loop_gpu, support_loop_w_dynamic_input_w_various_shapes1) {
    test_loop_gpu_wo_trip_count_w_multiple_shapes(
        { 1, -1, 4, 4 },
        {{ 1, 1, 4, 4 }, { 1, 2, 4, 4 }},   // axis value should be iter_num = (exit_value + 1)
        {input_data_4_4, input_data_2_4_4},
        std::vector<float>(),
        2, 3);
}

TEST(loop_gpu, support_loop_w_dynamic_input_w_various_shapes2) {
    test_loop_gpu_multiple_shapes(
        { 1, -1, -1, 4 },
        {{ 1, 1, 2, 4 }, { 1, 1, 4, 4 }, { 1, 2, 4, 4 }},
        {input_data_2_4, input_data_4_4, input_data_2_4_4},
        std::vector<float>(),
        -1, 10);
}

TEST(loop_gpu, support_loop_w_dynamic_input_w_various_shapes3) {
    test_loop_gpu_multiple_shapes_single_shared(
        { 1, -1, 560 },
        {{ 1, 58, 560 }, { 1, 87, 560 }, { 1, 72, 560 }, { 1, 88, 560 }, { 1, 89, 560 }},
        {input_data_2_4_4, input_data_2_4_4, input_data_2_4_4, input_data_2_4_4, input_data_2_4_4},
        std::vector<float>(),
        -1, 20);
}

static void test_loop_gpu_wo_trip_count_update_primitive_id(ov::PartialShape body_input_layout,
                                        std::vector<ov::PartialShape> whole_layouts,
                                        std::vector<std::vector<float>> input_data_list,
                                        std::vector<float> expected_output_data,
                                        size_t axis,
                                        size_t exit_value,
                                        bool is_caching_test = false) {
    auto& engine = get_test_engine();

    auto b_input_layout = cldnn::layout{ body_input_layout, data_types::f32, format::bfyx };

    ov::PartialShape sliced_input_shape = body_input_layout;
    sliced_input_shape[axis] = 1;
    auto sliced_input_layout = cldnn::layout{ sliced_input_shape, data_types::f32, format::bfyx };

    auto const_layout = cldnn::layout{ {}, data_types::i64, format::bfyx };

    auto e_initial_condition_mem = engine.allocate_memory(const_layout);
    auto e_num_iteration_mem = engine.allocate_memory(const_layout);
    auto b_exit_value_mem = engine.allocate_memory(const_layout);
    auto b_index_inc_mem = engine.allocate_memory(const_layout);

    // initialize input buffers
    set_values(e_initial_condition_mem, {1});
    set_values(b_exit_value_mem, {exit_value});
    set_values(b_index_inc_mem, {1});
    set_values(e_num_iteration_mem, {0});

    primitive_id body_current_iteration_id = "b_index";
    primitive_id body_execution_condition_id = "b_cond_exit_value";

    cldnn::topology body(
        input_layout(body_current_iteration_id, const_layout),
        input_layout("b_add_data", sliced_input_layout),
        input_layout("b_mul_data", sliced_input_layout),
        data("b_exit_value", b_exit_value_mem),
        data("b_index_inc", b_index_inc_mem),
        eltwise("b_index_update", input_info(body_current_iteration_id), input_info("b_index_inc"), eltwise_mode::sum),
        reorder("b_index_cast", input_info("b_index_update"),
                    cldnn::format::any, data_types::f32, {}, cldnn::reorder_mean_mode::subtract, cldnn::padding(), true),
        eltwise(body_execution_condition_id, input_info("b_index"), input_info("b_exit_value"), eltwise_mode::lt),
        eltwise("b_add", input_info("b_add_data"), input_info("b_index_cast"), eltwise_mode::sum),
        eltwise("b_mul", input_info("b_mul_data"), input_info("b_index_cast"), eltwise_mode::prod));

    primitive_id trip_count_id = "";
    primitive_id actual_iteration_count_id = "actual_iteration_count";
    primitive_id initial_mean = "initial_mean";

    primitive_id initial_condition_id = "initial_condition";
    primitive_id initial_condition_id_elt = "initial_condition_elt";
    primitive_id initial_condition_id_reorder = "initial_condition_reorder";
    primitive_id initial_condition_id_reorder2 = "initial_condition_reorder2";
    int64_t num_iterations = -1;

    std::vector<loop::io_primitive_map> input_primitive_maps {
        loop::io_primitive_map("input", "b_add_data", axis),
        loop::io_primitive_map("input", "b_mul_data", axis),
        loop::io_primitive_map(actual_iteration_count_id, body_current_iteration_id) };
    std::vector<loop::io_primitive_map> output_primitive_maps {
        loop::io_primitive_map(cldnn::input_info("loop", 0), cldnn::input_info("b_add", 0), axis),
        loop::io_primitive_map(cldnn::input_info("loop", 1), cldnn::input_info("b_mul", 0), axis) };
    std::vector<loop::backedge_mapping> back_edges {
        loop::backedge_mapping("b_index_update", body_current_iteration_id) };

    auto body_program = build_program(engine, body, body_execution_condition_id, output_primitive_maps, back_edges, true);

    auto const_shape = engine.allocate_memory({ov::PartialShape{4}, data_types::i32, format::bfyx});

    std::vector<int32_t> body_input_layouts;
    for (size_t i = 0; i < body_input_layout.size(); i++) {
        if (body_input_layout[i].is_dynamic())
            body_input_layouts.push_back(-1);
        else
            body_input_layouts.push_back(body_input_layout[i].get_length());
    }
    set_values<int32_t>(const_shape, body_input_layouts);
    const std::vector<float> values_to_subtract = {0.f};

    cldnn::topology topology(
        input_layout("input_origin", b_input_layout),
        input_layout(initial_condition_id, e_initial_condition_mem->get_layout()),
        mutable_data(actual_iteration_count_id, e_num_iteration_mem),

        reorder(initial_condition_id_reorder, input_info(initial_condition_id), cldnn::format::any, data_types::f32, values_to_subtract),
        reorder(initial_condition_id_reorder2, input_info(initial_condition_id_reorder), cldnn::format::any, data_types::i32),  // should be fused to test updating input id of loop

        shape_of("shape_of_input", input_info("input_origin"), data_types::i32),
        reduce("reduced_shape", input_info("shape_of_input"), reduce_mode::prod, {0}, true),
        reshape("reshape1", input_info("input_origin"), input_info("reduced_shape"), false, ov::PartialShape::dynamic(1)),
        data("const", const_shape),
        reshape("input", input_info("reshape1"), input_info("const"), false, ov::PartialShape::dynamic(4)),

        loop("loop", { input_info(actual_iteration_count_id), input_info(initial_condition_id_reorder2), input_info("input") }, body_program,
             trip_count_id, initial_condition_id_reorder2, actual_iteration_count_id,
             input_primitive_maps, output_primitive_maps, back_edges,
             num_iterations, body_current_iteration_id, body_execution_condition_id, 2),
        eltwise("out_sum", input_info("loop", 0), input_info("loop", 1), eltwise_mode::sum));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    for (size_t i = 0 ; i < whole_layouts.size(); i++) {
        auto whole_layout = whole_layouts[i];
        auto input_data = input_data_list[i];

        // initialize input buffers
        set_values(e_initial_condition_mem, {1});
        set_values(b_exit_value_mem, {exit_value});
        set_values(b_index_inc_mem, {1});
        set_values(e_num_iteration_mem, {0});

        auto e_input_layout = cldnn::layout{ whole_layout, data_types::f32, format::bfyx };
        auto e_input_mem = engine.allocate_memory(e_input_layout); // b,f,x,y
        auto expected_output_layout = whole_layout;
        set_values(e_input_mem, input_data);
        network->set_input_data("input_origin", e_input_mem);

        network->set_input_data(initial_condition_id, e_initial_condition_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), 1);

        auto expected_num_iterations = (exit_value + 1);
        expected_output_layout[axis] = expected_num_iterations;
        auto e_output_layout = cldnn::layout{ expected_output_layout, data_types::f32, format::bfyx };

        auto num_iter_mem = network->get_output_memory(actual_iteration_count_id);
        if (num_iter_mem != nullptr) {
            mem_lock<int64_t> num_iter_ptr{ num_iter_mem, get_test_stream() };
            ASSERT_EQ(num_iter_ptr.data()[0], expected_num_iterations);
        }

        std::vector<float> expected(input_data.size());
        if (expected_output_data.size() == 0) {
            size_t unit = 1;
            for (size_t k = axis; k < whole_layout.size(); k++) {
                unit *= whole_layout[k].get_length();
            }

            for (size_t j = 0; j < input_data.size(); j++) {
                auto val = static_cast<size_t>((j % unit) / 4) + 1;
                expected[j] = static_cast<float>(input_data[j] + val) + static_cast<float>(input_data[j] * val);
            }
        } else {
            expected = expected_output_data;
        }

        auto output_mem = outputs.begin()->second.get_memory();
        auto output_layout = output_mem->get_layout();
        ASSERT_EQ(output_layout.batch(), e_output_layout.batch());
        ASSERT_EQ(output_layout.feature(), e_output_layout.feature());
        ASSERT_EQ(output_layout.spatial(0), e_output_layout.spatial(0));
        ASSERT_EQ(output_layout.spatial(1), e_output_layout.spatial(1));
        // value check
        {
            mem_lock<float> output_ptr{ output_mem, get_test_stream() };
            for (size_t i = 0, iend = output_layout.count(); i < iend; ++i) {
                ASSERT_FLOAT_EQ(output_ptr[i], expected.at(i));
            }
        }
    }
}


TEST(loop_gpu, support_loop_w_dynamic_input_update_primitive_id) {
    test_loop_gpu_wo_trip_count_update_primitive_id(
        { 1, -1, 4, 4 },
        {{ 1, 1, 4, 4 }},   // axis value should be iter_num = (exit_value + 1)
        {input_data_4_4, input_data_2_4_4},
        std::vector<float>(),
        2, 3);
}

template <typename T>
void test_loop_gpu_zero_bytes_layout(bool is_caching_test)
{
    auto& engine = get_test_engine();

    // shape for zero bytes layout
    auto trip_count_mem = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 1}});

    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto operand_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
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
        loop("loop", { input_info("num_iteration"), input_info("trip_count"), input_info("initial_condition"), input_info("input") }, body_program,
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

TEST(loop_gpu, zero_bytes_layout) {
    test_loop_gpu_zero_bytes_layout<float>(false);
}
