// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "test_utils.h"

#include <intel_gpu/primitives/arg_max_min.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/concatenation.hpp>

using namespace cldnn;
using namespace tests;

template<typename T = float>
static std::vector<T> generateVector(size_t sz) {
    std::vector<T> vec(sz);
    T n = 0;
    std::generate(vec.begin(), vec.end(), [&n]() {
            return n++;
        });
    return vec;
}

template <typename T>
void test_basic(bool is_caching_test) {
    auto& engine = get_test_engine();

    const int b = 3;
    const int f = 2;
    const int y = 5;
    const int x = 5;

    auto input_data = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });

    const auto inputSize = input_data->get_layout().count();
    auto inputVals = generateVector(inputSize);
    set_values(input_data, inputVals);

    topology topology;
    topology.add(input_layout("Input", input_data->get_layout()));
    topology.add(
        reorder("reorder", input_info("Input"), input_data->get_layout())
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Input", input_data);
    network->set_output_memory("reorder", output_mem);

    auto outputs = network->execute();

    auto output = outputs.at("reorder").get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *output));

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < inputVals.size(); ++i) {
        ASSERT_TRUE(are_equal(inputVals[i], output_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, basic) {
    test_basic<float>(false);
}

TEST(set_output_memory_gpu, basic_const) {
    auto& engine = get_test_engine();

    const int b = 3;
    const int f = 2;
    const int y = 5;
    const int x = 5;

    auto input_data = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto const_data = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_const_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });

    const int inputSize = static_cast<int>(input_data->get_layout().count());
    auto inputVals = generateVector(inputSize);
    auto constVals = generateVector(inputSize);
    set_values(input_data, inputVals);
    set_values(const_data, constVals);

    topology topology;
    topology.add(input_layout("Input", input_data->get_layout()));
    topology.add(data("Const", const_data));
    topology.add(
            reorder("reorder_dyn", input_info("Input"), input_data->get_layout()),
            reorder("reorder_const", input_info("Const"), input_data->get_layout())
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input_data);
    network.set_output_memory("reorder_dyn", output_mem);
    network.set_output_memory("reorder_const", output_const_mem);

    auto outputs = network.execute();

    auto output_dyn = outputs.at("reorder_dyn").get_memory();
    auto output_const = outputs.at("reorder_const").get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *output_dyn));

    cldnn::mem_lock<float> output_dyn_ptr(output_dyn, get_test_stream());
    cldnn::mem_lock<float> output_const_ptr(output_const, get_test_stream());

    for (size_t i = 0; i < inputVals.size(); ++i) {
        ASSERT_TRUE(are_equal(inputVals[i], output_dyn_ptr[i])) << i;
    }

    for (size_t i = 0; i < inputVals.size(); ++i) {
        ASSERT_TRUE(are_equal(inputVals[i], output_const_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, basic_mutable) {
    auto& engine = get_test_engine();

    const int b = 3;
    const int f = 2;
    const int y = 5;
    const int x = 5;
    auto input_data = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto md = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mutable_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { b, f, x, y } });
    const auto inputSize = input_data->get_layout().count();
    auto inputVals = generateVector(inputSize);
    auto mutableVals = generateVector(inputSize);
    set_values(input_data, inputVals);
    set_values(md, mutableVals);

    topology topology;
    topology.add(input_layout("Input", input_data->get_layout()));
    topology.add(mutable_data("Mutable", md));
    topology.add(
            reorder("reorder_dyn", input_info("Input"), input_data->get_layout()),
            reorder("reorder_mutable", input_info("Mutable"), input_data->get_layout())
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input_data);
    network.set_output_memory("reorder_dyn", output_mem);
    network.set_output_memory("reorder_mutable", output_mutable_mem);

    auto outputs = network.execute();

    auto output_dyn = outputs.at("reorder_dyn").get_memory();
    auto output_mutable = outputs.at("reorder_mutable").get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *output_dyn));
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mutable_mem, *output_mutable));

    cldnn::mem_lock<float> output_dyn_ptr(output_dyn, get_test_stream());
    cldnn::mem_lock<float> output_mutable_mem_ptr(output_mutable_mem, get_test_stream());

    for (size_t i = 0; i < inputVals.size(); ++i) {
        ASSERT_TRUE(are_equal(inputVals[i], output_dyn_ptr[i])) << i;
    }

    for (size_t i = 0; i < inputVals.size(); ++i) {
        ASSERT_TRUE(are_equal(inputVals[i], output_mutable_mem_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, top_k1) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    auto top_k_input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});
    auto output_mem =
        engine.allocate_memory({data_types::f32, format::bfyx, {top_k, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(arg_max_min("arg_max", { input_info("input"), input_info("const") }, ov::op::TopKMode::MIN, top_k, 0));
    topology.add(reorder("reorder", input_info("arg_max"), output_mem->get_layout()));

    std::vector<float> input_vec = {
            //y0x0 y0x1 y1x0 y1x1
            /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
            /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,

            /*b1f0*/3.f,  0.5f,  7.f,   10.f,
            /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
            /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b1f3*/4.f,  0.5f,  8.f,   8.2f
    };
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_output_memory("reorder", output_mem);
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *output));

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_mem_ptr(output_mem, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_TRUE(are_equal(output_mem_ptr[i], output_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, top_k2) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto second_output = engine.allocate_memory({ data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    auto second_output_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max", { input_info("input"), input_info("const"), input_info("second_output") }, ov::op::TopKMode::MIN, top_k, 0));
    topology.add(reorder("reorder", input_info("arg_max"), second_output->get_layout()));

    std::vector<float> input_vec = {
            //y0x0 y0x1 y1x0 y1x1
            /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
            /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,

            /*b1f0*/3.f,  0.5f,  7.f,   10.f,
            /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
            /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b1f3*/4.f,  0.5f,  8.f,   8.2f
    };
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_output_memory("reorder", second_output_mem);
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    ASSERT_TRUE(engine.is_the_same_buffer(*second_output_mem, *output));

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_mem_ptr(second_output_mem, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_TRUE(are_equal(output_mem_ptr[i], output_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, basic_opt) {
    GTEST_SKIP();
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 1;
    auto& engine = get_test_engine();

    tensor ishape = { batch_num, feature_num, x_size , y_size };
    layout il = { data_types::f32, format::bfyx, ishape };

    tensor oshape = { batch_num*2, feature_num, x_size , y_size };
    layout ol = { data_types::f32, format::bfyx, oshape };

    auto input1 = engine.allocate_memory(il);
    std::vector<float> input_vec1 = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -2.1f, -3.1f, -4.1f,
        /*b0f1*/2.1f,  2.1f,  3.1f,  4.1f,
        /*b0f2*/3.1f, -3.1f,  3.1f,  5.1f,
        /*b0f3*/1.1f,  1.1f,  1.1f,  1.1f
    };
    set_values(input1, input_vec1);

    auto input2 = engine.allocate_memory(il);
    std::vector<float> input_vec2 = {
        //y0x0 y0x1 y1x0 y1x1
        /*b1f0*/0.2f, -2.2f, -3.2f, -4.2f,
        /*b1f1*/2.2f,  2.2f,  3.2f,  4.2f,
        /*b1f2*/3.2f, -3.2f,  3.2f,  5.2f,
        /*b1f3*/1.2f,  1.2f,  1.2f, -1.2f
    };
    set_values(input2, input_vec2);

    activation_additional_params params1 = { 0.5f, 2.5f };
    activation_additional_params params2 = { -2.5f, 0.5f };

    std::vector<float> output_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.5f, 0.5f, 0.5f, 0.5f,
        /*b0f1*/2.1f, 2.1f, 2.5f, 2.5f,
        /*b0f2*/2.5f, 0.5f, 2.5f, 2.5f,
        /*b0f3*/1.1f, 1.1f, 1.1f, 1.1f,

        /*b1f0*/0.2f, -2.2f, -2.5f, -2.5f,
        /*b1f1*/0.5f,  0.5f,  0.5f,  0.5f,
        /*b1f2*/0.5f, -2.5f,  0.5f,  0.5f,
        /*b1f3*/0.5f,  0.5f,  0.5f, -1.2f
    };
    auto output_mem = engine.allocate_memory(ol);

    topology topology;
    topology.add(input_layout("input1", il));
    topology.add(activation("clamp1", input_info("input1"), activation_func::clamp, params1));
    topology.add(input_layout("input2", il));
    topology.add(activation("clamp2", input_info("input2"), activation_func::clamp, params2));
    topology.add(reshape("reshape1", input_info("clamp1"), ishape));
    topology.add(reshape("reshape2", input_info("clamp2"), ishape));
    topology.add(concatenation("concat", { input_info("reshape1"), input_info("reshape2") }, 0, data_types::f32));
    topology.add(reshape("reshape3", input_info("concat"), oshape));
    topology.add(reorder("reorder", input_info("reshape3"), ol));
    topology.add(reorder("reorder2", input_info("reorder"), ol));

    primitive_id outputID = "reorder3";
    topology.add(reorder(outputID, input_info("concat"), ol));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_output_memory(outputID, output_mem);

    auto outputs = network.execute();
    auto output = outputs.at(outputID).get_memory();
    //  check for correct output memory setting
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *output));
    //  check for memory set propagation
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *network.get_output_memory("concat")));
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *network.get_output_memory("clamp1")));
    ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *network.get_output_memory("clamp2")));

    //  check for correct result
    cldnn::mem_lock<float> output_ptr(output_mem, get_test_stream());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_TRUE(are_equal(output_ptr[i], output_vec[i])) << i;
    }
}

TEST(set_output_memory_gpu, mutable_output_data) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto final_output = engine.allocate_memory({ data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    auto second_input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });

    topology topology;
    topology.add(input_layout("Add_1396", input->get_layout()));
    topology.add(cldnn::mutable_data("second_input", second_input));
    topology.add(cldnn::mutable_data("12220_md_write", final_output));
    topology.add(arg_max_min("arg_max", { input_info("Add_1396"), input_info("second_input"), input_info("12220_md_write") }, ov::op::TopKMode::MIN, top_k, 0));
    topology.add(cldnn::mutable_data("pred/sink_port_0", { input_info("arg_max")}, final_output) );

    std::vector<float> input_vec = {
            //y0x0 y0x1 y1x0 y1x1
            /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
            /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,

            /*b1f0*/3.f,  0.5f,  7.f,   10.f,
            /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
            /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b1f3*/4.f,  0.5f,  8.f,   8.2f
    };
    set_values(input, input_vec);
    auto prog = program::build_program(engine, topology, get_test_default_config(engine));
    network network(prog, 0);
    network.set_input_data("Add_1396", input);

    // to make _reset_arguments false
    network.execute();
    network.execute();
    network.set_output_memory("pred/sink_port_0", final_output);
}

TEST(set_output_memory_gpu, basic_cached) {
    test_basic<float>(true);
}
