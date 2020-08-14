/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/arg_max_min.hpp>
#include <api/data.hpp>
#include <api/mutable_data.hpp>
#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/network.hpp>
#include <api/topology.hpp>
#include <api/reorder.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

template<typename T = float>
static std::vector<T> generateVector(size_t sz) {
    std::vector<T> vec(sz);
    T n = 0;
    std::generate(vec.begin(), vec.end(), [&n]() {
            return n++;
        });
    return vec;
}

TEST(set_output_memory_gpu, basic) {
    const auto& engine = get_test_engine();

    const int b = 3;
    const int f = 2;
    const int y = 5;
    const int x = 5;
    auto input_data = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    const int inputSize = b * f * y * x;
    auto inputVals = generateVector(inputSize);
    set_values(input_data, inputVals);

    topology topology;
    topology.add(input_layout("Input", input_data.get_layout()));
    topology.add(
            reorder("reorder", "Input", input_data.get_layout())
    );

    network network(engine, topology);

    network.set_input_data("Input", input_data);
    network.set_output_memory("reorder", output_mem);

    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_mem_ptr = output_mem.pointer<float>();

    ASSERT_EQ(output_ptr, output_mem_ptr);
    for (size_t i = 0; i < inputVals.size(); ++i) {
        EXPECT_TRUE(are_equal(inputVals[i], output_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, basic_const) {
    const auto& engine = get_test_engine();

    const int b = 3;
    const int f = 2;
    const int y = 5;
    const int x = 5;
    auto input_data = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto const_data = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_const_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    const int inputSize = b * f * y * x;
    auto inputVals = generateVector(inputSize);
    auto constVals = generateVector(inputSize);
    set_values(input_data, inputVals);
    set_values(const_data, constVals);

    topology topology;
    topology.add(input_layout("Input", input_data.get_layout()));
    topology.add(data("Const", const_data));
    topology.add(
            reorder("reorder_dyn", "Input", input_data.get_layout()),
            reorder("reorder_const", "Const", input_data.get_layout())
    );

    network network(engine, topology);

    network.set_input_data("Input", input_data);
    network.set_output_memory("reorder_dyn", output_mem);
    network.set_output_memory("reorder_const", output_const_mem);

    auto outputs = network.execute();

    auto output_dyn = outputs.at("reorder_dyn").get_memory();
    auto output_const = outputs.at("reorder_const").get_memory();
    auto output_dyn_ptr = output_dyn.pointer<float>();
    auto output_dyn_mem_ptr = output_mem.pointer<float>();
    auto output_const_ptr = output_const.pointer<float>();
    auto output_const_mem_ptr = output_const_mem.pointer<float>();

    ASSERT_EQ(output_dyn_ptr, output_dyn_mem_ptr);
    for (size_t i = 0; i < inputVals.size(); ++i) {
        EXPECT_TRUE(are_equal(inputVals[i], output_dyn_ptr[i])) << i;
    }

    ASSERT_NE(output_const_ptr, output_const_mem_ptr);
    for (size_t i = 0; i < inputVals.size(); ++i) {
        EXPECT_TRUE(are_equal(inputVals[i], output_const_mem_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, basic_mutable) {
    const auto& engine = get_test_engine();

    const int b = 3;
    const int f = 2;
    const int y = 5;
    const int x = 5;
    auto input_data = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto mutable_data = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    auto output_mutable_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { b, f, x, y } });
    const int inputSize = b * f * y * x;
    auto inputVals = generateVector(inputSize);
    auto mutableVals = generateVector(inputSize);
    set_values(input_data, inputVals);
    set_values(mutable_data, mutableVals);

    topology topology;
    topology.add(input_layout("Input", input_data.get_layout()));
    topology.add(cldnn::mutable_data("Mutable", mutable_data));
    topology.add(
            reorder("reorder_dyn", "Input", input_data.get_layout()),
            reorder("reorder_mutable", "Mutable", input_data.get_layout())
    );

    network network(engine, topology);

    network.set_input_data("Input", input_data);
    network.set_output_memory("reorder_dyn", output_mem);
    network.set_output_memory("reorder_mutable", output_mutable_mem);

    auto outputs = network.execute();

    auto output_dyn = outputs.at("reorder_dyn").get_memory();
    auto output_mutable = outputs.at("reorder_mutable").get_memory();
    auto output_dyn_ptr = output_dyn.pointer<float>();
    auto output_dyn_mem_ptr = output_mem.pointer<float>();
    auto output_mutable_ptr = output_mutable.pointer<float>();
    auto output_mutable_mem_ptr = output_mutable_mem.pointer<float>();

    ASSERT_EQ(output_dyn_ptr, output_dyn_mem_ptr);
    for (size_t i = 0; i < inputVals.size(); ++i) {
        EXPECT_TRUE(are_equal(inputVals[i], output_dyn_ptr[i])) << i;
    }

    ASSERT_EQ(output_mutable_ptr, output_mutable_mem_ptr);
    for (size_t i = 0; i < inputVals.size(); ++i) {
        EXPECT_TRUE(are_equal(inputVals[i], output_mutable_mem_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, top_k1) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    auto top_k_input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, 1, 1}});
    auto output_mem =
        memory::allocate(engine, {data_types::f32, format::bfyx, {top_k, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(arg_max_min("arg_max", { "input", "const" }, arg_max_min::min, top_k, arg_max_min::batch));
    topology.add(reorder("reorder", "arg_max", output_mem.get_layout()));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_output_memory("reorder", output_mem);
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_mem_ptr = output_mem.pointer<float>();

    ASSERT_EQ(output_ptr, output_mem_ptr);
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        EXPECT_TRUE(are_equal(output_mem_ptr[i], output_ptr[i])) << i;
    }
}

TEST(set_output_memory_gpu, top_k2) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto second_output = memory::allocate(engine, { data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    auto second_output_mem = memory::allocate(engine, { data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max", { "input", "const", "second_output" }, arg_max_min::min, top_k, arg_max_min::batch));
    topology.add(reorder("reorder", "arg_max", second_output.get_layout()));

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

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_output_memory("reorder", second_output_mem);
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_mem_ptr = second_output_mem.pointer<float>();

    ASSERT_EQ(output_ptr, output_mem_ptr);
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        EXPECT_TRUE(are_equal(output_mem_ptr[i], output_ptr[i])) << i;
    }
}
