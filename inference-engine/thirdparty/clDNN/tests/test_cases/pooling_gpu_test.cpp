/*
// Copyright (c) 2016 Intel Corporation
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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/pooling.hpp"
#include "api/CPP/mutable_data.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "api/CPP/reorder.hpp"
#include <api/CPP/data.hpp>
#include "test_utils/float16.h"

using namespace cldnn;
using namespace tests;


TEST(pooling_forward_gpu, basic_max_byxf_f32_wsiz3x3_wstr1x1_i1x3x3x8_nopad) {
    //  Brief test description.
    //
    //  Pool window: 3x3
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [ 0.5, -0.5, -0.5, -0.5, 0.5f, -0.5, -0.5f, -0.5 ]
    //  [ 1.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 2.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 3.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 4.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 5.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 6.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 7.0, 0.0, 0.0, 0.0, 0.5, -0.5, -0.5, -0.5 ]
    //  [ 8.0, 0.0, 0.0, 4.0, 0.5, -0.5, -0.5, -0.5 ]
    //
    //  Expected output:
    //  [ 8.0, 0.0, 0.0, 4,0, 0,5, -0.5, -0.5, -0.5 ]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,  format::byxf,{ 1, 8, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,3,3 }, { 1,1,1,1 }));
    network network(engine, topology);
    set_values(input_prim, { 0.5f, -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, -0.5f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        2.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        3.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        4.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        6.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        7.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, -0.5f,
        8.0f, 0.0f, 0.0f, 4.0f, 0.5f, -0.5f, -0.5f, -0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(4.0f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz3x3_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 3x3
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,  format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,3,3 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.0f, output_ptr[0]);
}

TEST(pooling_forward_gpu, basic_max_pooling_int8) {

    engine engine;
    layout in_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,3,3 } };
    layout out_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,1,1 } };
    layout byte_layout = { type_to_data_type<int8_t>::value, format::bfyx,{ 1,1,3,3 } };
    std::initializer_list<float> input_f = { 1.0f, -2.5f, 3.1f, -4.0f, 5.03f, -6.99f, 7.0f, -8.0f, 9.5f };
    std::list<float> final_results = { 9.0f };

    // Allocate memory for input image.
    auto input_memory = memory::allocate(engine, in_layout);
    set_values(input_memory, input_f);

    // Create input_layout description
    // "input" - is the primitive id inside topology
    input_layout input("input", in_layout);

    topology topology(
        // 1. input layout primitive.
        input,
        // 2. reorder primitive with id "reorder_input"
        reorder("reorder_input", input, byte_layout),
        pooling("pool1", "reorder_input", pooling_mode::max, { 1,1,3,3 }, {1,1,1,1}),
        reorder("reorder2", "pool1", out_layout)
    );

    network network(
        engine,
        topology,
        {
            build_option::outputs({ "reorder2" })
        });

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    auto interm_ptr = interm.pointer<float>();
    auto output_size = outputs.at("reorder2").get_memory().count();
    unsigned int cntr = 0;
    for (const auto& exp : final_results)
    {
        EXPECT_EQ(exp, interm_ptr[cntr++]);
    }
}

TEST(pooling_forward_gpu, basic_avg_pooling_int8) {

    engine engine;
    layout in_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,3,3 } };
    layout out_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,1,1 } };
    layout byte_layout = { type_to_data_type<int8_t>::value, format::bfyx,{ 1,1,3,3 } };
    std::initializer_list<float> input_f = { 2.0f, -2.5f, 5.1f, -4.0f, 8.03f, -6.99f, 17.0f, -8.0f, 19.5f };
    auto final_result = 0.0f;
    for (const auto& val : input_f)
    {
        final_result += (float)((char)val);
    }
    final_result /= input_f.size();
    final_result = (float)((char)final_result);
    // Allocate memory for input image.
    auto input_memory = memory::allocate(engine, in_layout);
    set_values(input_memory, input_f);

    // Create input_layout description
    // "input" - is the primitive id inside topology
    input_layout input("input", in_layout);

    topology topology(
        // 1. input layout primitive.
        input,
        // 2. reorder primitive with id "reorder_input"
        reorder("reorder_input", input, byte_layout),
        pooling("pool1", "reorder_input", pooling_mode::average, { 1,1,3,3 }, { 1,1,1,1 }),
        reorder("reorder2", "pool1", out_layout)
    );

    network network(
        engine,
        topology,
        {
            build_option::outputs({ "reorder2" })
        });

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    auto interm_ptr = interm.pointer<float>();
    auto output_size = outputs.at("reorder2").get_memory().count();
    unsigned int cntr = 0;
    EXPECT_EQ(final_result, interm_ptr[0]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0,  1.5]
    //  [ 2.0,  1.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.0f, output_ptr[0]);
    EXPECT_EQ(1.5f, output_ptr[1]);
    EXPECT_EQ(2.0f, output_ptr[2]);
    EXPECT_EQ(1.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i4x4x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.25,  1.00,  0.50,  0.25]
    //  [ 2.00,  1.50, -0.50, -0.75]
    //  [ 0.00, -1.00,  0.50,  0.25]
    //  [ 0.50, -2.00, -1.50, -2.50]
    //
    //  Expected output:
    //  [ 2.0,  0.5]
    //  [ 0.5,  0.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }));

    network network(engine, topology);
    set_values(input_prim, { -0.25f, 1.00f, 0.50f, 0.25f, 2.00f, 1.50f, -0.50f, -0.75f, 0.00f, -1.00f, 0.50f, 0.25f, 0.50f, -2.00f, -1.50f, -2.50f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.0f, output_ptr[0]);
    EXPECT_EQ(0.5f, output_ptr[1]);
    EXPECT_EQ(0.5f, output_ptr[2]);
    EXPECT_EQ(0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x2x2_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [-0.5,  0.5,  0.0]   [-1.5, -0.5,  0.0]
    //  [ 1.0, -1.0, -2.0]   [ 0.0, -1.0,  1.5]
    //  [-1.0, -0.5, -0.5]   [-2.0,  1.0, -0.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  0.0, -0.5]   [ 0.0,  0.5, -0.5]
    //  [-2.0, -1.0,  1.0]   [ 1.0, -1.0,  0.0]
    //  [-0.5, -1.0,  1.5]   [ 0.5, -0.5,  0.0]
    //
    //  Expected output:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [ 1.0,  0.5]         [ 0.0,  1.5]
    //  [ 1.0, -0.5]         [ 1.0,  1.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  1.0]         [ 1.0,  0.5]
    //  [-0.5,  1.5]         [ 1.0,  0.0]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 0.5f, -1.5f, 0.0f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f, -0.5f, 0.0f, -0.5f, 1.0f, -2.0f, 0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, 1.0f, 1.5f, 0.0f, -1.0f, -0.5f, -2.0f, 0.5f, -0.5f, -1.0f, 1.0f, -0.5f, -0.5f, 1.5f, -0.5f, 0.0f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(1.0f, output_ptr[0]); EXPECT_EQ(0.0f, output_ptr[2]);
    EXPECT_EQ(0.5f, output_ptr[4]); EXPECT_EQ(1.5f, output_ptr[6]);
    EXPECT_EQ(1.0f, output_ptr[8]); EXPECT_EQ(1.0f, output_ptr[10]);
    EXPECT_EQ(-0.5f, output_ptr[12]); EXPECT_EQ(1.5f, output_ptr[14]);

    EXPECT_EQ(0.5f,  output_ptr[1]);  EXPECT_EQ(1.0f, output_ptr[3]);
    EXPECT_EQ(1.0f,  output_ptr[5]);  EXPECT_EQ(0.5f, output_ptr[7]);
    EXPECT_EQ(-0.5f, output_ptr[9]);  EXPECT_EQ(1.0f, output_ptr[11]);
    EXPECT_EQ(1.5f,  output_ptr[13]); EXPECT_EQ(0.0f, output_ptr[15]);
}

TEST(pooling_forward_gpu, offsets_max_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 1.5, -0.5]
    //  [   -1, 0.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0, 0, -1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.50f, -0.50f, -1.00f, 0.50f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ( 1.5f, output_ptr[0]);
    EXPECT_EQ(-0.5f, output_ptr[1]);
    EXPECT_EQ(-1.0f, output_ptr[2]);
    EXPECT_EQ( 0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_max_yxfb_f32_wsiz2x2_wstr2x2_i3x3x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 1.5,  -0.5]
    //  [   1,  -0.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);

    set_values(input_prim, { 
        1.50f, -1.00f, -0.50f,
        1.00f, -1.00f, -1.00f,
       -1.00f, -1.00f, -0.50f
    });

    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();
    EXPECT_EQ((int)output_prim.get_layout().size.count(), 4);

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(1.5f, get_value<float>(output_ptr, 0));
    EXPECT_EQ(-0.5f, get_value<float>(output_ptr, 1));
    EXPECT_EQ(1.0f, get_value<float>(output_ptr, 2));
    EXPECT_EQ(-0.5f, get_value<float>(output_ptr, 3));
}

TEST(pooling_forward_gpu, basic_avg_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: avg
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 4.0, -1.0,  3.5]
    //
    //  Expected output:
    //  [ 1.0,   0.625]
    //  [ 1.625, 0.875]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average,{ 1,1,2,2 },{ 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 4.0f, -1.0f, 3.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    
    EXPECT_EQ(1.0f,   output_ptr[0]);
    EXPECT_EQ(0.625f, output_ptr[1]);
    EXPECT_EQ(1.625f, output_ptr[2]);
    EXPECT_EQ(0.875f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 0.375, -0.125]
    //  [ -0.25,  0.125]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(0.375f,  output_ptr[0]);
    EXPECT_EQ(-0.125f, output_ptr[1]);
    EXPECT_EQ(-0.25f,  output_ptr[2]);
    EXPECT_EQ(0.125f,  output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_bfyx_f32_wsiz3x3_wstr3x3_i1x1x3x3_zeropad) {
    //  Test the corner case when average pooling window contains data from image, data from padding and data outside padding
    //
    //  Pool window: 3x3
    //  Pool stride: 3x3
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, -1.0, padd]
    //  [ padd,  0.5,  0.1,  0.2, padd]
    //  [ padd,  0.9,  1.1,  2.2, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 0.177777, -0.133333]
    //  [ 0.333333,  0.55]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,3,3 }, { 1,1,3,3 }, { 0,0,-1,-1 }));

    network network(engine, topology);

    std::vector<float> input_vec = { 1.5f, -0.5f, -1.0f, 0.5f, 0.1f, 0.2f, 0.9f, 1.1f, 2.2f };
    set_values(input_prim, input_vec);

    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_NEAR(output_ptr[0], 0.177777f, 1e-05F);
    EXPECT_NEAR(output_ptr[1], -0.133333f, 1e-05F);
    EXPECT_NEAR(output_ptr[2], 0.333333f, 1e-05F);
    EXPECT_NEAR(output_ptr[3], 0.55f, 1e-05F);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_f32_wsiz2x2_wstr2x2_i3x3x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5,  2.5]
    //  [ padd, -1.0,  0.5,  3.0]
    //  [ padd,  0.5,  0.0, -8.0]
    //
    //  Expected output:
    //  [  0.375,    0.5]
    //  [ -0.125, -1.125]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.5f, -0.5f, 2.5f, -1.0f, 0.5f, 3.0f, 0.5f, 0.0f, -8.0f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();
    EXPECT_EQ((int)output_prim.get_layout().size.count(), 4);

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(0.375f,  output_ptr[0]);
    EXPECT_EQ(0.5f,    output_ptr[1]);
    EXPECT_EQ(-0.125f, output_ptr[2]);
    EXPECT_EQ(-1.125f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 0.375, -0.125, 0, 0]
    //  [ 0, 0, -0.25,  0.125, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 2, 2 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.375f, -0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.25f, 0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1.5, -0.5, 0, 0]
    //  [0, 1, -0.5, 0, 0]
    //  [0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 3, 3 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.50f, -1.00f, -0.50f,
            1.00f, -1.00f, -1.00f,
            -1.00f, -1.00f, -0.50f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f,-0.5f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim.get_layout().size.count(), 4);
        EXPECT_EQ((int)output_prim.get_layout().get_buffer_size().count(), 16);

        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Out Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 0.375, -0.125, 0, 0]
    //  [ 0, 0, -0.25,  0.125, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 2, 2 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ {0,0,1,2}, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.375f, -0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.25f, 0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1.5, -0.5, 0]
    //  [0, 1, -0.5, 0]
    //  [0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 3, 3 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ { 0, 0, 1, 2 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.50f, -1.00f, -0.50f,
            1.00f, -1.00f, -1.00f,
            -1.00f, -1.00f, -0.50f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f, -0.5f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim.get_layout().size.count(), 4);
        EXPECT_EQ((int)output_prim.get_layout().get_buffer_size().count(), 16);

        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Out Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 0x0
    //  Input data:
    //  [ 1, 2, 3, 4]
    //  [ 5,  1.5, -0.5, 6]
    //  [ 7, -1.0,  0.5, 8]
    //  [ 9, 10, 11, 12]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 2.375, 3.125, 0, 0]
    //  [ 0, 0, 6.25,  7.875, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 4, 4 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,0,0 }, padding{ { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, {
            1.f, 2.f, 3.f, 4.f,
            5.f, 1.5f, -0.5f, 6.f,
            7.f, -1.0f, 0.5f, 8.f,
            9.f, 10.f, 11.f, 12.f});
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 2.375f, 3.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 6.25f, 7.875f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 0x0
    //  Input data:
    //  [ 1, 2, 3, 4, 5]
    //  [ 6,  1.5, -1.0, -0.5, 7]
    //  [ 8,  1.0, -1.0, -1.0, 9]
    //  [ 10, -1.0, -1.0, -0.5, 11]
    //  [ 12, 13, 14, 15, 16]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1, 3, 5, 0]
    //  [0, 8, 1.5, 9, 0]
    //  [0, 12, 14, 16, 0]
    //  [0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 5, 5 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, padding{ { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.f, 2.f, 3.f, 4.f, 5.f,
            6.f, 1.50f, -1.00f, -0.50f, 7.f,
            8.f, 1.00f, -1.00f, -1.00f, 9.f,
            10.f, -1.00f, -1.00f, -0.50f, 11.f,
            12.f, 13.f, 14.f, 15.f, 16.f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.f, 3.f, 5.f, 0.0f,
            0.0f, 8.f, 1.5f, 9.f, 0.0f,
            0.0f, 12.f, 14.f, 16.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        EXPECT_EQ((int)output_prim.get_layout().size.count(), 9);
        EXPECT_EQ((int)output_prim.get_layout().get_buffer_size().count(), 25);

        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "input", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();
    auto argmax_ptr = arg_max.pointer<float>();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax_input_padding) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2
    //  Input Padding : 2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 2 }, 0 })));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "reorder", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();
    auto argmax_ptr = arg_max.pointer<float>();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax_output_padding) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2
    //  Output Padding : 2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 2 }, 0 })));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "reorder", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, padding({ 0, 0, 1, 1 }, 0)));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();
    auto argmax_ptr = arg_max.pointer<float>();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 8.0f, 16.0f,0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 12.0f, 17.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }

    for (size_t i = 0; i < expected_argmax_vec.size(); ++i) {
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

TEST(pooling_forward_gpu, basic_in2x2x3x2_max_with_argmax_with_output_size) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        1.0f, 2.0f, -10.f,
        3.0f, 4.0f, -14.f,
        5.0f, 6.0f, -12.f,
        7.0f, 8.0f, 16.0f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling", "input", "arg_max", pooling_mode::max_with_argmax, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 2, 2, 2, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("pooling").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_layout = output.get_layout();
    auto argmax_ptr = arg_max.pointer<float>();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.size.spatial[1], 1);
    EXPECT_EQ(output_layout.size.spatial[0], 2);
    EXPECT_EQ(output_layout.size.feature[0], 2);
    EXPECT_EQ(output_layout.size.batch[0], 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}

template <class DataType>
static void generic_average_wo_padding_test(format fmt, tensor output, tensor input, tensor window, tensor stride, tensor offset)
{
    constexpr auto dt = std::is_same<DataType, float>::value ? data_types::f32 : data_types::f16;

    engine eng;

    if (!eng.get_info().supports_fp16)
    {
        if(dt == data_types::f16)
        {
            return;
        }
    }

    auto input_mem = memory::allocate(eng, layout{ dt, fmt, input });
    set_values(input_mem, std::vector<DataType>(input.count(), DataType(1)));
    std::vector<DataType> expected_output(output.count(), DataType(1));

    topology tpl;
    tpl.add(input_layout("in", input_mem.get_layout()));

    auto pool_in = "in";
    if (offset != tensor())
    {
        tpl.add(reorder("reorder", "in", input_mem.get_layout().with_padding(offset.negate().sizes())));
        pool_in = "reorder";
    }
    tpl.add(pooling("pool", pool_in, pooling_mode::average_no_padding, window, stride, offset));

    network net(eng, tpl);
    net.set_input_data("in", input_mem);
    auto output_mem = net.execute().at("pool").get_memory();

    ASSERT_TRUE(output_mem.count() == expected_output.size());
    EXPECT_TRUE(output_mem.get_layout().size == output);
    auto out_ptr = output_mem.pointer<DataType>();

    for (size_t i = 0; i < expected_output.size(); ++i)
        EXPECT_FLOAT_EQ(out_ptr[i], expected_output[i]);
}

//bfyx fp32
TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2)
{
    generic_average_wo_padding_test<float>(format::bfyx, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2_o1x1)
{
    generic_average_wo_padding_test<float>(format::bfyx, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s3x3_o1x1)
{
    generic_average_wo_padding_test<float>(format::bfyx, spatial(2, 2), spatial(3, 3), spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i1x1_w3x3_s1x1_o1x1)
{
    generic_average_wo_padding_test<float>(format::bfyx, spatial(1, 1), spatial(1, 1), spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//bfyx fp16
TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s2x2_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i3x3_w2x2_s3x3_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, spatial(2, 2), spatial(3, 3), spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, bfyx_average_without_padding_i1x1_w3x3_s1x1_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::bfyx, spatial(1, 1), spatial(1, 1), spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//yxfb fp32
TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2)
{
    generic_average_wo_padding_test<float>(format::yxfb, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2_o1x1)
{
    generic_average_wo_padding_test<float>(format::yxfb, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s3x3_o1x1)
{
    generic_average_wo_padding_test<float>(format::yxfb, spatial(2, 2), spatial(3, 3), spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i1x1_w3x3_s1x1_o1x1)
{
    generic_average_wo_padding_test<float>(format::yxfb, spatial(1, 1), spatial(1, 1), spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}

//yxfb fp16
TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{});
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s2x2_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, spatial(2, 2), spatial(3, 3), spatial(2, 2), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i3x3_w2x2_s3x3_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, spatial(2, 2), spatial(3, 3), spatial(3, 3), tensor{ 0,0,2,2 }, tensor{ 0,0,-1,-1 });
}

TEST(pooling_forward_gpu, yxfb_average_without_padding_i1x1_w3x3_s1x1_o1x1_fp16)
{
    generic_average_wo_padding_test<FLOAT16>(format::yxfb, spatial(1, 1), spatial(1, 1), spatial(3, 3), tensor{ 0,0,1,1 }, tensor{ 0,0,-1,-1 });
}


class pooling_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        for (auto generic_params : all_generic_params)
        {
            delete generic_params;
        }

        for (auto layer_params : all_layer_params)
        {
            delete layer_params;
        }
    }

    static tensor generate_input_offset(int x, int y, const tensor& window_size)
    {
        return tensor(0, 0, -std::min(x, window_size.spatial[0] - 1), -std::min(y, window_size.spatial[1] - 1));
    }

    static std::vector<cldnn::primitive*> generate_specific_test_params()
    {
        std::vector<pooling_mode> pooling_modes = { pooling_mode::max, pooling_mode::average, pooling_mode::average_no_padding };

        std::vector<tensor> sizes = { tensor(1, 1, 2, 2 ), tensor(1, 1, 3, 3), tensor(1, 1, 7, 4) };

        std::vector<tensor> strides = { tensor(1, 1, 1, 1), tensor(1, 1, 2, 2), tensor(1, 1, 4, 3) };

        for (auto pooling_mode : pooling_modes)
        {
            for (auto size : sizes)
            {
                for (auto stride : strides)
                {
                    // No padding
                    all_layer_params.push_back(new pooling("pooling", "input0", pooling_mode, size, stride));
                    all_layer_params.push_back(new pooling("pooling", "input0", pooling_mode, size, stride, generate_input_offset(4, 3, size)));

                    // Input padding
                    all_layer_params.push_back(new pooling("pooling", "reorder0", pooling_mode, size, stride));

                    // Output padding
                    all_layer_params.push_back(new pooling("pooling", "input0", pooling_mode, size, stride, generate_input_offset(2, 3, size), { { 0, 0, 1, 5 },{ 0, 0, 19, 4 } }));

                    // Input + output padding
                    all_layer_params.push_back(new pooling("pooling", "reorder0", pooling_mode, size, stride, generate_input_offset(2, 3, size), { { 0, 0, 2, 1 },{ 0, 0, 3, 4 } }));
                }
            }
        }

        // This case tests the pooling_gpu_bfyx_average_opt kernel.
        all_layer_params.push_back(new pooling("pooling", "input0", pooling_mode::average, tensor(1, 1, 3, 3), tensor(1, 1, 1, 1), generate_input_offset(1, 1, tensor(1, 1, 3, 3))));

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params()
    {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        if ((format == cldnn_format_type::cldnn_format_yxfb) || (format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_byxf))
        {
            return true;
        }
        return false;
    }

    virtual void prepare_input_for_test(std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            prepare_input_for_test_typed<float>(inputs);
        }
        else
        {
            prepare_input_for_test_typed<FLOAT16>(inputs);
        }
    }

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory>& inputs)
    {
        int k = (generic_params->data_type == data_types::f32) ? 8 : 4;
        auto input = inputs[0];
        auto input_size = inputs[0].get_layout().size;
        VVVVF<Type> input_rnd = generate_random_4d<Type>(input_size.batch[0], input_size.feature[0], input_size.spatial[1], input_size.spatial[0], -10, 10, k);
        VF<Type> input_rnd_vec = flatten_4d<Type>(input.get_layout().format, input_rnd);
        set_values(input, input_rnd_vec);
    }

    virtual cldnn::tensor get_expected_output_tensor()
    {
        const cldnn::pooling* pooling = (cldnn::pooling*)layer_params;

        int batch = generic_params->input_layouts[0].size.batch[0];
        int feature = generic_params->input_layouts[0].size.feature[0];
        int height = generic_params->input_layouts[0].size.spatial[1];
        int width = generic_params->input_layouts[0].size.spatial[0];

        int input_offset_height = pooling->input_offset.spatial[1];
        int input_offset_width = pooling->input_offset.spatial[0];

        int kernel_height = pooling->size.spatial[1];
        int kernel_width = pooling->size.spatial[0];

        int stride_height = pooling->stride.spatial[1];
        int stride_width = pooling->stride.spatial[0];

        int pooled_height = (int)(ceil((float)std::max(height - 2 * input_offset_height - kernel_height, 0) / stride_height)) + 1;
        int pooled_width = (int)(ceil((float)std::max(width - 2 * input_offset_width - kernel_width, 0) / stride_width)) + 1;
        
        // Make sure that the last pooling starts strictly inside the image.
        while ((pooled_height - 1) * stride_height >= height - input_offset_height) 
        {
            --pooled_height;
        }
        while ((pooled_width - 1) * stride_width >= width - input_offset_width) 
        {
            --pooled_width;
        }

        return cldnn::tensor(batch, feature, pooled_width, pooled_height);
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        const cldnn::pooling* pooling = (cldnn::pooling*)layer_params;

        int batch = inputs[0].get_layout().size.batch[0];
        int feature = inputs[0].get_layout().size.feature[0];
        int height = inputs[0].get_layout().size.spatial[1];
        int width = inputs[0].get_layout().size.spatial[0];



        cldnn::pooling_mode pooling_mode = pooling->mode;

        int input_offset_width = pooling->input_offset.spatial[0];
        int input_offset_height = pooling->input_offset.spatial[1];
        
        int kernel_width = pooling->size.spatial[0];
        int kernel_height = pooling->size.spatial[1];
        
        int stride_width = pooling->stride.spatial[0];
        int stride_height = pooling->stride.spatial[1];
        
        auto output_tensor = get_expected_output_tensor();

        int pooled_width = output_tensor.spatial[0];
        int pooled_height = output_tensor.spatial[1];
         
        //Output is bfyx
        auto output = memory::allocate(engine, cldnn::layout(inputs[0].get_layout().data_type, cldnn::format::bfyx, output_tensor, pooling->output_padding));

        auto input_mem = inputs[0].pointer<Type>();
        auto output_mem = output.pointer<Type>();

        int output_width = output.get_layout().get_buffer_size().spatial[0];
        int output_height = output.get_layout().get_buffer_size().spatial[1];

        const auto input_desc = get_linear_memory_desc(inputs[0].get_layout());
        const auto output_desc = get_linear_memory_desc(output.get_layout());

        switch (pooling_mode)
        {
            case cldnn::pooling_mode::max:
            {
                for (int i = 0; i < (int)output.get_layout().get_buffer_size().count(); i++)
                {
                    output_mem[i] = (generic_params->data_type == data_types::f32) ? -FLT_MAX : -65504;
                }
                for (int b = 0; b < batch; b++) 
                {
                    for (int f = 0; f < feature; f++) 
                    {
                        for (int h = 0; h < pooled_height; h++) 
                        {
                            for (int w = 0; w < pooled_width; w++) 
                            {
                                int input_offset_x_start = w * stride_width + input_offset_width;
                                int input_offset_x_end = std::min(input_offset_x_start + kernel_width, width);
                                input_offset_x_start = std::max(input_offset_x_start, 0);

                                int input_offset_y_start = h * stride_height + input_offset_height;
                                int input_offset_y_end = std::min(input_offset_y_start + kernel_height, height);
                                input_offset_y_start = std::max(input_offset_y_start, 0);

                                const size_t output_index = get_linear_index(output.get_layout(), b, f, h, w, output_desc);

                                for (int y = input_offset_y_start; y < input_offset_y_end; y++) 
                                {
                                    for (int x = input_offset_x_start; x < input_offset_x_end; x++) 
                                    {
                                        const size_t input_index = get_linear_index(inputs[0].get_layout(), b, f, y, x, input_desc);
                                        
                                        if (input_mem[input_index] > output_mem[output_index])
                                        {
                                            output_mem[output_index] = input_mem[input_index];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case cldnn::pooling_mode::average:
            case cldnn::pooling_mode::average_no_padding:
            {
                auto kernel_size =  kernel_width * kernel_height;
                auto dynamic_mode = (((output_tensor.spatial[0] - 1) * stride_width) + pooling->size.spatial[0]) > -2 * input_offset_width + width ||
                    (((output_tensor.spatial[1] - 1) * stride_height) + pooling->size.spatial[1]) > -2 * input_offset_width + height;

                auto divider = [=](int actual_x, int actual_y) {
                    auto x = kernel_width;
                    auto y = kernel_height;
                    if (dynamic_mode)
                    {
                        if (actual_x + kernel_width > width + std::abs(input_offset_width))
                        {
                            x = (width + std::abs(input_offset_width)) - actual_x;
                        }
                        if (actual_y + kernel_height > height + std::abs(input_offset_height))
                        {
                            y = (height + std::abs(input_offset_height)) - actual_y;
                        }
                    }
                    return y*x;
                };

                for (int i = 0; i < (int)output.get_layout().get_buffer_size().count(); i++)
                {
                    output_mem[i] = 0;
                }
                for (int b = 0; b < batch; b++) 
                {
                    for (int f = 0; f < feature; f++) 
                    {
                        for (int h = 0; h < pooled_height; h++) 
                        {
                            for (int w = 0; w < pooled_width; w++) 
                            {	
                                int input_offset_x_start = w * stride_width + input_offset_width;
                                int input_offset_x_end = std::min(input_offset_x_start + kernel_width, width);
                                input_offset_x_start = std::max(input_offset_x_start, 0);

                                int input_offset_y_start = h * stride_height + input_offset_height;
                                int input_offset_y_end = std::min(input_offset_y_start + kernel_height, height);
                                input_offset_y_start = std::max(input_offset_y_start, 0);	

                                int output_index = (b * feature + f) * output_height * output_width;
                                tensor lower_padding = pooling->output_padding.lower_size();
                                output_index += (lower_padding.spatial[1] + h) * output_width + lower_padding.spatial[0] + w;

                                int num_of_elements = 0;
                                for (int y = input_offset_y_start; y < input_offset_y_end; y++) 
                                {
                                    for (int x = input_offset_x_start; x < input_offset_x_end; x++) 
                                    {
                                        const size_t input_index = get_linear_index(inputs[0].get_layout(), b, f, y, x, input_desc);
                                        output_mem[output_index] += input_mem[input_index];
                                        if (!dynamic_mode || pooling_mode == cldnn::pooling_mode::average_no_padding)
                                        {
                                            num_of_elements++;
                                        }
                                    }
                                }
                                if (pooling_mode == cldnn::pooling_mode::average)
                                {
                                        num_of_elements = divider(input_offset_x_start, input_offset_y_start);
                                }
                                if (num_of_elements == 0)
                                {
                                    assert(0);
                                    return output;
                                }
                                output_mem[output_index] /= (Type)num_of_elements;
 
                            }
                        }
                    }
                }
                break;
            }
            default:
            {
                assert(0);
            }
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;

};

std::vector<cldnn::primitive*> pooling_test::all_layer_params = {};
std::vector<tests::test_params*> pooling_test::all_generic_params = {};

TEST_P(pooling_test, POOLING)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(DISABLED_POOLING,
                        pooling_test,
                        ::testing::Combine(::testing::ValuesIn(pooling_test::generate_generic_test_params()),
                                           ::testing::ValuesIn(pooling_test::generate_specific_test_params())),
                        tests::generic_test::custom_param_name_functor());
