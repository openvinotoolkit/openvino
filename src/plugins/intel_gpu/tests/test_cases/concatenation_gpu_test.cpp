// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include "concatenation_inst.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>

using namespace cldnn;
using namespace ::tests;

TEST(concat_gpu, mixed_input_types) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 4, 3 } });
    auto input4 = engine.allocate_memory({ data_types::i64, format::bfyx, { 1, 1, 4, 3 } });

    set_values<float>(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values<int32_t>(input1, { 11, 12, 13, 14, 12, 12, 13, 14, 13, 13, 13, 15 });
    set_values<int8_t>(input2, { 21, 22, 23, 24, 22, 22, 23, 24, 23, 23, 23, 25 });
    set_values(input3, { half_t(31.f), half_t(32.f), half_t(33.f),
                         half_t(34.f), half_t(32.f), half_t(32.f),
                         half_t(33.f), half_t(34.f), half_t(33.f),
                         half_t(33.f), half_t(33.f), half_t(35.f) });
    set_values<int64_t>(input4, { 41, 42, 43, 44, 42, 42, 43, 44, 43, 43, 43, 45 });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 42.0f, 42.0f, 43.0f, 44.0f, 43.0f, 43.0f, 43.0f, 45.0f };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            input_layout("input4", input4->get_layout()),
            concatenation("concat",
                          { "input0", "input1", "input2", "input3", "input4" },
                          1,
                          data_types::f32,
                          padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 4);
    EXPECT_EQ(f_size, 5);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, mixed_input_types_5d) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });

    set_values(input0, { half_t(1.0f), half_t(2.0f), half_t(3.0f),
                         half_t(4.0f), half_t(2.0f), half_t(2.0f),
                         half_t(3.0f), half_t(4.0f), half_t(3.0f),
                         half_t(3.0f), half_t(3.0f), half_t(5.0f) });
    set_values(input1, { half_t(11), half_t(12), half_t(13),
                         half_t(14), half_t(12), half_t(12),
                         half_t(13), half_t(14), half_t(13),
                         half_t(13), half_t(13), half_t(15) });
    set_values(input2, { half_t(21), half_t(22), half_t(23),
                         half_t(24), half_t(22), half_t(22),
                         half_t(23), half_t(24), half_t(23),
                         half_t(23), half_t(23), half_t(25) });
    set_values(input3, { half_t(31.f), half_t(32.f), half_t(33.f),
                         half_t(34.f), half_t(32.f), half_t(32.f),
                         half_t(33.f), half_t(34.f), half_t(33.f),
                         half_t(33.f), half_t(33.f), half_t(35.f) });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            concatenation("concat",
                          { "input0", "input1", "input2", "input3" },
                          1,
                          data_types::f32,
                          padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 1);
    EXPECT_EQ(f_size, 4);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, i8_optimization_with_pool) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 8, 3}});
    auto input1 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 8, 3}});


    set_values<int8_t>(input0, { 11, 12, 13,
                         14, 12, 12,
                         13, -14, 13,
                         13, -13, 15,
                         16, -16, -13,
                         -14, 12, 11,
                         16, -14, -13,
                         18, -13, -15, });
    set_values<int8_t>(input1, { 11, 12, 13,
                         15, 12, 12,
                         13, 14, 12,
                         13, 13, 15,
                         12, 14, 13,
                         14, 17, 18,
                         13, 14, 11,
                         13, 13, 15 });


    VF<int8_t> output_vec = {13, 13, 13, 13, 15, 15,
                        16, 15, 16, 14, 13, 14,
                        13, 14, 13, 18, 16, 18,
                        16, 15, 16, 15, 18, 14,
                        18, 14, -13, 15};

    layout reorder_layout(data_types::i8, format::yxfb, {7, 2, 2, 1});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      pooling("pool0", "input0", pooling_mode::max, {2, 2}, {1, 1}),
                      pooling("pool1", "input1", pooling_mode::max, {2, 2}, {1, 1}),
                      concatenation("concat",
                                    {"pool0", "pool1"},
                                    1,
                                    data_types::i8,
                                    padding{{0, 0, 0, 0}, 0}),
                      reorder("reorder", "concat", reorder_layout));
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output_memory = outputs.at("reorder").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(0);
    int x_size = output_layout.spatial(1);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 7);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, i8_optimization_with_conv) {
    //  Filter : 3x2x3
    //  Stride : 2x1
    //  Input1  : 4x5
    //  Input2  : 4x5
    //  Input3  : 4x5
    //  Concat output  : 3x4x5
    //  Conv input  : 3x4x5
    //  Output : 2x3
    //
    //  Input0:
    //  1  2  3  -4  5
    //  2  2  3  4  -6
    //  -3  3  3  5  1
    //  -1  1  1  1  -1
    //  Input1:
    //  5  5  3  -4  5
    //  2  -2  5  4  6
    //  6  1  3  5  1
    //  1  2  -3  -4  5
    //  Input2:
    //  -2  1  3  2  -5
    //  1  2  -2  4  2
    //  3  5  3  -3  1
    //  5  4  3  2  1
    //
    //  Filter:
    //  1  2  1     1  2  1     1  2  1
    //  2  1  2     2  1  2     2  1  2
    //
    //  Output:
    // 53  54  30
    // 52  47  37
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 5, 4}});
    auto input1 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 5, 4}});
    auto input2 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 5, 4}});
    auto weights = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 3, 3, 2 } });

    set_values<int8_t>(weights, { 1, 2, 1,
                          2, 1, 2, 1, 2, 1,
                          2, 1, 2, 1, 2, 1,
                          2, 1, 2 });

    set_values<int8_t>(input0, {  1, 2, 3, -4, 5,
                          2, 2, 3, 4, -6,
                          -3, 3, 3, 5, 1,
                          -1, 1, 1, 1, -1 });
    set_values<int8_t>(input1, { 5, 5, 3, -4, 5,
                         2, -2, 5, 4, 6,
                         6, 1, 3, 5, 1,
                         1, 2, -3, -4, 5 });
    set_values<int8_t>(input2, {  -2, 1, 3, 2, -5,
                          1, 2, -2, 4, 2,
                          3, 5, 3, -3, 1,
                          5, 4, 3, 2, 1 });

    VF<int8_t> output_vec = { 53, 54, 30, 52, 47, 37 };


    layout reorder_layout(data_types::i8, format::bfyx, {1, 1, 2, 3});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      input_layout("input2", input2->get_layout()),
                      concatenation("concat",
                                    {"input0", "input1", "input2"},
                                    1,
                                    data_types::i8,
                                    padding{{0, 0, 0, 0}, 0}),
                      data("weights", weights),
                      convolution("conv", "concat", { "weights" }, { 2, 1 }),
                      reorder("output", "conv", reorder_layout));
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, i8_optimization_with_pool_conv) {
    //  Filter : 32x2x1
    //  Input offset : 0x0x-1x0
    //  Stride : 1x1
    //  Input0  : 16x3x2
    //  Input1  : 16x3x2
    //  Output : 1x1x3
    //
    //  Input0:
    // -3 6 0 2 -1 -1 6 0 5 4 1 6 2 4 0 5
    // -2 -1 1 0 2 3 3 3 6 2 4 7 3 6 7 -1
    // 7 7 5 -3 1 -1 5 4 0 3 -2 6 2 5 2 4
    // 5 -1 3 6 2 0 -3 -1 0 3 0 -1 1 6 1 6
    // 5 -2 2 -1 5 6 3 4 1 0 6 6 7 2 6 3
    // 6 7 -1 5 5 6 -1 0 -1 5 5 2 3 -1 -3 4
    //
    //  Input1:
    //  4 -2 0 0 6 2 0 4 6 4 4 4 -3 -1 4 -3
    //  1 0 -1 5 -1 1 4 2 7 7 0 2 3 4 -1 3
    //  7 7 2 -3 -1 5 -2 2 6 -3 0 7 0 3 3 3
    //  -1 0 -2 -2 7 -3 -3 -1 5 0 3 4 0 -1 2 5
    //  2 -1 2 -3 0 -3 -3 2 4 3 3 5 5 7 5 1
    //  2 2 -3 6 6 7 1 -1 -2 5 1 -1 4 5 -3 -2
    //
    // Filters:
    // -1, 2, -2, 2, -2, 1, 1, 0, -1, 1, 2, -2, 2, 1, -2, 0,
    // 0, -2, -2, -2, -2, -1, 2, 1, 2, -1, -1, 0, 2, -2, -2, 1,
    // 0, -2, 0, 1, -2, -1, -2, 0, -1, -1, -2, 1, -2, 0, 1, 2,
    // 2, 2, 2, -2, 0, 2, 1, -2, -1, -1, 0, -2, 2, -1, 2, -1
    //
    //  Output:
    //  -14, -35, -10

    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 16, 3, 2}});
    auto input1 = engine.allocate_memory({data_types::i8, format::bfyx, {1, 16, 3, 2}});
    auto weights = engine.allocate_memory({data_types::i8, format::bfyx, {1, 32, 2, 1}});

    set_values<int8_t>(weights, {-1, 2, -2, 2, -2, 1, 1, 0, -1, 1, 2, -2, 2, 1, -2, 0, 0, -2, -2, -2, -2, -1, 2, 1, 2, -1, -1, 0, 2, -2, -2, 1,
                                0, -2, 0, 1, -2, -1, -2, 0, -1, -1, -2, 1, -2, 0, 1, 2, 2, 2, 2, -2, 0, 2, 1, -2, -1, -1, 0, -2, 2, -1, 2, -1});

    set_values<int8_t>(input0, {-3, 6, 0, 2, -1, -1, 6, 0, 5, 4, 1, 6, 2, 4, 0, 5,
                                -2, -1, 1, 0, 2, 3, 3, 3, 6, 2, 4, 7, 3, 6, 7, -1,
                                7, 7, 5, -3, 1, -1, 5, 4, 0, 3, -2, 6, 2, 5, 2, 4,
                                5, -1, 3, 6, 2, 0, -3, -1, 0, 3, 0, -1, 1, 6, 1, 6,
                                5, -2, 2, -1, 5, 6, 3, 4, 1, 0, 6, 6, 7, 2, 6, 3,
                                6, 7, -1, 5, 5, 6, -1, 0, -1, 5, 5, 2, 3, -1, -3, 4 });

    set_values<int8_t>(input1, { 4, -2, 0, 0, 6, 2, 0, 4, 6, 4, 4, 4, -3, -1, 4, -3,
                                 1, 0, -1, 5, -1, 1, 4, 2, 7, 7, 0, 2, 3, 4, -1, 3,
                                 7, 7, 2, -3, -1, 5, -2, 2, 6, -3, 0, 7, 0, 3, 3, 3,
                                 -1, 0, -2, -2, 7, -3, -3, -1, 5, 0, 3, 4, 0, -1, 2, 5,
                                 2, -1, 2, -3, 0, -3, -3, 2, 4, 3, 3, 5, 5, 7, 5, 1,
                                 2, 2, -3, 6, 6, 7, 1, -1, -2, 5, 1, -1, 4, 5, -3, -2});

    VF<int8_t> output_vec = { -14, -35, -10 };

    layout reorder_layout(data_types::i8, format::bfyx, {1, 1, 3, 1});
    topology topology(input_layout("input0", input0->get_layout()),
                      input_layout("input1", input1->get_layout()),
                      pooling("pool0", "input0", pooling_mode::max, {2, 2}, {1, 1}),
                      pooling("pool1", "input1", pooling_mode::max, {2, 2}, {1, 1}),
                      concatenation("concat",
                                    {"pool0", "pool1"},
                                    1,
                                    data_types::i8,
                                    padding{{0, 0, 0, 0}, 0}),
                      data("weights", weights),
                      convolution("conv", "concat", {"weights"}, {1, 1}, {0, 1}),
                      reorder("output", "conv", reorder_layout) );
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int8_t> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(0);
    int x_size = output_layout.spatial(1);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 1);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

using TestParamType_concat = ::testing::tuple<size_t,   // 0 - Input Batch size
        std::vector<size_t>,                            // 1 - Inputs Features Sizes
        size_t,                                         // 2 - Input Y Size
        size_t>;                                        // 3 - Input X Size


struct concat_gpu : public ::testing::TestWithParam<TestParamType_concat>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_concat> param_info)
    {
        std::string in;
        for (size_t i = 0; i < testing::get<1>(param_info.param).size() - 1; i++) {
            in += std::to_string(testing::get<1>(param_info.param)[i]) + "_";
        }
        in += std::to_string(testing::get<1>(param_info.param)[testing::get<1>(param_info.param).size() - 1]);

        return "in" + std::to_string(testing::get<0>(param_info.param))
               + "x" + in + "x" + std::to_string(testing::get<2>(param_info.param))
               + 'x' + std::to_string(testing::get<3>(param_info.param));
    }
};

static const auto concat_gpu_all_params = ::testing::Values(
    // Input Batch, Input Features, Input Y, Input X
    TestParamType_concat(2, { 2, 15 }, 2, 1),
    TestParamType_concat(2, { 2, 31 }, 2, 1),
    TestParamType_concat(2, { 2, 32 }, 2, 1),
    TestParamType_concat(2, { 2, 37 }, 2, 1),
    TestParamType_concat(2, { 2, 63 }, 2, 1),
    TestParamType_concat(2, { 2, 64 }, 2, 1),
    TestParamType_concat(2, { 2, 65 }, 2, 1),
    TestParamType_concat(2, { 2, 75 }, 2, 1),
    TestParamType_concat(2, { 15, 2 }, 2, 1),
    TestParamType_concat(2, { 31, 2 }, 2, 1),
    TestParamType_concat(2, { 32, 2 }, 2, 1),
    TestParamType_concat(2, { 37, 2 }, 2, 1),
    TestParamType_concat(2, { 63, 2 }, 2, 1),
    TestParamType_concat(2, { 64, 2 }, 2, 1),
    TestParamType_concat(2, { 65, 2 }, 2, 1),
    TestParamType_concat(2, { 75, 2 }, 2, 1),
    TestParamType_concat(2, { 2, 15 }, 1, 2),
    TestParamType_concat(2, { 2, 31 }, 1, 2),
    TestParamType_concat(2, { 2, 32 }, 1, 2),
    TestParamType_concat(2, { 2, 37 }, 1, 2),
    TestParamType_concat(2, { 2, 63 }, 1, 2),
    TestParamType_concat(2, { 2, 64 }, 1, 2),
    TestParamType_concat(2, { 2, 65 }, 1, 2),
    TestParamType_concat(2, { 2, 75 }, 1, 2),
    TestParamType_concat(2, { 15, 2 }, 1, 2),
    TestParamType_concat(2, { 31, 2 }, 1, 2),
    TestParamType_concat(2, { 32, 2 }, 1, 2),
    TestParamType_concat(2, { 37, 2 }, 1, 2),
    TestParamType_concat(2, { 63, 2 }, 1, 2),
    TestParamType_concat(2, { 64, 2 }, 1, 2),
    TestParamType_concat(2, { 65, 2 }, 1, 2),
    TestParamType_concat(2, { 75, 2 }, 1, 2),
    TestParamType_concat(2, { 32, 32 }, 1, 1),
    TestParamType_concat(2, { 64, 64 }, 1, 1),
    TestParamType_concat(2, { 2, 2, 2 }, 1, 1),
    TestParamType_concat(2, { 2, 32, 2 }, 1, 1),
    TestParamType_concat(2, { 31, 32, 32 }, 1, 1),
    TestParamType_concat(2, { 32, 31, 2 }, 1, 1),
    TestParamType_concat(2, { 32, 31, 32 }, 1, 1),
    TestParamType_concat(2, { 32, 32, 32 }, 1, 1),
    TestParamType_concat(2, { 33, 32, 32 }, 1, 1),
    TestParamType_concat(2, { 33, 3, 3 }, 1, 1),
    TestParamType_concat(2, { 33, 3, 33 }, 1, 1),
    TestParamType_concat(2, { 64, 64, 64, 64 }, 1, 1)
);

template <typename Type>
struct concat_gpu_4d : public concat_gpu {
public:

    void test(format::type fmt) {
        auto data_type = type_to_data_type<Type>::value;

        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<VVVVF<Type>> in_data;
        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);

                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            in_data.emplace_back(std::move(data));
            input_ids.push_back("input" + std::to_string(i));
        }

        topology.add(concatenation("concat", input_ids, 1));

        build_options options;
        options.set_option(build_option::optimize_data(true));
        network network(engine, topology, options);

        for (size_t i = 0; i < in_features.size(); i++) {
            network.set_input_data(input_ids[i], in_memory[i]);
        }

        network.execute();

        auto out_mem = network.get_output("concat").get_memory();
        cldnn::mem_lock<Type> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num; bi++) {
            size_t f_sum = 0;
            for (size_t in_i = 0; in_i < in_features.size(); in_i++) {
                for (size_t fi = 0; fi < in_features[in_i]; fi++) {
                    for (size_t yi = 0; yi < input_y; yi++) {
                        for (size_t xi = 0; xi < input_x; xi++) {
                            auto output_coords = tensor(batch(bi), feature(f_sum + fi), spatial(xi, yi, 0, 0));
                            auto output_offset = out_mem->get_layout().get_linear_offset(output_coords);

                            auto ref_val = in_data[in_i][bi][fi][yi][xi];
                            auto actual_val = out_ptr[output_offset];
                            EXPECT_EQ(ref_val, actual_val)
                                << " b=" << bi << ", f=" << f_sum + fi << "(input " << in_i << "), y=" << yi << ", x=" << xi;
                        }
                    }
                }
                f_sum += in_features[in_i];
            }
        }
    }
};

using concat_gpu_4d_f16 = concat_gpu_4d<FLOAT16>;
using concat_gpu_4d_i8 = concat_gpu_4d<int8_t>;
using concat_gpu_4d_u8 = concat_gpu_4d<uint8_t>;

TEST_P(concat_gpu_4d_f16, fs_b_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::fs_b_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_gpu_4d_f16,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_gpu_4d_i8, b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

TEST_P(concat_gpu_4d_i8, b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_gpu_4d_i8,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_gpu_4d_u8, b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_gpu_4d_u8,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

template <typename Type, typename OutputT>
struct concat_id_conv_gpu_4d : public concat_gpu {
public:

    void test(format::type fmt) {
        auto data_type = type_to_data_type<Type>::value;

        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<VVVVF<Type>> in_data;
        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -128, 128);
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);

                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            in_data.emplace_back(std::move(data));
            input_ids.push_back("input" + std::to_string(i));
        }

        topology.add(concatenation("concat", input_ids, 1));
        // Add identity convolution
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        weights_mem->fill(get_test_stream());
        get_test_stream().finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, get_test_stream());
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights", weights_mem));
        topology.add(convolution("conv", "concat", { "weights" }));

        build_options options;
        options.set_option(build_option::optimize_data(true));
        auto conv_forcing = implementation_desc{ fmt, std::string() };
        options.set_option(build_option::force_implementations({ {primitive_id("conv"), conv_forcing} }));
        network network(engine, topology, options);

        for (size_t i = 0; i < in_features.size(); i++) {
            network.set_input_data(input_ids[i], in_memory[i]);
        }

        network.execute();

        auto out_mem = network.get_output("conv").get_memory();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());
        ASSERT_EQ(out_mem->get_layout().format, fmt);

        for (size_t bi = 0; bi < batch_num; bi++) {
            size_t f_sum = 0;
            for (size_t in_i = 0; in_i < in_features.size(); in_i++) {
                for (size_t fi = 0; fi < in_features[in_i]; fi++) {
                    for (size_t yi = 0; yi < input_y; yi++) {
                        for (size_t xi = 0; xi < input_x; xi++) {
                            auto output_coords = tensor(batch(bi), feature(f_sum + fi), spatial(xi, yi, 0, 0));
                            auto output_offset = out_mem->get_layout().get_linear_offset(output_coords);

                            auto ref_val = in_data[in_i][bi][fi][yi][xi];
                            auto actual_val = static_cast<Type>(out_ptr[output_offset]);
                            ASSERT_EQ(ref_val, actual_val)
                                << " b=" << bi << ", f=" << f_sum + fi << "(input " << in_i << "), y=" << yi << ", x=" << xi;
                        }
                    }
                }
                f_sum += in_features[in_i];
            }
        }
    }
};

using concat_id_conv_gpu_4d_f16 = concat_id_conv_gpu_4d<FLOAT16, FLOAT16>;
using concat_id_conv_gpu_4d_i8 = concat_id_conv_gpu_4d<int8_t, float>;

TEST_P(concat_id_conv_gpu_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_id_conv_gpu_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(2, { 2, 32 }, 2, 1),
                            TestParamType_concat(2, { 31, 64 }, 2, 2),
                            TestParamType_concat(2, { 15, 15, 16 }, 2, 1),
                            TestParamType_concat(2, { 16, 15, 16 }, 2, 2),
                            TestParamType_concat(2, { 15, 2, 16, 64 }, 1, 2)
                        ),
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_id_conv_gpu_4d_i8, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke_low_precision,
                        concat_id_conv_gpu_4d_i8,
                        ::testing::Values(
                            TestParamType_concat(2, { 2, 32 }, 2, 1),
                            TestParamType_concat(2, { 31, 64 }, 2, 2),
                            TestParamType_concat(2, { 15, 15, 16 }, 2, 1),
                            TestParamType_concat(2, { 16, 15, 16 }, 2, 2),
                            TestParamType_concat(2, { 15, 2, 16, 64 }, 1, 2)
                        ),
                        concat_gpu::PrintToStringParamName);

template <typename Type>
struct concat_gpu_4d_implicit : public concat_gpu {
public:
    cldnn::memory::ptr run_concat_network(std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input, format::type fmt, build_options options) {
        auto data_type = type_to_data_type<Type>::value;
        auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        std::vector<primitive_id> pooling_ids;

        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = input[i];
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);
                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            topology.add(pooling("pool" +  std::to_string(i), "input" + std::to_string(i), pooling_mode::max, {1, 1}, {1, 1}));

            input_ids.push_back("input" + std::to_string(i));
            pooling_ids.push_back("pool" + std::to_string(i));
        }

        topology.add(concatenation("concat", pooling_ids, 1));
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        weights_mem->fill(get_test_stream());
        get_test_stream().finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, get_test_stream());
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights" , weights_mem));
        topology.add(convolution("conv", "concat", { "weights" }));
        topology.add(pooling("pool_final", "conv", pooling_mode::max, {1, 1}, {1, 1}));
        topology.add(reorder("reorder", "pool_final", layout(data_type, format::bfyx, {(int32_t)batch_num, (int32_t)output_f, (int32_t)input_y, (int32_t)input_x})));

        network concat_network(engine, topology, options);
        for (size_t i = 0; i < in_features.size(); i++) {
            concat_network.set_input_data(input_ids[i], in_memory[i]);
        }
        concat_network.execute();

        bool concat_opt_enabled = options.get<build_option_type::optimize_data>()->enabled();
        bool concat_opt_result = std::static_pointer_cast<concatenation_inst>(concat_network.get_primitive("concat"))->node.can_be_optimized();
        EXPECT_TRUE(concat_opt_enabled==concat_opt_result);

        return concat_network.get_output("reorder").get_memory();
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> generate_input() {
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());

        std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input(in_features.size());
        for (size_t i = 0; i < in_features.size(); ++i) {
            input[i] = generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
        }
        return input;
    }

    void test(format::type fmt) {
        auto input = generate_input();

        // implicit concat
        build_options options1;
        options1.set_option(build_option::optimize_data(true));
        auto out_mem1 = run_concat_network(input, fmt, options1);
        cldnn::mem_lock<Type> out_ptr1(out_mem1, get_test_stream());

        // explicit concat
        build_options options2;
        options2.set_option(build_option::optimize_data(false));
        auto out_mem2 = run_concat_network(input, fmt, options2);
        cldnn::mem_lock<Type> out_ptr2(out_mem2, get_test_stream());

        EXPECT_EQ(out_ptr1.size(), out_ptr2.size());
        size_t diff_count = 0;
        for (size_t i = 0; i < out_ptr1.size(); ++i) {
            if (out_ptr1[i] != out_ptr2[i]) diff_count++;
        }
        EXPECT_EQ(diff_count, 0);
    }
};

using concat_implicit_gpu_4d_f16 = concat_gpu_4d_implicit<FLOAT16>;
using concat_implicit_gpu_4d_i8 = concat_gpu_4d_implicit<int8_t>;

TEST_P(concat_implicit_gpu_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(1, { 16, 16 }, 2, 2),
                            TestParamType_concat(1, { 16, 8 }, 2, 2),
                            TestParamType_concat(1, { 8, 16 }, 2, 2)
                        ),
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_implicit_gpu_4d_i8, input_order_opt_b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(concat_gpu_onednn, basic_input_types) {
    auto& engine = get_onednn_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input4 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 3 } });

    set_values<float>(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values<float>(input1, { 11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f });
    set_values<float>(input2, { 21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f });
    set_values<float>(input3, { 31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f });
    set_values<float>(input4, { 41.0f, 42.0f, 43.0f, 44.0f, 42.0f, 42.0f, 43.0f, 44.0f, 43.0f, 43.0f, 43.0f, 45.0f });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f,
            41.0f, 42.0f, 43.0f, 44.0f, 42.0f, 42.0f, 43.0f, 44.0f, 43.0f, 43.0f, 43.0f, 45.0f };

    topology topology(
            input_layout("input0", input0->get_layout()),
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            input_layout("input3", input3->get_layout()),
            input_layout("input4", input4->get_layout()),
            concatenation("concat",
                          { "input0", "input1", "input2", "input3", "input4" },
                          1,
                          data_types::f32,
                          padding{ { 0,0,0,0 }, 0 })
    );

    build_options options_target;
    options_target.set_option(build_option::outputs({ "concat" }));
    implementation_desc impl = { format::bfyx, std::string(""), impl_types::onednn };
    options_target.set_option(build_option::force_implementations({ {"concat", impl} }));

    network network(engine, topology, options_target);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 4);
    EXPECT_EQ(f_size, 5);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

template <typename Type>
struct concat_gpu_4d_implicit_onednn : public concat_gpu {
public:
    cldnn::memory::ptr run_concat_network(std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input, format::type fmt, build_options options) {
        auto data_type = type_to_data_type<Type>::value;
        auto& engine = get_onednn_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<memory::ptr> in_memory;
        std::vector<primitive_id> input_ids;
        std::vector<primitive_id> pooling_ids;

        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = input[i];
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);
                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = engine.allocate_memory(in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            topology.add(pooling("pool" +  std::to_string(i), "input" + std::to_string(i), pooling_mode::max, {1, 1}, {1, 1}));

            input_ids.push_back("input" + std::to_string(i));
            pooling_ids.push_back("pool" + std::to_string(i));
        }

        topology.add(concatenation("concat", pooling_ids, 1));
        auto weights_lay = cldnn::layout(data_type, cldnn::format::bfyx, tensor(batch(output_f), feature(output_f)));
        auto weights_mem = engine.allocate_memory(weights_lay);
        weights_mem->fill(get_test_stream());
        get_test_stream().finish();
        {
            cldnn::mem_lock<Type> weights_ptr(weights_mem, get_test_stream());
            for (size_t fi = 0; fi < output_f; ++fi) {
                auto coords = tensor(batch(fi), feature(fi), spatial(0, 0, 0, 0));
                auto offset = weights_lay.get_linear_offset(coords);
                weights_ptr[offset] = static_cast<Type>(1.f);
            }
        }
        topology.add(data("weights" , weights_mem));
        topology.add(convolution("conv", "concat", { "weights" }));
        topology.add(pooling("pool_final", "conv", pooling_mode::max, {1, 1}, {1, 1}));
        topology.add(reorder("reorder", "pool_final", layout(data_type, format::bfyx, {(int32_t)batch_num, (int32_t)output_f, (int32_t)input_y, (int32_t)input_x})));

        network concat_network(engine, topology, options);
        for (size_t i = 0; i < in_features.size(); i++) {
            concat_network.set_input_data(input_ids[i], in_memory[i]);
        }
        concat_network.execute();

        bool concat_opt_enabled = options.get<build_option_type::optimize_data>()->enabled();
        bool concat_opt_result = std::static_pointer_cast<concatenation_inst>(concat_network.get_primitive("concat"))->node.can_be_optimized();
        EXPECT_TRUE(concat_opt_enabled==concat_opt_result);

        return concat_network.get_output("reorder").get_memory();
    }

    std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> generate_input() {
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());

        std::vector<std::vector<std::vector<std::vector<std::vector<Type>>>>> input(in_features.size());
        for (size_t i = 0; i < in_features.size(); ++i) {
            input[i] = generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
        }
        return input;
    }

    void test(format::type fmt) {
        auto& engine = get_onednn_test_engine();
        if (!engine.get_device_info().supports_immad) {
            // This case is only for device that uses onednn.
            return;
        }
        auto input = generate_input();

        // implicit concat
        build_options options1;
        options1.set_option(build_option::optimize_data(true));
        implementation_desc impl = { fmt, std::string(""), impl_types::onednn };
        options1.set_option(build_option::force_implementations({ {"conv", impl} }));
        auto out_mem1 = run_concat_network(input, fmt, options1);
        cldnn::mem_lock<Type> out_ptr1(out_mem1, get_test_stream());

        // explicit concat
        build_options options2;
        options2.set_option(build_option::optimize_data(false));
        auto out_mem2 = run_concat_network(input, fmt, options2);
        cldnn::mem_lock<Type> out_ptr2(out_mem2, get_test_stream());

        EXPECT_EQ(out_ptr1.size(), out_ptr2.size());
        size_t diff_count = 0;
        for (size_t i = 0; i < out_ptr1.size(); ++i) {
            if (out_ptr1[i] != out_ptr2[i]) diff_count++;
        }
        EXPECT_EQ(diff_count, 0);
    }
};


using concat_implicit_gpu_onednn_4d_f16 = concat_gpu_4d_implicit_onednn<FLOAT16>;
using concat_implicit_gpu_onednn_4d_i8 = concat_gpu_4d_implicit_onednn<int8_t>;

TEST_P(concat_implicit_gpu_onednn_4d_f16, input_order_opt_b_fs_yx_fsv16) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv16));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_onednn_4d_f16,
                        ::testing::Values(
                            TestParamType_concat(1, { 16, 16 }, 2, 2),
                            TestParamType_concat(1, { 16, 8 }, 2, 2),
                            TestParamType_concat(1, { 8, 16 }, 2, 2)
                        ),
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_implicit_gpu_onednn_4d_i8, input_order_opt_b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        concat_implicit_gpu_onednn_4d_i8,
                        ::testing::Values(
                            TestParamType_concat(1, { 32, 32 }, 2, 2),
                            TestParamType_concat(1, { 32, 8 }, 2, 2),
                            TestParamType_concat(1, { 8, 32 }, 2, 2)
                        ),
                        concat_gpu::PrintToStringParamName);
#endif
