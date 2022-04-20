// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/average_unpooling.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(average_unpooling_gpu, basic_in2x2x2x1) {
    //  Input  : 2x2x2x1
    //  Output : 2x2x3x2

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17.5
    //
    //  Input:
    //  f0: b0:  2.5 -4.5  b1:   0    -6.75
    //  f1: b0:  6    4    b1:   7     4.75
    //
    //  Output:
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });

    set_values(input, {
        2.5f, -4.5f,
        6.f, 4.f,
        0.f, -6.75f,
        7.0f, 4.75f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(average_unpooling("average_unpooling", "input", { 2, 2, 3, 2 }, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.625f, -0.5f, -1.125,
        0.625f, -0.5f, -1.125,
        1.5f, 2.5f, 1.f,
        1.5f, 2.5f, 1.f,
        0.f, -1.6875f, -1.6875f,
        0.f, -1.6875f, -1.6875f,
        1.75f, 2.9375f, 1.1875f,
        1.75f, 2.9375f, 1.1875f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(average_unpooling_gpu, basic_in2x2x3x2_with_average_pooling_unpooling) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x3x2

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  3    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17
    //
    //  Input:
    //  f0: b0:  2.5 -12  b1:   0    -13
    //  f1: b0:  6    1   b1:   7     2
    //
    //  Output:
    //  f0: b0:  0.625   0.625  -6   b1:   0      0     -6.5
    //  f0: b0:  0.625   0.625  -6   b1:   0      0     -6.5
    //  f1: b0:  1.5     1.5    0.5  b1:   1.75   1.75   1
    //  f1: b0:  1.5     1.5    0.5  b1:   1.75   1.75   1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        3.f, 6.f, -12.f,
        7.f, 8.f, 14.f,
        0.f, 0.f, -11.f,
        1.f, -1.f, -15.f,
        1.5f, 5.5f, -13.f,
        12.f, 9.f, 17.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(pooling("pooling", "input", pooling_mode::average_no_padding, { 2, 2 }, { 2, 2 }));
    topology.add(average_unpooling("average_unpooling", "pooling", input->get_layout().size, { 1, 1, 2, 2 }, { 1, 1, 2, 2 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.625f, 0.625f, -6,
        0.625f, 0.625f, -6,
        1.5f, 1.5f, 0.5f,
        1.5f, 1.5f, 0.5f,
        0.f, 0, -6.5f,
        0.f, 0, -6.5f,
        1.75f, 1.75f, 1.f,
        1.75f, 1.75f, 1.f
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(average_unpooling_gpu, basic_in2x2x2x1_output_padding) {
    //  Input  : 2x2x2x1
    //  Output Padding : 0x0x1x1
    //  Output : 2x2x3x2

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17.5
    //
    //  Input:
    //  f0: b0:  2.5 -4.5  b1:   0    -6.75
    //  f1: b0:  6    4    b1:   7     4.75
    //
    //  Output:
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        2.5f, -4.5f,
        6.f, 4.f,
        0.f, -6.75f,
        7.0f, 4.75f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(average_unpooling("average_unpooling", "input", { 2, 2, 3, 2 }, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, "", padding({ 0, 0, 1, 1 }, 0)));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.625f, -0.5f, -1.125, 0.f,
        0.f, 0.625f, -0.5f, -1.125, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 1.5f, 2.5f, 1.f, 0.f,
        0.f, 1.5f, 2.5f, 1.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, -1.6875f, -1.6875f, 0.f,
        0.f, 0.f, -1.6875f, -1.6875f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 1.75f, 2.9375f, 1.1875f, 0.f,
        0.f, 1.75f, 2.9375f, 1.1875f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f
    };
    std::vector<float> out;
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        out.push_back(output_ptr[i]);
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(average_unpooling_gpu, basic_in2x2x2x1_fp16) {
    //  Input  : 2x2x2x1
    //  Output Padding : 0x0x1x1
    //  Output : 2x2x3x2
    //  Input values in fp16

    //  Forward Average Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   1   -1     -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.5   -13
    //  f1: b0:  7    8   14   b1:   12   9      17.5
    //
    //  Input:
    //  f0: b0:  2.5 -4.5  b1:   0    -6.75
    //  f1: b0:  6    4    b1:   7     4.75
    //
    //  Output:
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f0: b0:  0.625   -0.5  -1.125   b1:   0  -1.6875  -1.6875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875
    //  f1: b0:  1.5     2.5    1       b1:   1.75   2.9375   1.1875

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        FLOAT16(2.5f), FLOAT16(-4.5f),
        FLOAT16(6.f), FLOAT16(4.f),
        FLOAT16(0.f), FLOAT16(-6.75f),
        FLOAT16(7.0f), FLOAT16(4.75f)
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(average_unpooling("average_unpooling", "input", { 2, 2, 3, 2 }, { 1, 1, 2, 2 }, { 1, 1, 1, 1 }));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("average_unpooling").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.625f, -0.5f, -1.125,
        0.625f, -0.5f, -1.125,
        1.5f, 2.5f, 1.f,
        1.5f, 2.5f, 1.f,
        0.f, -1.6875f, -1.6875f,
        0.f, -1.6875f, -1.6875f,
        1.75f, 2.9375f, 1.1875f,
        1.75f, 2.9375f, 1.1875f
    };
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}
