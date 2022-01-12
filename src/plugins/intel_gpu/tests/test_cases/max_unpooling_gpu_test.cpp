// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/max_unpooling.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/pooling.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(max_unpooling_gpu, basic_in2x3x2x2) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, ""));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(max_unpooling_gpu, basic_in2x3x2x2_output_padding) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2
    //  Output Padding : 0x0x1x1

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, "", padding({ 0, 0, 1, 1 }, 0)));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 4.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 8.f, 16.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.5f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 12.f, 0.f, 17.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
    };
    std::vector<float> out;
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        out.push_back(output_ptr[i]);
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(max_unpooling_gpu, basic_in2x3x2x2_output_size) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2
    //  Output size explicitly provided

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        4.0f, 4.0f,
        8.0f, 16.0f,
        0.5f, 0.0f,
        12.0f, 17.0f
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", {2, 2, 3, 2}));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(max_unpooling_gpu, basic_in2x3x2x2_fp16) {
    //  Input  : 2x2x2x1
    //  Argmax : 2x2x2x1
    //  Output : 2x2x3x2
    //  Input values in fp16

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Input:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 2, 1 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

    set_values(input, {
        FLOAT16(4.0f), FLOAT16(4.0f),
        FLOAT16(8.0f), FLOAT16(16.0f),
        FLOAT16(0.5f), FLOAT16(0.0f),
        FLOAT16(12.0f), FLOAT16(17.0f)
    });

    set_values(arg_max, {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("arg_max", arg_max));
    topology.add(max_unpooling("max_unpooling", "input", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, ""));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };
    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(max_unpooling_gpu, basic_in2x2x3x2_max_with_argmax_pooling_unpooling) {
    //  Input  : 2x2x3x2
    //  Argmax : 2x2x2x1
    //  Output : 2x2x2x2

    //  Forward Max Pooling Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8   16    b1:   12   9     17
    //
    //  Forward Max Pooling Output:
    //  f0: b0:  4    4   b1:   0.5    0
    //  f1: b0:  8   16   b1:   12    17
    //
    //  Argmax:
    //  f0: b0:  4    4   b1:   15    13
    //  f1: b0:  10  11   b1:   21    23
    //
    //  Unpooling output:
    //  f0: b0:  0    0  0   b1:   0    0    0
    //  f0: b0:  0    4  0   b1:   0.5  0    0
    //  f1: b0:  0    0  0   b1:   0    0    0
    //  f1: b0:  0    8  16  b1:   12   0    17

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto arg_max = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 1 } });

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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(mutable_data("arg_max", arg_max));
    topology.add(pooling("pooling_max_with_argmax", "input", "arg_max", pooling_mode::max_with_argmax, { 2, 2 }, { 1, 1 }));
    topology.add(max_unpooling("max_unpooling", "pooling_max_with_argmax", "arg_max", { 1, 1, 2, 2 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, ""));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("max_unpooling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_layout = output->get_layout();
    cldnn::mem_lock<float> argmax_ptr(arg_max, get_test_stream());

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(output_layout.spatial(1), 2);
    EXPECT_EQ(output_layout.spatial(0), 3);
    EXPECT_EQ(output_layout.feature(), 2);
    EXPECT_EQ(output_layout.batch(), 2);

    std::vector<float> expected_argmax_vec = {
        4.0f, 4.0f,
        10.0f, 11.0f,
        15.0f, 13.0f,
        21.0f, 23.0f
    };

    std::vector<float> expected_output_vec = {
        0.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        0.f, 0.f, 0.f,
        0.f, 8.f, 16.f,
        0.f, 0.f, 0.f,
        0.5f, 0.f, 0.f,
        0.f, 0.f, 0.f,
        12.f, 0.f, 17.f,
    };

    for (size_t i = 0; i < expected_output_vec.size(); ++i) {
        EXPECT_EQ(expected_output_vec[i], output_ptr[i]);
    }

    for (size_t i = 0; i < expected_argmax_vec.size(); ++i) {
        EXPECT_EQ(expected_argmax_vec[i], argmax_ptr[i]);
    }
}
