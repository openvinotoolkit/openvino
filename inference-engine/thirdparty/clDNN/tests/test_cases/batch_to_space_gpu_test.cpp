// Copyright (c) 2019 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/batch_to_space.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/data.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(batch_to_space_fp16_gpu, i8111_bs1222_cb0000_ce0000) {
    //  Input  :      8x1x1x1
    //  Block shape : 1x2x2x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x2x2x2
    //  Input values in fp16

    engine engine;
    tensor input_shape = tensor{batch(8), feature(1), spatial(1, 1)};
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f),
        FLOAT16(6.0f), FLOAT16(7.0f)
    });
    set_values(block_shape, {
            1, 2, 2, 2
    });
    set_values(crops_begin, {
            0, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 0, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i4321_bs1212_cb0000_ce0000) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x6x2x2
    //  Input values in fp16

    engine engine;
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
    });
    set_values(block_shape, {
            1, 2, 1, 2
    });
    set_values(crops_begin, {
            0, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 0, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 6.f, 1.f, 7.f, 12.f, 18.f,
        13.f, 19.f, 2.f, 8.f, 3.f, 9.f,
        14.f, 20.f, 15.f, 21.f, 4.f, 10.f,
        5.f, 11.f, 16.f, 22.f, 17.f, 23.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i4321_bs1212_cb0010_ce0101) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x1x0
    //  Crops end :   0x1x0x1
    //  Output :      1x5x1x1
    //  Input values in fp16

    engine engine;
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
    });
    set_values(block_shape, {
            1, 2, 1, 2
    });
    set_values(crops_begin, {
            0, 0, 1, 0
    });
    set_values(crops_end, {
            0, 1, 0, 1
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 13.f, 3.f, 15.f, 5.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i62121_bs12311_cb02000_ce00110) {
    //  Input  :      6x2x1x2x1
    //  Block shape : 1x2x3x1x1
    //  Crops begin : 0x2x0x0x0
    //  Crops end :   0x0x1x1x0
    //  Output :      1x2x2x1x1
    //  Input values in fp16

    engine engine;
    tensor input_shape = tensor{batch(6), feature(2), spatial(1, 2, 1)};
    auto input = memory::allocate(engine, { data_types::f16, format::bfzyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfzyx, { 5, 1, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfzyx, { 5, 1, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfzyx, { 5, 1, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
    });
    set_values(block_shape, {
            1, 2, 3, 1, 1
    });
    set_values(crops_begin, {
            0, 2, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 1, 1, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        2.f, 6.f, 14.f, 18.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i1212112_bs112321_cb02000_ce00110) {
    //  Input  :      12x1x2x1x1x2
    //  Block shape : 1x1x2x3x2x1
    //  Crops begin : 0x0x1x0x0x0
    //  Crops end :   0x0x0x2x0x0
    //  Output :      1x1x3x1x2x2
    //  Input values in fp16

    engine engine;
    tensor input_shape = tensor{batch(12), feature(1), spatial(2, 1, 1, 2)};
    auto input = memory::allocate(engine, { data_types::f16, format::bfwzyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfwzyx, { 6, 1, 1, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfwzyx, { 6, 1, 1, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfwzyx, { 6, 1, 1, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f),
        FLOAT16(24.0f), FLOAT16(25.0f), FLOAT16(26.0f), FLOAT16(27.0f),
        FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f), FLOAT16(31.0f)
    });
    set_values(block_shape, {
            1, 1, 2, 3, 2, 1
    });
    set_values(crops_begin, {
            0, 0, 1, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 0, 2, 0, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        24.f, 25.f, 28.f, 29.f,
        2.f, 3.f, 6.f, 7.f,
        26.f, 27.f, 30.f, 31.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(batch_to_space_fp32_gpu, i8111_bs1222_cb0000_ce0000) {
    //  Input  :      8x1x1x1
    //  Block shape : 1x2x2x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x2x2x2
    //  Input values in fp32

    engine engine;
    tensor input_shape = tensor{batch(8), feature(1), spatial(1, 1)};
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    });
    set_values(block_shape, {
            1, 2, 2, 2
    });
    set_values(crops_begin, {
            0, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 0, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}


TEST(batch_to_space_fp32_gpu, i4321_bs1212_cb0000_ce0000) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x6x2x2
    //  Input values in fp32

    engine engine;
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    });
    set_values(block_shape, {
            1, 2, 1, 2
    });
    set_values(crops_begin, {
            0, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 0, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 6.f, 1.f, 7.f, 12.f, 18.f,
        13.f, 19.f, 2.f, 8.f, 3.f, 9.f,
        14.f, 20.f, 15.f, 21.f, 4.f, 10.f,
        5.f, 11.f, 16.f, 22.f, 17.f, 23.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}


TEST(batch_to_space_fp32_gpu, i4321_bs1212_cb0010_ce0101) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x1x0
    //  Crops end :   0x1x0x1
    //  Output :      1x5x1x1
    //  Input values in fp32

    engine engine;
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfyx, { 4, 1, 1, 1 } });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    });
    set_values(block_shape, {
            1, 2, 1, 2
    });
    set_values(crops_begin, {
            0, 0, 1, 0
    });
    set_values(crops_end, {
            0, 1, 0, 1
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 13.f, 3.f, 15.f, 5.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}


TEST(batch_to_space_fp32_gpu, i62121_bs12311_cb02000_ce00110) {
    //  Input  :      6x2x1x2x1
    //  Block shape : 1x2x3x1x1
    //  Crops begin : 0x2x0x0x0
    //  Crops end :   0x0x1x1x0
    //  Output :      1x2x2x1x1
    //  Input values in fp32

    engine engine;
    tensor input_shape = tensor{batch(6), feature(2), spatial(1, 2, 1)};
    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfzyx, { 5, 1, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfzyx, { 5, 1, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfzyx, { 5, 1, 1, 1, 1 } });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    });

    set_values(block_shape, {
            1, 2, 3, 1, 1
    });
    set_values(crops_begin, {
            0, 2, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 1, 1, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        2.f, 6.f, 14.f, 18.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i1212112_bs112321_cb02000_ce00110) {
    //  Input  :      12x1x2x1x1x2
    //  Block shape : 1x1x2x3x2x1
    //  Crops begin : 0x0x1x0x0x0
    //  Crops end :   0x0x0x2x0x0
    //  Output :      1x1x3x1x2x2
    //  Input values in fp32

    engine engine;
    tensor input_shape = tensor{batch(12), feature(1), spatial(2, 1, 1, 2)};
    auto input = memory::allocate(engine, { data_types::f32, format::bfwzyx, input_shape });

    auto block_shape = memory::allocate(engine, { data_types::i32, format::bfwzyx, { 6, 1, 1, 1, 1, 1 } });
    auto crops_begin = memory::allocate(engine, { data_types::i32, format::bfwzyx, { 6, 1, 1, 1, 1, 1 } });
    auto crops_end = memory::allocate(engine, { data_types::i32, format::bfwzyx, { 6, 1, 1, 1, 1, 1 } });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f,
        28.0f, 29.0f, 30.0f, 31.0f
    });
    set_values(block_shape, {
            1, 1, 2, 3, 2, 1
    });
    set_values(crops_begin, {
            0, 0, 1, 0, 0, 0
    });
    set_values(crops_end, {
            0, 0, 0, 2, 0, 0
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(data("input1", block_shape));
    topology.add(data("input2", crops_begin));
    topology.add(data("input3", crops_end));
    topology.add(batch_to_space("batch_to_space", "Input", "input1", "input2", "input3"));

    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        24.f, 25.f, 28.f, 29.f,
        2.f, 3.f, 6.f, 7.f,
        26.f, 27.f, 30.f, 31.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}
