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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/space_to_batch.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/data.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(space_to_batch_fp16_gpu, i1222_bs1222_pb0000_pe0000) {
    // Input :       1x2x2x2
    // Block shape : 1x2x2x2
    // Pads begin :  0x0x0x0
    // Pads end :    0x0x0x0
    // Output :      8x1x1x1
    // Input values in fp16

    engine engine;
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, {1,2,2,2} });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f),
        FLOAT16(6.0f), FLOAT16(7.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfyx, {1,2,2,2}, 1),
                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                           tensor(format::bfyx, {8,1,1,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(space_to_batch_fp16_gpu, i1242_bs1221_pb0020_pe0000) {
    // Input :       1x2x4x2
    // Block shape : 1x2x2x1
    // Pads begin :  0x0x2x0
    // Pads end :    0x0x0x0
    // Output :      4x1x3x2
    // Input values in fp16

    engine engine;
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, {1,2,2,4} });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfyx, {1,2,2,1}, 1),
                                                           tensor(format::bfyx, {0,0,2,0}, 0),
                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                           tensor(format::bfyx, {4,1,3,2}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 0.f, 0.f, 1.f, 4.f, 5.f,
        0.f, 0.f, 2.f, 3.f, 6.f, 7.f,
        0.f, 0.f, 8.f, 9.f, 12.f, 13.f,
        0.f, 0.f, 10.f, 11.f, 14.f, 15.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(space_to_batch_fp16_gpu, i2132_bs1222_pb0010_pe0100) {
    // Input :       2x1x3x2
    // Block shape : 1x2x2x2
    // Pads begin :  0x0x1x0
    // Pads end :    0x1x0x0
    // Output :      16x1x2x1
    // Input values in fp16

    engine engine;
    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, {2,1,2,3} });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfyx, {1,2,2,2}, 1),
                                                           tensor(format::bfyx, {0,0,1,0}, 0),
                                                           tensor(format::bfyx, {0,1,0,0}, 0),
                                                           tensor(format::bfyx, {16,1,2,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 2.f, 0.f, 8.f, 0.f, 3.f, 0.f, 9.f,
        0.f, 4.f, 6.f, 10.f, 1.f, 5.f, 7.f, 11.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(space_to_batch_fp16_gpu, i12132_bs12122_pb00010_pe00000) {
    // Input :       1x2x1x3x2
    // Block shape : 1x2x1x2x2
    // Pads begin :  0x0x0x1x0
    // Pads end :    0x0x0x0x0
    // Output :      8x1x1x2x1
    // Input values in fp16

    engine engine;
    auto input = memory::allocate(engine, { data_types::f16, format::bfzyx, {1,2,2,3,1} });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfzyx, {1,2,1,2,2}, 1),
                                                           tensor(format::bfzyx, {0,0,0,1,0}, 0),
                                                           tensor(format::bfzyx, {0,0,0,0,0}, 0),
                                                           tensor(format::bfzyx, {8,1,1,2,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 2.f, 0.f, 3.f, 0.f, 4.f, 1.f, 5.f,
        0.f, 8.f, 0.f, 9.f, 6.f, 10.f, 7.f, 11.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(space_to_batch_fp16_gpu, i134121_bs142121_pb010100_pe000000) {
    // Input :       1x3x4x1x2x1
    // Block shape : 1x4x2x1x2x1
    // Pads begin :  0x1x0x1x0x0
    // Pads end :    0x0x0x0x0x0
    // Output :      16x1x2x2x1x1
    // Input values in fp16

    engine engine;
    tensor input_shape = tensor{ batch(1), feature(3), spatial(1, 2, 1, 4) };
    auto input = memory::allocate(engine, { data_types::f16, format::bfwzyx, input_shape });

    set_values(input, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfwzyx, {1,4,2,1,2,1}, 1),
                                                           tensor(format::bfwzyx, {0,1,0,1,0,0}, 0),
                                                           tensor(format::bfwzyx, {0,0,0,0,0,0}, 0),
                                                           tensor(format::bfwzyx, {16,1,2,2,1,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 4.f, 0.f, 1.f, 0.f, 5.f,
        0.f, 2.f, 0.f, 6.f, 0.f, 3.f, 0.f, 7.f,
        0.f, 8.f, 0.f, 12.f, 0.f, 9.f, 0.f, 13.f,
        0.f, 10.f, 0.f, 14.f, 0.f, 11.f, 0.f, 15.f,
        0.f, 16.f, 0.f, 20.f, 0.f, 17.f, 0.f, 21.f,
        0.f, 18.f, 0.f, 22.f, 0.f, 19.f, 0.f, 23.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(space_to_batch_fp32_gpu, i1222_bs1222_pb0000_pe0000) {
    // Input :       1x2x2x2
    // Block shape : 1x2x2x2
    // Pads begin :  0x0x0x0
    // Pads end :    0x0x0x0
    // Output :      8x1x1x1
    // Input values in fp32

    engine engine;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, {1,2,2,2} });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfyx, {1,2,2,2}, 1),
                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                           tensor(format::bfyx, {8,1,1,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(space_to_batch_fp32_gpu, i1242_bs1221_pb0020_pe0000) {
    // Input :       1x2x4x2
    // Block shape : 1x2x2x1
    // Pads begin :  0x0x2x0
    // Pads end :    0x0x0x0
    // Output :      4x1x3x2
    // Input values in fp32

    engine engine;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, {1,2,2,4} });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfyx, {1,2,2,1}, 1),
                                                           tensor(format::bfyx, {0,0,2,0}, 0),
                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                           tensor(format::bfyx, {4,1,3,2}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 0.f, 0.f, 1.f, 4.f, 5.f,
        0.f, 0.f, 2.f, 3.f, 6.f, 7.f,
        0.f, 0.f, 8.f, 9.f, 12.f, 13.f,
        0.f, 0.f, 10.f, 11.f, 14.f, 15.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(space_to_batch_fp32_gpu, i2132_bs1222_pb0010_pe0100) {
    // Input :       2x1x3x2
    // Block shape : 1x2x2x2
    // Pads begin :  0x0x1x0
    // Pads end :    0x1x0x0
    // Output :      16x1x2x1
    // Input values in fp32

    engine engine;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, {2,1,2,3} });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfyx, {1,2,2,2}, 1),
                                                           tensor(format::bfyx, {0,0,1,0}, 0),
                                                           tensor(format::bfyx, {0,1,0,0}, 0),
                                                           tensor(format::bfyx, {16,1,2,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 2.f, 0.f, 8.f, 0.f, 3.f, 0.f, 9.f,
        0.f, 4.f, 6.f, 10.f, 1.f, 5.f, 7.f, 11.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(space_to_batch_fp32_gpu, i12132_bs12122_pb00010_pe00000) {
    // Input :       1x2x1x3x2
    // Block shape : 1x2x1x2x2
    // Pads begin :  0x0x0x1x0
    // Pads end :    0x0x0x0x0
    // Output :      8x1x1x2x1
    // Input values in fp32

    engine engine;
    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx, {1,2,2,3,1} });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfzyx, {1,2,1,2,2}, 1),
                                                           tensor(format::bfzyx, {0,0,0,1,0}, 0),
                                                           tensor(format::bfzyx, {0,0,0,0,0}, 0),
                                                           tensor(format::bfzyx, {8,1,1,2,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 2.f, 0.f, 3.f, 0.f, 4.f, 1.f, 5.f,
        0.f, 8.f, 0.f, 9.f, 6.f, 10.f, 7.f, 11.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(space_to_batch_fp32_gpu, i134121_bs142121_pb010100_pe000000) {
    // Input :       1x3x4x1x2x1
    // Block shape : 1x4x2x1x2x1
    // Pads begin :  0x1x0x1x0x0
    // Pads end :    0x0x0x0x0x0
    // Output :      16x1x2x2x1x1
    // Input values in fp32

    engine engine;
    tensor input_shape = tensor{ batch(1), feature(3), spatial(1, 2, 1, 4) };
    auto input = memory::allocate(engine, { data_types::f32, format::bfwzyx, input_shape });

    set_values(input, {
       0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
       6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
       18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input", input.get_layout()));
    topology.add(space_to_batch("space_to_batch", "Input", tensor(format::bfwzyx, {1,4,2,1,2,1}, 1),
                                                           tensor(format::bfwzyx, {0,1,0,1,0,0}, 0),
                                                           tensor(format::bfwzyx, {0,0,0,0,0,0}, 0),
                                                           tensor(format::bfwzyx, {16,1,2,2,1,1}, 1)));
    network network(engine, topology);

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("space_to_batch").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 4.f, 0.f, 1.f, 0.f, 5.f,
        0.f, 2.f, 0.f, 6.f, 0.f, 3.f, 0.f, 7.f,
        0.f, 8.f, 0.f, 12.f, 0.f, 9.f, 0.f, 13.f,
        0.f, 10.f, 0.f, 14.f, 0.f, 11.f, 0.f, 15.f,
        0.f, 16.f, 0.f, 20.f, 0.f, 17.f, 0.f, 21.f,
        0.f, 18.f, 0.f, 22.f, 0.f, 19.f, 0.f, 23.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}
