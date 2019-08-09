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

#include <api/CPP/input_layout.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/reverse_sequence.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(reverese_sequence_gpu_test, fp32_d2_2_ba1_sa0) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 1;
    int32_t seq_axis = 0;

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            0.0f, 3.0f, 2.0f, 1.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d3_3_3_ba0_sa1) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 1;

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f
    });

    set_values(seq_lengths, {
        2.0f, 2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f,
            12.0f, 13.0f, 14.0f, 9.0f, 10.0f, 11.0f, 15.0f, 16.0f, 17.0f,
            21.0f, 22.0f, 23.0f, 18.0f, 19.0f, 20.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d3_3_3_ba2_sa0) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 2;
    int32_t seq_axis = 0;

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f
    });

    set_values(seq_lengths, {
            2.0f, 2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
            18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_3_2ba0_sa3) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 3;

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 
            13.0f, 12.0f, 15.0f, 14.0f, 17.0f, 16.0f, 
            19.0f, 18.0f, 21.0f, 20.0f, 23.0f, 22.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_3_2ba0_sa2) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 2;

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    set_values(seq_lengths, {
            2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f, 
            8.0f, 9.0f, 6.0f, 7.0f, 10.0f, 11.0f, 
            14.0f, 15.0f, 12.0f, 13.0f, 16.0f, 17.0f, 
            20.0f, 21.0f, 18.0f, 19.0f, 22.0f, 23.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_3_2ba2_sa0) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 2;
    int32_t seq_axis = 0;

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    set_values(seq_lengths, {
            1.0f, 1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 16.0f, 17.0f, 
            6.0f, 7.0f, 8.0f, 9.0f, 22.0f, 23.0f, 
            12.0f, 13.0f, 14.0f, 15.0f, 4.0f, 5.0f, 
            18.0f, 19.0f, 20.0f, 21.0f, 10.0f, 11.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_ba1_sa0) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 1;
    int32_t seq_axis = 0;

    set_values(input, {
            FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            0.0f, 3.0f, 2.0f, 1.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d3_3_3_ba0_sa1) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 1;

    set_values(input, {
            FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f), FLOAT16(9.0f),
            FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
            FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f), FLOAT16(26.0f)
    });

    set_values(seq_lengths, {
            2.0f, 2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<uint16_t >();

    std::vector<float> expected_results = {
            3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f,
            12.0f, 13.0f, 14.0f, 9.0f, 10.0f, 11.0f, 15.0f, 16.0f, 17.0f,
            21.0f, 22.0f, 23.0f, 18.0f, 19.0f, 20.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d3_3_3_ba2_sa0) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 2;
    int32_t seq_axis = 0;

    set_values(input, {
            FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f), FLOAT16(9.0f),
            FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
            FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f), FLOAT16(26.0f)
    });

    set_values(seq_lengths, {
            2.0f, 2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_3_2ba0_sa3) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 3;

    set_values(input, {
            FLOAT16(0.0f), FLOAT16( 1.0f), FLOAT16( 2.0f), FLOAT16( 3.0f), FLOAT16( 4.0f), FLOAT16( 5.0f), FLOAT16( 6.0f), FLOAT16( 7.0f), FLOAT16( 8.0f), FLOAT16( 9.0f),
            FLOAT16(10.0f), FLOAT16( 11.0f), FLOAT16( 12.0f), FLOAT16( 13.0f), FLOAT16( 14.0f), FLOAT16( 15.0f), FLOAT16( 16.0f), FLOAT16( 17.0f), FLOAT16( 18.0f), FLOAT16( 19.0f),
            FLOAT16(20.0f), FLOAT16( 21.0f), FLOAT16( 22.0f), FLOAT16( 23.0f)
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
            13.0f, 12.0f, 15.0f, 14.0f, 17.0f, 16.0f,
            19.0f, 18.0f, 21.0f, 20.0f, 23.0f, 22.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_3_2ba0_sa2) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 2;

    set_values(input, {
            FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f), FLOAT16(9.0f),
            FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
            FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
    });

    set_values(seq_lengths, {
            2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f,
            8.0f, 9.0f, 6.0f, 7.0f, 10.0f, 11.0f,
            14.0f, 15.0f, 12.0f, 13.0f, 16.0f, 17.0f,
            20.0f, 21.0f, 18.0f, 19.0f, 22.0f, 23.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_3_2ba2_sa0) {
    engine engine;

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 2;
    int32_t seq_axis = 0;

    set_values(input, {
            FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f), FLOAT16(9.0f),
            FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f),
            FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
    });

    set_values(seq_lengths, {
            1.0f, 1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths.get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", "input", "seq_lengths", seq_axis, batch_axis)
    );

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 16.0f, 17.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 22.0f, 23.0f,
            12.0f, 13.0f, 14.0f, 15.0f, 4.0f, 5.0f,
            18.0f, 19.0f, 20.0f, 21.0f, 10.0f, 11.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}
