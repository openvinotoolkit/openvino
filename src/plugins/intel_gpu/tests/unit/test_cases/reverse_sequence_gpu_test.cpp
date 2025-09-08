// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reverse_sequence.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template <typename T>
void test_fp32_d2_2_ba1_sa0(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 1;
    int32_t seq_axis = 0;

    set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("seq_lengths", seq_lengths);

    auto outputs = network->execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
            0.0f, 3.0f, 2.0f, 1.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_ba1_sa0) {
    test_fp32_d2_2_ba1_sa0<float>(false);
}

template <typename T>
void test_fp32_d3_3_3_ba0_sa1(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("seq_lengths", seq_lengths);

    auto outputs = network->execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
            3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f,
            12.0f, 13.0f, 14.0f, 9.0f, 10.0f, 11.0f, 15.0f, 16.0f, 17.0f,
            21.0f, 22.0f, 23.0f, 18.0f, 19.0f, 20.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d3_3_3_ba0_sa1) {
    test_fp32_d3_3_3_ba0_sa1<float>(false);
}

TEST(reverese_sequence_gpu_test, fp32_d3_3_3_ba2_sa0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_3_2ba0_sa3) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
            13.0f, 12.0f, 15.0f, 14.0f, 17.0f, 16.0f,
            19.0f, 18.0f, 21.0f, 20.0f, 23.0f, 22.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_3_2ba0_sa2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f,
            8.0f, 9.0f, 6.0f, 7.0f, 10.0f, 11.0f,
            14.0f, 15.0f, 12.0f, 13.0f, 16.0f, 17.0f,
            20.0f, 21.0f, 18.0f, 19.0f, 22.0f, 23.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp32_d2_2_3_2ba2_sa0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 16.0f, 17.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 22.0f, 23.0f,
            12.0f, 13.0f, 14.0f, 15.0f, 4.0f, 5.0f,
            18.0f, 19.0f, 20.0f, 21.0f, 10.0f, 11.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_ba1_sa0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 1;
    int32_t seq_axis = 0;

    set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f)
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.0f, 3.0f, 2.0f, 1.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16x2_d2_2_ba1_sa0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 1;
    int32_t seq_axis = 0;

    set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f)
        });

    set_values(seq_lengths, {
            ov::float16(1.0f), ov::float16(2.0f)
        });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
        reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.0f, 3.0f, 2.0f, 1.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d3_3_3_ba0_sa1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 1;

    set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
            ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
            ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f)
    });

    set_values(seq_lengths, {
            2.0f, 2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f, 6.0f, 7.0f, 8.0f,
            12.0f, 13.0f, 14.0f, 9.0f, 10.0f, 11.0f, 15.0f, 16.0f, 17.0f,
            21.0f, 22.0f, 23.0f, 18.0f, 19.0f, 20.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d3_3_3_ba2_sa0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 1, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 2;
    int32_t seq_axis = 0;

    set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
            ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
            ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f)
    });

    set_values(seq_lengths, {
            2.0f, 2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_3_2ba0_sa3) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 3;

    set_values(input, {
            ov::float16(0.0f), ov::float16( 1.0f), ov::float16( 2.0f), ov::float16( 3.0f), ov::float16( 4.0f), ov::float16( 5.0f), ov::float16( 6.0f), ov::float16( 7.0f), ov::float16( 8.0f), ov::float16( 9.0f),
            ov::float16(10.0f), ov::float16( 11.0f), ov::float16( 12.0f), ov::float16( 13.0f), ov::float16( 14.0f), ov::float16( 15.0f), ov::float16( 16.0f), ov::float16( 17.0f), ov::float16( 18.0f), ov::float16( 19.0f),
            ov::float16(20.0f), ov::float16( 21.0f), ov::float16( 22.0f), ov::float16( 23.0f)
    });

    set_values(seq_lengths, {
            1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
            13.0f, 12.0f, 15.0f, 14.0f, 17.0f, 16.0f,
            19.0f, 18.0f, 21.0f, 20.0f, 23.0f, 22.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_3_2ba0_sa2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    int32_t batch_axis = 0;
    int32_t seq_axis = 2;

    set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
            ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
            ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f)
    });

    set_values(seq_lengths, {
            2.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f,
            8.0f, 9.0f, 6.0f, 7.0f, 10.0f, 11.0f,
            14.0f, 15.0f, 12.0f, 13.0f, 16.0f, 17.0f,
            20.0f, 21.0f, 18.0f, 19.0f, 22.0f, 23.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(reverese_sequence_gpu_test, fp16_d2_2_3_2ba2_sa0) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 2, 3 } });
    auto seq_lengths = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    int32_t batch_axis = 2;
    int32_t seq_axis = 0;

    set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
            ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
            ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f)
    });

    set_values(seq_lengths, {
            1.0f, 1.0f, 2.0f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("seq_lengths", seq_lengths->get_layout()));
    topology.add(
            reverse_sequence("reverse_sequence", input_info("input"), input_info("seq_lengths"), seq_axis, batch_axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("seq_lengths", seq_lengths);

    auto outputs = network.execute();

    auto output = outputs.at("reverse_sequence").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.0f, 1.0f, 2.0f, 3.0f, 16.0f, 17.0f,
            6.0f, 7.0f, 8.0f, 9.0f, 22.0f, 23.0f,
            12.0f, 13.0f, 14.0f, 15.0f, 4.0f, 5.0f,
            18.0f, 19.0f, 20.0f, 21.0f, 10.0f, 11.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(reverese_sequence_gpu_test, fp32_d2_2_ba1_sa0_cached) {
    test_fp32_d2_2_ba1_sa0<float>(true);
}
#endif
TEST(reverese_sequence_gpu_test, fp32_d3_3_3_ba0_sa1_cached) {
    test_fp32_d3_3_3_ba0_sa1<float>(true);
}
