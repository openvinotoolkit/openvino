// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/shuffle_channels.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template <typename T>
void test_d1_15_2_2_ax1_g5(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 15, 2, 2 } });
    int32_t axis = 1;
    int32_t group = 5;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
            50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Input0", input0);

    auto outputs = network->execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
            0.f, 1.f, 2.f, 3.f, 12.f, 13.f, 14.f, 15.f, 24.f, 25.f, 26.f, 27.f, 36.f, 37.f, 38.f, 39.f, 48.f, 49.f, 50.f, 51.f,
            4.f, 5.f, 6.f, 7.f, 16.f, 17.f, 18.f, 19.f, 28.f, 29.f, 30.f, 31.f, 40.f, 41.f, 42.f, 43.f, 52.f, 53.f, 54.f, 55.f,
            8.f, 9.f, 10.f, 11.f, 20.f, 21.f, 22.f, 23.f, 32.f, 33.f, 34.f, 35.f, 44.f, 45.f, 46.f, 47.f, 56.f, 57.f, 58.f, 59.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d1_15_2_2_ax1_g5) {
        test_d1_15_2_2_ax1_g5<float>(false);
}

TEST(shuffle_channels_fp32_gpu, d1_15_2_2_axm3_g5) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 15, 2, 2 } });
    int32_t axis = -3;
    int32_t group = 5;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
            50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 1.f, 2.f, 3.f, 12.f, 13.f, 14.f, 15.f, 24.f, 25.f, 26.f, 27.f, 36.f, 37.f, 38.f, 39.f, 48.f, 49.f, 50.f, 51.f,
            4.f, 5.f, 6.f, 7.f, 16.f, 17.f, 18.f, 19.f, 28.f, 29.f, 30.f, 31.f, 40.f, 41.f, 42.f, 43.f, 52.f, 53.f, 54.f, 55.f,
            8.f, 9.f, 10.f, 11.f, 20.f, 21.f, 22.f, 23.f, 32.f, 33.f, 34.f, 35.f, 44.f, 45.f, 46.f, 47.f, 56.f, 57.f, 58.f, 59.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d15_2_2_ax0_g5) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 15, 2, 1, 2 } });
    int32_t axis = 0;
    int32_t group = 5;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
            50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 1.f, 2.f, 3.f, 12.f, 13.f, 14.f, 15.f, 24.f, 25.f, 26.f, 27.f, 36.f, 37.f, 38.f, 39.f, 48.f, 49.f, 50.f, 51.f,
            4.f, 5.f, 6.f, 7.f, 16.f, 17.f, 18.f, 19.f, 28.f, 29.f, 30.f, 31.f, 40.f, 41.f, 42.f, 43.f, 52.f, 53.f, 54.f, 55.f,
            8.f, 9.f, 10.f, 11.f, 20.f, 21.f, 22.f, 23.f, 32.f, 33.f, 34.f, 35.f, 44.f, 45.f, 46.f, 47.f, 56.f, 57.f, 58.f, 59.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d15_2_2_axm4_g5) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 15, 2, 1, 2 } });
    int32_t axis = -4;
    int32_t group = 5;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
            50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 1.f, 2.f, 3.f, 12.f, 13.f, 14.f, 15.f, 24.f, 25.f, 26.f, 27.f, 36.f, 37.f, 38.f, 39.f, 48.f, 49.f, 50.f, 51.f,
            4.f, 5.f, 6.f, 7.f, 16.f, 17.f, 18.f, 19.f, 28.f, 29.f, 30.f, 31.f, 40.f, 41.f, 42.f, 43.f, 52.f, 53.f, 54.f, 55.f,
            8.f, 9.f, 10.f, 11.f, 20.f, 21.f, 22.f, 23.f, 32.f, 33.f, 34.f, 35.f, 44.f, 45.f, 46.f, 47.f, 56.f, 57.f, 58.f, 59.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d2_2_6_axm2_g3) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 6 } });
    int32_t axis = -2;
    int32_t group = 3;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 2.f, 4.f, 1.f, 3.f, 5.f, 6.f, 8.f, 10.f, 7.f, 9.f, 11.f,
            12.f, 14.f, 16.f, 13.f, 15.f, 17.f, 18.f, 20.f, 22.f, 19.f, 21.f, 23.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d2_6_2_axm3_g3) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 6, 1, 2 } });
    int32_t axis = -3;
    int32_t group = 3;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 1.f, 4.f, 5.f, 8.f, 9.f, 2.f, 3.f, 6.f, 7.f, 10.f, 11.f,
            12.f, 13.f, 16.f, 17.f, 20.f, 21.f, 14.f, 15.f, 18.f, 19.f, 22.f, 23.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d2_2_6_axm2_g2) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 6 } });
    int32_t axis = -2;
    int32_t group = 2;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 3.f, 1.f, 4.f, 2.f, 5.f, 6.f, 9.f, 7.f, 10.f, 8.f, 11.f,
            12.f, 15.f, 13.f, 16.f, 14.f, 17.f, 18.f, 21.f, 19.f, 22.f, 20.f, 23.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d2_6_2_axm3_g2) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 6, 1, 2 } });
    int32_t axis = -3;
    int32_t group = 2;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 1.f, 6.f, 7.f, 2.f, 3.f, 8.f, 9.f, 4.f, 5.f, 10.f, 11.f,
            12.f, 13.f, 18.f, 19.f, 14.f, 15.f, 20.f, 21.f, 16.f, 17.f, 22.f, 23.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d6_axm0_g2) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, { 6, 1, 1, 1 } });
    int32_t axis = 0;
    int32_t group = 2;

    set_values(input0, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input0->get_layout()));
    topology.add(
            shuffle_channels("shuffle_channels", input_info("Input0"), group, axis)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input0);

    auto outputs = network.execute();

    auto output = outputs.at("shuffle_channels").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            0.f, 3.f, 1.f, 4.f, 2.f, 5.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(shuffle_channels_fp32_gpu, d1_15_2_2_ax1_g5_cached) {
    test_d1_15_2_2_ax1_g5<float>(true);
}
