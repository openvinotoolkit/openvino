// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/space_to_depth.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

class space_to_depth_fp16_gpu: public ::testing::Test {
public:
    void test_d1122_bs2_mbf(bool is_caching_test) {
        //  Input  : 1x1x2x2
        //  Block size : 2
        //  Output : 1x4x1x1
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 2, 2 } });
        size_t block_size = 2;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f),
                ov::float16(2.0f), ov::float16(3.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.f, 1.f, 2.f, 3.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1142_bs2_mbf(bool is_caching_test) {
        //  Input  : 1x1x4x2
        //  Block size : 2
        //  Output : 1x4x2x1
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 2, 4 } });
        size_t block_size = 2;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f),
                ov::float16(2.0f), ov::float16(3.0f),
                ov::float16(4.0f), ov::float16(5.0f),
                ov::float16(6.0f), ov::float16(7.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 4.0f, 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1264_bs2_mbf(bool is_caching_test) {
        //  Input  : 1x2x6x4
        //  Block size : 2
        //  Output : 1x8x3x2
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 2, 4, 6 } });
        size_t block_size = 2;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
                ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
                ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f),
                ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
                ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f),
                ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f), ov::float16(29.0f),
                ov::float16(30.0f), ov::float16(31.0f), ov::float16(32.0f), ov::float16(33.0f), ov::float16(34.0f),
                ov::float16(35.0f), ov::float16(36.0f), ov::float16(37.0f), ov::float16(38.0f), ov::float16(39.0f),
                ov::float16(40.0f), ov::float16(41.0f), ov::float16(42.0f), ov::float16(43.0f), ov::float16(44.0f),
                ov::float16(45.0f), ov::float16(46.0f), ov::float16(47.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
        space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 2.0f, 8.0f, 10.0f, 16.0f, 18.0f,
                24.0f, 26.0f, 32.0f, 34.0f, 40.0f, 42.0f,
                1.0f, 3.0f, 9.0f, 11.0f, 17.0f, 19.0f,
                25.0f, 27.0f, 33.0f, 35.0f, 41.0f, 43.0f,
                4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f,
                28.0f, 30.0f, 36.0f, 38.0f, 44.0f, 46.0f,
                5.0f, 7.0f, 13.0f, 15.0f, 21.0f, 23.0f,
                29.0f, 31.0f, 37.0f, 39.0f, 45.0f, 47.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1199_bs3_mbf(bool is_caching_test) {
        //  Input  : 1x1x9x9
        //  Block size : 3
        //  Output : 1x9x3x3
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 9, 9 } });
        size_t block_size = 3;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
                ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
                ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f),
                ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
                ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f),
                ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f), ov::float16(29.0f),
                ov::float16(30.0f), ov::float16(31.0f), ov::float16(32.0f), ov::float16(33.0f), ov::float16(34.0f),
                ov::float16(35.0f), ov::float16(36.0f), ov::float16(37.0f), ov::float16(38.0f), ov::float16(39.0f),
                ov::float16(40.0f), ov::float16(41.0f), ov::float16(42.0f), ov::float16(43.0f), ov::float16(44.0f),
                ov::float16(45.0f), ov::float16(46.0f), ov::float16(47.0f), ov::float16(48.0f), ov::float16(49.0f),
                ov::float16(50.0f), ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),
                ov::float16(55.0f), ov::float16(56.0f), ov::float16(57.0f), ov::float16(58.0f), ov::float16(59.0f),
                ov::float16(60.0f), ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),
                ov::float16(65.0f), ov::float16(66.0f), ov::float16(67.0f), ov::float16(68.0f), ov::float16(69.0f),
                ov::float16(70.0f), ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),
                ov::float16(75.0f), ov::float16(76.0f), ov::float16(77.0f), ov::float16(78.0f), ov::float16(79.0f),
                ov::float16(80.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f, 1.0f,
                4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f, 2.0f, 5.0f,
                8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f, 9.0f, 12.0f, 15.0f,
                36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f, 10.0f, 13.0f, 16.0f, 37.0f,
                40.0f, 43.0f, 64.0f, 67.0f, 70.0f, 11.0f, 14.0f, 17.0f, 38.0f, 41.0f,
                44.0f, 65.0f, 68.0f, 71.0f, 18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f,
                72.0f, 75.0f, 78.0f, 19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f,
                76.0f, 79.0f, 20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f,
                80.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1122_bs2_mdf(bool is_caching_test) {
        //  Input  : 1x1x2x2
        //  Block size : 2
        //  Output : 1x4x1x1
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 2, 2 } });
        size_t block_size = 2;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f),
                ov::float16(2.0f), ov::float16(3.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.f, 1.f, 2.f, 3.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1142_bs2_mdf(bool is_caching_test) {
        //  Input  : 1x1x4x2
        //  Block size : 2
        //  Output : 1x4x2x1
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 2, 4 } });
        size_t block_size = 2;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f),
                ov::float16(2.0f), ov::float16(3.0f),
                ov::float16(4.0f), ov::float16(5.0f),
                ov::float16(6.0f), ov::float16(7.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 4.0f, 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1264_bs2_mdf(bool is_caching_test) {
        //  Input  : 1x2x6x4
        //  Block size : 2
        //  Output : 1x8x3x2
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 2, 4, 6 } });
        size_t block_size = 2;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
                ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
                ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f),
                ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
                ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f),
                ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f), ov::float16(29.0f),
                ov::float16(30.0f), ov::float16(31.0f), ov::float16(32.0f), ov::float16(33.0f), ov::float16(34.0f),
                ov::float16(35.0f), ov::float16(36.0f), ov::float16(37.0f), ov::float16(38.0f), ov::float16(39.0f),
                ov::float16(40.0f), ov::float16(41.0f), ov::float16(42.0f), ov::float16(43.0f), ov::float16(44.0f),
                ov::float16(45.0f), ov::float16(46.0f), ov::float16(47.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 2.0f, 8.0f, 10.0f, 16.0f, 18.0f,
                1.0f, 3.0f, 9.0f, 11.0f, 17.0f, 19.0f,
                4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f,
                5.0f, 7.0f, 13.0f, 15.0f, 21.0f, 23.0f,
                24.0f, 26.0f, 32.0f, 34.0f, 40.0f, 42.0f,
                25.0f, 27.0f, 33.0f, 35.0f, 41.0f, 43.0f,
                28.0f, 30.0f, 36.0f, 38.0f, 44.0f, 46.0f,
                29.0f, 31.0f, 37.0f, 39.0f, 45.0f, 47.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_d1199_bs3_mdf(bool is_caching_test) {
        //  Input  : 1x1x9x9
        //  Block size : 3
        //  Output : 1x9x3x3
        //  Input values in fp16

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 9, 9 } });
        size_t block_size = 3;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
                ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f), ov::float16(9.0f),
                ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f),
                ov::float16(15.0f), ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
                ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f),
                ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f), ov::float16(29.0f),
                ov::float16(30.0f), ov::float16(31.0f), ov::float16(32.0f), ov::float16(33.0f), ov::float16(34.0f),
                ov::float16(35.0f), ov::float16(36.0f), ov::float16(37.0f), ov::float16(38.0f), ov::float16(39.0f),
                ov::float16(40.0f), ov::float16(41.0f), ov::float16(42.0f), ov::float16(43.0f), ov::float16(44.0f),
                ov::float16(45.0f), ov::float16(46.0f), ov::float16(47.0f), ov::float16(48.0f), ov::float16(49.0f),
                ov::float16(50.0f), ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),
                ov::float16(55.0f), ov::float16(56.0f), ov::float16(57.0f), ov::float16(58.0f), ov::float16(59.0f),
                ov::float16(60.0f), ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),
                ov::float16(65.0f), ov::float16(66.0f), ov::float16(67.0f), ov::float16(68.0f), ov::float16(69.0f),
                ov::float16(70.0f), ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),
                ov::float16(75.0f), ov::float16(76.0f), ov::float16(77.0f), ov::float16(78.0f), ov::float16(79.0f),
                ov::float16(80.0f)
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f, 1.0f,
                4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f, 2.0f, 5.0f,
                8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f, 9.0f, 12.0f, 15.0f,
                36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f, 10.0f, 13.0f, 16.0f, 37.0f,
                40.0f, 43.0f, 64.0f, 67.0f, 70.0f, 11.0f, 14.0f, 17.0f, 38.0f, 41.0f,
                44.0f, 65.0f, 68.0f, 71.0f, 18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f,
                72.0f, 75.0f, 78.0f, 19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f,
                76.0f, 79.0f, 20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f,
                80.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }
};

class space_to_depth_fp32_gpu: public ::testing::Test {
public:
    void test_d1122_bs2_mbf(bool is_caching_test) {
        //  Input  : 1x1x2x2
        //  Block size : 2
        //  Output : 1x4x1x1
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
        size_t block_size = 2;

        set_values(input1, {
                0.f, 1.f, 2.f, 3.f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.f, 1.f, 2.f, 3.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1142_bs2_mbf(bool is_caching_test) {
        //  Input  : 1x1x4x2
        //  Block size : 2
        //  Output : 1x4x2x1
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 4 } });
        size_t block_size = 2;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 4.0f, 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1264_bs2_mbf(bool is_caching_test) {
        //  Input  : 1x2x6x4
        //  Block size : 2
        //  Output : 1x8x3x2
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 6 } });
        size_t block_size = 2;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f,
                35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f,
                45.0f, 46.0f, 47.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
            space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 2.0f, 8.0f, 10.0f, 16.0f, 18.0f,
                24.0f, 26.0f, 32.0f, 34.0f, 40.0f, 42.0f,
                1.0f, 3.0f, 9.0f, 11.0f, 17.0f, 19.0f,
                25.0f, 27.0f, 33.0f, 35.0f, 41.0f, 43.0f,
                4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f,
                28.0f, 30.0f, 36.0f, 38.0f, 44.0f, 46.0f,
                5.0f, 7.0f, 13.0f, 15.0f, 21.0f, 23.0f,
                29.0f, 31.0f, 37.0f, 39.0f, 45.0f, 47.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1199_bs3_mbf(bool is_caching_test) {
        //  Input  : 1x1x9x9
        //  Block size : 3
        //  Output : 1x9x3x3
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 9, 9 } });
        size_t block_size = 3;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
                50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
                70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
                80.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f, 1.0f,
                4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f, 2.0f, 5.0f,
                8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f, 9.0f, 12.0f, 15.0f,
                36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f, 10.0f, 13.0f, 16.0f, 37.0f,
                40.0f, 43.0f, 64.0f, 67.0f, 70.0f, 11.0f, 14.0f, 17.0f, 38.0f, 41.0f,
                44.0f, 65.0f, 68.0f, 71.0f, 18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f,
                72.0f, 75.0f, 78.0f, 19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f,
                76.0f, 79.0f, 20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f,
                80.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1122_bs2_mdf(bool is_caching_test) {
        //  Input  : 1x1x2x2
        //  Block size : 2
        //  Output : 1x4x1x1
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
        size_t block_size = 2;

        set_values(input1, {
                0.f, 1.f, 2.f, 3.f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.f, 1.f, 2.f, 3.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1142_bs2_mdf(bool is_caching_test) {
        //  Input  : 1x1x4x2
        //  Block size : 2
        //  Output : 1x4x2x1
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 4 } });
        size_t block_size = 2;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 4.0f, 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1264_bs2_mdf(bool is_caching_test) {
        //  Input  : 1x2x6x4
        //  Block size : 2
        //  Output : 1x8x3x2
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 6 } });
        size_t block_size = 2;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f,
                35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f,
                45.0f, 46.0f, 47.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 2.0f, 8.0f, 10.0f, 16.0f, 18.0f,
                1.0f, 3.0f, 9.0f, 11.0f, 17.0f, 19.0f,
                4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f,
                5.0f, 7.0f, 13.0f, 15.0f, 21.0f, 23.0f,
                24.0f, 26.0f, 32.0f, 34.0f, 40.0f, 42.0f,
                25.0f, 27.0f, 33.0f, 35.0f, 41.0f, 43.0f,
                28.0f, 30.0f, 36.0f, 38.0f, 44.0f, 46.0f,
                29.0f, 31.0f, 37.0f, 39.0f, 45.0f, 47.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1199_bs3_mdf(bool is_caching_test) {
        //  Input  : 1x1x9x9
        //  Block size : 3
        //  Output : 1x9x3x3
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 9, 9 } });
        size_t block_size = 3;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
                50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
                70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
                80.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(
                space_to_depth("space_to_depth", input_info("Input0"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_depth").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f, 1.0f,
                4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f, 2.0f, 5.0f,
                8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f, 9.0f, 12.0f, 15.0f,
                36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f, 10.0f, 13.0f, 16.0f, 37.0f,
                40.0f, 43.0f, 64.0f, 67.0f, 70.0f, 11.0f, 14.0f, 17.0f, 38.0f, 41.0f,
                44.0f, 65.0f, 68.0f, 71.0f, 18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f,
                72.0f, 75.0f, 78.0f, 19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f,
                76.0f, 79.0f, 20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f,
                80.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1199_bs3_mdf_fsv16(bool is_caching_test) {
        //  Input  : 1x1x9x9
        //  Block size : 3
        //  Output : 1x9x3x3
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 9, 9 } });
        size_t block_size = 3;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
                50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
                70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
                80.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(reorder("reorder", input_info("Input0"), format::b_fs_yx_fsv16, data_types::f32));
        topology.add(space_to_depth("space_to_depth", input_info("reorder"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size));
        topology.add(reorder("reorder_out", input_info("space_to_depth"), format::bfyx, data_types::f32));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("reorder_out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f, 1.0f,
                4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f, 2.0f, 5.0f,
                8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f, 9.0f, 12.0f, 15.0f,
                36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f, 10.0f, 13.0f, 16.0f, 37.0f,
                40.0f, 43.0f, 64.0f, 67.0f, 70.0f, 11.0f, 14.0f, 17.0f, 38.0f, 41.0f,
                44.0f, 65.0f, 68.0f, 71.0f, 18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f,
                72.0f, 75.0f, 78.0f, 19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f,
                76.0f, 79.0f, 20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f,
                80.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_d1199_bs3_mdf_fsv4(bool is_caching_test) {
        //  Input  : 1x1x9x9
        //  Block size : 3
        //  Output : 1x9x3x3
        //  Input values in fp32

        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 9, 9 } });
        size_t block_size = 3;

        set_values(input1, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
                50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
                70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
                80.0f
        });

        topology topology;
        topology.add(input_layout("Input0", input1->get_layout()));
        topology.add(reorder("reorder", input_info("Input0"), format::b_fs_yx_fsv4, data_types::f32));
        topology.add(space_to_depth("space_to_depth", input_info("reorder"), SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size));
        topology.add(reorder("reorder_out", input_info("space_to_depth"), format::bfyx, data_types::f32));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input1);

        auto outputs = network->execute();

        auto output = outputs.at("reorder_out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f, 1.0f,
                4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f, 2.0f, 5.0f,
                8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f, 9.0f, 12.0f, 15.0f,
                36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f, 10.0f, 13.0f, 16.0f, 37.0f,
                40.0f, 43.0f, 64.0f, 67.0f, 70.0f, 11.0f, 14.0f, 17.0f, 38.0f, 41.0f,
                44.0f, 65.0f, 68.0f, 71.0f, 18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f,
                72.0f, 75.0f, 78.0f, 19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f,
                76.0f, 79.0f, 20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f,
                80.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Test cases for mode "blocks first ".
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(space_to_depth_fp16_gpu, d1122_bs2_mbf) {
    this->test_d1122_bs2_mbf(false);
}

TEST_F(space_to_depth_fp16_gpu, d1142_bs2_mbf) {
    this->test_d1142_bs2_mbf(false);
}

TEST_F(space_to_depth_fp16_gpu, d1264_bs2_mbf) {
    this->test_d1264_bs2_mbf(false);
}

TEST_F(space_to_depth_fp16_gpu, d1199_bs3_mbf) {
    this->test_d1199_bs3_mbf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1122_bs2_mbf) {
    this->test_d1122_bs2_mbf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1142_bs2_mbf) {
    test_d1142_bs2_mbf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1264_bs2_mbf) {
    this->test_d1264_bs2_mbf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mbf) {
    this->test_d1199_bs3_mbf(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2. Test cases for mode "depth first ".
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(space_to_depth_fp16_gpu, d1122_bs2_mdf) {
    this->test_d1122_bs2_mdf(false);
}

TEST_F(space_to_depth_fp16_gpu, d1142_bs2_mdf) {
    this->test_d1142_bs2_mdf(false);
}

TEST_F(space_to_depth_fp16_gpu, d1264_bs2_mdf) {
    this->test_d1264_bs2_mdf(false);
}

TEST_F(space_to_depth_fp16_gpu, d1199_bs3_mdf) {
    this->test_d1199_bs3_mdf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1122_bs2_mdf) {
    this->test_d1122_bs2_mdf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1142_bs2_mdf) {
    this->test_d1142_bs2_mdf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1264_bs2_mdf) {
    this->test_d1264_bs2_mdf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mdf) {
    this->test_d1199_bs3_mdf(false);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mdf_fsv16) {
    this->test_d1199_bs3_mdf_fsv16(false);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mdf_fsv4) {
    this->test_d1199_bs3_mdf_fsv4(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Test cases for mode "blocks first ".
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(space_to_depth_fp16_gpu, d1122_bs2_mbf_cached) {
    this->test_d1122_bs2_mbf(true);
}

TEST_F(space_to_depth_fp16_gpu, d1142_bs2_mbf_cached) {
    this->test_d1142_bs2_mbf(true);
}

TEST_F(space_to_depth_fp16_gpu, d1264_bs2_mbf_cached) {
    this->test_d1264_bs2_mbf(true);
}

TEST_F(space_to_depth_fp16_gpu, d1199_bs3_mbf_cached) {
    this->test_d1199_bs3_mbf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1122_bs2_mbf_cached) {
    this->test_d1122_bs2_mbf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1142_bs2_mbf_cached) {
    test_d1142_bs2_mbf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1264_bs2_mbf_cached) {
    this->test_d1264_bs2_mbf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mbf_cached) {
    this->test_d1199_bs3_mbf(true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2. Test cases for mode "depth first ".
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(space_to_depth_fp16_gpu, d1122_bs2_mdf_cached) {
    this->test_d1122_bs2_mdf(true);
}

TEST_F(space_to_depth_fp16_gpu, d1142_bs2_mdf_cached) {
    this->test_d1142_bs2_mdf(true);
}

TEST_F(space_to_depth_fp16_gpu, d1264_bs2_mdf_cached) {
    this->test_d1264_bs2_mdf(true);
}

TEST_F(space_to_depth_fp16_gpu, d1199_bs3_mdf_cached) {
    this->test_d1199_bs3_mdf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1122_bs2_mdf_cached) {
    this->test_d1122_bs2_mdf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1142_bs2_mdf_cached) {
    this->test_d1142_bs2_mdf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1264_bs2_mdf_cached) {
    this->test_d1264_bs2_mdf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mdf_cached) {
    this->test_d1199_bs3_mdf(true);
}

TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mdf_fsv16_cached) {
    this->test_d1199_bs3_mdf_fsv16(true);
}
#endif
TEST_F(space_to_depth_fp32_gpu, d1199_bs3_mdf_fsv4_cached) {
    this->test_d1199_bs3_mdf_fsv4(true);
}
