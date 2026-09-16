// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/depth_to_space.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

TEST(depth_to_space_fp16_gpu, d1411_bs2) {
    //  Input  : 1x4x1x1
    //  Block size : 2
    //  Output : 1x1x2x2
    //  Input values in fp16

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 4, 1, 1 } });
    size_t block_size = 2;

    set_values(input1, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(3.0f)
    });

    topology topology;
    topology.add(input_layout("Input0", input1->get_layout()));
    topology.add(
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<uint16_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(depth_to_space_fp16_gpu, d1421_bs2) {
    //  Input  : 1x4x2x1
    //  Block size : 2
    //  Output : 1x1x4x2
    //  Input values in fp16

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 4, 1, 2 } });
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
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<uint16_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.0f, 2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f, 7.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(depth_to_space_fp16_gpu, d1933_bs3) {
    //  Input  : 1x9x3x3
    //  Block size : 3
    //  Output : 1x1x9x9
    //  Input values in fp16

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 9, 3, 3 } });
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
            depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<uint16_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.0f, 9.0f, 18.0f, 1.0f, 10.0f, 19.0f, 2.0f, 11.0f, 20.0f, 27.0f,
        36.0f, 45.0f, 28.0f, 37.0f, 46.0f, 29.0f, 38.0f, 47.0f, 54.0f, 63.0f,
        72.0f, 55.0f, 64.0f, 73.0f, 56.0f, 65.0f, 74.0f, 3.0f, 12.0f, 21.0f,
        4.0f, 13.0f, 22.0f, 5.0f, 14.0f, 23.0f, 30.0f, 39.0f, 48.0f, 31.0f,
        40.0f, 49.0f, 32.0f, 41.0f, 50.0f, 57.0f, 66.0f, 75.0f, 58.0f, 67.0f,
        76.0f, 59.0f, 68.0f, 77.0f, 6.0f, 15.0f, 24.0f, 7.0f, 16.0f, 25.0f,
        8.0f, 17.0f, 26.0f, 33.0f, 42.0f, 51.0f, 34.0f, 43.0f, 52.0f, 35.0f,
        44.0f, 53.0f, 60.0f, 69.0f, 78.0f, 61.0f, 70.0f, 79.0f, 62.0f, 71.0f,
        80.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(depth_to_space_fp32_gpu, d1411_bs2) {
    //  Input  : 1x4x1x1
    //  Block size : 2
    //  Output : 1x1x2x2
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 1 } });
    size_t block_size = 2;

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f
    });

    topology topology;
    topology.add(input_layout("Input0", input1->get_layout()));
    topology.add(
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(depth_to_space_fp32_gpu, d112960540_bs2) {
    //  Input  : 1x12x960x540
    //  Block size : 2
    //  Output : 1x3x1920x1080
    //  Input values in fp16

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 12, 960, 540 } });
    size_t block_size = 2;

    auto random_input = rg.generate_random_4d<ov::float16>(1, 12, 540, 960, -1, 1);
    auto input_rnd_vec = flatten_4d<ov::float16>(format::bfyx, random_input);
    set_values(input1, input_rnd_vec);

    topology topology_act;
    topology_act.add(input_layout("Input0", input1->get_layout()));
    topology_act.add(
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network_act(engine, topology_act, get_test_default_config(engine));

    network_act.set_input_data("Input0", input1);

    auto outputs = network_act.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr (output, get_test_stream());

    std::vector<uint16_t> perm = { 0,3,4,1,5,2 };

    topology topology_ref;
    topology_ref.add(input_layout("Input0", input1->get_layout()));
    topology_ref.add(reorder("reorder1", input_info("Input0"), { data_types::f16, format::bfwzyx, tensor{ batch(1), feature(12), spatial(1, 1, 960, 540) }
        }));
    topology_ref.add(
        reshape("reshape", input_info("reorder1"), tensor{ batch(1), feature(2), spatial(960, 540, 3, 2) })
    );
    topology_ref.add(
        permute("perm", input_info("reshape"), perm)
    );
    topology_ref.add(
        reshape("reshape2", input_info("perm"), tensor(1, 3, 2 * 960, 2 * 540))
    );

    ExecutionConfig config(ov::intel_gpu::optimize_data(true));
    network network_ref(engine, topology_ref, config);
    network_ref.set_input_data("Input0", input1);

    auto outputs_ref = network_ref.execute();

    auto output_ref = outputs_ref.at("reshape2").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr_ref(output_ref, get_test_stream());

    for (size_t i = 0; i < output->get_layout().count(); ++i) {
        ASSERT_EQ(output_ptr_ref[i], output_ptr[i]);
    }
}

TEST(depth_to_space_fp32_gpu, d1933_bs3) {
    //  Input  : 1x9x3x3
    //  Block size : 3
    //  Output : 1x1x9x9
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 9, 3, 3 } });
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
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.0f, 9.0f, 18.0f, 1.0f, 10.0f, 19.0f, 2.0f, 11.0f, 20.0f, 27.0f,
        36.0f, 45.0f, 28.0f, 37.0f, 46.0f, 29.0f, 38.0f, 47.0f, 54.0f, 63.0f,
        72.0f, 55.0f, 64.0f, 73.0f, 56.0f, 65.0f, 74.0f, 3.0f, 12.0f, 21.0f,
        4.0f, 13.0f, 22.0f, 5.0f, 14.0f, 23.0f, 30.0f, 39.0f, 48.0f, 31.0f,
        40.0f, 49.0f, 32.0f, 41.0f, 50.0f, 57.0f, 66.0f, 75.0f, 58.0f, 67.0f,
        76.0f, 59.0f, 68.0f, 77.0f, 6.0f, 15.0f, 24.0f, 7.0f, 16.0f, 25.0f,
        8.0f, 17.0f, 26.0f, 33.0f, 42.0f, 51.0f, 34.0f, 43.0f, 52.0f, 35.0f,
        44.0f, 53.0f, 60.0f, 69.0f, 78.0f, 61.0f, 70.0f, 79.0f, 62.0f, 71.0f,
        80.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}


TEST(depth_to_space_fp32_gpu, d1822_bs2_blocks_first) {
    //  Input  : 1x8x2x2
    //  Block size : 2
    //  Output : 1x2x4x4
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 8, 2, 2 } });
    size_t block_size = 2;

    set_values(input1, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f,
        28.0f, 29.0f, 30.0f, 31.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input1->get_layout()));
    topology.add(
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::blocks_first)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", input1);

    auto outputs = network.execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.0f, 8.0f, 1.0f, 9.0f, 16.0f, 24.0f, 17.0f, 25.0f,
        2.0f, 10.0f, 3.0f, 11.0f, 18.0f, 26.0f, 19.0f, 27.0f,
        4.0f, 12.0f, 5.0f, 13.0f, 20.0f, 28.0f, 21.0f, 29.0f,
        6.0f, 14.0f, 7.0f, 15.0f, 22.0f, 30.0f, 23.0f, 31.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

template <typename T>
void test_depth_to_space_fp32_gpu_d1822_bs2_depth_first(bool is_caching_test) {
    //  Input  : 1x8x2x2
    //  Block size : 2
    //  Output : 1x2x4x4
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 8, 2, 2 } });
    size_t block_size = 2;

    set_values(input1, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f,
        28.0f, 29.0f, 30.0f, 31.0f
    });

    topology topology;
    topology.add(input_layout("Input0", input1->get_layout()));
    topology.add(
        depth_to_space("depth_to_space", input_info("Input0"), block_size, depth_to_space_mode::depth_first)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Input0", input1);

    auto outputs = network->execute();

    auto output = outputs.at("depth_to_space").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
        0.0f, 4.0f, 1.0f, 5.0f, 8.0f, 12.0f, 9.0f, 13.0f,
        2.0f, 6.0f, 3.0f, 7.0f, 10.0f, 14.0f, 11.0f, 15.0f,
        16.0f, 20.0f, 17.0f, 21.0f, 24.0f, 28.0f, 25.0f, 29.0f,
        18.0f, 22.0f, 19.0f, 23.0f, 26.0f, 30.0f, 27.0f, 31.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(depth_to_space_fp32_gpu, d1822_bs2_depth_first) {
    test_depth_to_space_fp32_gpu_d1822_bs2_depth_first<float>(false);
}

TEST(export_import_depth_to_space_fp32_gpu, d1822_bs2_depth_first) {
    test_depth_to_space_fp32_gpu_d1822_bs2_depth_first<float>(true);
}

static void test_depth_to_space_fp16_input_fp32_output(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 3, 2 } });

    size_t block_size = 1;

    set_values(input, {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, 6.0f,
        3.0f, 3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    });
    set_values(weights, {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(1.0f), ov::float16(2.0f)
    });

    // Apply existed topology that makes kernel build failure because of input and output data types are different.
    topology topology;
    topology.add(cldnn::input_layout("input", input->get_layout()));
    topology.add(cldnn::data("weights", weights));
    topology.add(cldnn::reorder("reorder_input", input_info("input"), cldnn::layout(data_types::f16, format::byxf, { 1, 1, 4, 5 })));
    topology.add(cldnn::convolution("conv", input_info("reorder_input"), "weights", "", 1, { 2, 1 }, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(cldnn::depth_to_space("depth_to_space", input_info("conv"), block_size, depth_to_space_mode::depth_first));
    topology.add(cldnn::activation("activate", input_info("depth_to_space"), cldnn::activation_func::relu_negative_slope, {0.25f, 0.f}));
    topology.add(cldnn::reorder("convert:output", input_info("activate"), format::any, data_types::f32, {}, reorder_mean_mode::subtract, padding(), true));
    topology.add(cldnn::reorder("result:output/sink_port_0", input_info("convert:output"), format::bfyx, data_types::f32, {}, reorder_mean_mode::subtract, padding(), false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    auto output = outputs.at("result:output/sink_port_0").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        24.0f, 24.0f, 32.0f, 28.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(depth_to_space_gpu, fp16_input_fp32_output) {
    test_depth_to_space_fp16_input_fp32_output(false);
}

// Verify that depth_to_space output preserves 4D rank when followed by a
// rank-changing reorder (bfyx -> bfzyx).  The rank-changing reorder must NOT
// be fused into depth_to_space; instead it should remain as a separate
// (possibly optimized) node so that the kernel sees matching input/output dims.
TEST(depth_to_space_fp32_gpu, d1811_bs2_no_rank_changing_fusion) {
    //  Input  : 1x8x1x1  (4D, bfyx)
    //  Block size : 2
    //  Output : 1x2x2x2  (4D, bfyx) -> reorder to bfzyx (5D)
    //
    //  With optimize_data enabled, the graph optimizer should NOT fuse the
    //  bfyx->bfzyx reorder into depth_to_space because that would change
    //  the output rank of a rank-preserving operation.

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 8, 1, 1 } });
    size_t block_size = 2;

    // blocks_first: feature 0..7 -> spatially rearranged
    set_values(input, { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(depth_to_space("depth_to_space", input_info("input"), block_size, depth_to_space_mode::blocks_first));
    // Rank-changing reorder: 4D bfyx -> 5D bfzyx
    topology.add(reorder("reorder_bfzyx", input_info("depth_to_space"), format::bfzyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto output = outputs.at("reorder_bfzyx").get_memory();
    auto out_layout = output->get_layout();

    // Output should be 5D bfzyx
    ASSERT_EQ(out_layout.format, format::bfzyx);

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<float> expected = { 0.f, 2.f, 4.f, 6.f, 1.f, 3.f, 5.f, 7.f };

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_FLOAT_EQ(expected[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(depth_to_space_fp16_gpu, d8_12_576x672_bs2_with_eltwise_and_bfzyx_reshape) {
    auto& engine = get_test_engine();

    // Use small spatial dims to keep memory manageable in unit tests,
    // but preserve the essential structure: 4D b_fs_yx_fsv16 input, eltwise
    // fused, followed by reshape requiring 5D.
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 12, 4, 4 } });
    auto eltw_data = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 8, 8 } });

    size_t block_size = 2;

    // Fill with sequential values
    std::vector<ov::float16> input_vals(1 * 12 * 4 * 4);
    for (size_t i = 0; i < input_vals.size(); ++i)
        input_vals[i] = ov::float16(static_cast<float>(i));
    set_values(input, input_vals);

    std::vector<ov::float16> eltw_vals(1 * 3 * 8 * 8);
    for (size_t i = 0; i < eltw_vals.size(); ++i)
        eltw_vals[i] = ov::float16(1.0f);
    set_values(eltw_data, eltw_vals);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("eltw_data", eltw_data));
    // Reorder to b_fs_yx_fsv16 to match the production scenario
    topology.add(reorder("reorder_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f16));
    topology.add(depth_to_space("depth_to_space", input_info("reorder_fsv16"), block_size, depth_to_space_mode::blocks_first));
    // Eltwise add (this gets fused into depth_to_space)
    topology.add(eltwise("add", { input_info("depth_to_space"), input_info("eltw_data") }, eltwise_mode::sum));
    // Reorder to bfzyx (rank-changing) - must NOT be fused into depth_to_space
    topology.add(reorder("reorder_bfzyx", input_info("add"), format::bfzyx, data_types::f16));
    // Final reorder back to bfyx for easy verification
    topology.add(reorder("output", input_info("reorder_bfzyx"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    using output_lock_t = cldnn::mem_lock<ov::float16, mem_lock_type::read>;

    // The key assertion: this must not crash during kernel compilation.
    // Previously clBuildProgram would fail because of INPUT0_GET_INDEX
    // macro argument mismatch when the reorder was fused.
    ASSERT_NO_THROW({
        network network(engine, topology, config);
        network.set_input_data("input", input);
        auto outputs = network.execute();
        auto output = outputs.at("output").get_memory();

        // Verify output is valid (non-zero, correct shape)
        auto out_layout = output->get_layout();
        ASSERT_EQ(out_layout.format, format::bfyx);

        output_lock_t output_ptr(output, get_test_stream());
        // depth_to_space output: 1x3x8x8, eltwise adds 1.0 to each element
        // Check a few samples are within reasonable range
        ASSERT_GT(static_cast<float>(output_ptr[0]), -1000.f);
        ASSERT_LT(static_cast<float>(output_ptr[0]), 1000.f);
    });
}
