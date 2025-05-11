// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/batch_to_space.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

TEST(batch_to_space_fp16_gpu, i8111_bs1222_cb0000_ce0000) {
    //  Input  :      8x1x1x1
    //  Block shape : 1x2x2x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x2x2x2
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(8), feature(1), spatial(1, 1)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(3.0f),
        ov::float16(4.0f), ov::float16(5.0f),
        ov::float16(6.0f), ov::float16(7.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {1,2,2,2}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i4321_bs1212_cb0000_ce0000) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x6x2x2
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
        ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
        ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
        ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f),
        ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
        ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfyx, {1,2,1,2}, 1),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {1,6,2,2}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 6.f, 1.f, 7.f, 12.f, 18.f,
        13.f, 19.f, 2.f, 8.f, 3.f, 9.f,
        14.f, 20.f, 15.f, 21.f, 4.f, 10.f,
        5.f, 11.f, 16.f, 22.f, 17.f, 23.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i4321_bs1212_cb0010_ce0101) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x1x0
    //  Crops end :   0x1x0x1
    //  Output :      1x5x1x1
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
        ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
        ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
        ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f),
        ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
        ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfyx, {1,2,1,2}, 1),
                                                                       tensor(format::bfyx, {0,0,1,0}, 0),
                                                                       tensor(format::bfyx, {0,1,0,1}, 0),
                                                                       tensor(format::bfyx, {1,5,1,1}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 13.f, 3.f, 15.f, 5.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i62121_bs12311_cb02000_ce00110) {
    //  Input  :      6x2x1x2x1
    //  Block shape : 1x2x3x1x1
    //  Crops begin : 0x2x0x0x0
    //  Crops end :   0x0x1x1x0
    //  Output :      1x2x2x1x1
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(6), feature(2), spatial(1, 2, 1)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfzyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
        ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
        ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
        ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f),
        ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
        ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfzyx, {1,2,3,1,1}, 1),
                                                                       tensor(format::bfzyx, {0,2,0,0,0}, 0),
                                                                       tensor(format::bfzyx, {0,0,1,1,0}, 0),
                                                                       tensor(format::bfzyx, {1,2,2,1,1}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        2.f, 6.f, 14.f, 18.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i1212112_bs112321_cb02000_ce00110) {
    //  Input  :      12x1x2x1x1x2
    //  Block shape : 1x1x2x3x2x1
    //  Crops begin : 0x0x1x0x0x0
    //  Crops end :   0x0x0x2x0x0
    //  Output :      1x1x3x1x2x2
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(12), feature(1), spatial(2, 1, 1, 2)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfwzyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
        ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
        ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
        ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f),
        ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f),
        ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f),
        ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f),
        ov::float16(28.0f), ov::float16(29.0f), ov::float16(30.0f), ov::float16(31.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfwzyx, {1,1,2,3,2,1}, 1),
                                                                       tensor(format::bfwzyx, {0,0,1,0,0,0}, 0),
                                                                       tensor(format::bfwzyx, {0,0,0,2,0,0}, 0),
                                                                       tensor(format::bfwzyx, {1,1,3,1,2,2}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        24.f, 25.f, 28.f, 29.f,
        2.f, 3.f, 6.f, 7.f,
        26.f, 27.f, 30.f, 31.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i21611_bs1112_cb0000_ce0000_b_fs_yx_fsv16) {
    //  Input  :      2x16x1x1
    //  Block shape : 1x1x1x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x16x1x2
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(2), feature(16), spatial(1, 1)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
        ov::float16(8.0f), ov::float16(9.0f),  ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f),
        ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f), ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f),
        ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f), ov::float16(29.0f), ov::float16(30.0f), ov::float16(31.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f16));
    topology.add(batch_to_space("batch_to_space", input_info("input_fsv"), tensor(format::bfyx, {1,1,1,2}, 1),
                                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                                           tensor(format::bfyx, {1,16,1,2}, 1)));
    topology.add(reorder("bts_to_bfyx", input_info("batch_to_space"), format::bfyx, data_types::f16));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("bts_to_bfyx").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 16.f, 1.f, 17.f, 2.f, 18.f, 3.f, 19.f,
        4.f, 20.f, 5.f, 21.f, 6.f, 22.f, 7.f, 23.f,
        8.f, 24.f, 9.f, 25.f, 10.f, 26.f, 11.f, 27.f,
        12.f, 28.f, 13.f, 29.f, 14.f, 30.f, 15.f, 31.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp16_gpu, i2812_bs1112_cb0000_ce0000_b_fs_yx_fsv16) {
    //  Input  :      2x8x1x2
    //  Block shape : 1x1x1x2
    //  Crops begin : 0x2x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x6x1x4
    //  Input values in fp16

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(2), feature(8), spatial(2, 1)};
    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, input_shape });

    set_values(input, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
        ov::float16(8.0f), ov::float16(9.0f),  ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f),
        ov::float16(16.0f), ov::float16(17.0f), ov::float16(18.0f), ov::float16(19.0f), ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f),
        ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f), ov::float16(29.0f), ov::float16(30.0f), ov::float16(31.0f)
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f16));
    topology.add(batch_to_space("batch_to_space", input_info("input_fsv"), tensor(format::bfyx, {1,1,1,2}, 1),
                                                                           tensor(format::bfyx, {0,2,0,0}, 0),
                                                                           tensor(format::bfyx, {0,0,0,0}, 0),
                                                                           tensor(format::bfyx, {1,6,1,4}, 1)));
    topology.add(reorder("bts_to_bfyx", input_info("batch_to_space"), format::bfyx, data_types::f16));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("bts_to_bfyx").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        4.f, 20.f, 5.f, 21.f, 6.f, 22.f, 7.f, 23.f,
        8.f, 24.f, 9.f, 25.f, 10.f, 26.f, 11.f, 27.f,
        12.f, 28.f, 13.f, 29.f, 14.f, 30.f, 15.f, 31.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(batch_to_space_fp32_gpu, i8111_bs1222_cb0000_ce0000) {
    //  Input  :      8x1x1x1
    //  Block shape : 1x2x2x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x2x2x2
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(8), feature(1), spatial(1, 1)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_shape });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {1,2,2,2}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i4321_bs1212_cb0000_ce0000) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x0x0
    //  Crops end :   0x0x0x0
    //  Output :      1x6x2x2
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_shape });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfyx, {1,2,1,2}, 1),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {0,0,0,0}, 0),
                                                                       tensor(format::bfyx, {1,6,2,2}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 6.f, 1.f, 7.f, 12.f, 18.f,
        13.f, 19.f, 2.f, 8.f, 3.f, 9.f,
        14.f, 20.f, 15.f, 21.f, 4.f, 10.f,
        5.f, 11.f, 16.f, 22.f, 17.f, 23.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i4321_bs1212_cb0010_ce0101) {
    //  Input  :      4x3x2x1
    //  Block shape : 1x2x1x2
    //  Crops begin : 0x0x1x0
    //  Crops end :   0x1x0x1
    //  Output :      1x5x1x1
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(4), feature(3), spatial(1, 2)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_shape });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfyx, {1,2,1,2}, 1),
                                                                       tensor(format::bfyx, {0,0,1,0}, 0),
                                                                       tensor(format::bfyx, {0,1,0,1}, 0),
                                                                       tensor(format::bfyx, {1,5,1,1}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 13.f, 3.f, 15.f, 5.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i62121_bs12311_cb02000_ce00110) {
    //  Input  :      6x2x1x2x1
    //  Block shape : 1x2x3x1x1
    //  Crops begin : 0x2x0x0x0
    //  Crops end :   0x0x1x1x0
    //  Output :      1x2x2x1x1
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(6), feature(2), spatial(1, 2, 1)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, input_shape });

    set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfzyx, {1,2,3,1,1}, 1),
                                                                       tensor(format::bfzyx, {0,2,0,0,0}, 0),
                                                                       tensor(format::bfzyx, {0,0,1,1,0}, 0),
                                                                       tensor(format::bfzyx, {1,2,2,1,1}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        2.f, 6.f, 14.f, 18.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i1212112_bs112321_cb02000_ce00110) {
    //  Input  :      12x1x2x1x1x2
    //  Block shape : 1x1x2x3x2x1
    //  Crops begin : 0x0x1x0x0x0
    //  Crops end :   0x0x0x2x0x0
    //  Output :      1x1x3x1x2x2
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(12), feature(1), spatial(2, 1, 1, 2)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, input_shape });

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

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(batch_to_space("batch_to_space", input_info("Input"), tensor(format::bfwzyx, {1,1,2,3,2,1}, 1),
                                                                       tensor(format::bfwzyx, {0,0,1,0,0,0}, 0),
                                                                       tensor(format::bfwzyx, {0,0,0,2,0,0}, 0),
                                                                       tensor(format::bfwzyx, {1,1,3,1,2,2}, 1)));
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("batch_to_space").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        24.f, 25.f, 28.f, 29.f,
        2.f, 3.f, 6.f, 7.f,
        26.f, 27.f, 30.f, 31.f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i21621_bs1112_cb0201_ce0810_b_fs_yx_fsv16) {
    //  Input  :      2x16x2x1
    //  Block shape : 1x1x1x2
    //  Crops begin : 0x2x0x1
    //  Crops end :   0x8x1x0
    //  Output :      1x6x1x1
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(2), feature(16), spatial(1, 2)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_shape });

    set_values(input, {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
        8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
        32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
        40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
        48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
        56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f32));
    topology.add(batch_to_space("batch_to_space", input_info("input_fsv"), tensor(format::bfyx, {1,1,1,2}, 1),
                                                                           tensor(format::bfyx, {0,2,0,1}, 0),
                                                                           tensor(format::bfyx, {0,8,1,0}, 0),
                                                                           tensor(format::bfyx, {1,6,1,1}, 1)));
    topology.add(reorder("bts_to_bfyx", input_info("batch_to_space"), format::bfyx, data_types::f32));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input", input);

    auto outputs = network.execute();

    auto output = outputs.at("bts_to_bfyx").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        36.0f, 38.0f, 40.0f, 42.0f, 44.0f, 46.0f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

template <typename T>
void test_batch_to_space_fp32_gpu_i41021_bs1221_cb0201_ce0810_b_fs_yx_fsv16(bool is_caching_test) {
    //  Input  :      4x10x2x1
    //  Block shape : 1x2x2x1
    //  Crops begin : 0x8x1x0
    //  Crops end :   0x4x0x0
    //  Output :      1x8x3x1
    //  Input values in fp32

    auto& engine = get_test_engine();
    tensor input_shape = tensor{batch(4), feature(10), spatial(1, 2)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, input_shape });

    set_values(input, {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
        20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
        30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
        40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f,
        50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
        60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f,
        70.0f, 71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f
    });

    topology topology;
    topology.add(input_layout("Input", input->get_layout()));
    topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f32));
    topology.add(batch_to_space("batch_to_space", input_info("input_fsv"), tensor(format::bfyx, {1,2,2,1}, 1),
                                                                           tensor(format::bfyx, {0,8,1,0}, 0),
                                                                           tensor(format::bfyx, {0,4,0,0}, 0),
                                                                           tensor(format::bfyx, {1,8,3,1}, 1)));
    topology.add(reorder("bts_to_bfyx", input_info("batch_to_space"), format::bfyx, data_types::f32));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Input", input);

    auto outputs = network->execute();

    auto output = outputs.at("bts_to_bfyx").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
        28.0f, 9.0f,  29.0f, 68.0f, 49.0f, 69.0f,
        30.0f, 11.0f, 31.0f, 70.0f, 51.0f, 71.0f,
        32.0f, 13.0f, 33.0f, 72.0f, 53.0f, 73.0f,
        34.0f, 15.0f, 35.0f, 74.0f, 55.0f, 75.0f
    };

    ASSERT_EQ(output_ptr.size(), expected_results.size());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(batch_to_space_fp32_gpu, i41021_bs1221_cb0201_ce0810_b_fs_yx_fsv16) {
    test_batch_to_space_fp32_gpu_i41021_bs1221_cb0201_ce0810_b_fs_yx_fsv16<float>(false);
}

TEST(export_import_batch_to_space_fp32_gpu, i41021_bs1221_cb0201_ce0810_b_fs_yx_fsv16) {
    test_batch_to_space_fp32_gpu_i41021_bs1221_cb0201_ce0810_b_fs_yx_fsv16<float>(true);
}
