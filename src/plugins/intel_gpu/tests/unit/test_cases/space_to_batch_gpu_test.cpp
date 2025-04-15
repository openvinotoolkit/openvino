// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/space_to_batch.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

class space_to_batch_fp16_gpu: public ::testing::Test {
public:
    void test_i1222_bs1222_pb0000_pe0000(bool is_caching_test) {
        // Input :       1x2x2x2
        // Block shape : 1x2x2x2
        // Pads begin :  0x0x0x0
        // Pads end :    0x0x0x0
        // Output :      8x1x1x1
        // Input values in fp16

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f16, format::bfyx, {1,2,2,2} });

        set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f),
            ov::float16(2.0f), ov::float16(3.0f),
            ov::float16(4.0f), ov::float16(5.0f),
            ov::float16(6.0f), ov::float16(7.0f)
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                        tensor(format::bfyx, {0,0,0,0}, 0),
                                                                        tensor(format::bfyx, {0,0,0,0}, 0),
                                                                        tensor(format::bfyx, {8,1,1,1}, 1)));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_i1242_bs1221_pb0020_pe0000(bool is_caching_test) {
        // Input :       1x2x4x2
        // Block shape : 1x2x2x1
        // Pads begin :  0x0x2x0
        // Pads end :    0x0x0x0
        // Output :      4x1x3x2
        // Input values in fp16

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f16, format::bfyx, {1,2,2,4} });

        set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
            ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
            ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
            ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f)
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfyx, {1,2,2,1}, 1),
                                                                        tensor(format::bfyx, {0,0,2,0}, 0),
                                                                        tensor(format::bfyx, {0,0,0,0}, 0),
                                                                        tensor(format::bfyx, {4,1,3,2}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 0.f, 0.f, 1.f, 4.f, 5.f,
            0.f, 0.f, 2.f, 3.f, 6.f, 7.f,
            0.f, 0.f, 8.f, 9.f, 12.f, 13.f,
            0.f, 0.f, 10.f, 11.f, 14.f, 15.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_i2132_bs1222_pb0010_pe0100(bool is_caching_test) {
        // Input :       2x1x3x2
        // Block shape : 1x2x2x2
        // Pads begin :  0x0x1x0
        // Pads end :    0x1x0x0
        // Output :      16x1x2x1
        // Input values in fp16

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f16, format::bfyx, {2,1,2,3} });

        set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
            ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
            ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f)
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                        tensor(format::bfyx, {0,0,1,0}, 0),
                                                                        tensor(format::bfyx, {0,1,0,0}, 0),
                                                                        tensor(format::bfyx, {16,1,2,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 2.f, 0.f, 8.f, 0.f, 3.f, 0.f, 9.f,
            0.f, 4.f, 6.f, 10.f, 1.f, 5.f, 7.f, 11.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_i12132_bs12122_pb00010_pe00000(bool is_caching_test) {
        // Input :       1x2x1x3x2
        // Block shape : 1x2x1x2x2
        // Pads begin :  0x0x0x1x0
        // Pads end :    0x0x0x0x0
        // Output :      8x1x1x2x1
        // Input values in fp16

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f16, format::bfzyx, {1,2,2,3,1} });

        set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
            ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
            ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f)
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfzyx, {1,2,1,2,2}, 1),
                                                                        tensor(format::bfzyx, {0,0,0,1,0}, 0),
                                                                        tensor(format::bfzyx, {0,0,0,0,0}, 0),
                                                                        tensor(format::bfzyx, {8,1,1,2,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 2.f, 0.f, 3.f, 0.f, 4.f, 1.f, 5.f,
            0.f, 8.f, 0.f, 9.f, 6.f, 10.f, 7.f, 11.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_i134121_bs142121_pb010100_pe000000(bool is_caching_test) {
        // Input :       1x3x4x1x2x1
        // Block shape : 1x4x2x1x2x1
        // Pads begin :  0x1x0x1x0x0
        // Pads end :    0x0x0x0x0x0
        // Output :      16x1x2x2x1x1
        // Input values in fp16

        auto& engine = get_test_engine();
        tensor input_shape = tensor{ batch(1), feature(3), spatial(1, 2, 1, 4) };
        auto input = engine.allocate_memory({ data_types::f16, format::bfwzyx, input_shape });

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
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfwzyx, {1,4,2,1,2,1}, 1),
                                                                        tensor(format::bfwzyx, {0,1,0,1,0,0}, 0),
                                                                        tensor(format::bfwzyx, {0,0,0,0,0,0}, 0),
                                                                        tensor(format::bfwzyx, {16,1,2,2,1,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

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
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_i11611_bs1222_pb0010_pe0001_b_fs_yx_fsv16(bool is_caching_test) {
        // Input :       1x16x1x1
        // Block shape : 1x2x2x2
        // Pads begin :  0x0x1x0
        // Pads end :    0x0x0x1
        // Output :      8x8x1x1
        // Input values in fp16

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f16, format::bfyx, {1,16,1,1} });

        set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
            ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
            ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
            ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f)
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f16));
        topology.add(space_to_batch("space_to_batch", input_info("input_fsv"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                            tensor(format::bfyx, {0,0,1,0}, 0),
                                                                            tensor(format::bfyx, {0,0,0,1}, 0),
                                                                            tensor(format::bfyx, {8,8,1,1}, 1)));
        topology.add(reorder("stb_to_bfyx", input_info("space_to_batch"), format::bfyx, data_types::f16));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("stb_to_bfyx").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            1.f, 3.f, 5.f, 7.f, 9.f, 11.f, 13.f, 15.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }

    void test_i1812_bs1221_pb0010_pe0200_b_fs_yx_fsv16(bool is_caching_test) {
        // Input :       1x8x1x2
        // Block shape : 1x2x2x1
        // Pads begin :  0x0x1x0
        // Pads end :    0x2x0x0
        // Output :      4x5x1x2
        // Input values in fp16

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f16, format::bfyx, {1,8,2,1} });

        set_values(input, {
            ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
            ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f),
            ov::float16(8.0f), ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f),
            ov::float16(12.0f), ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f)
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f16));
        topology.add(space_to_batch("space_to_batch", input_info("input_fsv"), tensor(format::bfyx, {1,2,2,1}, 1),
                                                                            tensor(format::bfyx, {0,0,1,0}, 0),
                                                                            tensor(format::bfyx, {0,2,0,0}, 0),
                                                                            tensor(format::bfyx, {4,5,1,2}, 1)));
        topology.add(reorder("stb_to_bfyx", input_info("space_to_batch"), format::bfyx, data_types::f16));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("stb_to_bfyx").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 4.f, 5.f, 8.f, 9.f, 12.f, 13.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            2.f, 3.f, 6.f, 7.f, 10.f, 11.f, 14.f, 15.f, 0.f, 0.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
        }
    }
};

TEST_F(space_to_batch_fp16_gpu, i1222_bs1222_pb0000_pe0000) {
    this->test_i1222_bs1222_pb0000_pe0000(false);
}

TEST_F(space_to_batch_fp16_gpu, i1242_bs1221_pb0020_pe0000) {
    this->test_i1242_bs1221_pb0020_pe0000(false);
}

TEST_F(space_to_batch_fp16_gpu, i2132_bs1222_pb0010_pe0100) {
    this->test_i2132_bs1222_pb0010_pe0100(false);
}

TEST_F(space_to_batch_fp16_gpu, i12132_bs12122_pb00010_pe00000) {
    this->test_i12132_bs12122_pb00010_pe00000(false);
}

TEST_F(space_to_batch_fp16_gpu, i134121_bs142121_pb010100_pe000000) {
    this->test_i134121_bs142121_pb010100_pe000000(false);
}

TEST_F(space_to_batch_fp16_gpu, i11611_bs1222_pb0010_pe0001_b_fs_yx_fsv16) {
    this->test_i11611_bs1222_pb0010_pe0001_b_fs_yx_fsv16(false);
}

TEST_F(space_to_batch_fp16_gpu, i1812_bs1221_pb0010_pe0200_b_fs_yx_fsv16) {
    this->test_i1812_bs1221_pb0010_pe0200_b_fs_yx_fsv16(false);
}

class space_to_batch_fp32_gpu: public ::testing::Test {
public:
    void test_i1222_bs1222_pb0000_pe0000(bool is_caching_test) {
        // Input :       1x2x2x2
        // Block shape : 1x2x2x2
        // Pads begin :  0x0x0x0
        // Pads end :    0x0x0x0
        // Output :      8x1x1x1
        // Input values in fp32

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1,2,2,2} });

        set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                        tensor(format::bfyx, {0,0,0,0}, 0),
                                                                        tensor(format::bfyx, {0,0,0,0}, 0),
                                                                        tensor(format::bfyx, {8,1,1,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_i1242_bs1221_pb0020_pe0000(bool is_caching_test) {
        // Input :       1x2x4x2
        // Block shape : 1x2x2x1
        // Pads begin :  0x0x2x0
        // Pads end :    0x0x0x0
        // Output :      4x1x3x2
        // Input values in fp32

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1,2,2,4} });

        set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f, 15.0f
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfyx, {1,2,2,1}, 1),
                                                                        tensor(format::bfyx, {0,0,2,0}, 0),
                                                                        tensor(format::bfyx, {0,0,0,0}, 0),
                                                                        tensor(format::bfyx, {4,1,3,2}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 0.f, 0.f, 1.f, 4.f, 5.f,
            0.f, 0.f, 2.f, 3.f, 6.f, 7.f,
            0.f, 0.f, 8.f, 9.f, 12.f, 13.f,
            0.f, 0.f, 10.f, 11.f, 14.f, 15.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_i2132_bs1222_pb0010_pe0100(bool is_caching_test) {
        // Input :       2x1x3x2
        // Block shape : 1x2x2x2
        // Pads begin :  0x0x1x0
        // Pads end :    0x1x0x0
        // Output :      16x1x2x1
        // Input values in fp32

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {2,1,2,3} });

        set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfyx, {1,2,2,2}, 1),
                                                                        tensor(format::bfyx, {0,0,1,0}, 0),
                                                                        tensor(format::bfyx, {0,1,0,0}, 0),
                                                                        tensor(format::bfyx, {16,1,2,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 2.f, 0.f, 8.f, 0.f, 3.f, 0.f, 9.f,
            0.f, 4.f, 6.f, 10.f, 1.f, 5.f, 7.f, 11.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_i12132_bs12122_pb00010_pe00000(bool is_caching_test) {
        // Input :       1x2x1x3x2
        // Block shape : 1x2x1x2x2
        // Pads begin :  0x0x0x1x0
        // Pads end :    0x0x0x0x0
        // Output :      8x1x1x2x1
        // Input values in fp32

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, {1,2,2,3,1} });

        set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfzyx, {1,2,1,2,2}, 1),
                                                                        tensor(format::bfzyx, {0,0,0,1,0}, 0),
                                                                        tensor(format::bfzyx, {0,0,0,0,0}, 0),
                                                                        tensor(format::bfzyx, {8,1,1,2,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 2.f, 0.f, 3.f, 0.f, 4.f, 1.f, 5.f,
            0.f, 8.f, 0.f, 9.f, 6.f, 10.f, 7.f, 11.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_i134121_bs142121_pb010100_pe000000(bool is_caching_test) {
        // Input :       1x3x4x1x2x1
        // Block shape : 1x4x2x1x2x1
        // Pads begin :  0x1x0x1x0x0
        // Pads end :    0x0x0x0x0x0
        // Output :      16x1x2x2x1x1
        // Input values in fp32

        auto& engine = get_test_engine();
        tensor input_shape = tensor{ batch(1), feature(3), spatial(1, 2, 1, 4) };
        auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, input_shape });

        set_values(input, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
        18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(space_to_batch("space_to_batch", input_info("Input"), tensor(format::bfwzyx, {1,4,2,1,2,1}, 1),
                                                                        tensor(format::bfwzyx, {0,1,0,1,0,0}, 0),
                                                                        tensor(format::bfwzyx, {0,0,0,0,0,0}, 0),
                                                                        tensor(format::bfwzyx, {16,1,2,2,1,1}, 1)));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("space_to_batch").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_i11622_bs1421_pb0000_pe0000_b_fs_yx_fsv16(bool is_caching_test) {
        // Input :       1x16x2x2
        // Block shape : 1x4x2x1
        // Pads begin :  0x0x0x0
        // Pads end :    0x0x0x0
        // Output :      8x4x1x2
        // Input values in fp32

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1,16,2,2} });

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
        topology.add(space_to_batch("space_to_batch", input_info("input_fsv"), tensor(format::bfyx, {1,4,2,1}, 1),
                                                                            tensor(format::bfyx, {0,0,0,0}, 0),
                                                                            tensor(format::bfyx, {0,0,0,0}, 0),
                                                                            tensor(format::bfyx, {8,4,1,2}, 1)));
        topology.add(reorder("stb_to_bfyx", input_info("space_to_batch"), format::bfyx, data_types::f32));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("stb_to_bfyx").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 1.f, 16.f, 17.f, 32.f, 33.f, 48.f, 49.f,
            2.f, 3.f, 18.f, 19.f, 34.f, 35.f, 50.f, 51.f,
            4.f, 5.f, 20.f, 21.f, 36.f, 37.f, 52.f, 53.f,
            6.f, 7.f, 22.f, 23.f, 38.f, 39.f, 54.f, 55.f,
            8.f, 9.f, 24.f, 25.f, 40.f, 41.f, 56.f, 57.f,
            10.f, 11.f, 26.f, 27.f, 42.f, 43.f, 58.f, 59.f,
            12.f, 13.f, 28.f, 29.f, 44.f, 45.f, 60.f, 61.f,
            14.f, 15.f, 30.f, 31.f, 46.f, 47.f, 62.f, 63.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }

    void test_i1623_bs1312_pb0001_pe0000_b_fs_yx_fsv16(bool is_caching_test) {
        // Input :       1x6x2x3
        // Block shape : 1x3x1x2
        // Pads begin :  0x0x0x1
        // Pads end :    0x0x0x0
        // Output :      6x2x2x2
        // Input values in fp32

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1,6,3,2} });

        set_values(input, {
            0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
            8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
            32.0f, 33.0f, 34.0f, 35.0f
        });

        topology topology;
        topology.add(input_layout("Input", input->get_layout()));
        topology.add(reorder("input_fsv", input_info("Input"), format::b_fs_yx_fsv16, data_types::f32));
        topology.add(space_to_batch("space_to_batch", input_info("input_fsv"), tensor(format::bfyx, {1,3,1,2}, 1),
                                                                            tensor(format::bfyx, {0,0,0,1}, 0),
                                                                            tensor(format::bfyx, {0,0,0,0}, 0),
                                                                            tensor(format::bfyx, {6,2,2,2}, 1)));
        topology.add(reorder("stb_to_bfyx", input_info("space_to_batch"), format::bfyx, data_types::f32));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input", input);

        auto outputs = network->execute();

        auto output = outputs.at("stb_to_bfyx").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
            0.f, 1.f, 0.f, 4.f, 0.f, 19.f, 0.f, 22.f,
            0.f, 2.f, 3.f, 5.f, 18.f, 20.f, 21.f, 23.f,
            0.f, 7.f, 0.f, 10.f, 0.f, 25.f, 0.f, 28.f,
            6.f, 8.f, 9.f, 11.f, 24.f, 26.f, 27.f, 29.f,
            0.f, 13.f, 0.f, 16.f, 0.f, 31.f, 0.f, 34.f,
            12.f, 14.f, 15.f, 17.f, 30.f, 32.f, 33.f, 35.f
        };

        ASSERT_EQ(output_ptr.size(), expected_results.size());

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i]);
        }
    }
};

TEST_F(space_to_batch_fp32_gpu, i1222_bs1222_pb0000_pe0000) {
    this->test_i1222_bs1222_pb0000_pe0000(false);
}

TEST_F(space_to_batch_fp32_gpu, i1242_bs1221_pb0020_pe0000) {
    this->test_i1242_bs1221_pb0020_pe0000(false);
}

TEST_F(space_to_batch_fp32_gpu, i2132_bs1222_pb0010_pe0100) {
    this->test_i2132_bs1222_pb0010_pe0100(false);
}

TEST_F(space_to_batch_fp32_gpu, i12132_bs12122_pb00010_pe00000) {
    this->test_i12132_bs12122_pb00010_pe00000(false);
}

TEST_F(space_to_batch_fp32_gpu, i134121_bs142121_pb010100_pe000000) {
    this->test_i134121_bs142121_pb010100_pe000000(false);
}

TEST_F(space_to_batch_fp32_gpu, i11622_bs1421_pb0000_pe0000_b_fs_yx_fsv16) {
    this->test_i11622_bs1421_pb0000_pe0000_b_fs_yx_fsv16(false);
}

TEST_F(space_to_batch_fp32_gpu, i1623_bs1312_pb0001_pe0000_b_fs_yx_fsv16) {
    this->test_i1623_bs1312_pb0001_pe0000_b_fs_yx_fsv16(false);
}

TEST_F(space_to_batch_fp16_gpu, i1222_bs1222_pb0000_pe0000_cached) {
    this->test_i1222_bs1222_pb0000_pe0000(true);
}
#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(space_to_batch_fp16_gpu, i1242_bs1221_pb0020_pe0000_cached) {
    this->test_i1242_bs1221_pb0020_pe0000(true);
}

TEST_F(space_to_batch_fp16_gpu, i2132_bs1222_pb0010_pe0100_cached) {
    this->test_i2132_bs1222_pb0010_pe0100(true);
}

TEST_F(space_to_batch_fp16_gpu, i12132_bs12122_pb00010_pe00000_cached) {
    this->test_i12132_bs12122_pb00010_pe00000(true);
}

TEST_F(space_to_batch_fp16_gpu, i134121_bs142121_pb010100_pe000000_cached) {
    this->test_i134121_bs142121_pb010100_pe000000(true);
}

TEST_F(space_to_batch_fp16_gpu, i11611_bs1222_pb0010_pe0001_b_fs_yx_fsv16_cached) {
    this->test_i11611_bs1222_pb0010_pe0001_b_fs_yx_fsv16(true);
}

TEST_F(space_to_batch_fp16_gpu, i1812_bs1221_pb0010_pe0200_b_fs_yx_fsv16_cached) {
    this->test_i1812_bs1221_pb0010_pe0200_b_fs_yx_fsv16(true);
}

TEST_F(space_to_batch_fp32_gpu, i1222_bs1222_pb0000_pe0000_cached) {
    this->test_i1222_bs1222_pb0000_pe0000(true);
}

TEST_F(space_to_batch_fp32_gpu, i1242_bs1221_pb0020_pe0000_cached) {
    this->test_i1242_bs1221_pb0020_pe0000(true);
}

TEST_F(space_to_batch_fp32_gpu, i2132_bs1222_pb0010_pe0100_cached) {
    this->test_i2132_bs1222_pb0010_pe0100(true);
}

TEST_F(space_to_batch_fp32_gpu, i12132_bs12122_pb00010_pe00000_cached) {
    this->test_i12132_bs12122_pb00010_pe00000(true);
}

TEST_F(space_to_batch_fp32_gpu, i134121_bs142121_pb010100_pe000000_cached) {
    this->test_i134121_bs142121_pb010100_pe000000(true);
}

TEST_F(space_to_batch_fp32_gpu, i11622_bs1421_pb0000_pe0000_b_fs_yx_fsv16_cached) {
    this->test_i11622_bs1421_pb0000_pe0000_b_fs_yx_fsv16(true);
}

TEST_F(space_to_batch_fp32_gpu, i1623_bs1312_pb0001_pe0000_b_fs_yx_fsv16_cached) {
    this->test_i1623_bs1312_pb0001_pe0000_b_fs_yx_fsv16(true);
}
#endif
