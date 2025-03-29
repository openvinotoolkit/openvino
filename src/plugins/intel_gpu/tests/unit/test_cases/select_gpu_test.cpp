// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/select.hpp>
#include "select_inst.h"

using namespace cldnn;
using namespace ::tests;

// select_gpu_f32
template <typename T>
void test_select_basic(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
         5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        0.f,   0.f,  0.f,  0.f,
        1.f,   1.f,  1.f,  1.f,
        0.f,   1.f,  0.f,  1.f,
        1.f,   0.f,  1.f,  0.f });

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("input2", input2);
    network->set_input_data("mask", mask);
    auto outputs = network->execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {  0.5f,  2.5f,   0.5f,  2.5f,
                           2.f,   0.f,    6.f,   5.2f,
                          15.f,   0.5f,   8.f,  12.f,
                           4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic) {
    test_select_basic<float>(false);
}

TEST(select_gpu_f32, select_basic_negative) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        -0.f,   -0.f,  -0.f,  -0.f,
        -1.f,   -1.f,  -1.f,  -1.f,
        -0.f,   -1.f,  -0.f,  -1.f,
        -1.f,   -0.f,  -1.f,  -0.f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_mask_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 1, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f,

        3.f,  0.5f,
        7.f,  12.f,

        4.f,  -0.5f,
        8.f,  8.f
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f,
        0.f,

        1.f,
        1.f,

        0.f,
        1.f,

        1.f,
        0.f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  3.f,

        2.f,   0.f,
        6.f,   5.2f,

        15.f,  17.f,
        7.f,   12.f,

        4.f,   -0.5f,
        -0.5f, -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_mask_1x1x1x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f,

        3.f,  0.5f,
        7.f,  12.f,

        4.f,  -0.5f,
        8.f,  8.f
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma_byxf_2x2x2x2_bcast_mask_2x2x2x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf ,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 1 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,   0.f,
        5.f,   1.5f,

        2.f,   0.f,
        6.f,   5.2f,

        3.f,   0.5f,
        7.f,   12.f,

        4.f,   -0.5f,
        8.f,   8.f
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.1f,  0.0f,
        0.5f,  0.0f,

        -0.f,  -0.1f,
        -0.f,  -0.5f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        1.f,  2.5f,
        5.f,  3.f,

        2.f,  7.f,
        6.f,  4.f,

        15.f, 0.5f,
        8.f,  12.f,

        -2.f, -0.5f,
        -0.5f, 8.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_in2_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f,

        3.f,  0.5f,
        7.f,  12.f,

        4.f,  -0.5f,
        8.f,  8.f
    });

    set_values(input2, {
        0.5f,
        1.5f,

        5.f,
        2.f,

        15.f,
        8.f,

        -2.f,
        -0.5f,
    });

    set_values(mask, {
        0.f,  0.f,
        0.f,  0.f,

        1.f,  1.f,
        1.f,  1.f,

        0.f,  1.f,
        0.f,  1.f,

        1.f,  0.f,
        1.f,  0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  0.5f,
        1.5f,  1.5f,

        2.f,   0.f,
        6.f,   5.2f,

        15.f,  0.5f,
        8.f,   12.f,

        4.f,   -2.f,
        8.f,   -0.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_in1_2x2x2x1_bcast_in2_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,

        2.f,  0.f,

        3.f,  0.5f,

        4.f,  -0.5f,
    });

    set_values(input2, {
        0.5f,
        1.5f,

        5.f,
        2.f,

        15.f,
        8.f,

        -2.f,
        -0.5f,
    });

    set_values(mask, {
        0.f,  0.f,
        0.f,  0.f,

        1.f,  1.f,
        1.f,  1.f,

        0.f,  1.f,
        0.f,  1.f,

        1.f,  0.f,
        1.f,  0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  0.5f,
        1.5f,  1.5f,

        2.f,   0.f,
        2.f,   0.f,

        15.f,  0.5f,
        8.f,   0.5f,

        4.f,   -2.f,
        4.f,   -0.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_mask_2x1x2x2_in1_1x2x2x2_in2_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f
    });

    set_values(input2, {
        0.5f,
        1.5f,

        5.f,
        2.f,

        15.f,
        8.f,

        -2.f,
        -0.5f,
    });

    set_values(mask, {
        1.f,  0.f,
        1.f,  0.f,

        0.f,  1.f,
        0.f,  1.f,
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        1.f,   0.5f,
        5.f,   1.5f,

        2.f,   5.f,
        6.f,   2.f,

        15.f,  0.f,
        8.f,   1.5f,

        -2.f,  0.f,
        -0.5f, 5.2f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma_byxf_2x2x2x2_bcast_mask_2x1x2x2_in1_2x2x2x1_in2_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf ,{ 2, 2, 1, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        3.f,  0.5f,
        7.f,  12.f,
    });

    set_values(input2, {
        0.5f,  2.5f,

        5.f,   7.f,

        15.f,  17.f,

        -2.f,  6.5f,
    });

    set_values(mask, {
        0.f,
        0.f,

        0.1f,
        0.5f,

        -0.f,
        -0.5f,

        -0.7f,
        -0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        0.5f,  2.5f,

        1.f,   0.f,
        5.f,   1.5f,

        15.f,  17.f,
        7.f,   12.f,

        3.f,   0.5f,
        -2.f,  6.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_in2_1x1x1x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f,

        3.f,  0.5f,
        7.f,  12.f,

        4.f,  -0.5f,
        8.f,  8.f
    });

    set_values(input2, {
        1.f
    });

    set_values(mask, {
        0.f,  0.f,
        0.f,  0.f,

        1.f,  1.f,
        1.f,  1.f,

        0.f,  1.f,
        0.f,  1.f,

        1.f,  0.f,
        1.f,  0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        1.f,   1.f,
        1.f,   1.f,

        2.f,   0.f,
        6.f,   5.2f,

        1.f,   0.5f,
        1.f,   12.f,

        4.f,   1.f,
        8.f,   1.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma_byxf_2x2x2x2_bcast_in2_2x2x2x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf ,{ 2, 2, 2, 1 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,   0.f,
        5.f,   1.5f,

        2.f,   0.f,
        6.f,   5.2f,

        3.f,   0.5f,
        7.f,   12.f,

        4.f,   -0.5f,
        8.f,   8.f
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        15.f,  17.f,
        8.f,   10.f,
    });

    set_values(mask, {
        0.1f,  0.3f,
        0.5f,  0.7f,

        0.f,   0.f,
        0.f,   0.f,

        -0.f,  -0.1f,
        -0.f,  -0.5f,

        -0.7f, -0.f,
        -1.5f, -0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        1.f,  0.f,
        5.f,  1.5f,

        0.5f, 2.5f,
        1.5f, 3.f,

        15.f, 0.5f,
        8.f,  12.f,

        4.f,  17.0f,
        8.f,  10.0f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_in1_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,
        5.f,

        2.f,
        6.f,

        3.f,
        7.f,

        4.f,
        8.f,
    });

    set_values(input2, {
        0.5f, 2.5f,
        1.5f, 1.f,

        5.f,  7.f,
        2.f,  4.f,

        15.f, 17.f,
        8.f,  10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f,  0.f,
        0.f,  0.f,

        1.f,  1.f,
        1.f,  1.f,

        0.f,  1.f,
        0.f,  1.f,

        1.f,  0.f,
        1.f,  0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  1.f,

        2.f,   2.f,
        6.f,   6.f,

        15.f,  3.f,
        8.f,   7.f,

        4.f,   6.5f,
        8.f,   -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_2x2x2x2_bcast_in1_1x1x1x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f
    });

    set_values(input2, {
        0.5f, 2.5f,
        1.5f, 1.f,

        5.f,  7.f,
        2.f,  4.f,

        15.f, 17.f,
        8.f,  10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f,  0.f,
        0.f,  0.f,

        1.f,  1.f,
        1.f,  1.f,

        0.f,  1.f,
        0.f,  1.f,

        1.f,  0.f,
        1.f,  0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  1.f,

        1.f,   1.f,
        1.f,   1.f,

        15.f,  1.f,
        8.f,   1.f,

        1.f,   6.5f,
        1.f,   -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma_byxf_2x2x2x2_bcast_in1_2x2x2x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf ,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        3.f,  0.5f,
        7.f,  12.f,
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f,   0.f,
        0.f,   0.f,

        0.1f,  0.3f,
        0.5f,  0.7f,

        -0.f,  -0.1f,
        -0.f,  -0.5f,

        -0.7f, -0.f,
        -1.5f, -0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f, 2.5f,
        1.5f, 3.f,

        1.f,  0.f,
        5.f,  1.5f,

        15.f, 0.5f,
        8.f,  12.f,

        3.f,  6.5f,
        7.f,  -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma_byxf_2x2x2x2_bcast_mask_2x1x2x2_in1_2x2x2x1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 2, 2, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf ,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        3.f,  0.5f,
        7.f,  12.f,
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f,
        0.f,

        0.1f,
        0.5f,

        -0.f,
        -0.5f,

        -0.7f,
        -0.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  3.f,

        1.f,   0.f,
        5.f,   1.5f,

        15.f,  17.f,
        7.f,   12.f,

        3.f,   0.5f,
        -0.5f, -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_comma) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        0.f,   0.f,  0.f,  0.f,
        0.1f,   0.3f,  0.5f,  0.7f,
        -0.f,   -0.1f,  -0.f,  -0.5f,
        -0.7f,   -0.f,  -1.5f,  -0.f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_error_input_sizes) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 3, 4, 5, 6 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    EXPECT_ANY_THROW(network(engine, topology, get_test_default_config(engine)));
}

TEST(select_gpu_f32, select_basic_error_mask_sizes) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 3, 4, 5, 6 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    EXPECT_ANY_THROW(network(engine, topology, get_test_default_config(engine)));
}

TEST(select_gpu_f32, select_basic_error_input_types) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));
    EXPECT_ANY_THROW(network(engine, topology, get_test_default_config(engine)));
}

TEST(select_gpu_f32, select_basic_byxf) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::byxf,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values(mask, {
        0.f,   0.f,  0.f,  0.f,
        1.f,   1.f,  1.f,  1.f,
        0.f,   1.f,  0.f,  1.f,
        1.f,   0.f,  1.f,  0.f });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_mask_f16) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f16, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values<uint16_t>(mask, {
        0,   0,  0,  0,
        1,   1,  1,  1,
        0,   1,  0,  1,
        1,   0,  1,  0 });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_mask_i8) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::i8, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values<char>(mask, {
        0,   0,  0,  0,
        1,   1,  1,  1,
        0,   1,  0,  1,
        1,   0,  1,  0 });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_mask_u8) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::u8, format::yxfb,{ 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f, 5.f, 1.5f,
        2.f,   0.f, 6.f, 5.2f,
        3.f,  0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f,  8.f
    });

    set_values(input2, {
        0.5f,   2.5f,  0.5f,  2.5f,
        5.f,   7.f,    2.f,   4.f,
        15.f,  17.f,    8.f,  10.f,
        -2.f,  6.5f,  -0.5f, -2.5f });

    set_values<unsigned char>(mask, {
        0,   0,  0,  0,
        128,   210,  150,  177,
        0,   211,  0,  255,
        199,   0,  160,  0 });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = { 0.5f,  2.5f,   0.5f,  2.5f,
        2.f,   0.f,    6.f,   5.2f,
        15.f,   0.5f,   8.f,  12.f,
        4.f,   6.5f,   8.f,  -2.5f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,    0.f,    2.f,    0.f
    });

    set_values(input2, {
        0.5f,    2.5f,    5.f,    7.f
    });

    set_values(mask, {
        0.f,    0.f,    1.f,    1.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[4] = {
        0.5f,    2.5f,    2.f,    0.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_bfyx_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f,
        2.f,   0.f
    });

    set_values(input2, {
        0.5f,   2.5f,
        5.f,   7.f
    });

    set_values(mask, {
        0.f,   0.f,
        1.f,   1.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[4] = {
        0.5f,  2.5f,
        2.f,   0.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f32, select_basic_byxf_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,   0.f,
        2.f,   0.f
    });

    set_values(input2, {
        0.5f,   2.5f,
        5.f,   7.f
    });

    set_values(mask, {
        0.f,   0.f,
        1.f,   1.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[4] = {
        0.5f,  2.5f,
        2.f,   0.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

// select_gpu_f16
template <typename T>
void test_f16_select_basic_1x1x2x2(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<T>(input, {
        1,   0,
        2,   0
    });

    set_values<T>(input2, {
        0,   2,
        5,   7
    });

    set_values<T>(mask, {
        0,   0,
        1,   1
    });

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("input2", input2);
    network->set_input_data("mask", mask);
    auto outputs = network->execute();

    auto output = outputs.at("select").get_memory();

    T answers[4] = {
        0,  2,
        2,   0
    };

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f16, select_basic_1x1x2x2) {
    test_f16_select_basic_1x1x2x2<uint16_t>(false);
}

TEST(select_gpu_f16, select_basic_mask_f32_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<float>(mask, {
        0.f,   0.f,
        1.5f,   0.4f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f16, select_basic_mask_i8_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<char>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(select_gpu_f16, select_basic_mask_u8_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<uint16_t>(input, {
        1,   0,
        2,   0
    });

    set_values<uint16_t>(input2, {
        0,   2,
        5,   7
    });

    set_values<unsigned char>(mask, {
        0,   0,
        128,   255
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    uint16_t answers[4] = {
        0,  2,
        2,   0
    };

    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

// select_gpu_i8
template <typename T>
void test_i8_select_basic_1x1x2x2(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<T>(input, {
        1,   0,
        2,   0
    });

    set_values<T>(input2, {
        0,   2,
        5,   7
    });

    set_values<T>(mask, {
        0,   0,
        3,   5
    });

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("input2", input2);
    network->set_input_data("mask", mask);
    auto outputs = network->execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_i8, select_basic_1x1x2x2) {
    test_i8_select_basic_1x1x2x2<char>(false);
}

TEST(select_gpu_i8, select_basic_mask_f32_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<float>(mask, {
        0.f,   0.f,
        1.5f,  0.4f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    cldnn::mem_lock<char> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_i8, select_basic_mask_f16_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<uint16_t>(mask, {
        0,   0,
        3,   5
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    cldnn::mem_lock<char> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_i8, select_basic_mask_u8_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<char>(input, {
        1,   0,
        2,   0
    });

    set_values<char>(input2, {
        0,   2,
        5,   7
    });

    set_values<unsigned char>(mask, {
        0,   0,
        128,   255
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  2,
        2,  0
    };

    cldnn::mem_lock<char> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

// select_gpu_u8
template <typename T>
void test_u8_select_basic_1x1x2x2(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<T>(input, {
        128,   0,
        255,   0
    });

    set_values<T>(input2, {
        0,   255,
        205,   128
    });

    set_values<T>(mask, {
        0,   0,
        128,   255
    });

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    network->set_input_data("input2", input2);
    network->set_input_data("mask", mask);
    auto outputs = network->execute();

    auto output = outputs.at("select").get_memory();

    T answers[4] = {
        0,  255,
        255,  0
    };

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_u8, select_basic_1x1x2x2) {
    test_u8_select_basic_1x1x2x2<unsigned char>(false);
}

TEST(select_gpu_u8, select_basic_mask_f32_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<float>(mask, {
        0.f,   0.f,
        1.5f,  0.4f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    int answers[4] = {
        0,  255,
        255,  0
    };

    cldnn::mem_lock<unsigned char> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_u8, select_basic_mask_f16_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<uint16_t>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    unsigned char answers[4] = {
        0,  255,
        255,  0
    };

    cldnn::mem_lock<unsigned char> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_u8, select_basic_mask_i8_1x1x2x2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::u8, format::yxfb,{ 1, 1, 2, 2 } });
    auto mask = engine.allocate_memory({ data_types::i8, format::yxfb,{ 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values<unsigned char>(input, {
        128,   0,
        255,   0
    });

    set_values<unsigned char>(input2, {
        0,   255,
        205,   128
    });

    set_values<char>(mask, {
        0,   0,
        1,   1
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    unsigned char answers[4] = {
        0,  255,
        255,  0
    };

    cldnn::mem_lock<unsigned char> output_ptr(output, get_test_stream());

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_fp32, select_numpy_broadcast_mask_u8_1x1x3) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } });
    auto mask = engine.allocate_memory({ data_types::u8, format::bfyx, { 1, 1, 3, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input"), input_info("input2")));

    set_values(input, {
        1.f,    0.f,    2.f
    });

    set_values(input2, {
        0.5f,    2.5f,    5.f
    });

    set_values<unsigned char>(mask, {
        1,   0,   1
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[27] = {
        1.f, 0.5f, 1.f,
        0.f, 0.5f, 0.f,
        2.f, 0.5f, 2.f,
        1.f, 2.5f, 1.f,
        0.f, 2.5f, 0.f,
        2.f, 2.5f, 2.f,
        1.f, 5.f, 1.f,
        0.f, 5.f, 0.f,
        2.f, 5.f, 2.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 27; i++)
    {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_f32, select_different_formats) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::byxf, { 2, 1, 2, 2 } });
    auto mask   = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f, 2.f,
        3.f, 4.f,

        5.f, 6.f,
        7.f, 8.f
    });

    set_values(input2, {
        9.f,  10.f,
        11.f, 12.f,

        13.f, 14.f,
        15.f, 16.f
    });

    set_values(mask, {
        0.f, 0.f,
        1.f, 1.f
    });

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    std::vector<float> answers {
        9.f,  10.f,
        3.f,  4.f,

        13.f, 14.f,
        7.f,  8.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(select_gpu_f32, dynamic) {
    auto& engine = get_test_engine();

    ov::PartialShape in1_shape  = { 2, 2, 2, 2 };
    ov::PartialShape in2_shape  = { 2, 2, 2, 2 };
    ov::PartialShape mask_shape = { 2, 2, 2, 1 };

    layout input1_layout { ov::PartialShape::dynamic(in1_shape.size()),  data_types::f32, format::bfyx };
    layout input2_layout { ov::PartialShape::dynamic(in2_shape.size()),  data_types::f32, format::bfyx };
    layout mask_layout   { ov::PartialShape::dynamic(mask_shape.size()), data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory({ in1_shape,  data_types::f32, format::bfyx });
    auto input2 = engine.allocate_memory({ in2_shape,  data_types::f32, format::bfyx });
    auto mask   = engine.allocate_memory({ mask_shape, data_types::f32, format::bfyx });

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f,

        3.f,  0.5f,
        7.f,  12.f,

        4.f,  -0.5f,
        8.f,  8.f
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values(mask, {
        0.f,
        0.f,

        1.f,
        1.f,

        0.f,
        1.f,

        1.f,
        0.f,
    });

    topology topology;
    topology.add(input_layout("input1", input1_layout));
    topology.add(input_layout("input2", input2_layout));
    topology.add(input_layout("mask", mask_layout));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);

    auto inst = network.get_primitive("select");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  3.f,

        2.f,   0.f,
        6.f,   5.2f,

        15.f,  17.f,
        7.f,   12.f,

        4.f,   -0.5f,
        -0.5f, -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(select_gpu_f32, select_basic_cached) {
    test_select_basic<float>(true);
}

TEST(select_gpu_f16, select_basic_1x1x2x2_cached) {
    test_f16_select_basic_1x1x2x2<uint16_t>(true);
}

TEST(select_gpu_i8, select_basic_1x1x2x2_cached) {
    test_i8_select_basic_1x1x2x2<char>(true);
}
#endif
TEST(select_gpu_u8, select_basic_1x1x2x2_cached) {
    test_u8_select_basic_1x1x2x2<unsigned char>(true);
}

TEST(select_cpu_impl_f32, select_basic_bfyx_2x2x2x2_bcast_mask_2x2x1x2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ { 2, 2, 2, 2 }, data_types::f32, format::bfyx });
    auto input2 = engine.allocate_memory({ { 2, 2, 2, 2 }, data_types::f32, format::bfyx });
    auto mask = engine.allocate_memory({ { 2, 2, 2, 1 }, data_types::u8, format::bfyx });

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("mask", mask->get_layout()));
    topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

    set_values(input1, {
        1.f,  0.f,
        5.f,  1.5f,

        2.f,  0.f,
        6.f,  5.2f,

        3.f,  0.5f,
        7.f,  12.f,

        4.f,  -0.5f,
        8.f,  8.f
    });

    set_values(input2, {
        0.5f,  2.5f,
        1.5f,  3.f,

        5.f,   7.f,
        2.f,   4.f,

        15.f,  17.f,
        8.f,   10.f,

        -2.f,  6.5f,
        -0.5f, -2.5f
    });

    set_values<uint8_t>(mask, {
        0, 0,

        1, 1,

        0, 1,

        1, 0,
    });

    auto config = get_test_default_config(engine);
    auto forcing_map = ov::intel_gpu::ImplForcingMap{{"select", {format::bfyx, "", impl_types::cpu}}};
    config.set_property(ov::intel_gpu::force_implementations(forcing_map));

    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("mask", mask);
    auto outputs = network.execute();

    auto output = outputs.at("select").get_memory();

    float answers[16] = {
        0.5f,  2.5f,
        1.5f,  3.f,

        2.f,   0.f,
        6.f,   5.2f,

        15.f,  17.f,
        7.f,   12.f,

        4.f,   -0.5f,
        -0.5f, -2.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}
