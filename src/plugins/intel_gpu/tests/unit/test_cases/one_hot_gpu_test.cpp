// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/one_hot.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template <typename T>
VVVVF<T> one_hot_cpu(VVVVF<T> &input, uint16_t axis,
    int32_t one_hot_limit, int input_padding_y = 0,
    int input_padding_x = 0, int output_padding_y = 0,
    int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t out_sizes[4];
    out_sizes[0] = input.size();
    out_sizes[1] = input[0].size();
    out_sizes[2] = input[0][0].size() + 2 * padding_y;
    out_sizes[3] = input[0][0][0].size() + 2 * padding_x;
    for (uint16_t i = 3; i > axis; --i)
        out_sizes[i] = out_sizes[i - 1];
    out_sizes[axis] = one_hot_limit;
    VVVVF<T> output(out_sizes[0], VVVF<T>(out_sizes[1], VVF<T>(out_sizes[2], VF<T>(out_sizes[3]))));

    switch (axis) {
    case 0:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[f][y][x][0] == (T)b ? 1 : 0;
        break;
    case 1:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[b][y][x][0] == (T)f ? 1 : 0;
        break;
    case 2:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[b][f][x][0] == (T)y ? 1 : 0;
        break;
    case 3:
        for (size_t b = 0; b < out_sizes[0]; ++b)
            for (size_t f = 0; f < out_sizes[1]; ++f)
                for (size_t y = 0; y < out_sizes[2]; ++y)
                    for (size_t x = 0; x < out_sizes[3]; ++x)
                        output[b][f][y][x] = input[b][f][y][0] == (T)x ? 1 : 0;
        break;
    default: break;
    }
    return output;
}

template <typename T>
void generic_one_hot_test_int(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, tensor shape,
    uint16_t one_hot_axis, int input_padding_y, int input_padding_x, int output_padding_y, int output_padding_x, bool is_caching_test) {
    tests::random_generator rg(GET_SUITE_NAME);
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
        shape.spatial[1], shape.spatial[0] };
    int32_t one_hot_limit = output_dims[one_hot_axis];

    int min_random = 0, max_random = one_hot_limit + 2;
    VVVVF<T> input_rnd = rg.generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);

    auto& engine = get_test_engine();
    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = engine.allocate_memory({ ov::element::from<T>(), test_input_fmt, input_tensor });
    set_values(input, input_rnd_vec);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, one_hot_limit));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    VVVVF<T> output_cpu = one_hot_cpu<T>(input_rnd, one_hot_axis, one_hot_limit, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    ASSERT_EQ(output_layout.format.value, test_input_fmt.value);
    auto output_tensor = output_layout.get_padded_dims();
    int y_size = output_tensor[2];
    int x_size = output_tensor[3];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(y_size, (int)output_cpu[0][0].size());
    ASSERT_EQ(x_size, (int)output_cpu[0][0][0].size());
    ASSERT_EQ(f_size, (int)output_cpu[0].size());
    ASSERT_EQ(b_size, (int)output_cpu.size());

    bool test_is_correct = true;
    VF<T> output_cpu_vec = flatten_4d<T>(test_input_fmt, output_cpu);

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
            break;
        }
    }
    ASSERT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "input_b = " << input_b << std::endl
        << "input_f = " << input_f << std::endl
        << "input_y = " << input_y << std::endl
        << "input_x = " << input_x << std::endl
        << "one_hot_limit = " << one_hot_limit << std::endl
        << "one_hot_axis = " << one_hot_axis << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl;
}

TEST(one_hot_gpu_i32, generic) {
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(5, 2, 1, 2), 0, 0, 0, 0, 0, false);
    generic_one_hot_test_int<int32_t>(format::bfyx, 1, 2, 3, 1, tensor(1, 5, 3, 2), 1, 0, 0, 0, 0, false);
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 4), 2, 0, 0, 0, 0, false);
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 4, 1), 3, 0, 0, 0, 0, false);
}

TEST(one_hot_gpu_i64, generic) {
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(5, 2, 1, 2), 0, 0, 0, 0, 0, false);
    generic_one_hot_test_int<int64_t>(format::bfyx, 1, 2, 3, 1, tensor(1, 5, 3, 2), 1, 0, 0, 0, 0, false);
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 4), 2, 0, 0, 0, 0, false);
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 4, 1), 3, 0, 0, 0, 0, false);
}

TEST(one_hot_gpu_i32, generic_cached) {
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(5, 2, 1, 2), 0, 0, 0, 0, 0, true);
#ifdef RUN_ALL_MODEL_CACHING_TESTS
    generic_one_hot_test_int<int32_t>(format::bfyx, 1, 2, 3, 1, tensor(1, 5, 3, 2), 1, 0, 0, 0, 0, true);
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 4), 2, 0, 0, 0, 0, true);
    generic_one_hot_test_int<int32_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 4, 1), 3, 0, 0, 0, 0, true);
}

TEST(one_hot_gpu_i64, generic_cached) {
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(5, 2, 1, 2), 0, 0, 0, 0, 0, true);
    generic_one_hot_test_int<int64_t>(format::bfyx, 1, 2, 3, 1, tensor(1, 5, 3, 2), 1, 0, 0, 0, 0, true);
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 1, 4), 2, 0, 0, 0, 0, true);
    generic_one_hot_test_int<int64_t>(format::bfyx, 2, 2, 1, 1, tensor(2, 2, 4, 1), 3, 0, 0, 0, 0, true);
#endif
}

TEST(one_hot_gpu_i32, bfzyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 2);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 2);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32_to_f32, bfyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, data_types::f32, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(z_size, 2);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    std::vector<float> output_cpu_vec = {1.f, 0.f, 0.f, 0.f, 0.f,
                                         0.f, 1.f, 0.f, 0.f, 0.f};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

TEST(one_hot_gpu_i64_to_f32, bfyx_ax4) {
    // input: 1x1x2x1
    // axis: 4
    // output: 1x1x2x1x5
    int in_b = 1;
    int in_f = 1;
    int in_y = 2;
    int in_x = 1;
    tensor shape(in_b, in_f, 5, in_x, in_y);
    uint16_t one_hot_axis = 4;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, data_types::f32, one_hot_axis, 5));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 2);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 5);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    std::vector<float> output_cpu_vec = {1.f, 0.f, 0.f, 0.f, 0.f,
                                         0.f, 1.f, 0.f, 0.f, 0.f};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        ASSERT_EQ(output_cpu_vec[i], output_ptr[i]);
    }
}

TEST(one_hot_gpu_i32, bfzyx_ax0) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(3, in_b, in_x, in_y, in_f);
    uint16_t one_hot_axis = 0;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 3);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax0) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(3, in_b, in_x, in_y, in_f);
    uint16_t one_hot_axis = 0;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 3);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32, bfzyx_ax1) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, 3, in_x, in_y, in_f);
    uint16_t one_hot_axis = 1;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 3);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax1) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, 3, in_x, in_y, in_f);
    uint16_t one_hot_axis = 1;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 3);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32, bfzyx_ax2) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, in_y, 3);
    uint16_t one_hot_axis = 2;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 3);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax2) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, in_y, 3);
    uint16_t one_hot_axis = 2;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 3);
    ASSERT_EQ(y_size, 1);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i32, bfzyx_ax3) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, 3, in_y);
    uint16_t one_hot_axis = 3;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int32_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int32_t> output_ptr(output_memory, get_test_stream());

    auto output_tensor = output_layout.get_padded_dims();
    int z_size = output_tensor[2];
    int y_size = output_tensor[3];
    int x_size = output_tensor[4];
    int f_size = output_tensor[1];
    int b_size = output_tensor[0];
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int32_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_gpu_i64, bfzyx_ax3) {
    int in_b = 1;
    int in_f = 1;
    int in_y = 1;
    int in_x = 2;
    tensor shape(in_b, in_f, in_x, 3, in_y);
    uint16_t one_hot_axis = 3;
    std::vector<tensor::value_type> output_dims = { shape.batch[0], shape.feature[0],
                                                    shape.spatial[2], shape.spatial[1], shape.spatial[0] };

    VF<int64_t> input_rnd_vec = {0, 1};

    auto& engine = get_test_engine();
    tensor input_tensor(in_b, in_f, in_x, in_y);
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), shape, one_hot_axis, 3));

    set_values(input, input_rnd_vec);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<int64_t> output_ptr(output_memory, get_test_stream());

    int z_size = output_layout.spatial(2);
    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(z_size, 1);
    ASSERT_EQ(y_size, 3);
    ASSERT_EQ(x_size, 2);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);

    bool test_is_correct = true;

    std::vector<int64_t> output_cpu_vec = {1, 0, 0, 1, 0, 0};

    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (output_cpu_vec[i] != output_ptr[i]) {
            test_is_correct = false;
        }
    }
    ASSERT_EQ(test_is_correct, true);
}

TEST(one_hot_error, basic_error_wrong_axis) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), tensor(1, 1, 1, 50), 5, 2));

    std::string msg_to_find = "Incorrect parameters configuration: one_hot_axis should be less or equal to 4.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(one_hot_error, basic_error_bad_shape) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(one_hot("output", input_info("input"), tensor(1, 5, 1, 50), 2, 2));

    std::string msg_to_find = "Incorrect parameters configuration: shape does not fit input size.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}
