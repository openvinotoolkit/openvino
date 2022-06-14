// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/broadcast.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template<typename T>
void start_broadcast_test(format cldnn_format, data_types cldnn_data_type, std::vector<size_t> output_shape,
                          std::vector<size_t> input_shape, std::vector<size_t> broadcast_axes,
                          std::vector<T> golden_data) {
    size_t input_data_size = accumulate(input_shape.rbegin(), input_shape.rend(), (size_t)1, std::multiplies<size_t>());
    EXPECT_GE(input_data_size, (size_t)1);
    std::vector<T> input_data = {};
    for (size_t i = 1; i <= input_data_size; ++i) {
        input_data.push_back((T)i);
    }

    EXPECT_EQ(golden_data.size(), accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>()));

    std::vector<tensor::value_type> output_4d(4, 1);
    for (size_t i = 0; i < output_shape.size(); ++i) {
        output_4d.at(4 - output_shape.size() + i) = (tensor::value_type)output_shape.at(i);
    }
    std::vector<tensor::value_type> input_4d(4, 1);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        input_4d.at(4 - input_shape.size() + i) = (tensor::value_type)input_shape.at(i);
    }
    std::vector<uint16_t> fixed_b_axes;
    size_t shift = 4 - output_shape.size();
    for (size_t i = 0; i < shift; ++i) {
        fixed_b_axes.push_back((uint16_t) i);
    }
    for (size_t i = 0; i < broadcast_axes.size(); ++i) {
        fixed_b_axes.push_back((uint16_t) (broadcast_axes.at(i) + shift));
    }

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({cldnn_data_type, format::bfyx, {input_4d.at(0), input_4d.at(1), input_4d.at(3), input_4d.at(2)}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", "input", cldnn_format, cldnn_data_type));
    topology.add(broadcast("broadcast", "reorder", {output_4d.at(0), output_4d.at(1), output_4d.at(3), output_4d.at(2)}, fixed_b_axes));
    topology.add(reorder("output", "broadcast", format::bfyx, cldnn_data_type));


    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (tensor::value_type b = 0; b < output_4d.at(0); ++b) {
        for (tensor::value_type f = 0; f < output_4d.at(1); ++f) {
            for (tensor::value_type y = 0; y < output_4d.at(2); ++y) {
                for (tensor::value_type x = 0; x < output_4d.at(3); ++x) {
                    auto output_off = ((b * output_4d.at(1) + f) * output_4d.at(2) + y) * output_4d.at(3) + x;
                    EXPECT_EQ(output_ptr[output_off], golden_data[output_off]);
                }
            }
        }
    }
}

template<typename T>
void start_broadcast_test_5d(format cldnn_format, data_types cldnn_data_type, std::vector<size_t> output_shape,
                             std::vector<size_t> input_shape, std::vector<size_t> broadcast_axes,
                             std::vector<T> golden_data)
{
    size_t input_data_size = accumulate(input_shape.rbegin(), input_shape.rend(), (size_t)1, std::multiplies<size_t>());
    EXPECT_GE(input_data_size, (size_t)1);
    std::vector<T> input_data = {};
    for (size_t i = 1; i <= input_data_size; ++i) {
        input_data.push_back((T)i);
    }

    EXPECT_EQ(golden_data.size(), accumulate(output_shape.rbegin(), output_shape.rend(), (size_t)1, std::multiplies<size_t>()));

    std::vector<tensor::value_type> output_5d(5, 1);
    for (size_t i = 0; i < output_shape.size(); ++i) {
        output_5d.at(5 - output_shape.size() + i) = (tensor::value_type)output_shape.at(i);
    }
    std::vector<tensor::value_type> input_5d(5, 1);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        input_5d.at(5 - input_shape.size() + i) = (tensor::value_type)input_shape.at(i);
    }
    std::vector<uint16_t> fixed_b_axes;
    size_t shift = 5 - output_shape.size();
    for (size_t i = 0; i < shift; ++i) {
        fixed_b_axes.push_back((uint16_t)i);
    }
    for (size_t i = 0; i < broadcast_axes.size(); ++i) {
        fixed_b_axes.push_back((uint16_t)(broadcast_axes.at(i) + shift));
    }

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ cldnn_data_type, format::bfzyx, { input_5d.at(0), input_5d.at(1), input_5d.at(4), input_5d.at(3), input_5d.at(2) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", "input", cldnn_format, cldnn_data_type));
    topology.add(broadcast("broadcast", "reorder", { output_5d.at(0), output_5d.at(1), output_5d.at(4), output_5d.at(3), output_5d.at(2) }, fixed_b_axes));
    topology.add(reorder("output", "broadcast", format::bfzyx, cldnn_data_type));


    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (tensor::value_type b = 0; b < output_5d.at(0); ++b) {
        for (tensor::value_type f = 0; f < output_5d.at(1); ++f) {
            for (tensor::value_type z = 0; z < output_5d.at(2); ++z) {
                for (tensor::value_type y = 0; y < output_5d.at(3); ++y) {
                    for (tensor::value_type x = 0; x < output_5d.at(4); ++x) {
                        auto output_off = (((b * output_5d.at(1) + f) * output_5d.at(2) + z) * output_5d.at(3) + y) * output_5d.at(4) + x;
                        EXPECT_EQ(output_ptr[output_off], golden_data[output_off]);
                    }
                }
            }
        }
    }
}

TEST(broadcast_gpu_float, bfyx_1_to_5_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_5_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_5_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_5_w_b_axes_0) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_5_w_b_axes_0) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_5_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_5_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, bfyx_1_to_5_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_5_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {5}, {1}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_5_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {5}, {1}, {0}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_1_to_4x5_w_b_axes_0x1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_4x5_w_b_axes_0x1) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_4x5_w_b_axes_0x1) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_4x5_w_b_axes_0x1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1_to_4x5_w_b_axes_0x1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {4, 5}, {1}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {4, 5}, {1}, {0, 1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_3x4x5_w_b_axes_0x1x2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {3, 4, 5}, {1}, {0, 1, 2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv16, data_types::i8, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_2x3x4x5_w_b_axes_0x1x2x3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {1}, {0, 1, 2, 3}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_1_to_5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1_to_5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1_to_5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_5_w_o_b_axes) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_5_w_o_b_axes) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1_to_5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1_to_5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1_to_5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1_to_5_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {5}, {1}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1_to_5_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {5}, {1}, {}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3_to_12_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_12_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_12_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_12_w_o_b_axes) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_12_w_o_b_axes) {
    std::vector<int64_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_12_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_12_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_12_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_12_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {12}, {3}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_12_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {12}, {3}, {}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_1x1_to_4x5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_1x1_to_4x5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_1x1_to_4x5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1x1_to_4x5_w_o_b_axes) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1x1_to_4x5_w_o_b_axes) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_1x1_to_4x5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_1x1_to_4x5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_1x1_to_4x5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_1x1_to_4x5_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {4, 5}, {1, 1}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_1x1_to_4x5_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {4, 5}, {1, 1}, {}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x3_to_8x6_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3_to_8x6_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3_to_8x6_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
                                      1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3_to_8x6_w_o_b_axes) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3_to_8x6_w_o_b_axes) {
    std::vector<int64_t> golden_data = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3_to_8x6_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3_to_8x6_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3_to_8x6_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                                        1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3_to_8x6_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3_to_8x6_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {8, 6}, {2, 3}, {}, golden_data);
}

TEST(broadcast_gpu_float, bfyx_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3x4_to_6x6x4_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
                                        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {6, 6, 4}, {2, 3, 4}, {}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
                                      51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
                                      71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
                                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                                      101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                                      111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                        91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                        111, 112, 113, 114, 115, 116, 117, 118, 119, 120};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                         FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3x4x5_to_2x9x8x5_w_o_b_axes) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(1.0f),   FLOAT16(2.0f),   FLOAT16(3.0f),   FLOAT16(4.0f),   FLOAT16(5.0f),
                                        FLOAT16(6.0f),   FLOAT16(7.0f),   FLOAT16(8.0f),   FLOAT16(9.0f),   FLOAT16(10.0f),
                                        FLOAT16(11.0f),  FLOAT16(12.0f),  FLOAT16(13.0f),  FLOAT16(14.0f),  FLOAT16(15.0f),
                                        FLOAT16(16.0f),  FLOAT16(17.0f),  FLOAT16(18.0f),  FLOAT16(19.0f),  FLOAT16(20.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(21.0f),  FLOAT16(22.0f),  FLOAT16(23.0f),  FLOAT16(24.0f),  FLOAT16(25.0f),
                                        FLOAT16(26.0f),  FLOAT16(27.0f),  FLOAT16(28.0f),  FLOAT16(29.0f),  FLOAT16(30.0f),
                                        FLOAT16(31.0f),  FLOAT16(32.0f),  FLOAT16(33.0f),  FLOAT16(34.0f),  FLOAT16(35.0f),
                                        FLOAT16(36.0f),  FLOAT16(37.0f),  FLOAT16(38.0f),  FLOAT16(39.0f),  FLOAT16(40.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(41.0f),  FLOAT16(42.0f),  FLOAT16(43.0f),  FLOAT16(44.0f),  FLOAT16(45.0f),
                                        FLOAT16(46.0f),  FLOAT16(47.0f),  FLOAT16(48.0f),  FLOAT16(49.0f),  FLOAT16(50.0f),
                                        FLOAT16(51.0f),  FLOAT16(52.0f),  FLOAT16(53.0f),  FLOAT16(54.0f),  FLOAT16(55.0f),
                                        FLOAT16(56.0f),  FLOAT16(57.0f),  FLOAT16(58.0f),  FLOAT16(59.0f),  FLOAT16(60.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(61.0f),  FLOAT16(62.0f),  FLOAT16(63.0f),  FLOAT16(64.0f),  FLOAT16(65.0f),
                                        FLOAT16(66.0f),  FLOAT16(67.0f),  FLOAT16(68.0f),  FLOAT16(69.0f),  FLOAT16(70.0f),
                                        FLOAT16(71.0f),  FLOAT16(72.0f),  FLOAT16(73.0f),  FLOAT16(74.0f),  FLOAT16(75.0f),
                                        FLOAT16(76.0f),  FLOAT16(77.0f),  FLOAT16(78.0f),  FLOAT16(79.0f),  FLOAT16(80.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(81.0f),  FLOAT16(82.0f),  FLOAT16(83.0f),  FLOAT16(84.0f),  FLOAT16(85.0f),
                                        FLOAT16(86.0f),  FLOAT16(87.0f),  FLOAT16(88.0f),  FLOAT16(89.0f),  FLOAT16(90.0f),
                                        FLOAT16(91.0f),  FLOAT16(92.0f),  FLOAT16(93.0f),  FLOAT16(94.0f),  FLOAT16(95.0f),
                                        FLOAT16(96.0f),  FLOAT16(97.0f),  FLOAT16(98.0f),  FLOAT16(99.0f),  FLOAT16(100.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f),
                                        FLOAT16(101.0f), FLOAT16(102.0f), FLOAT16(103.0f), FLOAT16(104.0f), FLOAT16(105.0f),
                                        FLOAT16(106.0f), FLOAT16(107.0f), FLOAT16(108.0f), FLOAT16(109.0f), FLOAT16(110.0f),
                                        FLOAT16(111.0f), FLOAT16(112.0f), FLOAT16(113.0f), FLOAT16(114.0f), FLOAT16(115.0f),
                                        FLOAT16(116.0f), FLOAT16(117.0f), FLOAT16(118.0f), FLOAT16(119.0f), FLOAT16(120.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 9, 8, 5}, {2, 3, 4, 5}, {}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3_to_2x3_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_2x3_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_2x3_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x3_w_b_axes_0) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 1, 2, 3};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x3_w_b_axes_0) {
    std::vector<int64_t> golden_data = {1, 2, 3, 1, 2, 3};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_2x3_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_2x3_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_2x3_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_2x3_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_2x3_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3}, {3}, {0}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3_to_2x6_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_2x6_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_2x6_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x6_w_b_axes_0) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x6_w_b_axes_0) {
    std::vector<int64_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_2x6_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_2x6_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_2x6_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_2x6_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 6}, {3}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_2x6_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 6}, {3}, {0}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2_to_2x3_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_2x3_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_2x3_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_2x3_w_b_axes_1) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 2, 2, 2};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_2x3_w_b_axes_1) {
    std::vector<int64_t> golden_data = {1, 1, 1, 2, 2, 2};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_2x3_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_2x3_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2_to_2x3_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 2, 2, 2};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_2x3_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_2x3_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3}, {2}, {1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2_to_6x3_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_6x3_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_6x3_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_6x3_w_b_axes_1) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 2, 2, 2, 1, 1, 1,
                                        2, 2, 2, 1, 1, 1, 2, 2, 2};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_6x3_w_b_axes_1) {
    std::vector<int64_t> golden_data = {1, 1, 1, 2, 2, 2, 1, 1, 1,
                                        2, 2, 2, 1, 1, 1, 2, 2, 2};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_6x3_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 2, 2, 2, 1, 1, 1,
                                        2, 2, 2, 1, 1, 1, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_6x3_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 2, 2, 2, 1, 1, 1,
                                        2, 2, 2, 1, 1, 1, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2_to_6x3_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 1, 1, 2, 2, 2, 1, 1, 1,
                                        2, 2, 2, 1, 1, 1, 2, 2, 2};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_6x3_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {6, 3}, {2}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_6x3_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {6, 3}, {2}, {1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {3, 4}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3x4_to_2x3x4_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {3, 4}, {0}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                                      5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                                      5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                                      5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {2, 4}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
                                        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {2, 4}, {1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                      4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                      4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                      4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(4.0f), FLOAT16(4.0f), FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(5.0f), FLOAT16(5.0f), FLOAT16(5.0f),
                                        FLOAT16(6.0f), FLOAT16(6.0f), FLOAT16(6.0f), FLOAT16(6.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {2, 3}, {2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4_w_b_axes_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f),
                                        FLOAT16(4.0f), FLOAT16(4.0f), FLOAT16(4.0f), FLOAT16(4.0f),
                                        FLOAT16(5.0f), FLOAT16(5.0f), FLOAT16(5.0f), FLOAT16(5.0f),
                                        FLOAT16(6.0f), FLOAT16(6.0f), FLOAT16(6.0f), FLOAT16(6.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {2, 3}, {2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                                      1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                                      1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                                      1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                        1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {4}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_4_to_2x3x4_w_b_axes_0_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
                                        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {4}, {0, 1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                      1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                      1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                                      1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {3}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_2x3x4_w_b_axes_0_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {3}, {0, 2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4}, {2}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_2x3x4_w_b_axes_1_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
                                        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4}, {2}, {1, 2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                                      37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                                      49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                                      37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                                      49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                                      37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                                      49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                                      37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                                      49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                                      37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                                      49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                                      25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                                      37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                                      49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f), FLOAT16(45.0f),
                                        FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f), FLOAT16(49.0f), FLOAT16(50.0f),
                                        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f), FLOAT16(55.0f),
                                        FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f), FLOAT16(59.0f), FLOAT16(60.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f), FLOAT16(45.0f),
                                        FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f), FLOAT16(49.0f), FLOAT16(50.0f),

                                        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f), FLOAT16(55.0f),
                                        FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f), FLOAT16(59.0f), FLOAT16(60.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3x4x5_to_2x3x4x5_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f), FLOAT16(45.0f),
                                        FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f), FLOAT16(49.0f), FLOAT16(50.0f),
                                        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f), FLOAT16(55.0f),
                                        FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f), FLOAT16(59.0f), FLOAT16(60.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f), FLOAT16(45.0f),
                                        FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f), FLOAT16(49.0f), FLOAT16(50.0f),
                                        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f), FLOAT16(55.0f),
                                        FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f), FLOAT16(59.0f), FLOAT16(60.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 4, 5}, {0}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x4x5_to_2x3x4x5_w_b_axes_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(31.0f), FLOAT16(32.0f), FLOAT16(33.0f), FLOAT16(34.0f), FLOAT16(35.0f),
                                        FLOAT16(36.0f), FLOAT16(37.0f), FLOAT16(38.0f), FLOAT16(39.0f), FLOAT16(40.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4, 5}, {1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                      26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                      26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      16.0, 17.0, 18.0, 19.0, 20.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                      21.0, 22.0, 23.0, 24.0, 25.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                                      26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                                      26.0, 27.0, 28.0, 29.0, 30.0, 26.0, 27.0, 28.0, 29.0, 30.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        16, 17, 18, 19, 20, 16, 17, 18, 19, 20,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        21, 22, 23, 24, 25, 21, 22, 23, 24, 25,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30,
                                        26, 27, 28, 29, 30, 26, 27, 28, 29, 30};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3x5_to_2x3x4x5_w_b_axes_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f), FLOAT16(25.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f),
                                        FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f), FLOAT16(29.0f), FLOAT16(30.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 5}, {2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                                      13.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 14.0,
                                      15.0, 15.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0,
                                      17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                                      19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                                      21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                                      23.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                                      13.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 14.0,
                                      15.0, 15.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0,
                                      17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                                      19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                                      21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                                      23.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                                      13.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 14.0,
                                      15.0, 15.0, 15.0, 15.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0,
                                      17.0, 17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 18.0,
                                      19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                                      21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                                      23.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
                                        15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
                                        17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
                                        19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
                                        21, 21, 21, 21, 21, 22, 22, 22, 22, 22,
                                        23, 23, 23, 23, 23, 24, 24, 24, 24, 24};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
                                        15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
                                        17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
                                        19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
                                        21, 21, 21, 21, 21, 22, 22, 22, 22, 22,
                                        23, 23, 23, 23, 23, 24, 24, 24, 24, 24};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
                                        15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
                                        17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
                                        19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
                                        21, 21, 21, 21, 21, 22, 22, 22, 22, 22,
                                        23, 23, 23, 23, 23, 24, 24, 24, 24, 24};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
                                        15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
                                        17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
                                        19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
                                        21, 21, 21, 21, 21, 22, 22, 22, 22, 22,
                                        23, 23, 23, 23, 23, 24, 24, 24, 24, 24};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
                                        15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
                                        17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
                                        19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
                                        21, 21, 21, 21, 21, 22, 22, 22, 22, 22,
                                        23, 23, 23, 23, 23, 24, 24, 24, 24, 24};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),
                                        FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f),
                                        FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(13.0f), FLOAT16(13.0f), FLOAT16(13.0f), FLOAT16(13.0f),
                                        FLOAT16(14.0f), FLOAT16(14.0f), FLOAT16(14.0f), FLOAT16(14.0f), FLOAT16(14.0f),
                                        FLOAT16(15.0f), FLOAT16(15.0f), FLOAT16(15.0f), FLOAT16(15.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(16.0f), FLOAT16(16.0f), FLOAT16(16.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(17.0f), FLOAT16(17.0f), FLOAT16(17.0f), FLOAT16(17.0f),
                                        FLOAT16(18.0f), FLOAT16(18.0f), FLOAT16(18.0f), FLOAT16(18.0f), FLOAT16(18.0f),
                                        FLOAT16(19.0f), FLOAT16(19.0f), FLOAT16(19.0f), FLOAT16(19.0f), FLOAT16(19.0f),
                                        FLOAT16(20.0f), FLOAT16(20.0f), FLOAT16(20.0f), FLOAT16(20.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(21.0f), FLOAT16(21.0f), FLOAT16(21.0f), FLOAT16(21.0f),
                                        FLOAT16(22.0f), FLOAT16(22.0f), FLOAT16(22.0f), FLOAT16(22.0f), FLOAT16(22.0f),
                                        FLOAT16(23.0f), FLOAT16(23.0f), FLOAT16(23.0f), FLOAT16(23.0f), FLOAT16(23.0f),
                                        FLOAT16(24.0f), FLOAT16(24.0f), FLOAT16(24.0f), FLOAT16(24.0f), FLOAT16(24.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3x4_to_2x3x4x5_w_b_axes_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),
                                        FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f),
                                        FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f),
                                        FLOAT16(13.0f), FLOAT16(13.0f), FLOAT16(13.0f), FLOAT16(13.0f), FLOAT16(13.0f),
                                        FLOAT16(14.0f), FLOAT16(14.0f), FLOAT16(14.0f), FLOAT16(14.0f), FLOAT16(14.0f),
                                        FLOAT16(15.0f), FLOAT16(15.0f), FLOAT16(15.0f), FLOAT16(15.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(16.0f), FLOAT16(16.0f), FLOAT16(16.0f), FLOAT16(16.0f),
                                        FLOAT16(17.0f), FLOAT16(17.0f), FLOAT16(17.0f), FLOAT16(17.0f), FLOAT16(17.0f),
                                        FLOAT16(18.0f), FLOAT16(18.0f), FLOAT16(18.0f), FLOAT16(18.0f), FLOAT16(18.0f),
                                        FLOAT16(19.0f), FLOAT16(19.0f), FLOAT16(19.0f), FLOAT16(19.0f), FLOAT16(19.0f),
                                        FLOAT16(20.0f), FLOAT16(20.0f), FLOAT16(20.0f), FLOAT16(20.0f), FLOAT16(20.0f),
                                        FLOAT16(21.0f), FLOAT16(21.0f), FLOAT16(21.0f), FLOAT16(21.0f), FLOAT16(21.0f),
                                        FLOAT16(22.0f), FLOAT16(22.0f), FLOAT16(22.0f), FLOAT16(22.0f), FLOAT16(22.0f),
                                        FLOAT16(23.0f), FLOAT16(23.0f), FLOAT16(23.0f), FLOAT16(23.0f), FLOAT16(23.0f),
                                        FLOAT16(24.0f), FLOAT16(24.0f), FLOAT16(24.0f), FLOAT16(24.0f), FLOAT16(24.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3, 4}, {3}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv16, data_types::i8, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_4x5_to_2x3x4x5_w_b_axes_0_1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(16.0f), FLOAT16(17.0f), FLOAT16(18.0f), FLOAT16(19.0f), FLOAT16(20.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {4, 5}, {0, 1}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                                      11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
                                        11, 12, 13, 14, 15, 11, 12, 13, 14, 15};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}

TEST(broadcast_gpu_f16, bs_fs_yx_bsv32_fsv16_3x5_to_2x3x4x5_w_b_axes_0_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f),
                                        FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 5}, {0, 2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                      11.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
                                        11, 11, 11, 11, 11, 12, 12, 12, 12, 12};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),
                                        FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f),
                                        FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),
                                        FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f),
                                        FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3x4_to_2x3x4x5_w_b_axes_0_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),
                                        FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f),
                                        FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),  FLOAT16(9.0f),
                                        FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f), FLOAT16(10.0f),
                                        FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f), FLOAT16(11.0f),
                                        FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f), FLOAT16(12.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3, 4}, {0, 3}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                      6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10,
                                        6, 7, 8, 9, 10, 6, 7, 8, 9, 10};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x5_to_2x3x4x5_w_b_axes_1_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f),
                                        FLOAT16(6.0f),  FLOAT16(7.0f),  FLOAT16(8.0f),  FLOAT16(9.0f),  FLOAT16(10.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 5}, {1, 2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                        7, 7, 7, 7, 7, 8, 8, 8, 8, 8};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x4_to_2x3x4x5_w_b_axes_1_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),  FLOAT16(7.0f),
                                        FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f),  FLOAT16(8.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 4}, {1, 3}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                      6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                      6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                      6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                                      6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                        6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2x3_to_2x3x4x5_w_b_axes_2_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),  FLOAT16(5.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),
                                        FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f),  FLOAT16(6.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2, 3}, {2, 3}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<float> golden_data = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                      1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<uint8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<int64_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<int8_t> golden_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                                        1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_5_to_2x3x4x5_w_b_axes_0_1_2) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f),
                                        FLOAT16(1.0f),  FLOAT16(2.0f),  FLOAT16(3.0f),  FLOAT16(4.0f),  FLOAT16(5.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {5}, {0, 1, 2}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_4_to_2x3x4x5_w_b_axes_0_1_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f),  FLOAT16(4.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {4}, {0, 1, 3}, golden_data);
}

TEST(broadcast_gpu_float, bfyx_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_3_to_2x3x4x5_w_b_axes_0_2_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),
                                        FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f),  FLOAT16(3.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {3}, {0, 2, 3}, golden_data);
}


TEST(broadcast_gpu_float, bfyx_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bfyx, data_types::f32, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_float, b_fs_yx_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::b_fs_yx_fsv16, data_types::f32, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_float, bs_fs_yx_bsv32_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<float> golden_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    start_broadcast_test<float>(format::bs_fs_yx_bsv32_fsv16, data_types::f32, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<uint8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<uint8_t>(format::bfyx, data_types::u8, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<int64_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int64_t>(format::bfyx, data_types::i64, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv16, data_types::i8, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_yx_fsv32_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int8_t>(format::b_fs_yx_fsv32, data_types::i8, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_int8_t, bs_fs_yx_bsv32_fsv32_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<int8_t> golden_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    start_broadcast_test<int8_t>(format::bs_fs_yx_bsv32_fsv32, data_types::i8, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_yx_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::b_fs_yx_fsv16, data_types::f16, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu_fp16, bs_fs_yx_bsv32_fsv16_2_to_2x3x4x5_w_b_axes_1_2_3) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),  FLOAT16(1.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),
                                        FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f),  FLOAT16(2.0f)};
    start_broadcast_test<FLOAT16>(format::bs_fs_yx_bsv32_fsv16, data_types::f16, {2, 3, 4, 5}, {2}, {1, 2, 3}, golden_data);
}

TEST(broadcast_gpu, basic_error_wrong_b_axes_size) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {2, 3, 4, 5}, {0, 1, 2, 3, 4}));

    std::string msg_to_find = "Incorrect parameters configuration: broadcast_axes size should be less or equal 4.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu, basic_error_wrong_b_axis_value) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {2, 3, 4, 5}, {0, 4}));

    std::string msg_to_find = "Incorrect parameters configuration: broadcast_axes index should be within broadcast_sizes range.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu, basic_error_duplicate_b_axis_values) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {2, 3, 4, 5}, {0, 1, 1}));

    std::string msg_to_find = "Incorrect parameters configuration: Duplicate axes numbers was found in broadcast_axes.";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu, basic_error_wrong_input_dimension_0) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {2, 3, 4, 5}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {2, 3, 4, 5}, {1}));

    std::string msg_to_find = "Input size on dimension number 0(=2) is not equal to: (=1)";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu, basic_error_not_dividable_2x3x4x5_to_3x3x4x5) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {2, 3, 4, 5}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {3, 3, 4, 5}, {}));

    std::string msg_to_find = "Invalid broadcast size: not dividable by input size";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu, basic_error_not_dividable_3_to_2x3x4x5_w_b_axes_0x1x3) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 3, 1}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {2, 3, 4, 5}, {0, 1, 3}));

    std::string msg_to_find = "Invalid broadcast size: not dividable by input size";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu, basic_error_not_dividable_4x5_to_3x4x5_w_b_axes_1) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 3, 5, 4}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(broadcast("output", "input", {2, 3, 4, 5}, {1}));

    std::string msg_to_find = "Invalid broadcast size: not dividable by input size";
    EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
}

TEST(broadcast_gpu_float, bfzyx_1_to_5_w_b_axes_0) {
    std::vector<float> golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0 };
    start_broadcast_test_5d<float>(format::bfzyx, data_types::f32, { 5 }, { 1 }, { 0 }, golden_data);
}

TEST(broadcast_gpu_float, b_fs_zyx_fsv16_1_to_5_w_b_axes_0) {
    std::vector<float> golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0 };
    start_broadcast_test_5d<float>(format::b_fs_zyx_fsv16, data_types::f32, { 5 }, { 1 }, { 0 }, golden_data);
}


TEST(broadcast_gpu_uint8_t, bfzyx_1_to_5_w_b_axes_0) {
    std::vector<uint8_t> golden_data = { 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<uint8_t>(format::bfzyx, data_types::u8, { 5 }, { 1 }, { 0 }, golden_data);
}

TEST(broadcast_gpu_int64_t, bfzyx_1_to_5_w_b_axes_0) {
    std::vector<int64_t> golden_data = { 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<int64_t>(format::bfzyx, data_types::i64, { 5 }, { 1 }, { 0 }, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv16_1_to_5_w_b_axes_0) {
    std::vector<int8_t> golden_data = { 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv16, data_types::i8, { 5 }, { 1 }, { 0 }, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv32_1_to_5_w_b_axes_0) {
    std::vector<int8_t> golden_data = { 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv32, data_types::i8, { 5 }, { 1 }, { 0 }, golden_data);
}


TEST(broadcast_gpu_fp16, b_fs_zyx_fsv16_1_to_5_w_b_axes_0) {
    std::vector<FLOAT16> golden_data = { FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f) };
    start_broadcast_test_5d<FLOAT16>(format::b_fs_zyx_fsv16, data_types::f16, { 5 }, { 1 }, { 0 }, golden_data);
}

TEST(broadcast_gpu_float, bfzyx_1_to_4x5_w_b_axes_0x1) {
    std::vector<float> golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    start_broadcast_test_5d<float>(format::bfzyx, data_types::f32, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_float, b_fs_zyx_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<float> golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    start_broadcast_test_5d<float>(format::b_fs_zyx_fsv16, data_types::f32, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfzyx_1_to_4x5_w_b_axes_0x1) {
    std::vector<uint8_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1 };
    start_broadcast_test_5d<uint8_t>(format::bfzyx, data_types::u8, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_int64_t, bfzyx_1_to_4x5_w_b_axes_0x1) {
    std::vector<int64_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1 };
    start_broadcast_test_5d<int64_t>(format::bfzyx, data_types::i64, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<int8_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1 };
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv16, data_types::i8, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv32_1_to_4x5_w_b_axes_0x1) {
    std::vector<int8_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1 };
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv32, data_types::i8, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_zyx_fsv16_1_to_4x5_w_b_axes_0x1) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)};
    start_broadcast_test_5d<FLOAT16>(format::b_fs_zyx_fsv16, data_types::f16, { 4, 5 }, { 1 }, { 0, 1 }, golden_data);
}

TEST(broadcast_gpu_float, bfyx_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<float> golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    start_broadcast_test_5d<float>(format::bfzyx, data_types::f32, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}

TEST(broadcast_gpu_float, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<float> golden_data = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    start_broadcast_test_5d<float>(format::b_fs_zyx_fsv16, data_types::f32, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}

TEST(broadcast_gpu_uint8_t, bfyx_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<uint8_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<uint8_t>(format::bfzyx, data_types::u8, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}

TEST(broadcast_gpu_int64_t, bfyx_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<int64_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<int64_t>(format::bfzyx, data_types::i64, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<int8_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv16, data_types::i8, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}

TEST(broadcast_gpu_int8_t, b_fs_zyx_fsv32_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<int8_t> golden_data = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    start_broadcast_test_5d<int8_t>(format::b_fs_zyx_fsv32, data_types::i8, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}

TEST(broadcast_gpu_fp16, b_fs_zyx_fsv16_1_to_2x3x4x5x2_w_b_axes_0x1x2x3x4) {
    std::vector<FLOAT16> golden_data = {FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
                                        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f) };
    start_broadcast_test_5d<FLOAT16>(format::b_fs_zyx_fsv16, data_types::f16, { 2, 3, 4, 5, 2 }, { 1 }, { 0, 1, 2, 3, 4 }, golden_data);
}
