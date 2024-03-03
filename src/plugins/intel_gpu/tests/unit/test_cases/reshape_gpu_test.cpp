// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/activation.hpp>

#include "reshape_inst.h"

using namespace cldnn;
using namespace ::tests;
using namespace testing;

namespace {
void verify_float(const float& output_value, const float& value) {
    ASSERT_FLOAT_EQ(output_value, value);
}

void verify_int(const int32_t& output_value, const int32_t& value) {
    ASSERT_EQ(output_value, value);
}

template <class ElemType>
void generic_reshape_test(format fmt, tensor const& input_size, tensor const& reshape_size,
    bool /* in_place */, padding const& input_padd = padding(),
    padding const& output_padd = padding(), bool is_caching_test = false) {
    auto& engine = get_test_engine();

    //allocate input memory
    auto data_type = data_types::f32;
    if (std::is_same<ElemType, ov::float16>::value)
        data_type = data_types::f16;
    else if (std::is_same<ElemType, int8_t>::value)
        data_type = data_types::i8;
    else if (std::is_same<ElemType, int32_t>::value)
        data_type = data_types::i32;
    else if (std::is_same<ElemType, int64_t>::value)
        data_type = data_types::i64;

    auto input = engine.allocate_memory({data_type, fmt, input_size});

    {
        cldnn::mem_lock<ElemType> input_ptr(input, get_test_stream());
        auto input_itr = input_ptr.begin();

        auto elements = input_size.count();

        int value = 1;
        for (size_t i = 0; i < elements; ++i)
            *input_itr++ = (ElemType)value++;
    }

    topology tpl;
    std::string reshape_input = "input";

    tpl.add(input_layout("input", input->get_layout()));
    if (input_padd) {
        auto padded_input_layout = input->get_layout();
        padded_input_layout.data_padding = input_padd;
        tpl.add(reorder("reorder", input_info("input"), padded_input_layout));
        reshape_input = "reorder";
    }
    tpl.add(reshape("reshape", reshape_input, reshape_size, cldnn::reshape::reshape_mode::base, output_padd));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{reshape_input, "reshape"}));

    cldnn::network::ptr net = get_network(engine, tpl, config, get_test_stream_ptr(), is_caching_test);
    net->set_input_data("input", input);
    auto outputs = net->execute();

    ASSERT_TRUE(outputs.size() == 2 && outputs.count("reshape") == 1 && outputs.count(reshape_input) == 1);
    auto net_input = outputs.at(reshape_input).get_memory();
    auto output = outputs.at("reshape").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);        //reshape should not change data_type
    ASSERT_TRUE(output->get_layout().format.value == input->get_layout().format.value);  //reshape should not change format

    //output size should be equal to requested plus output padding
    ASSERT_TRUE(output->get_layout().get_tensor() == reshape_size);
    ASSERT_TRUE(output->get_layout().get_buffer_size() == reshape_size.add(output_padd.lower_size()).add(output_padd.upper_size()));

    {
        cldnn::mem_lock<const ElemType> output_ptr(output, get_test_stream());
        auto output_itr = output_ptr.begin();

        auto sizes = reshape_size.sizes(fmt);
        auto lower = output_padd.lower_size().sizes(fmt);
        auto upper = output_padd.upper_size().sizes(fmt);
        auto buffer_sizes = sizes;
        int32_t accum = 1;
        for (size_t i = 1; i <= sizes.size(); ++i) {
            buffer_sizes[sizes.size() - i] = accum;
            accum *= lower[sizes.size() - i] + sizes[sizes.size() - i] + upper[sizes.size() - i];
        }

        int value = 1;

        output_itr += lower[0] * buffer_sizes[0];
        for (int d1 = 0; d1 < sizes[0]; ++d1) {
            output_itr += lower[1] * buffer_sizes[1];
            for (int d2 = 0; d2 < sizes[1]; ++d2) {
                output_itr += lower[2] * buffer_sizes[2];
                for (int d3 = 0; d3 < sizes[2]; ++d3) {
                    output_itr += lower[3] * buffer_sizes[3];
                    for (int d4 = 0; d4 < sizes[3]; ++d4) {
                        auto& output_value = *output_itr;
                        ++output_itr;
                        if (data_type == data_types::f16 || data_type == data_types::f32)
                            verify_float(static_cast<float>(output_value), static_cast<float>((ElemType)value));
                        else
                            verify_int(static_cast<int32_t>(output_value), static_cast<int32_t>(value));
                        ++value;
                    }

                    output_itr += upper[3] * buffer_sizes[3];
                }

                output_itr += upper[2] * buffer_sizes[2];
            }

            output_itr += upper[1] * buffer_sizes[1];
        }
    }
}
}  // namespace

TEST(reshape_gpu_f32, basic_2dim_in_place) {
    generic_reshape_test<float>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 4, 1),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_in_place) {
    generic_reshape_test<ov::float16>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_i8, basic_2dim_in_place) {
    generic_reshape_test<int8_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_i32, basic_2dim_in_place) {
    generic_reshape_test<int32_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_i64, basic_2dim_in_place) {
    generic_reshape_test<int64_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_in_place) {
    generic_reshape_test<float>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(27, 2, 3, 4),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_in_place) {
    generic_reshape_test<ov::float16>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshape_gpu_i32, basic_4dim_in_place) {
    generic_reshape_test<int32_t>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshape_gpu_i64, basic_4dim_in_place) {
    generic_reshape_test<int64_t>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshpape_gpu_f32, basic_2dim_output_padd) {
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 4, 2),
        tensor(1, 1, 8, 1),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 1, 1}));
}

TEST(reshape_gpu_f16, basic_2dim_output_padd) {
    generic_reshape_test<ov::float16>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_i8, basic_2dim_output_padd) {
    generic_reshape_test<int8_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_i32, basic_2dim_output_padd) {
    generic_reshape_test<int32_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_i64, basic_2dim_output_padd) {
    generic_reshape_test<int64_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_f32, basic_2dim_input_padd) {
    generic_reshape_test<float>(
        format::fyxb,
        tensor(1, 1, 2, 5),
        tensor(1, 1, 5, 2),
        false,
        padding({0, 0, 3, 2}, {0, 0, 1, 4}));
}

TEST(reshape_gpu_f16, basic_2dim_input_padd) {
    generic_reshape_test<ov::float16>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_i8, basic_2dim_input_padd) {
    generic_reshape_test<int8_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_i32, basic_2dim_input_padd) {
    generic_reshape_test<int32_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_i64, basic_2dim_input_padd) {
    generic_reshape_test<int64_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_f32, basic_2dim_input_output_padd) {
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshape_gpu_f16, basic_2dim_input_output_padd) {
    generic_reshape_test<ov::float16>(
        format::byxf,
        tensor(1, 1, 6, 6),
        tensor(1, 1, 3, 12),
        false,
        padding({0, 0, 1, 1}, {0, 0, 0, 0}),
        padding({0, 0, 2, 1}, {0, 0, 1, 2}));
}

TEST(reshape_gpu_i8, basic_2dim_input_output_padd) {
    generic_reshape_test<int8_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshape_gpu_i32, basic_2dim_input_output_padd) {
    generic_reshape_test<int32_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshape_gpu_i64, basic_2dim_input_output_padd) {
    generic_reshape_test<int64_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshpape_gpu_f32, basic_4dim_output_padd) {
    generic_reshape_test<float>(
        format::bfyx,
        tensor(2, 5, 7, 3),
        tensor(1, 14, 15, 1),
        false,
        padding(),
        padding({1, 0, 0, 1}, {0, 2, 3, 0}));
}

TEST(reshape_gpu_f16, basic_4dim_output_padd) {
    generic_reshape_test<ov::float16>(
        format::bfyx,
        tensor(5, 4, 2, 2),
        tensor(40, 2, 1, 1),
        false,
        padding(),
        padding({0, 2, 0, 1}, {0, 2, 3, 0}));
}

TEST(reshape_gpu_f32, basic_4dim_input_padd) {
    generic_reshape_test<float>(
        format::yxfb,
        tensor(8, 128, 3, 3),
        tensor(16, 8, 8, 9),
        false,
        padding({0, 1, 3, 3}, {0, 1, 1, 1}));
}

TEST(reshape_gpu_f16, basic_4dim_input_padd) {
    generic_reshape_test<ov::float16>(
        format::yxfb,
        tensor(2, 32, 8, 8),
        tensor(8, 128, 1, 4),
        false,
        padding({2, 2, 1, 0}, {1, 2, 2, 0}));
}

TEST(reshape_gpu_f32, basic_4dim_input_output_padd) {
    generic_reshape_test<float>(
        format::fyxb,
        tensor(8, 1024, 25, 25),
        tensor(8, 64, 100, 100),
        false,
        padding({2, 0, 2, 1}, {0, 1, 4, 0}),
        padding({1, 2, 3, 4}, {0, 4, 1, 1}));
}

TEST(reshape_gpu_f16, basic_4dim_input_output_padd) {
    generic_reshape_test<ov::float16>(
        format::byxf,
        tensor(32, 3, 227, 227),
        tensor(8, 12, 227, 227),
        false,
        padding({0, 1, 4, 4}, {0, 1, 1, 1}),
        padding({0, 29, 29, 0}, {0, 0, 0, 0}));
}

TEST(reshape_gpu_f32, basic_5dim_in_place) {
    generic_reshape_test<float>(
        format::bfzyx,
        tensor(9, 9, 2, 4, 2),
        tensor(27, 2, 1, 4, 6),
        true);
}

template <typename T>
void test_multiple_users_with_reorder(bool is_caching_test) {
    // Tests split with crop implementation
    //                                                   _ REORDER(yxfb) --> RELU(yxfb)
    //                                                  |
    //  INPUT(bfyx,2x2x1x1)--RELU(bfyx)--RESHAPE(4x1x1x1)
    //                                                  |_
    //                                                     RELU(bfyx)

    //  Input:
    //  b0f0: -1.0
    //  b0f1:  2.0
    //  b1f0: -3.0
    //  b1f1:  4.0

    //  Out1:
    //  b0f0:  0.0
    //  b0f1:  0.0
    //  b1f0:  2.0
    //  b1f1:  4.0

    //  Out2:
    //  b0f0:  0.0
    //  b0f1:  2.0
    //  b1f0:  0.0
    //  b1f1:  4.0

    auto& engine = get_test_engine();
    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 1;
    auto y_size = 1;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num))}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("relu", input_info("input"), activation_func::relu));
    topology.add(reshape("reshape", input_info("relu"), tensor(batch(4))));
    topology.add(reorder("reorder1", input_info("reshape"), format::yxfb, data_types::f32));
    topology.add(activation("relu1", input_info("reorder1"), activation_func::relu));
    topology.add(activation("relu2", input_info("reshape"), activation_func::relu));

    std::vector<T> input_vec = {-1.f, 2.f, -3.f, 4.f};
    std::vector<T> out1 = {0.f, 2.f, 0.f, 4.0f};
    std::vector<T> out2 = {0.f, 2.f, 0.f, 4.0f};
    set_values(input, input_vec);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("relu1").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("relu2").get_memory();
    cldnn::mem_lock<T> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(reshape_gpu_f32, multiple_users_with_reorder) {
    test_multiple_users_with_reorder<float>(false);
}

template <typename T>
void test_calc_output_shape(bool is_caching_test) {
    //  INPUT(bfyx,2x2x1x1) -- RESHAPE(1, 1, 0, -1)

    //  Input:
    //  b0f0: -1.0
    //  b0f1:  2.0
    //  b1f0: -3.0
    //  b1f1:  4.0
    //
    // output_shape (1, 1, 1, 4)

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {2, 2, 1, 1}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reshape("reshape", input_info("input"), tensor(1, 1, 1, -1)));

    set_values(input, {-1.f, 2.f, -3.f, 4.f});

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, input->get_layout().format);

    ASSERT_TRUE(output->get_layout().get_tensor() == tensor(1, 1, 1, 4));

    T answers[4] = {-1.f, 2.f, -3.f, 4.f};

    cldnn::mem_lock<T> output_ptr(output, get_test_stream());
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, calc_output_shape) {
    test_calc_output_shape<float>(false);
}

template <typename T>
void test_basic_bfwzyx(bool is_caching_test) {
    // input:  bfwzyx, (3, 3, 2, 2, 1, 1)
    // reshape: (1, 1, 2, 2, 3, 3), pad (0, 0, 0, 0, 0, 1)

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{data_types::f32, format::bfwzyx, tensor{batch(3), feature(3), spatial(1, 1, 2, 2)}});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reshape("reshape", input_info("input"), tensor(batch(1), feature(1), spatial(2, 2, 3, 3)), cldnn::reshape::reshape_mode::base, padding({0, 0, 0, 0, 0, 1}, 0.f)));

    // clang-format off
    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    std::vector<float> expected_out = {
        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,

        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 1.f, 2.f, 3.f,

        4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 6.f,

        7.f, 8.f, 9.f, 1.f,
        2.f, 3.f, 4.f, 5.f,
        6.f, 7.f, 8.f, 9.f,

        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
    };
    // clang-format on

    set_values(input, input_data);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, input->get_layout().format);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), expected_out.size());

    for (size_t i = 0; i < expected_out.size(); i++) {
        ASSERT_TRUE(are_equal(expected_out[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, basic_bfwzyx) {
    test_basic_bfwzyx<float>(false);
}

template <typename T>
void test_shrink_chain_partial(bool is_caching_test) {
    auto& engine = get_test_engine();
    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 1;
    auto y_size = 1;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num))}});
    auto scale_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});
    auto shift_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});

    std::vector<T> scale_vals = {0.f, 1.f, 2.f, 3.f};
    std::vector<T> scale_shifts = {5.f, 10.f, 15.f, 20.0f};
    set_values(scale_in, scale_vals);
    set_values(shift_in, scale_shifts);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("scale_in", scale_in));
    topology.add(data("shift_in", shift_in));
    topology.add(activation("relu", input_info("input"), activation_func::relu));
    topology.add(reshape("reshape", input_info("relu"), tensor(spatial(2, 2))));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32));
    topology.add(reshape("reshape1", input_info("reorder"), tensor(feature(4))));
    topology.add(eltwise("scale", { input_info("reshape1"), input_info("scale_in") }, eltwise_mode::prod));
    topology.add(eltwise("shift", { input_info("scale"), input_info("shift_in") }, eltwise_mode::sum));
    topology.add(reorder("out_reorder", input_info("shift"), format::yxfb, data_types::f32));

    std::vector<T> input_vec = {-1.f, 2.f, -3.f, 4.f};
    std::vector<T> out = {5.f, 12.f, 15.f, 32.0f};
    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("out_reorder").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out.size(); i++)
        ASSERT_EQ(output_ptr[i], out[i]) << " i=" << i;
}

TEST(reshape_gpu_f32, shrink_chain_partial) {
    test_shrink_chain_partial<float>(false);
}

template <typename T>
void test_shrink_chain_full(bool is_caching_test) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});
    auto scale_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});
    auto shift_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});

    std::vector<T> scale_vals = {0.f, 1.f, 2.f, 3.f};
    std::vector<T> scale_shifts = {5.f, 10.f, 15.f, 20.0f};
    set_values(scale_in, scale_vals);
    set_values(shift_in, scale_shifts);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("scale_in", scale_in));
    topology.add(data("shift_in", shift_in));
    topology.add(activation("relu", input_info("input"), activation_func::relu));
    topology.add(reshape("reshape", input_info("relu"), tensor(spatial(2, 2))));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32));
    topology.add(reshape("reshape1", input_info("reorder"), tensor(feature(4))));
    topology.add(eltwise("scale", { input_info("reshape1"), input_info("scale_in") }, eltwise_mode::prod));
    topology.add(eltwise("shift", { input_info("scale"), input_info("shift_in") }, eltwise_mode::sum));
    topology.add(reorder("out_reorder", input_info("shift"), format::yxfb, data_types::f32));

    std::vector<T> input_vec = {-1.f, 2.f, -3.f, 4.f};
    std::vector<T> out = {5.f, 12.f, 15.f, 32.0f};
    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("out_reorder").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out.size(); i++)
        ASSERT_EQ(output_ptr[i], out[i]) << " i=" << i;
}

TEST(reshape_gpu_f32, shrink_chain_full) {
    test_shrink_chain_full<float>(false);
}

template <typename T>
void test_shrink_chain_out(bool is_caching_test) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});
    auto scale_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});
    auto shift_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});

    std::vector<T> scale_vals = {0.f, 1.f, 2.f, 3.f};
    std::vector<T> scale_shifts = {5.f, 10.f, 15.f, 20.0f};
    set_values(scale_in, scale_vals);
    set_values(shift_in, scale_shifts);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("relu", input_info("input"), activation_func::relu));
    topology.add(reshape("reshape", input_info("relu"), tensor(spatial(2, 2))));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32));
    topology.add(reshape("reshape1", input_info("reorder"), tensor(feature(4))));

    std::vector<T> input_vec = {-1.f, 2.f, -3.f, 4.f};
    std::vector<T> out = {0.f, 2.f, 0.f, 4.0f};
    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("reshape1").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out.size(); i++)
        ASSERT_EQ(output_ptr[i], out[i]) << " i=" << i;
}

TEST(reshape_gpu_f32, shrink_chain_out) {
    test_shrink_chain_out<float>(false);
}

template <typename T>
void test_shrink_chain_partial_reorder_truncate(bool is_caching_test) {
    auto& engine = get_test_engine();
    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 1;
    auto y_size = 1;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num))}});
    auto scale_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});
    auto shift_in = engine.allocate_memory({data_types::f32, format::bfyx, { tensor(feature(4)) }});

    std::vector<T> scale_vals = {0.f, 1.f, 2.f, 3.f};
    std::vector<T> scale_shifts = {5.f, 10.f, 15.f, 20.0f};
    set_values(scale_in, scale_vals);
    set_values(shift_in, scale_shifts);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("scale_in", scale_in));
    topology.add(data("shift_in", shift_in));
    topology.add(activation("relu", input_info("input"), activation_func::relu));
    topology.add(reshape("reshape", input_info("relu"), tensor(spatial(2, 2))));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32, {}, reorder_mean_mode::subtract, padding(), true));
    topology.add(reshape("reshape1", input_info("reorder"), tensor(feature(4))));
    topology.add(eltwise("scale", { input_info("reshape1"), input_info("scale_in") }, eltwise_mode::prod));
    topology.add(eltwise("shift", { input_info("scale"), input_info("shift_in") }, eltwise_mode::sum));
    topology.add(reorder("out_reorder", input_info("shift"), format::yxfb, data_types::f32));

    std::vector<T> input_vec = {-1.f, 2.f, -3.f, 4.f};
    std::vector<T> out = {5.f, 12.f, 15.f, 32.0f};
    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("out_reorder").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out.size(); i++)
        ASSERT_EQ(output_ptr[i], out[i]) << " i=" << i;
}

TEST(reshape_gpu_f32, shrink_chain_partial_reorder_truncate) {
    test_shrink_chain_partial_reorder_truncate<float>(false);
}

TEST(reshape_gpu_f32, basic_runtime_static_shape) {
    // input:  bfwzyx, (3, 3, 2, 2, 1, 1)
    // reshape: (1, 1, 2, 2, 3, 3), pad (0, 0, 0, 0, 0, 1)

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ov::PartialShape{3, 3, 2, 2, 1, 1}, data_types::f32, format::bfwzyx});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(shape_of("shape_of_input", input_info("input"), data_types::i32));
    topology.add(reduce("reduced_shape", input_info("shape_of_input"), reduce_mode::prod, {0}, true));
    topology.add(reshape("reshape", input_info("input"), input_info("reduced_shape"), false, ov::PartialShape::dynamic(1)));

    // clang-format off
    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    // clang-format on

    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, format::bfyx);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), input_data.size());

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_TRUE(are_equal(input_data[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, basic_runtime_dynamic_shape) {
    // input:  bfwzyx, (3, 3, 2, 2, 1, 1)
    // reshape: (1, 1, 2, 2, 3, 3), pad (0, 0, 0, 0, 0, 1)

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ov::PartialShape{3, 3, 2, 2, 1, 1}, data_types::f32, format::bfwzyx});

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(6), data_types::f32, format::bfwzyx }));
    topology.add(shape_of("shape_of_input", input_info("input"), data_types::i32));
    topology.add(reduce("reduced_shape", input_info("shape_of_input"), reduce_mode::prod, {0}, true));
    topology.add(reshape("reshape", input_info("input"), input_info("reduced_shape"), false, ov::PartialShape::dynamic(1)));

    // clang-format off
    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    // clang-format on

    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, format::bfyx);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), input_data.size());

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_TRUE(are_equal(input_data[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, basic_runtime_dynamic_shape_with_const) {
    // input:  bfwzyx, (3, 3, 2, 2, 1, 1)
    // reshape: (1, 1, 2, 2, 3, 3), pad (0, 0, 0, 0, 0, 1)

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ov::PartialShape{3, 3, 2, 2, 1, 1}, data_types::f32, format::bfwzyx});
    auto const_shape = engine.allocate_memory({ov::PartialShape{2}, data_types::i32, format::bfyx});

    set_values<int32_t>(const_shape, {-1, 3});

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(6), data_types::f32, format::bfwzyx}));
    topology.add(data("const", const_shape));
    topology.add(reshape("reshape", input_info("input"), input_info("const"), false, ov::PartialShape::dynamic(1)));

    // clang-format off
    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    // clang-format on

    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, format::bfyx);
    ASSERT_TRUE(output->get_layout().is_static());
    std::vector<int32_t> ref_dims = {12, 3, 1, 1};
    ASSERT_EQ(output->get_layout().get_dims(), ref_dims);
    ov::PartialShape ref_pshape = {12, 3};
    ASSERT_EQ(output->get_layout().get_partial_shape(), ref_pshape);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), input_data.size());

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_TRUE(are_equal(input_data[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, basic_runtime_dynamic_shape_with_const_optimized_out) {
    // input:  bfwzyx, (3, 3, 2, 2, 1, 1)
    // reshape: (1, 1, 2, 2, 3, 3), pad (0, 0, 0, 0, 0, 1)

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ov::PartialShape{3, 3, 2, 2, 1, 1}, data_types::f32, format::bfwzyx});
    auto const_shape = engine.allocate_memory({ov::PartialShape{2}, data_types::i32, format::bfyx});

    set_values<int32_t>(const_shape, {-1, 3});

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(6), data_types::f32, format::bfwzyx}));
    topology.add(data("const", const_shape));
    topology.add(reshape("reshape", input_info("input"), input_info("const"), false, ov::PartialShape::dynamic(2)));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f32));

    // clang-format off
    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    // clang-format on

    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.at("reorder").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, format::bfyx);
    ASSERT_TRUE(output->get_layout().is_static());
    std::vector<int32_t> ref_dims = {12, 3, 1, 1};
    ASSERT_EQ(output->get_layout().get_dims(), ref_dims);
    ov::PartialShape ref_pshape = {12, 3};
    ASSERT_EQ(output->get_layout().get_partial_shape(), ref_pshape);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), input_data.size());

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_TRUE(are_equal(input_data[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, basic_dynamic_shape_to_static_optimized_out) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ov::PartialShape{2, 10}, data_types::f32, format::bfyx});
    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}));
    topology.add(reshape("reshape", input_info("input"), false, {2, 10}, {2, 10}));
    topology.add(reduce("reduce", input_info("reshape"), reduce_mode::max, {1}, true));

    // clang-format off
    std::vector<float> input_data = {
        0.0, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        0.0, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };
    // clang-format on

    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_TRUE(network.get_primitive("reshape")->can_be_optimized());

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, format::bfyx);
    ASSERT_TRUE(output->get_layout().is_static());
    ov::PartialShape expected_shape = {2, 1};
    ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    std::vector<float> expected_res = {9.f, 9.f};
    ASSERT_EQ(output_ptr.size(), expected_res.size());


    for (size_t i = 0; i < expected_res.size(); i++) {
        ASSERT_EQ(expected_res[i], output_ptr[i]);
    }
}

TEST(reshape_gpu_f32, basic_runtime_dynamic_shape_activation_fusion) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ov::PartialShape{3, 3, 2, 2, 1, 1}, data_types::f32, format::bfwzyx});

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(6), data_types::f32, format::bfwzyx }));
    topology.add(reorder("input_reorder", input_info("input"), format::bfwzyx, data_types::f16));
    topology.add(shape_of("shape_of_input", input_info("input"), data_types::i32));
    topology.add(reduce("reduced_shape", input_info("shape_of_input"), reduce_mode::prod, {0}, true));
    topology.add(reshape("reshape", input_info("input_reorder"), input_info("reduced_shape"), false, ov::PartialShape::dynamic(1)));
    topology.add(activation("activation", input_info("reshape"), activation_func::pow, {2.0f, 0.0f}));
    topology.add(reorder("output_reorder", input_info("activation"), format::bfyx, data_types::f32));

    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output_reorder");

    auto output = outputs.at("output_reorder").get_memory();

    ASSERT_TRUE(network.get_primitive("reshape")->can_be_optimized());

    ASSERT_EQ(output->get_layout().data_type, input->get_layout().data_type);
    ASSERT_EQ(output->get_layout().format, format::bfyx);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), input_data.size());

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_TRUE(are_equal(input_data[i] * input_data[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, reshape_reorder_trucation_mode)
{
    auto& engine = get_test_engine();
    const int b = 1;
    const int f = 4;
    const int x = 2;
    const int y = 2;

    const int f_reshape = 1;
    const int w_reshape = 2;
    const int z_reshape = 2;

    std::vector<uint16_t> permute_order = { 0, 1, 4, 5, 3, 2 };

    auto input_size = cldnn::tensor(batch(b), feature(f), spatial(x, y));
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, input_size });
    std::vector<float> input_data = {
        0.f, 0.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 1.f,
        2.f, 2.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 3.f
    };

    std::vector<float> expected_out = {
        0.f, 2.f, 1.f, 3.f,
        0.f, 2.f, 1.f, 3.f,
        0.f, 2.f, 1.f, 3.f,
        0.f, 2.f, 1.f, 3.f
    };

    set_values(input_mem, input_data);

    topology topology(
        input_layout("input", input_mem->get_layout()),
        reorder("input_6d", input_info("input"), { data_types::f32, format::bfwzyx, cldnn::tensor(batch(b), feature(f), spatial(x, y)) }),
        activation("relu", input_info("input_6d"), activation_func::relu),
        reshape("reshape_4_to_6", input_info("relu"), cldnn::tensor(batch(b), feature(f_reshape), spatial(x, y, z_reshape, w_reshape))),
        reorder("reorder_i32", input_info("reshape_4_to_6"), format::bfwzyx, data_types::i32, {}, reorder_mean_mode::subtract, padding(), true),
        permute("permute", input_info("reorder_i32"), permute_order),
        reshape("reshape_6_to_4", input_info("permute"), cldnn::tensor(batch(b), feature(f), spatial(x, y))),
        reorder("output_4d", input_info("reshape_6_to_4"), { data_types::f32, format::bfyx, cldnn::tensor(batch(b), feature(f), spatial(x, y)) })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    EXPECT_NO_THROW(network.get_primitive_info("reorder_i32")); // To check whether the reoder node is not moved in front of reshape

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output_4d");

    auto output = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i)
    {
        ASSERT_EQ(expected_out[i], output_ptr[i]);
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(reshape_gpu_f32, basic_2dim_in_place_cached) {
    generic_reshape_test<float>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 4, 1),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_in_place_cached) {
    generic_reshape_test<ov::float16>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_i8, basic_2dim_in_place_cached) {
    generic_reshape_test<int8_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_i32, basic_2dim_in_place_cached) {
    generic_reshape_test<int32_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_i64, basic_2dim_in_place_cached) {
    generic_reshape_test<int64_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_in_place_cached) {
    generic_reshape_test<float>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(27, 2, 3, 4),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_in_place_cached) {
    generic_reshape_test<ov::float16>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_i32, basic_4dim_in_place_cached) {
    generic_reshape_test<int32_t>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_i64, basic_4dim_in_place_cached) {
    generic_reshape_test<int64_t>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshpape_gpu_f32, basic_2dim_output_padd_cached) {
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 4, 2),
        tensor(1, 1, 8, 1),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 1, 1}),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_output_padd_cached) {
    generic_reshape_test<ov::float16>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}),
        true);
}

TEST(reshape_gpu_i8, basic_2dim_output_padd_cached) {
    generic_reshape_test<int8_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}),
        true);
}

TEST(reshape_gpu_i32, basic_2dim_output_padd_cached) {
    generic_reshape_test<int32_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}),
        true);
}

TEST(reshape_gpu_i64, basic_2dim_output_padd_cached) {
    generic_reshape_test<int64_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}),
        true);
}

TEST(reshape_gpu_f32, basic_2dim_input_padd_cached) {
    generic_reshape_test<float>(
        format::fyxb,
        tensor(1, 1, 2, 5),
        tensor(1, 1, 5, 2),
        false,
        padding({0, 0, 3, 2}, {0, 0, 1, 4}),
        padding(),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_input_padd_cached) {
    generic_reshape_test<ov::float16>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}),
        padding(),
        true);
}

TEST(reshape_gpu_i8, basic_2dim_input_padd_cached) {
    generic_reshape_test<int8_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}),
        padding(),
        true);
}

TEST(reshape_gpu_i32, basic_2dim_input_padd_cached) {
    generic_reshape_test<int32_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}),
        padding(),
        true);
}

TEST(reshape_gpu_i64, basic_2dim_input_padd_cached) {
    generic_reshape_test<int64_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}),
        padding(),
        true);
}

TEST(reshape_gpu_f32, basic_2dim_input_output_padd_cached) {
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_input_output_padd_cached) {
    generic_reshape_test<ov::float16>(
        format::byxf,
        tensor(1, 1, 6, 6),
        tensor(1, 1, 3, 12),
        false,
        padding({0, 0, 1, 1}, {0, 0, 0, 0}),
        padding({0, 0, 2, 1}, {0, 0, 1, 2}),
        true);
}

TEST(reshape_gpu_i8, basic_2dim_input_output_padd_cached) {
    generic_reshape_test<int8_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}),
        true);
}

TEST(reshape_gpu_i32, basic_2dim_input_output_padd_cached) {
    generic_reshape_test<int32_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}),
        true);
}

TEST(reshape_gpu_i64, basic_2dim_input_output_padd_cached) {
    generic_reshape_test<int64_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}),
        true);
}

TEST(reshpape_gpu_f32, basic_4dim_output_padd_cached) {
    generic_reshape_test<float>(
        format::bfyx,
        tensor(2, 5, 7, 3),
        tensor(1, 14, 15, 1),
        false,
        padding(),
        padding({1, 0, 0, 1}, {0, 2, 3, 0}),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_output_padd_cached) {
    generic_reshape_test<ov::float16>(
        format::bfyx,
        tensor(5, 4, 2, 2),
        tensor(40, 2, 1, 1),
        false,
        padding(),
        padding({0, 2, 0, 1}, {0, 2, 3, 0}),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_input_padd_cached) {
    generic_reshape_test<float>(
        format::yxfb,
        tensor(8, 128, 3, 3),
        tensor(16, 8, 8, 9),
        false,
        padding({0, 1, 3, 3}, {0, 1, 1, 1}),
        padding(),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_input_padd_cached) {
    generic_reshape_test<ov::float16>(
        format::yxfb,
        tensor(2, 32, 8, 8),
        tensor(8, 128, 1, 4),
        false,
        padding({2, 2, 1, 0}, {1, 2, 2, 0}),
        padding(),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_input_output_padd_cached) {
    generic_reshape_test<float>(
        format::fyxb,
        tensor(8, 1024, 25, 25),
        tensor(8, 64, 100, 100),
        false,
        padding({2, 0, 2, 1}, {0, 1, 4, 0}),
        padding({1, 2, 3, 4}, {0, 4, 1, 1}),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_input_output_padd_cached) {
    generic_reshape_test<ov::float16>(
        format::byxf,
        tensor(32, 3, 227, 227),
        tensor(8, 12, 227, 227),
        false,
        padding({0, 1, 4, 4}, {0, 1, 1, 1}),
        padding({0, 29, 29, 0}, {0, 0, 0, 0}),
        true);
}

TEST(reshape_gpu_f32, basic_5dim_in_place_cached) {
    generic_reshape_test<float>(
        format::bfzyx,
        tensor(9, 9, 2, 4, 2),
        tensor(27, 2, 1, 4, 6),
        true,
        padding(),
        padding(),
        true);
}

TEST(reshape_gpu_f32, multiple_users_with_reorder_cached) {
    test_multiple_users_with_reorder<float>(true);
}

TEST(reshape_gpu_f32, calc_output_shape_cached) {
    test_calc_output_shape<float>(true);
}

TEST(reshape_gpu_f32, basic_bfwzyx_cached) {
    test_basic_bfwzyx<float>(true);
}

TEST(reshape_gpu_f32, shrink_chain_partial_cached) {
    test_shrink_chain_partial<float>(true);
}

TEST(reshape_gpu_f32, shrink_chain_full_cached) {
    test_shrink_chain_full<float>(true);
}
#endif
TEST(reshape_gpu_f32, shrink_chain_out_cached) {
    test_shrink_chain_out<float>(true);
}
