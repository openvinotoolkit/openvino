// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/arg_max_min.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <arg_max_min_inst.h>
#include "arg_max_min/arg_max_min_kernel_axis.h"
#include "arg_max_min/arg_max_min_kernel_topk_radix.h"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"

#include "program_wrapper.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>

using namespace cldnn;
using namespace ::tests;

template <format::type layoutFormat, typename DataType>
struct arg_max_input_types {
    static const auto format = layoutFormat;
    using input_type = DataType;
};

template <typename ArgMaxInput>
struct argmax_gpu_test : public testing::Test {
    static const auto format = ArgMaxInput::format;
    using input_type = typename ArgMaxInput::input_type;
    const data_types data_type = ov::element::from<input_type>();
    std::vector<input_type> getTypedVector(const std::vector<float>& input) {
        return std::vector<input_type>(input.begin(), input.end());
    }

    void checkOutput(std::shared_ptr<memory> mem, size_t out_size) {
        cldnn::mem_lock<input_type> out_ptr(mem, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            float out_value = get_value<input_type>(out_ptr.data(), i);
            ASSERT_EQ(out_value, i < (out_size / 2) ? 0 : 1);
        }
    }
};

using format_types = testing::Types<arg_max_input_types<format::bfyx, float>,
                                    arg_max_input_types<format::b_fs_yx_fsv16, float>,
                                    arg_max_input_types<format::b_fs_yx_fsv32, float>,
                                    arg_max_input_types<format::bs_fs_yx_bsv16_fsv16, float>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv16, float>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv32, float>,
                                    arg_max_input_types<format::bfyx, int32_t>,
                                    arg_max_input_types<format::b_fs_yx_fsv16, int32_t>,
                                    arg_max_input_types<format::b_fs_yx_fsv32, int32_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv16_fsv16, int32_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv16, int32_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv32, int32_t>,
                                    arg_max_input_types<format::bfyx, ov::float16>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv16, ov::float16>,
                                    arg_max_input_types<format::bfyx, int8_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv16, int8_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv32, int8_t>,
                                    arg_max_input_types<format::bfyx, uint8_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv16, uint8_t>,
                                    arg_max_input_types<format::bs_fs_yx_bsv32_fsv32, uint8_t>>;

TYPED_TEST_SUITE(argmax_gpu_test, format_types);

// Helper trait to check for uint8_t input_type
template<typename T>
struct is_uint8_input : std::false_type {};

template<format::type Fmt>
struct is_uint8_input<arg_max_input_types<Fmt, uint8_t>> : std::true_type {};

TYPED_TEST(argmax_gpu_test, base) {
    //  Input  : 2x4x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({this->data_type, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reordered_input", input_info("input"), this->format, this->data_type));
    topology.add(arg_max_min("arg_max", { input_info("reordered_input") }, ov::op::TopKMode::MIN, top_k, 0));
    topology.add(reorder("plane_arg_max", input_info("arg_max"), format::bfyx, this->data_type));

    std::vector<float> input_vec = {// y0x0 y0x1 y1x0 y1x1
                                    /*b0f0*/ 0.1f, -0.1f, 0.9f,  1.5f,
                                    /*b0f1*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b0f2*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b0f3*/ 0.2f, 0.2f,  -10.f, 4.2f,

                                    /*b1f0*/ 3.f,  0.5f,  7.f,   10.f,
                                    /*b1f1*/ 4.f,  0.5f,  8.f,   8.2f,
                                    /*b1f2*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b1f3*/ 4.f,  0.5f,  8.f,   8.2f};

    // Positive values for u8 input type test
    std::vector<float> input_vec_u8 = {// y0x0 y0x1 y1x0 y1x1
                                    /*b0f0*/ 0.1f, 0.1f, 0.9f,  1.5f,
                                    /*b0f1*/ 0.2f, 0.2f,  0.1f, 5.2f,
                                    /*b0f2*/ 0.2f, 0.2f,  0.1f, 5.2f,
                                    /*b0f3*/ 0.2f, 0.2f,  0.1f, 4.2f,

                                    /*b1f0*/ 3.f,  0.5f,  7.f,   10.f,
                                    /*b1f1*/ 4.f,  0.5f,  8.f,   8.2f,
                                    /*b1f2*/ 0.2f, 0.2f,  0.1f, 5.2f,
                                    /*b1f3*/ 4.f,  0.5f,  8.f,   8.2f};

    // If format is of type u8 then use non negative values as input.
    if (is_uint8_input<TypeParam>::value) {
        set_values(input, this->getTypedVector(input_vec_u8));
    } else {
        set_values(input, this->getTypedVector(input_vec));
    }

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("plane_arg_max").get_memory();

    this->checkOutput(output, out_size);
}

// TODO: extend test with layouts to 3d case in scope of arg_max operation layouts support
TEST(arg_max_gpu_min_axis_batch_bfzyx, i32) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 1, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input =
        engine.allocate_memory({data_types::f32, format::bfzyx, {batch_num, feature_num, x_size, y_size, z_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MIN,
                             top_k,
                             0,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::i32));

    std::vector<float> input_vec = {// y0x0 y0x1 y1x0 y1x1
                                    /*b0f0*/ 0.1f, -0.1f, 0.9f,  1.5f,
                                    /*b0f1*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b0f2*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b0f3*/ 0.2f, 0.2f,  -10.f, 4.2f,

                                    /*b1f0*/ 3.f,  0.5f,  7.f,   10.f,
                                    /*b1f1*/ 4.f,  0.5f,  8.f,   8.2f,
                                    /*b1f2*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b1f3*/ 4.f,  0.5f,  8.f,   8.2f};

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output, get_test_stream());
    int32_t out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<int32_t>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             2,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::f32));

    std::vector<float> input_vec = {0.1f, -0.1f, 0.9f,  1.5f, 0.2f, 0.2f, -10.f, 5.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 0.2f, 0.2f, -10.f, 4.2f,

                                    3.f,  0.5f,  7.f,   10.f, 4.f,  0.5f, 8.f,   8.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 4.f,  0.5f, 8.f,   8.2f};

    std::vector<float> ref_vec = {
        1.f,
        1.f,
        1.f,
        1.f,
        1.f,
        1.f,
        1.f,
        1.f,

        0.f,
        0.f,
        0.f,
        0.f,
        1.f,
        1.f,
        1.f,
        1.f,
    };

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_batch_yxfb, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             0,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::f32));

    std::vector<float> input_vec = {0.1f, -0.1f, 0.9f,  1.5f, 0.2f, 0.2f, -10.f, 5.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 0.2f, 0.2f, -10.f, 4.2f,

                                    3.f,  0.5f,  7.f,   10.f, 4.f,  0.5f, 8.f,   8.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 4.f,  0.5f, 8.f,   8.2f};

    std::vector<float> ref_vec = {0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,

                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f};

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             2,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::f32));

    std::vector<float> input_vec = {0.1f, -0.1f, 0.9f,  1.5f, 0.2f, 0.2f, -10.f, 5.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 0.2f, 0.2f, -10.f, 4.2f,

                                    3.f,  0.5f,  7.f,   10.f, 4.f,  0.5f, 8.f,   8.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 4.f,  0.5f, 8.f,   8.2f};

    std::vector<float> ref_vec = {
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

        0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f,

        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

        1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
    };

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(top_k_layer_tests, second_output) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    auto top_k_input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});
    auto second_output = engine.allocate_memory({data_types::f32, format::bfyx, {top_k, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max", { input_info("input"), input_info("const"), input_info("second_output") }, ov::op::TopKMode::MIN, top_k, 0));

    std::vector<float> input_vec = {// y0x0 y0x1 y1x0 y1x1
                                    /*b0f0*/ 0.1f, -0.1f, 0.9f,  1.5f,
                                    /*b0f1*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b0f2*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b0f3*/ 0.2f, 0.2f,  -10.f, 4.2f,

                                    /*b1f0*/ 3.f,  0.5f,  7.f,   10.f,
                                    /*b1f1*/ 4.f,  0.5f,  8.f,   8.2f,
                                    /*b1f2*/ 0.2f, 0.2f,  -10.f, 5.2f,
                                    /*b1f3*/ 4.f,  0.5f,  8.f,   8.2f};
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> second_output_ptr(second_output, get_test_stream());

    float out_buffer[out_size];
    float second_out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
        second_out_buffer[i] = get_value<float>(second_output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
        ASSERT_EQ(second_out_buffer[i], input_vec[i]);
    }
}

TEST(top_k_layer_tests, second_output2) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    auto top_k_input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});
    auto second_output = engine.allocate_memory({data_types::f32, format::yxfb, {top_k, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));

    topology.add(arg_max_min("arg_max",
                             { input_info("input"), input_info("const"), input_info("second_output") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             0,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::f32));

    std::vector<float> input_vec = {0.1f, -0.1f, 0.9f,  1.5f, 0.2f, 0.2f, -10.f, 5.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 0.2f, 0.2f, -10.f, 4.2f,

                                    3.f,  0.5f,  7.f,   10.f, 4.f,  0.5f, 8.f,   8.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 4.f,  0.5f, 8.f,   8.2f};

    std::vector<float> ref_vec = {0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,

                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f,
                                  0.f,
                                  1.f};

    std::vector<float> second_ref_vec = {0.1f,
                                         1.5f,
                                         0.2f,
                                         5.2f,

                                         0.2f,
                                         5.2f,
                                         0.2f,
                                         4.2f,

                                         3.f,
                                         10.f,
                                         4.f,
                                         8.2f,

                                         0.2f,
                                         5.2f,
                                         4.f,
                                         8.2f};

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> second_output_ptr(second_output, get_test_stream());
    float out_buffer[out_size];
    float second_out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
        second_out_buffer[i] = get_value<float>(second_output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
        ASSERT_EQ(second_out_buffer[i], second_ref_vec[i]);
    }
}

TEST(top_k_layer_tests, multiple_outputs) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", {top_k_input}));
    auto arg_max_min_prim = arg_max_min("arg_max",
                                        { input_info("input"), input_info("const") },
                                        ov::op::TopKMode::MAX, top_k,
                                        0,
                                        ov::op::TopKSortType::SORT_VALUES,
                                        false,
                                        false,
                                        data_types::f32,
                                        2);
    arg_max_min_prim.output_paddings = {padding(), padding()};
    arg_max_min_prim.output_data_types = {optional_data_type{data_types::f32}, optional_data_type{data_types::f32}};
    topology.add(arg_max_min_prim);
    topology.add(permute("permute_1", input_info("arg_max", 0), {0, 1, 2, 3}));
    topology.add(permute("permute_2", input_info("arg_max", 1), {0, 1, 2, 3}));
    topology.add(concatenation("concat", { input_info("permute_1"), input_info("permute_2") }, 0));

    std::vector<float> input_vec = {
            //y0x0 y0x1 y1x0 y1x1
            /*b0f0*/0.1f, 0.2f, 0.3f,  0.4f,
            /*b0f1*/0.5f, 0.6f,  0.7f, 0.8f,
            /*b0f2*/0.9f, 1.0f,  1.1f, 1.2f,
            /*b0f3*/1.3f, 1.4f,  1.5f, 1.6f,

            /*b1f0*/2.1f, 2.2f, 2.3f, 2.4f,
            /*b1f1*/2.5f, 2.6f, 2.7f, 2.8f,
            /*b1f2*/2.9f, 3.0f, 3.1f, 3.2f,
            /*b1f3*/3.3f, 3.4f, 3.5f, 3.6f,
    };

    std::vector<float> ref_result = {
            /*indexes*/
            /*b0*/
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            /*b1*/
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            /**values*/
            /*b0*/
            2.1f, 2.2f, 2.3f, 2.4f,
            2.5f, 2.6f, 2.7f, 2.8f,
            2.9f, 3.0f, 3.1f, 3.2f,
            3.3f, 3.4f, 3.5f, 3.6f,
            /*b1*/
            0.1f, 0.2f, 0.3f,  0.4f,
            0.5f, 0.6f,  0.7f, 0.8f,
            0.9f, 1.0f,  1.1f, 1.2f,
            1.3f, 1.4f,  1.5f, 1.6f,
    };

    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "concat");
    const int out_size = y_size * feature_num * x_size * top_k * 2;
    auto output = outputs.at("concat").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_result[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, sort_by_values) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             2,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::f32));

    std::vector<float> input_vec = {0.1f, -0.1f, 0.9f,  1.5f, 0.2f, 0.2f, -10.f, 5.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 0.2f, 0.2f, -10.f, 4.2f,

                                    3.f,  0.5f,  7.f,   10.f, 4.f,  0.5f, 8.f,   8.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 4.f,  0.5f, 8.f,   8.2f};

    std::vector<float> ref_vec = {
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

        0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f,

        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

        1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f,
    };

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, sort_by_indices) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             2,
                             ov::op::TopKSortType::SORT_INDICES,
                             false,
                             false,
                             data_types::f32));

    std::vector<float> input_vec = {0.1f, -0.1f, 0.9f,  1.5f, 0.2f, 0.2f, -10.f, 5.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 0.2f, 0.2f, -10.f, 4.2f,

                                    3.f,  0.5f,  7.f,   10.f, 4.f,  0.5f, 8.f,   8.2f,

                                    0.2f, 0.2f,  -10.f, 5.2f, 4.f,  0.5f, 8.f,   8.2f};

    std::vector<float> ref_vec = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
    };

    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_large_output_size, sort_by_indices) {
    static const int32_t x_size = 1, y_size = 1, feature_num = 1, batch_num = 20000;
    auto& engine = get_test_engine();
    const int top_k = 6000;
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             0,
                             ov::op::TopKSortType::SORT_INDICES,
                             false,
                             false,
                             data_types::f32));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");

    // No data checking.  The test will fail to compile if kernel not switch to use global memory
}

TEST(arg_max_gpu_topk_axis, large_output_size_internal_buffer_sizes_match_update_path) {
    kernel_selector::arg_max_min_params params;
    params.inputs[0] = kernel_selector::DataTensor(std::vector<size_t>{20000, 1, 1, 1},
                                                   kernel_selector::Datatype::F32,
                                                   kernel_selector::DataLayout::yxfb);
    params.outputs[0] = kernel_selector::DataTensor(std::vector<size_t>{6000, 1, 1, 1},
                                                    kernel_selector::Datatype::F32,
                                                    kernel_selector::DataLayout::yxfb);
    params.argMaxMinAxis = kernel_selector::ArgMaxMinAxis::BATCH;
    params.argMaxMinOut = kernel_selector::ArgMaxMinOut::MAX;
    params.argMaxMinSortType = kernel_selector::ArgMaxMinSortType::INDEX;
    params.topK = 6000;
    params.outputs_num = 1;
    // Selector-level tests construct EngineInfo manually. We only need a non-zero
    // maxWorkGroupSize so GetOptimalLocalWorkGroupSizes() can derive valid LWS;
    // 256 is the smallest stable bucket in the selector heuristics and avoids
    // baking in 512/1024-specific behavior that is irrelevant to this test.
    params.engineInfo.maxWorkGroupSize = 256;

    kernel_selector::ArgMaxMinKernelAxis kernel;
    auto kernels_data = kernel.GetKernelsData(params);

    ASSERT_EQ(kernels_data.size(), size_t(1));
    auto& kd = kernels_data[0];
    ASSERT_FALSE(kd.internalBuffers.empty());
    ASSERT_TRUE(static_cast<bool>(kd.update_dispatch_data_func));

    const auto initial_buffers = kd.internalBuffers;
    kd.update_dispatch_data_func(params, kd);

    ASSERT_EQ(initial_buffers.size(), kd.internalBuffers.size());
    for (size_t i = 0; i < initial_buffers.size(); ++i) {
        ASSERT_EQ(initial_buffers[i].byte_count, kd.internalBuffers[i].byte_count) << "buffer index=" << i;
        ASSERT_EQ(initial_buffers[i].lockable, kd.internalBuffers[i].lockable) << "buffer index=" << i;
    }
}

TEST(arg_max_gpu_topk_axis, zero_sort_size_internal_buffers_are_clamped) {
    kernel_selector::arg_max_min_params init_params;
    init_params.inputs[0] = kernel_selector::DataTensor(std::vector<size_t>{20000, 1, 1, 1},
                                                        kernel_selector::Datatype::F32,
                                                        kernel_selector::DataLayout::yxfb);
    init_params.outputs[0] = kernel_selector::DataTensor(std::vector<size_t>{6000, 1, 1, 1},
                                                         kernel_selector::Datatype::F32,
                                                         kernel_selector::DataLayout::yxfb);
    init_params.argMaxMinAxis = kernel_selector::ArgMaxMinAxis::BATCH;
    init_params.argMaxMinOut = kernel_selector::ArgMaxMinOut::MAX;
    init_params.argMaxMinSortType = kernel_selector::ArgMaxMinSortType::INDEX;
    init_params.topK = 6000;
    init_params.outputs_num = 1;
    init_params.engineInfo.maxWorkGroupSize = 256;

    kernel_selector::arg_max_min_params zero_params = init_params;
    zero_params.inputs[0] = kernel_selector::DataTensor(std::vector<size_t>{0, 1, 1, 1},
                                                        kernel_selector::Datatype::F32,
                                                        kernel_selector::DataLayout::yxfb);
    zero_params.outputs[0] = kernel_selector::DataTensor(std::vector<size_t>{0, 1, 1, 1},
                                                         kernel_selector::Datatype::F32,
                                                         kernel_selector::DataLayout::yxfb);

    kernel_selector::ArgMaxMinKernelAxis kernel;
    auto kernels_data = kernel.GetKernelsData(init_params);

    ASSERT_EQ(kernels_data.size(), size_t(1));
    auto& kd = kernels_data[0];
    ASSERT_TRUE(static_cast<bool>(kd.update_dispatch_data_func));

    kd.update_dispatch_data_func(zero_params, kd);

    ASSERT_EQ(kd.internalBuffers.size(), size_t(3));
    for (size_t i = 0; i < kd.internalBuffers.size(); ++i) {
        ASSERT_EQ(kd.internalBuffers[i].byte_count, size_t(0)) << "buffer index=" << i;
    }
}

template <typename T>
void test_top_k_layer_tests_sort_probabilities_by_indices(bool is_caching_test) {
    static const int32_t x_size = 10, y_size = 1, feature_num = 1, batch_num = 1;
    auto& engine = get_test_engine();
    const int top_k = 5;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));

    topology.add(arg_max_min("arg_max",
                             { input_info("input") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             3,
                             ov::op::TopKSortType::SORT_VALUES,
                             false,
                             false,
                             data_types::i32));
    std::vector<T> input_vec = {0.9f, 0.1f, 0.2f, 0.8f, 0.5f, 0.6f, 0.3f, 0.4f, 0.7f, 0.95f};

    std::vector<int> ref_vec = {9, 0, 3, 8, 5};

    set_values(input, input_vec);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<int, mem_lock_type::read> output_ptr(output, get_test_stream());
    int out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<int>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(top_k_layer_tests, sort_probabilities_by_indices) {
    test_top_k_layer_tests_sort_probabilities_by_indices<float>(false);
}

TEST(export_import_top_k_layer_tests, sort_probabilities_by_indices) {
    test_top_k_layer_tests_sort_probabilities_by_indices<float>(true);
}

const std::vector<float> input_vec1 = {
    0.000109, 0.000282, 0.000112, 0.000108, 0.000154, 0.000026, 0.000103, 0.000072, 0.000138, 0.000098, 0.001701,
    0.000206, 0.000554, 0.000135, 0.000058, 0.000190, 0.000051, 0.000043, 0.000062, 0.000262, 0.000232, 0.000112,
    0.000105, 0.000107, 0.000227, 0.000104, 0.000075, 0.000076, 0.000076, 0.000143, 0.000135, 0.000073, 0.000126,
    0.000120, 0.000100, 0.000249, 0.000144, 0.000507, 0.000185, 0.000293, 0.000198, 0.000033, 0.000240, 0.000064,
    0.000128, 0.000094, 0.000114, 0.000058, 0.000148, 0.000103, 0.000262, 0.000278, 0.000123, 0.000051, 0.000058,
    0.000148, 0.000052, 0.000454, 0.000090, 0.000071, 0.000100, 0.000143, 0.000148, 0.000123, 0.000257, 0.000060,
    0.000157, 0.000159, 0.000135, 0.000155, 0.000495, 0.000246, 0.000234, 0.000387, 0.000449, 0.000261, 0.000059,
    0.000155, 0.000086, 0.000230, 0.000197, 0.000221, 0.000034, 0.000093, 0.000499, 0.000103, 0.000165, 0.000102,
    0.000435, 0.000239, 0.000061, 0.000039, 0.000047, 0.000036, 0.000224, 0.000055, 0.000041, 0.000082, 0.000129,
    0.000346, 0.000640, 0.000288, 0.000231, 0.000184, 0.000104, 0.000156, 0.000225, 0.000493, 0.000147, 0.000101,
    0.000032, 0.000222, 0.000117, 0.000246, 0.000110, 0.000106, 0.000130, 0.000127, 0.000154, 0.000136, 0.000228,
    0.000177, 0.000239, 0.000209, 0.000113, 0.000076, 0.000151, 0.000260, 0.000123, 0.000150, 0.000034, 0.000180,
    0.000111, 0.000052, 0.000096, 0.000345, 0.000095, 0.000221, 0.000171, 0.000461, 0.000080, 0.000103, 0.000081,
    0.000132, 0.000138, 0.000161, 0.000159, 0.000109, 0.000140, 0.000481, 0.000172, 0.000067, 0.000157, 0.000491,
    0.000117, 0.000070, 0.000270, 0.000156, 0.000229, 0.000184, 0.000130, 0.000049, 0.000157, 0.000144, 0.000143,
    0.000203, 0.000134, 0.000512, 0.000172, 0.000187, 0.000102, 0.000202, 0.000100, 0.000107, 0.000060, 0.000196,
    0.000109, 0.000137, 0.000270, 0.000180, 0.000124, 0.000144, 0.000098, 0.000095, 0.000272, 0.000169, 0.000269,
    0.000370, 0.000212, 0.000323, 0.000391, 0.000055, 0.000114, 0.000338, 0.000208, 0.000067, 0.000457, 0.000129,
    0.000175, 0.000295, 0.000185, 0.000269, 0.000090, 0.000094, 0.000312, 0.000112, 0.000266, 0.000493, 0.000186,
    0.000127, 0.000070, 0.000265, 0.000160, 0.000243, 0.000067, 0.000118, 0.000295, 0.000083, 0.000239, 0.000134,
    0.000161, 0.000127, 0.000179, 0.000120, 0.000180, 0.000163, 0.000043, 0.000135, 0.000170, 0.000132, 0.000291,
    0.000234, 0.000513, 0.000800, 0.000257, 0.000140, 0.000106, 0.000049, 0.000075, 0.000070, 0.000239, 0.000187,
    0.000118, 0.000056, 0.000088, 0.000152, 0.000224, 0.000124, 0.000092, 0.000218, 0.000194, 0.000259, 0.000409,
    0.000207, 0.000191, 0.000085, 0.000047, 0.000518, 0.000088, 0.000367, 0.000203, 0.000388, 0.000197, 0.000145,
    0.000207, 0.000108, 0.000150, 0.000142, 0.000332, 0.000276, 0.000434, 0.000240, 0.000139, 0.000435, 0.000170,
    0.000331, 0.000363, 0.002229, 0.000178, 0.000104, 0.000049, 0.000390, 0.000655, 0.001461, 0.000378, 0.000060,
    0.000893, 0.000110, 0.000231, 0.001369, 0.000158, 0.001266, 0.000297, 0.001158, 0.001414, 0.000334, 0.000410,
    0.000143, 0.000275, 0.000303, 0.000066, 0.000120, 0.000220, 0.000095, 0.000101, 0.000184, 0.000053, 0.000086,
    0.000212, 0.000118, 0.000215, 0.000104, 0.000072, 0.000139, 0.000076, 0.000152, 0.000083, 0.000105, 0.000329,
    0.000192, 0.000149, 0.000170, 0.000066, 0.000097, 0.000285, 0.000146, 0.000236, 0.000129, 0.000091, 0.000076,
    0.000100, 0.000134, 0.000079, 0.000125, 0.000272, 0.000185, 0.000187, 0.000086, 0.000149, 0.000045, 0.000561,
    1.813452, 0.000385, 0.000380, 0.001857, 0.000571, 0.000130, 0.000650, 0.000133, 0.000147, 0.000210, 0.000342,
    0.002329, 0.000712, 0.001396, 0.000610, 0.000405, 0.000096, 0.000120, 0.000102, 0.000091, 0.000078, 0.001877,
    0.000361, 0.000724, 0.000161, 0.000082, 0.000243, 0.000173, 0.000075, 0.000182, 0.000085, 0.000205, 0.000199,
    0.000085, 0.000040, 0.000737, 0.000237, 0.000108, 0.000219, 0.000099, 0.000156, 0.000038, 0.000059, 0.000474,
    0.000527, 0.000265, 0.000683, 0.000070, 0.000165, 0.000362, 0.000083, 0.000138, 0.000213, 0.000085, 0.000118,
    0.000165, 0.000186, 0.000181, 0.000112, 0.000119, 0.000249, 0.000402, 0.000347, 0.000110, 0.000122, 0.000293,
    0.000054, 0.000112, 0.000148, 0.000167, 0.000226, 0.000188, 0.000097, 0.000127, 0.000172, 0.000047, 0.000054,
    0.000195, 0.000239, 0.000254, 0.000175, 0.000108, 0.000123, 0.000131, 0.000102, 0.000200, 0.000088, 0.000090,
    0.000083, 0.000150, 0.000169, 0.000225, 0.000212, 0.000077, 0.000267, 0.000259, 0.000106, 0.000487, 0.000287,
    0.000262, 0.000070, 0.000187, 0.000147, 0.000272, 0.000179, 0.000127, 0.000130, 0.000079, 0.000289, 0.000094,
    0.000049, 0.000197, 0.000131, 0.000145, 0.000047, 0.000075, 0.000105, 0.000344, 0.000033, 0.000107, 0.000126,
    0.000068, 0.000123, 0.000103, 0.000120, 0.000141, 0.000078, 0.000083, 0.000079, 0.000094, 0.000096, 0.000105,
    0.000115, 0.000348, 0.000072, 0.000102, 0.000246, 0.000105, 0.000089, 0.000425, 0.000387, 0.000077, 0.000201,
    0.000121, 0.000083, 0.000234, 0.000351, 0.000328, 0.000135, 0.000080, 0.000155, 0.000061, 0.000041, 0.000289,
    0.000071, 0.000066, 0.000377, 0.000077, 0.000114, 0.000133, 0.000090, 0.000213, 0.000088, 0.000156, 0.000153,
    0.000079, 0.000155, 0.000123, 0.000268, 0.000173, 0.000050, 0.000136, 0.000153, 0.000074, 0.000106, 0.000173,
    0.000111, 0.000196, 0.000285, 0.000066, 0.000190, 0.000094, 0.000306, 0.000327, 0.000085, 0.000082, 0.000200,
    0.000602, 0.000138, 0.000207, 0.000178, 0.000101, 0.000190, 0.000152, 0.000153, 0.000088, 0.000051, 0.000141,
    0.000128, 0.000220, 0.000095, 0.000148, 0.000300, 0.000171, 0.000053, 0.000212, 0.000282, 0.000142, 0.000175,
    0.000151, 0.000084, 0.000118, 0.000205, 0.000429, 0.000044, 0.000112, 0.000107, 0.000397, 0.000087, 0.000208,
    0.000116, 0.000069, 0.000037, 0.000178, 0.000060, 0.000107, 0.000124, 0.000208, 0.000115, 0.000051, 0.000093,
    0.000150, 0.000152, 0.000104, 0.000165, 0.000189, 0.000417, 0.000081, 0.000052, 0.000027, 0.000075, 0.000158,
    0.000073, 0.000067, 0.000159, 0.000062, 0.000112, 0.000058, 0.000116, 0.000100, 0.000167, 0.000314, 0.000089,
    0.000095, 0.000126, 0.000112, 0.000074, 0.000106, 0.000129, 0.000253, 0.000252, 0.000136, 0.000107, 0.000110,
    0.000183, 0.000096, 0.000092, 0.000148, 0.000138, 0.000098, 0.000107, 0.000202, 0.000180, 0.000111, 0.000053,
    0.000145, 0.000096, 0.000113, 0.000215, 0.000124, 0.000059, 0.000093, 0.000382, 0.000133, 0.000079, 0.000097,
    0.000284, 0.000105, 0.000098, 0.000180, 0.000071, 0.000104, 0.000472, 0.000068, 0.000041, 0.000063, 0.000179,
    0.000128, 0.000169, 0.000219, 0.000110, 0.000294, 0.000199, 0.000403, 0.000189, 0.000126, 0.000209, 0.000230,
    0.000108, 0.000192, 0.000344, 0.000156, 0.000112, 0.000101, 0.000207, 0.000125, 0.000233, 0.000114, 0.000258,
    0.000174, 0.000207, 0.000112, 0.000242, 0.000272, 0.000151, 0.000107, 0.000134, 0.000147, 0.000346, 0.000040,
    0.000102, 0.000191, 0.000082, 0.000267, 0.000172, 0.000063, 0.000180, 0.000115, 0.000233, 0.000098, 0.000264,
    0.000071, 0.000120, 0.000140, 0.000160, 0.000288, 0.000028, 0.000080, 0.000084, 0.000327, 0.000091, 0.000100,
    0.000209, 0.000087, 0.000150, 0.000064, 0.000110, 0.000096, 0.000198, 0.000246, 0.000290, 0.000130, 0.000143,
    0.000130, 0.000120, 0.000283, 0.000092, 0.000186, 0.000159, 0.000181, 0.000114, 0.000058, 0.000165, 0.000153,
    0.000260, 0.000079, 0.000302, 0.000222, 0.000173, 0.000091, 0.000081, 0.000133, 0.000163, 0.000115, 0.000156,
    0.000188, 0.000049, 0.000109, 0.000159, 0.000088, 0.000163, 0.000103, 0.000203, 0.000199, 0.000098, 0.000258,
    0.000138, 0.000080, 0.000079, 0.000199, 0.000084, 0.000308, 0.000166, 0.000169, 0.000065, 0.000102, 0.000189,
    0.000249, 0.000067, 0.000069, 0.000241, 0.000155, 0.000109, 0.000095, 0.000172, 0.000131, 0.000081, 0.000221,
    0.000046, 0.000338, 0.000135, 0.000207, 0.000094, 0.000026, 0.000055, 0.000297, 0.000107, 0.000113, 0.000105,
    0.000069, 0.000150, 0.000179, 0.000161, 0.000041, 0.000205, 0.000193, 0.000265, 0.000274, 0.000057, 0.000157,
    0.000120, 0.000186, 0.000141, 0.000261, 0.000086, 0.000289, 0.000050, 0.000069, 0.000103, 0.000087, 0.000087,
    0.000050, 0.000066, 0.000188, 0.000152, 0.000162, 0.000308, 0.000102, 0.000146, 0.000096, 0.000158, 0.000085,
    0.000110, 0.000046, 0.000342, 0.000231, 0.000333, 0.000071, 0.000199, 0.000209, 0.000126, 0.000264, 0.000124,
    0.000190, 0.000139, 0.000209, 0.000018, 0.000317, 0.000111, 0.000054, 0.000078, 0.000089, 0.000132, 0.000053,
    0.000218, 0.000126, 0.000243, 0.000105, 0.000128, 0.000120, 0.000070, 0.000054, 0.000077, 0.000140, 0.000170,
    0.000091, 0.000212, 0.000179, 0.000159, 0.000112, 0.000098, 0.000242, 0.000292, 0.000365, 0.000311, 0.000046,
    0.000080, 0.000084, 0.000197, 0.000068, 0.000157, 0.000181, 0.000150, 0.000084, 0.000095, 0.000114, 0.000118,
    0.000149, 0.000299, 0.000061, 0.000122, 0.000091, 0.000083, 0.000277, 0.000335, 0.000104, 0.000153, 0.000088,
    0.000094, 0.000128, 0.000088, 0.000208, 0.000364, 0.000202, 0.000116, 0.000168, 0.000117, 0.000110, 0.000149,
    0.000128, 0.000160, 0.000126, 0.000089, 0.000322, 0.000112, 0.000253, 0.000218, 0.000100, 0.000244, 0.000260,
    0.000055, 0.000059, 0.000198, 0.000236, 0.000606, 0.000110, 0.000184, 0.000123, 0.000149, 0.000169, 0.000147,
    0.000131, 0.000146, 0.000078, 0.000317, 0.000326, 0.000411, 0.000113, 0.000093, 0.000054, 0.000219, 0.000119,
    0.000203, 0.000210, 0.000099, 0.000101, 0.000047, 0.000059, 0.000102, 0.000128, 0.000176, 0.000043, 0.000072,
    0.000189, 0.000180, 0.000448, 0.000198, 0.000117, 0.000060, 0.000153, 0.000137, 0.000069, 0.000362, 0.000150,
    0.000144, 0.000163, 0.000116, 0.000171, 0.000128, 0.000124, 0.000295, 0.000078, 0.000265, 0.000072, 0.000096,
    0.000156, 0.000205, 0.000154, 0.000072, 0.000069, 0.000279, 0.000141, 0.000117, 0.000078, 0.000178, 0.000106,
    0.000118, 0.000204, 0.000286, 0.000362, 0.000089, 0.000102, 0.000223, 0.000187, 0.000269, 0.000413, 0.000165,
    0.000059, 0.000104, 0.000264, 0.000212, 0.000096, 0.000148, 0.000066, 0.000120, 0.000097, 0.000161, 0.000140,
    0.000266, 0.000106, 0.000300, 0.000202, 0.000033, 0.000050, 0.000136, 0.000161, 0.000142, 0.000299, 0.000088,
    0.000233, 0.000149, 0.000104, 0.000190, 0.000320, 0.000101, 0.000199, 0.000110, 0.000070, 0.000264, 0.000069};
const int output_ref = 341;

template <typename T>
void test_top_k_layer_md_sync(bool is_caching_test) {
    static const int32_t x_size = 1, y_size = 1, feature_num = 1001, batch_num = 1;
    const int top_k = 1;
    layout inp_l = {data_types::f32, format::yxfb, {batch_num, feature_num, x_size, y_size}};
    layout mutableLayout = {data_types::i32, format::bfyx, {1, 1, 1, 1}};

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory(inp_l);
    set_values(input1, input_vec1);

    auto shared_memory = engine.allocate_memory(mutableLayout);
    const std::vector<T> topk_vec = {1};
    auto top_k_input = engine.allocate_memory(mutableLayout);
    set_values(top_k_input, topk_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(mutable_data("arg_max_md_write", shared_memory));
    topology.add(data("const", top_k_input));
    topology.add(arg_max_min("arg_max.0",
                             { input_info("input1"), input_info("const"), input_info("arg_max_md_write") },
                             ov::op::TopKMode::MAX,
                             top_k,
                             1,
                             ov::op::TopKSortType::SORT_INDICES,
                             true));
    topology.add(mutable_data("arg_max.1", { input_info("arg_max.0") }, shared_memory));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input1", input1);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(2));
    auto output = outputs.at("arg_max.1").get_memory();
    mem_lock<T> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr[0], output_ref);
}

TEST(top_k_layer_tests, md_sync) {
    test_top_k_layer_md_sync<int>(false);
}

TEST(export_import_top_k_layer_tests, md_sync) {
    test_top_k_layer_md_sync<int>(true);
}

struct arg_max_min_dynamic_params {
    format::type input_format;
    int64_t axis;
    uint32_t top_k;
    std::vector<float> expected_output;
};

class arg_max_min_gpu_dynamic : public ::testing::TestWithParam<arg_max_min_dynamic_params> {};

TEST_P(arg_max_min_gpu_dynamic, basic) {
    const auto& p = GetParam();
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, p.input_format};
    auto input_layout_static = layout{ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, p.input_format};
    auto input = engine.allocate_memory(input_layout_static);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(arg_max_min("arg_max", { input_info("input") }, ov::op::TopKMode::MIN, p.top_k, p.axis));

    std::vector<float> input_vec = {// y0x0 y0x1 y1x0 y1x1
                                    /*b0f0*/ 0.1f, -0.1f, 0.9f, 1.5f,
                                    /*b0f1*/ 0.2f, 0.2f, -10.f, 5.2f,
                                    /*b0f2*/ 0.2f, 0.2f, -10.f, 5.2f,
                                    /*b0f3*/ 0.2f, 0.2f, -10.f, 4.2f,

                                    /*b1f0*/ 3.f,  0.5f, 7.f, 10.f,
                                    /*b1f1*/ 4.f,  0.5f, 8.f, 8.2f,
                                    /*b1f2*/ 0.2f, 0.2f, -10.f, 5.2f,
                                    /*b1f3*/ 4.f,  0.5f, 8.f, 8.2f};

    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("arg_max");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "arg_max");

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), p.expected_output.size());
    for (size_t i = 0; i < p.expected_output.size(); i++) {
        ASSERT_FLOAT_EQ(output_ptr[i], p.expected_output[i]) << "i=" << i;
    }
}

INSTANTIATE_TEST_SUITE_P(arg_max_min_gpu,
                         arg_max_min_gpu_dynamic,
                         ::testing::Values(arg_max_min_dynamic_params{format::bfyx,
                                                                      0,
                                                                      2,
                                                                      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
                                           // input_vec is laid out as b,y,x,f here, so features are the innermost 4 values
                                           arg_max_min_dynamic_params{format::byxf,
                                                                      1,
                                                                      1,
                                                                      {1, 2, 2, 2,
                                                                       1, 1, 2, 1}}),
                         [](const ::testing::TestParamInfo<arg_max_min_dynamic_params>& info) {
                             return format(info.param.input_format).to_string();
                         });

TEST(arg_max_min_test, check_second_output_data_type) {
    auto& engine = get_test_engine();

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::program prog(engine, config);
    std::vector<std::shared_ptr<primitive>> input_prims;
    std::vector<input_info> input_prim_ids;
    {
        auto prim_id = "input";
        static const int32_t x_size = 1, y_size = 1, feature_num = 2000, batch_num = 1;
        auto input_static = layout{{batch_num, feature_num, y_size, x_size}, data_types::f16, format::bfyx};
        auto input_layout_prim = std::make_shared<input_layout>(prim_id, input_static);
        input_prims.push_back(input_layout_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }
    {
        auto prim_id = "top_k";
        auto top_k_input = layout{{1,1,1,1}, data_types::f16, format::bfyx};
        auto top_k_prim = std::make_shared<input_layout>(prim_id, top_k_input);
        input_prims.push_back(top_k_prim);
        input_prim_ids.push_back(input_info(prim_id));
    }

    auto arg_max_min_prim = std::make_shared<arg_max_min>("output", input_prim_ids,
                                                        ov::op::TopKMode::MAX, 400, 1,
                                                        ov::op::TopKSortType::SORT_VALUES, true, false,
                                                        data_types::f16, 2);

    arg_max_min_prim->output_paddings = {padding(), padding()};
    arg_max_min_prim->output_data_types = {data_types::f16, data_types::i32};
    auto& arg_max_min_node = prog.get_or_create(arg_max_min_prim);
    for (auto& prim : input_prims) {
        auto& input_layout_node = prog.get_or_create(prim);
        program_wrapper::add_connection(prog, input_layout_node, arg_max_min_node);
    }

    auto second_output_layout = arg_max_min_node.get_output_layout(false, 1);
    ASSERT_EQ(second_output_layout.data_type, data_types::i32);
}

// =============================================================================
// Tests for arg_max_min_topk_radix kernel
// Conditions: f16/f32 input, N >= 2, SLM fits
// =============================================================================

inline ExecutionConfig get_radix_topk_config(const cldnn::engine& engine, format::type fmt = format::bfyx) {
    auto config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc radix_impl = {fmt, "arg_max_min_topk_radix", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"arg_max", radix_impl}}));
    return config;
}

// Helper: check that forcing actually resulted in the radix kernel being used
inline void assert_radix_topk_selected(const cldnn::network& net) {
    auto info = net.get_primitive_info("arg_max");
    ASSERT_TRUE(info.find("arg_max_min_topk_radix") != std::string::npos)
        << "Expected arg_max_min_topk_radix, got: " << info;
}

TEST(arg_max_gpu_topk_radix, dynamic_f32_dflash_shape_transitions_use_radix) {
    constexpr int32_t batch_num = 1;
    constexpr int32_t vocab_size = 248320;
    constexpr int32_t top_k = 16;
    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), vocab_size, 1},
        data_types::f32,
        format::bfyx};
    const std::vector<float> sorted_values = {
        4.4388075f,
        4.3951770f,
        4.2429514f,
        4.2423444f,
        4.2085915f,
        4.1768341f,
        4.0288038f,
        3.9938712f,
        3.9857004f,
        3.9501023f,
        3.9219890f,
        3.8840609f,
        3.8789275f,
        3.8753984f,
        3.7988553f,
        3.7939982f,
    };

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(arg_max_min("arg_max",
                             {input_info("input")},
                             ov::op::TopKMode::MAX,
                             top_k,
                             2,
                             ov::op::TopKSortType::SORT_VALUES,
                             true,
                             false,
                             data_types::f32,
                             2));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));

    for (const int32_t feature_num : {1, 7, 15, 7, 1}) {
        auto input_layout_static =
            layout{ov::PartialShape{batch_num, feature_num, vocab_size, 1}, data_types::f32, format::bfyx};
        auto input = engine.allocate_memory(input_layout_static);
        std::vector<float> input_values(batch_num * feature_num * vocab_size, -1.0f);
        for (int32_t feature = 0; feature < feature_num; ++feature) {
            for (int32_t rank = 0; rank < top_k; ++rank) {
                input_values[feature * vocab_size + rank * 7919] = sorted_values[rank] + feature;
            }
        }
        set_values(input, input_values);

        network.set_input_data("input", input);
        auto outputs = network.execute();
        auto output = outputs.at("arg_max").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (int32_t feature = 0; feature < feature_num; ++feature) {
            for (int32_t rank = 0; rank < top_k; ++rank) {
                EXPECT_FLOAT_EQ(output_ptr[feature * top_k + rank],
                                sorted_values[rank] + feature)
                    << "value mismatch at feature_num=" << feature_num << ", feature=" << feature << ", rank=" << rank;
            }
        }
    }
}

namespace {
// The radix kernel supports both floating point input types, so every case below runs for both.
using topk_radix_data_types = ::testing::Types<ov::float16, float>;

struct topk_radix_type_names {
    template <typename T>
    static std::string GetName(int) {
        return std::is_same<T, ov::float16>::value ? "f16" : "f32";
    }
};

template <typename T>
std::vector<T> topk_radix_typed_vec(const std::vector<float>& values) {
    return std::vector<T>(values.begin(), values.end());
}

// f16 storage rounds the reference values, f32 results are expected to be bit-exact.
template <typename T>
constexpr float topk_radix_tolerance() {
    return std::is_same<T, ov::float16>::value ? 0.02f : 0.01f;
}
}  // namespace

template <typename T>
class arg_max_gpu_topk_radix : public ::testing::Test {};
TYPED_TEST_SUITE(arg_max_gpu_topk_radix, topk_radix_data_types, topk_radix_type_names);

// --------------------------------------------------------------------------
// Parameterized test: value-only verification with various configurations
// --------------------------------------------------------------------------
struct TopKRadixValueTestParam {
    ov::op::TopKMode mode;             // MAX or MIN
    int32_t batch_num;
    int32_t feature_num;
    int top_k;
    int axis;
    std::vector<float> input_vec;
    std::vector<float> ref_values;     // expected top-K values in sorted order
    std::string name;
};

class TopKRadixValueTest : public testing::TestWithParam<std::tuple<TopKRadixValueTestParam, data_types>> {};

template <typename T>
void run_topk_radix_value_case(const TopKRadixValueTestParam& p) {
    auto& engine = get_test_engine();
    const data_types dt = ov::element::from<T>();

    auto input = engine.allocate_memory({dt, format::bfyx,
                                         {p.batch_num, p.feature_num, 1, 1}});
    set_values(input, topk_radix_typed_vec<T>(p.input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max",
                             {input_info("input")},
                             p.mode,
                             p.top_k,
                             p.axis,
                             ov::op::TopKSortType::SORT_VALUES,
                             true,
                             false,
                             dt,
                             2));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < p.ref_values.size(); i++) {
        float val = static_cast<float>(output_ptr[i]);
        ASSERT_NEAR(val, p.ref_values[i], topk_radix_tolerance<T>()) << "value mismatch at i=" << i;
    }
}

TEST_P(TopKRadixValueTest, values_match) {
    const auto& p = std::get<0>(GetParam());
    if (std::get<1>(GetParam()) == data_types::f16)
        run_topk_radix_value_case<ov::float16>(p);
    else
        run_topk_radix_value_case<float>(p);
}

INSTANTIATE_TEST_SUITE_P(arg_max_gpu_topk_radix, TopKRadixValueTest,
    testing::Combine(
        testing::Values(
            TopKRadixValueTestParam{
                ov::op::TopKMode::MAX, 1, 16, 4, 1,
                {1.0f, 5.0f, 3.0f, 9.0f,
                 2.0f, 7.0f, 4.0f, 8.0f,
                 6.0f, 0.5f, 10.0f, 0.1f,
                 15.0f, 12.0f, 11.0f, 0.2f},
                {15.0f, 12.0f, 11.0f, 10.0f},
                "max_axis_feature"
            },
            TopKRadixValueTestParam{
                ov::op::TopKMode::MIN, 1, 16, 4, 1,
                {1.0f, 5.0f, 3.0f, 9.0f,
                 2.0f, 7.0f, 4.0f, 8.0f,
                 6.0f, 0.5f, -1.0f, 0.1f,
                 15.0f, 12.0f, 11.0f, -2.0f},
                {-2.0f, -1.0f, 0.1f, 0.5f},
                "min_axis_feature"
            },
            TopKRadixValueTestParam{
                ov::op::TopKMode::MAX, 1, 8, 3, 1,
                {-5.0f, -1.0f, -8.0f, -0.5f,
                 -3.0f, -10.0f, -2.0f, -0.1f},
                {-0.1f, -0.5f, -1.0f},
                "max_negatives"
            }
        ),
        testing::Values(data_types::f16, data_types::f32)
    ),
    [](const testing::TestParamInfo<TopKRadixValueTest::ParamType>& info) {
        return std::get<0>(info.param).name +
               (std::get<1>(info.param) == data_types::f16 ? "_f16" : "_f32");
    }
);

// Separate case for large N on batch axis (generated input)
TYPED_TEST(arg_max_gpu_topk_radix, max_large_n_axis_batch) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    static const int32_t batch_num = 1000;
    auto& engine = get_test_engine();
    const int top_k = 10;
    auto input = engine.allocate_memory({dt, format::bfyx, {batch_num, 1, 1, 1}});

    std::vector<float> input_vec(batch_num);
    for (int i = 0; i < batch_num; i++) {
        input_vec[i] = static_cast<float>(i) * 0.01f;
    }
    set_values(input, topk_radix_typed_vec<T>(input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max",
                             {input_info("input")},
                             ov::op::TopKMode::MAX,
                             top_k,
                             0,
                             ov::op::TopKSortType::SORT_VALUES,
                             true, false, dt));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < top_k; i++) {
        float expected = input_vec[batch_num - 1 - i];
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), expected, topk_radix_tolerance<T>())
            << "mismatch at i=" << i;
    }
}

// --------------------------------------------------------------------------
// Non-parameterized tests: each has unique verification logic
// --------------------------------------------------------------------------

// Radix test: MAX, second output (indices), axis=feature(1)
TYPED_TEST(arg_max_gpu_topk_radix, max_second_output_indices) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    static const int32_t feature_num = 32;
    auto& engine = get_test_engine();
    const int top_k = 8;
    auto input = engine.allocate_memory({dt, format::bfyx, {1, feature_num, 1, 1}});
    auto top_k_input = engine.allocate_memory({dt, format::bfyx, {1, 1, 1, 1}});
    auto second_output = engine.allocate_memory({data_types::f32, format::bfyx, {1, top_k, 1, 1}});

    std::vector<float> input_vec(feature_num);
    for (int i = 0; i < feature_num; i++)
        input_vec[i] = static_cast<float>(i * 3 % 17);
    set_values(input, topk_radix_typed_vec<T>(input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max",
                             {input_info("input"), input_info("const"), input_info("second_output")},
                             ov::op::TopKMode::MAX, top_k, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, dt));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 1; i < top_k; i++) {
        ASSERT_GE(static_cast<float>(output_ptr[i - 1]), static_cast<float>(output_ptr[i]))
            << "values not sorted descending at i=" << i;
    }
    ASSERT_NEAR(static_cast<float>(output_ptr[0]), 16.0f, 0.01f);

    cldnn::mem_lock<float> idx_ptr(second_output, get_test_stream());
    ASSERT_EQ(static_cast<int>(idx_ptr[0]), 11);  // first 16 is at idx=11
}

// Radix test: duplicate values tiebreaking (smaller index first)
TYPED_TEST(arg_max_gpu_topk_radix, max_duplicate_values_tiebreak) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    auto& engine = get_test_engine();
    const int top_k = 5;
    auto input = engine.allocate_memory({dt, format::bfyx, {1, 10, 1, 1}});
    auto top_k_input = engine.allocate_memory({dt, format::bfyx, {1, 1, 1, 1}});
    auto second_output = engine.allocate_memory({data_types::f32, format::bfyx, {1, top_k, 1, 1}});

    // 5,5,5,5,5,3,3,3,1,1 — top5 are all 5.0, indices 0..4
    set_values(input, topk_radix_typed_vec<T>({5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                                               3.0f, 3.0f, 3.0f, 1.0f, 1.0f}));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max",
                             {input_info("input"), input_info("const"), input_info("second_output")},
                             ov::op::TopKMode::MAX, top_k, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, dt));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < top_k; i++)
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), 5.0f, 0.01f) << "value mismatch at i=" << i;

    cldnn::mem_lock<float> idx_ptr(second_output, get_test_stream());
    for (int i = 0; i < top_k; i++)
        ASSERT_EQ(static_cast<int>(idx_ptr[i]), i) << "index tiebreak mismatch at i=" << i;
}

// Radix test: YOLOv26n-like scenario (N=8400, K=300, axis=feature)
TYPED_TEST(arg_max_gpu_topk_radix, max_yolov26n_like) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    static const int32_t feature_num = 8400;
    auto& engine = get_test_engine();
    const int top_k = 300;
    auto input = engine.allocate_memory({dt, format::bfyx, {1, feature_num, 1, 1}});

    std::vector<float> input_vec(feature_num);
    for (int i = 0; i < feature_num; i++)
        input_vec[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < 300; i++)
        input_vec[i * 28] = 100.0f + static_cast<float>(300 - i) * 0.1f;
    set_values(input, topk_radix_typed_vec<T>(input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max",
                             {input_info("input")},
                             ov::op::TopKMode::MAX, top_k, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, dt));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 1; i < top_k; i++) {
        ASSERT_GE(static_cast<float>(output_ptr[i - 1]), static_cast<float>(output_ptr[i]))
            << "values not sorted descending at i=" << i;
    }
    ASSERT_NEAR(static_cast<float>(output_ptr[0]), 130.0f, 0.2f);
    ASSERT_GT(static_cast<float>(output_ptr[top_k - 1]), 50.0f);
}

// Radix test: CI failure reproducer - many duplicate values, verify indices tiebreaking
TYPED_TEST(arg_max_gpu_topk_radix, max_duplicates_n70_k5_indices_tiebreak) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    static const int32_t batch_num = 4, feature_num = 70;
    auto& engine = get_test_engine();
    const int top_k = 5;
    auto input = engine.allocate_memory({dt, format::bfyx, {batch_num, feature_num, 1, 1}});
    auto top_k_input = engine.allocate_memory({dt, format::bfyx, {1, 1, 1, 1}});
    auto second_output = engine.allocate_memory({data_types::f32, format::bfyx, {batch_num, top_k, 1, 1}});

    std::vector<float> input_vec(batch_num * feature_num);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 14);
    for (size_t i = 0; i < input_vec.size(); i++)
        input_vec[i] = static_cast<float>(dist(rng));
    set_values(input, topk_radix_typed_vec<T>(input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max",
                             {input_info("input"), input_info("const"), input_info("second_output")},
                             ov::op::TopKMode::MAX, top_k, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, dt));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> val_ptr(output, get_test_stream());
    cldnn::mem_lock<float> idx_ptr(second_output, get_test_stream());

    for (int b = 0; b < batch_num; b++) {
        int input_offset = b * feature_num;
        int out_offset = b * top_k;

        std::vector<std::pair<float, int>> ref;
        for (int i = 0; i < feature_num; i++)
            ref.push_back({input_vec[input_offset + i], i});
        std::stable_sort(ref.begin(), ref.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int k = 0; k < top_k; k++) {
            ASSERT_NEAR(static_cast<float>(val_ptr[out_offset + k]), ref[k].first, 0.01f)
                << "value mismatch at b=" << b << " k=" << k;
            ASSERT_EQ(static_cast<int>(idx_ptr[out_offset + k]), ref[k].second)
                << "index mismatch at b=" << b << " k=" << k;
        }
    }
}

// Radix test: very large sort size (N=100000, K=256, axis=feature)
TYPED_TEST(arg_max_gpu_topk_radix, max_n100k_k256_axis_feature) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    static const int32_t feature_num = 100000;
    auto& engine = get_test_engine();
    const int top_k = 256;
    const int stride = feature_num / top_k;
    auto input = engine.allocate_memory({dt, format::bfyx, {1, feature_num, 1, 1}});
    auto top_k_input = engine.allocate_memory({dt, format::bfyx, {1, 1, 1, 1}});
    auto second_output = engine.allocate_memory({data_types::f32, format::bfyx, {1, top_k, 1, 1}});

    // Background stays below 1.0, the K planted values are distinct integers
    // (exactly representable in f16) so both values and indices are unambiguous.
    std::vector<float> input_vec(feature_num);
    for (int i = 0; i < feature_num; i++)
        input_vec[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < top_k; i++)
        input_vec[i * stride] = 100.0f + static_cast<float>(top_k - i);
    set_values(input, topk_radix_typed_vec<T>(input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max",
                             {input_info("input"), input_info("const"), input_info("second_output")},
                             ov::op::TopKMode::MAX, top_k, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, dt));

    network network(engine, topology, get_radix_topk_config(engine));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(network));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> val_ptr(output, get_test_stream());
    cldnn::mem_lock<float> idx_ptr(second_output, get_test_stream());

    for (int k = 0; k < top_k; k++) {
        ASSERT_NEAR(static_cast<float>(val_ptr[k]), 100.0f + static_cast<float>(top_k - k),
                    topk_radix_tolerance<T>()) << "value mismatch at k=" << k;
        ASSERT_EQ(static_cast<int>(idx_ptr[k]), k * stride) << "index mismatch at k=" << k;
    }
}


namespace {
inline std::vector<float> make_radix_topk_input(int batch_num,
                                                int feature_num,
                                                int y_size,
                                                int x_size,
                                                int top_k) {
    std::vector<float> values(static_cast<size_t>(batch_num) * feature_num * y_size * x_size);

    for (int b = 0; b < batch_num; ++b) {
        for (int f = 0; f < feature_num; ++f) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    const size_t offset = ((static_cast<size_t>(b) * feature_num + f) * y_size + y) * x_size + x;
                    values[offset] = f < top_k ? 100.0f : 0.0f;
                }
            }
        }
    }

    return values;
}

template <typename T>
void run_radix_topk_case(cldnn::engine& engine,
                         const std::vector<float>& input_vec,
                         data_types dt,
                         int batch_num,
                         int feature_num,
                         int y_size,
                         int x_size,
                         int top_k,
                         format::type fmt,
                         std::vector<float>& result) {
    const size_t out_size = static_cast<size_t>(batch_num) * top_k * y_size * x_size;
    auto input = engine.allocate_memory({dt, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    set_values(input, topk_radix_typed_vec<T>(input_vec));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("input_byxf", input_info("input"), format::byxf, dt));
    topology.add(arg_max_min("arg_max", {input_info("input_byxf")},
                             ov::op::TopKMode::MAX, top_k, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, dt));
    topology.add(reorder("plane_arg_max", input_info("arg_max"), format::bfyx, dt));

    network net(engine, topology, get_radix_topk_config(engine, fmt));
    ASSERT_NO_FATAL_FAILURE(assert_radix_topk_selected(net));
    net.set_input_data("input", input);
    auto outputs = net.execute();

    auto output = outputs.at("plane_arg_max").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> out_ptr(output, get_test_stream());

    result.resize(out_size);
    for (size_t i = 0; i < out_size; i++)
        result[i] = static_cast<float>(out_ptr[i]);
}
}  // namespace

TYPED_TEST(arg_max_gpu_topk_radix, byxf_matches_bfyx_reference) {
    using T = TypeParam;
    const data_types dt = ov::element::from<T>();
    auto& engine = get_test_engine();

    const int batch_num = 1, feature_num = 384, y_size = 2, x_size = 3;
    const int top_k = 6;

    const std::vector<float> input_vec = make_radix_topk_input(batch_num, feature_num, y_size, x_size, top_k);

    std::vector<float> byxf_result;
    run_radix_topk_case<T>(engine,
                           input_vec,
                           dt,
                           batch_num,
                           feature_num,
                           y_size,
                           x_size,
                           top_k,
                           format::byxf,
                           byxf_result);

    for (int b = 0; b < batch_num; b++) {
        for (int y = 0; y < y_size; y++) {
            for (int x = 0; x < x_size; x++) {
                for (int k = 0; k < top_k; k++) {
                    const size_t out_idx = ((static_cast<size_t>(b) * top_k + k) * y_size + y) * x_size + x;
                    ASSERT_NEAR(byxf_result[out_idx], 100.0f, topk_radix_tolerance<T>())
                        << "value mismatch at b=" << b << " y=" << y << " x=" << x << " k=" << k;
                }
            }
        }
    }
}

namespace {
// Ordering key of the GPU TopK kernels: -0.0 equals +0.0 and NaN ranks above +inf, as in PyTorch.
uint32_t topk_radix_order_key(float value) {
    if (std::isnan(value)) {
        return 0xFFFFFFFFu;
    }
    uint32_t bits = 0;
    if (value != 0.0f) {
        std::memcpy(&bits, &value, sizeof(bits));
    }
    return (bits >> 31) ? ~bits : (bits | 0x80000000u);
}

// Rows are contiguous in `values`; equal values are ordered by the smallest index. Without
// SORT_VALUES the selected indices are expected in ascending order.
std::vector<int32_t> topk_radix_reference_indices(const std::vector<float>& values,
                                                  size_t rows,
                                                  size_t row_size,
                                                  size_t top_k,
                                                  ov::op::TopKMode mode,
                                                  ov::op::TopKSortType sort) {
    std::vector<int32_t> result;
    result.reserve(rows * top_k);
    std::vector<uint32_t> keys(row_size);
    std::vector<int32_t> order(row_size);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t i = 0; i < row_size; ++i) {
            keys[i] = topk_radix_order_key(values[row * row_size + i]);
        }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return mode == ov::op::TopKMode::MAX ? keys[a] > keys[b] : keys[a] < keys[b];
        });
        if (sort != ov::op::TopKSortType::SORT_VALUES) {
            std::sort(order.begin(), order.begin() + top_k);
        }
        result.insert(result.end(), order.begin(), order.begin() + top_k);
    }
    return result;
}

// TopK copies its values from the input, so they must keep their bits, including the sign of zero.
bool topk_radix_same_value(float expected, float actual) {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    return expected == actual && std::signbit(expected) == std::signbit(actual);
}

template <typename T>
std::vector<float> topk_radix_rounded(const std::vector<float>& values) {
    std::vector<float> rounded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        rounded[i] = static_cast<float>(static_cast<T>(values[i]));
    }
    return rounded;
}

// Runs TopK on `shapes` one after another, with networks built for `network_shape`, and checks that
// `kernel` ran, that the indices match the reference and that every value is the input element at
// its index. The kernel is forced unless `force_kernel` is false, which checks the default choice.
template <typename T>
void run_topk_radix_case(const ov::PartialShape& network_shape,
                         const std::vector<ov::PartialShape>& shapes,
                         int64_t axis,
                         ov::op::TopKMode mode,
                         int32_t top_k,
                         const std::function<std::vector<float>(size_t rows, size_t row_size)>& generate,
                         ov::op::TopKSortType sort = ov::op::TopKSortType::SORT_VALUES,
                         const std::string& kernel = "arg_max_min_topk_radix",
                         bool force_kernel = true) {
    auto& engine = get_test_engine();
    const data_types dt = ov::element::from<T>();

    auto config = get_test_default_config(engine);
    if (force_kernel) {
        ov::intel_gpu::ImplementationDesc impl = {format::bfyx, kernel, impl_types::ocl};
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"arg_max", impl}}));
    }
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    // A single-output TopK returns either the values or the indices.
    auto make_network = [&](bool values) {
        topology topology;
        topology.add(input_layout("input", layout{network_shape, dt, format::bfyx}));
        topology.add(arg_max_min("arg_max",
                                 {input_info("input")},
                                 mode,
                                 top_k,
                                 axis,
                                 sort,
                                 values,
                                 false,
                                 values ? dt : data_types::i32));
        return std::make_shared<cldnn::network>(engine, topology, config);
    };
    auto indices_network = make_network(false);
    auto values_network = make_network(true);

    for (const auto& shape : shapes) {
        const size_t row_size = static_cast<size_t>(shape[static_cast<size_t>(axis)].get_length());
        const size_t rows = ov::shape_size(shape.to_shape()) / row_size;
        const auto values = topk_radix_rounded<T>(generate(rows, row_size));
        const auto expected = topk_radix_reference_indices(values, rows, row_size, top_k, mode, sort);

        auto input = engine.allocate_memory(layout{shape, dt, format::bfyx});
        set_values(input, topk_radix_typed_vec<T>(values));
        indices_network->set_input_data("input", input);
        values_network->set_input_data("input", input);
        auto indices_outputs = indices_network->execute();
        auto values_outputs = values_network->execute();
        for (const auto& net : {indices_network, values_network}) {
            const auto info = net->get_primitive_info("arg_max");
            ASSERT_NE(info.find(kernel), std::string::npos) << "Expected " << kernel << ", got: " << info;
        }

        auto indices = indices_outputs.at("arg_max").get_memory();
        auto out_values = values_outputs.at("arg_max").get_memory();
        ASSERT_EQ(indices->get_layout().count(), expected.size());
        ASSERT_EQ(out_values->get_layout().count(), expected.size());
        cldnn::mem_lock<int32_t, mem_lock_type::read> indices_ptr(indices, get_test_stream());
        cldnn::mem_lock<T, mem_lock_type::read> values_ptr(out_values, get_test_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_EQ(indices_ptr[i], expected[i])
                << "index mismatch for shape " << shape << " at row=" << i / top_k << ", rank=" << i % top_k;
            const float expected_value = values[(i / top_k) * row_size + static_cast<size_t>(expected[i])];
            ASSERT_TRUE(topk_radix_same_value(expected_value, static_cast<float>(values_ptr[i])))
                << "value mismatch for shape " << shape << " at row=" << i / top_k << ", rank=" << i % top_k
                << ": expected " << expected_value << ", got " << static_cast<float>(values_ptr[i]);
        }
    }
}

// Few distinct levels, so many equal values compete for the last top-K slots.
std::vector<float> topk_radix_tied_values(size_t rows, size_t row_size, int levels, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, levels - 1);
    std::vector<float> values(rows * row_size);
    for (auto& value : values) {
        value = static_cast<float>(dist(rng)) * 0.25f;
    }
    return values;
}

std::vector<float> topk_radix_random_values(size_t rows, size_t row_size, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 2.5f);
    std::vector<float> values(rows * row_size);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

// Every row of at least 64 values holds NaNs of both signs, both infinities and a block of
// alternating -0.0 and +0.0. The other values lie on the losing side of zero, so the top-K
// reaches into the zeros.
std::vector<float> topk_radix_special_values(size_t rows, size_t row_size, ov::op::TopKMode mode) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float losing_sign = mode == ov::op::TopKMode::MAX ? -1.f : 1.f;
    const size_t zeros_begin = std::max<size_t>(16, row_size / 8);
    const size_t zeros_end = zeros_begin + std::min<size_t>(40, row_size / 4);
    auto values = topk_radix_random_values(rows, row_size, 17);
    for (size_t row = 0; row < rows; ++row) {
        float* row_values = values.data() + row * row_size;
        for (size_t i = 0; i < row_size; ++i) {
            row_values[i] = losing_sign * (1.f + std::fabs(row_values[i]));
        }
        for (size_t i = zeros_begin; i < zeros_end; ++i) {
            row_values[i] = std::copysign(0.f, i % 2 ? 1.f : -1.f);
        }
        row_values[5] = nan;
        row_values[row_size * 3 / 4] = std::copysign(nan, -1.f);
        row_values[row_size - 1] = nan;
        row_values[7] = inf;
        row_values[9] = -inf;
    }
    return values;
}
}  // namespace

// Chunked mode with dynamic rows: the maximum level repeats in every chunk, so the first
// top-K indices come from equal values spread over several chunks.
TYPED_TEST(arg_max_gpu_topk_radix, chunked_dynamic_rows_tied_indices) {
    using T = TypeParam;
    constexpr int32_t vocab_size = 248320;
    run_topk_radix_case<T>(
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), vocab_size, 1},
        {{1, 1, vocab_size, 1}, {1, 7, vocab_size, 1}, {1, 3, vocab_size, 1}},
        2,
        ov::op::TopKMode::MAX,
        16,
        [](size_t rows, size_t row_size) {
            return topk_radix_tied_values(rows, row_size, 2000, 7);
        });
}

// Chunked mode with distinct random values, checked against the exact reference order.
TYPED_TEST(arg_max_gpu_topk_radix, chunked_dynamic_rows_random_indices) {
    using T = TypeParam;
    constexpr int32_t vocab_size = 248320;
    run_topk_radix_case<T>(
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), vocab_size, 1},
        {{1, 7, vocab_size, 1}, {1, 15, vocab_size, 1}},
        2,
        ov::op::TopKMode::MAX,
        16,
        [](size_t rows, size_t row_size) {
            return topk_radix_random_values(rows, row_size, 11);
        });
}

// Chunked MIN mode on the feature axis, where the last chunk is shorter than the others.
TYPED_TEST(arg_max_gpu_topk_radix, chunked_static_min_tied_indices) {
    using T = TypeParam;
    constexpr int32_t sort_size = 40001;
    run_topk_radix_case<T>(
        ov::PartialShape{2, sort_size, 1, 1},
        {{2, sort_size, 1, 1}},
        1,
        ov::op::TopKMode::MIN,
        20,
        [](size_t rows, size_t row_size) {
            return topk_radix_tied_values(rows, row_size, 64, 3);
        });
}

// Single-stage mode where the threshold value overflows the remaining slots in many blocks.
TYPED_TEST(arg_max_gpu_topk_radix, single_stage_ordered_ties) {
    using T = TypeParam;
    constexpr int32_t sort_size = 10000;
    run_topk_radix_case<T>(
        ov::PartialShape{3, sort_size, 1, 1},
        {{3, sort_size, 1, 1}},
        1,
        ov::op::TopKMode::MAX,
        64,
        [](size_t rows, size_t row_size) {
            return topk_radix_tied_values(rows, row_size, 8, 5);
        });
}

// NaN ranks above +inf and -0.0 ties with +0.0, in single-stage, chunked and dynamic-rows mode.
TYPED_TEST(arg_max_gpu_topk_radix, nan_and_signed_zero_order) {
    using T = TypeParam;
    constexpr int32_t vocab_size = 248320;
    for (const auto mode : {ov::op::TopKMode::MAX, ov::op::TopKMode::MIN}) {
        SCOPED_TRACE(mode == ov::op::TopKMode::MAX ? "MAX" : "MIN");
        const auto generate = [mode](size_t rows, size_t row_size) {
            return topk_radix_special_values(rows, row_size, mode);
        };
        ASSERT_NO_FATAL_FAILURE(
            run_topk_radix_case<T>(ov::PartialShape{2, 4096, 1, 1}, {{2, 4096, 1, 1}}, 1, mode, 16, generate));
        ASSERT_NO_FATAL_FAILURE(
            run_topk_radix_case<T>(ov::PartialShape{1, 40000, 1, 1}, {{1, 40000, 1, 1}}, 1, mode, 16, generate));
        ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(ov::PartialShape{ov::Dimension::dynamic(), vocab_size, 1, 1},
                                                       {{3, vocab_size, 1, 1}},
                                                       1,
                                                       mode,
                                                       16,
                                                       generate));
    }
}

// The axis kernel orders NaN and signed zeros like radix in all of its sorting paths.
TYPED_TEST(arg_max_gpu_topk_radix, axis_kernel_nan_and_signed_zero_order) {
    using T = TypeParam;
    const std::string axis_kernel = "arg_max_min_axis";
    for (const auto mode : {ov::op::TopKMode::MAX, ov::op::TopKMode::MIN}) {
        SCOPED_TRACE(mode == ov::op::TopKMode::MAX ? "MAX" : "MIN");
        const auto generate = [mode](size_t rows, size_t row_size) {
            return topk_radix_special_values(rows, row_size, mode);
        };
        for (const auto sort : {ov::op::TopKSortType::SORT_VALUES, ov::op::TopKSortType::SORT_INDICES}) {
            SCOPED_TRACE(sort == ov::op::TopKSortType::SORT_VALUES ? "SORT_VALUES" : "SORT_INDICES");
            // Rank-based, partial and merge sorting, the k=1 search, and K=N, where MIN also selects the NaNs
            for (const auto& [row_size, top_k] :
                 std::vector<std::pair<int32_t, int32_t>>{{4096, 16}, {64, 40}, {4096, 1}, {64, 64}}) {
                ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(ov::PartialShape{2, row_size, 1, 1},
                                                               {{2, row_size, 1, 1}},
                                                               1,
                                                               mode,
                                                               top_k,
                                                               generate,
                                                               sort,
                                                               axis_kernel));
            }
        }
        // Short MoE router rows keep the axis kernel by default
        ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(ov::PartialShape{ov::Dimension::dynamic(), 128, 1, 1},
                                                       {{64, 128, 1, 1}},
                                                       1,
                                                       mode,
                                                       8,
                                                       generate,
                                                       ov::op::TopKSortType::SORT_VALUES,
                                                       axis_kernel,
                                                       false));
    }
}

// Without SORT_VALUES the selected elements come out in index order. SORT_NONE is treated as SORT_INDICES.
TYPED_TEST(arg_max_gpu_topk_radix, sort_by_indices) {
    using T = TypeParam;
    constexpr int32_t vocab_size = 248320;
    for (const auto sort : {ov::op::TopKSortType::SORT_INDICES, ov::op::TopKSortType::NONE}) {
        SCOPED_TRACE(sort == ov::op::TopKSortType::NONE ? "NONE" : "SORT_INDICES");
        ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
            ov::PartialShape{3, 10000, 1, 1}, {{3, 10000, 1, 1}}, 1, ov::op::TopKMode::MAX, 64,
            [](size_t rows, size_t row_size) { return topk_radix_tied_values(rows, row_size, 8, 5); },
            sort));
        ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
            ov::PartialShape{2, 40001, 1, 1}, {{2, 40001, 1, 1}}, 1, ov::op::TopKMode::MIN, 20,
            [](size_t rows, size_t row_size) { return topk_radix_tied_values(rows, row_size, 64, 3); },
            sort));
        ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
            ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), vocab_size, 1},
            {{1, 1, vocab_size, 1}, {1, 7, vocab_size, 1}},
            2,
            ov::op::TopKMode::MAX,
            16,
            [](size_t rows, size_t row_size) { return topk_radix_tied_values(rows, row_size, 2000, 7); },
            sort));
    }
}

// Index-ordered output when 30 values tie at the K-th value ahead of 18 better values in the middle of
// the row: every better value is selected, and only the ties with the smallest indices fill the rest.
TYPED_TEST(arg_max_gpu_topk_radix, sort_by_indices_ties_before_better_values) {
    using T = TypeParam;
    for (const auto mode : {ov::op::TopKMode::MAX, ov::op::TopKMode::MIN}) {
        SCOPED_TRACE(mode == ov::op::TopKMode::MAX ? "MAX" : "MIN");
        const float sign = mode == ov::op::TopKMode::MAX ? 1.f : -1.f;
        const auto generate = [sign](size_t rows, size_t row_size) {
            std::vector<float> values(rows * row_size, -2.f * sign);
            for (size_t row = 0; row < rows; ++row) {
                float* row_values = values.data() + row * row_size;
                std::fill_n(row_values, 30, sign);
                std::fill_n(row_values + row_size / 2, 18, 2.f * sign);
            }
            return values;
        };
        for (const auto sort : {ov::op::TopKSortType::SORT_INDICES, ov::op::TopKSortType::NONE}) {
            SCOPED_TRACE(sort == ov::op::TopKSortType::NONE ? "NONE" : "SORT_INDICES");
            ASSERT_NO_FATAL_FAILURE(
                run_topk_radix_case<T>(ov::PartialShape{2, 1000, 1, 1}, {{2, 1000, 1, 1}}, 1, mode, 20, generate, sort));
            ASSERT_NO_FATAL_FAILURE(
                run_topk_radix_case<T>(ov::PartialShape{1, 40000, 1, 1}, {{1, 40000, 1, 1}}, 1, mode, 20, generate, sort));
            ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
                ov::PartialShape{ov::Dimension::dynamic(), 40000, 1, 1}, {{3, 40000, 1, 1}}, 1, mode, 20, generate, sort));
        }
    }
}

// Index-ordered output when 30 ties at indices 0..29 come before 18 better zeros at 500..517, K=20. The
// 20000-value row is chunked on every device, with the ties and the zeros in its first chunk; the
// 1000-value row runs in a single stage.
TYPED_TEST(arg_max_gpu_topk_radix, sort_by_indices_ties_before_better_zeros) {
    using T = TypeParam;
    for (const auto mode : {ov::op::TopKMode::MAX, ov::op::TopKMode::MIN}) {
        SCOPED_TRACE(mode == ov::op::TopKMode::MAX ? "MAX" : "MIN");
        const auto generate = [mode](size_t rows, size_t row_size) {
            const float sign = mode == ov::op::TopKMode::MAX ? -1.0f : 1.0f;
            std::vector<float> values(rows * row_size, sign);
            for (size_t row = 0; row < rows; ++row) {
                std::fill_n(values.begin() + row * row_size, 30, 0.25f * sign);
                std::fill_n(values.begin() + row * row_size + 500, 18, 0.0f);
            }
            return values;
        };
        for (const auto sort : {ov::op::TopKSortType::SORT_INDICES, ov::op::TopKSortType::NONE}) {
            SCOPED_TRACE(sort == ov::op::TopKSortType::NONE ? "NONE" : "SORT_INDICES");
            for (const int32_t row_size : {1000, 20000}) {
                ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(ov::PartialShape{2, row_size, 1, 1},
                                                               {{2, row_size, 1, 1}},
                                                               1,
                                                               mode,
                                                               20,
                                                               generate,
                                                               sort));
            }
        }
    }
}

// Rows of 1024 values or more use radix for k=1 by default, with equal maxima resolved to the smallest index.
TYPED_TEST(arg_max_gpu_topk_radix, topk1_selects_radix_from_1024_values) {
    using T = TypeParam;
    constexpr int32_t vocab_size = 248320;
    const std::string radix_kernel = "arg_max_min_topk_radix";
    ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
        ov::PartialShape{3, 32000, 1, 1}, {{3, 32000, 1, 1}}, 1, ov::op::TopKMode::MAX, 1,
        [](size_t rows, size_t row_size) { return topk_radix_tied_values(rows, row_size, 3, 9); },
        ov::op::TopKSortType::SORT_VALUES, radix_kernel, false));
    ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
        ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), vocab_size, 1},
        {{1, 1, vocab_size, 1}, {1, 4, vocab_size, 1}},
        2,
        ov::op::TopKMode::MIN,
        1,
        [](size_t rows, size_t row_size) { return topk_radix_random_values(rows, row_size, 13); },
        ov::op::TopKSortType::SORT_VALUES, radix_kernel, false));
    ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
        ov::PartialShape{4, 1024, 1, 1}, {{4, 1024, 1, 1}}, 1, ov::op::TopKMode::MAX, 1,
        [](size_t rows, size_t row_size) { return topk_radix_random_values(rows, row_size, 15); },
        ov::op::TopKSortType::SORT_VALUES, radix_kernel, false));
    ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(
        ov::PartialShape{4, 1023, 1, 1}, {{4, 1023, 1, 1}}, 1, ov::op::TopKMode::MAX, 1,
        [](size_t rows, size_t row_size) { return topk_radix_random_values(rows, row_size, 15); },
        ov::op::TopKSortType::SORT_VALUES, "arg_max_min_axis", false));
}

// Static shapes whose chunk count depends on the device and the row count, plus large K where
// the merge stage needs long chunks.
TYPED_TEST(arg_max_gpu_topk_radix, static_chunk_configs) {
    using T = TypeParam;
    const auto random = [](size_t rows, size_t row_size) { return topk_radix_random_values(rows, row_size, 21); };
    const auto tied = [](size_t rows, size_t row_size) { return topk_radix_tied_values(rows, row_size, 40, 23); };
    ASSERT_NO_FATAL_FAILURE(
        run_topk_radix_case<T>(ov::PartialShape{1, 8192, 1, 1}, {{1, 8192, 1, 1}}, 1, ov::op::TopKMode::MAX, 16, random));
    ASSERT_NO_FATAL_FAILURE(
        run_topk_radix_case<T>(ov::PartialShape{1, 16383, 1, 1}, {{1, 16383, 1, 1}}, 1, ov::op::TopKMode::MAX, 16, tied));
    ASSERT_NO_FATAL_FAILURE(
        run_topk_radix_case<T>(ov::PartialShape{2, 5000, 1, 1}, {{2, 5000, 1, 1}}, 1, ov::op::TopKMode::MIN, 32, tied));
    ASSERT_NO_FATAL_FAILURE(
        run_topk_radix_case<T>(ov::PartialShape{2, 81900, 1, 1}, {{2, 81900, 1, 1}}, 1, ov::op::TopKMode::MAX, 300, tied));
    ASSERT_NO_FATAL_FAILURE(
        run_topk_radix_case<T>(ov::PartialShape{1, 262144, 1, 1}, {{1, 262144, 1, 1}}, 1, ov::op::TopKMode::MAX, 2048, random));
    ASSERT_NO_FATAL_FAILURE(run_topk_radix_case<T>(ov::PartialShape{ov::Dimension::dynamic(), 81900, 1, 1},
                                                   {{2, 81900, 1, 1}, {5, 81900, 1, 1}},
                                                   1,
                                                   ov::op::TopKMode::MAX,
                                                   300,
                                                   tied));
}

// The chunk count follows the device size and a static row count, and large K limits it.
TEST(arg_max_gpu_topk_radix, chunk_config_follows_device_rows_and_k) {
    struct chunk_case {
        size_t rows;
        size_t sort_size;
        size_t top_k;
        uint32_t threads;
        size_t chunks;
    };
    // 960 threads keep 60 work-groups of 256 SIMD16 work-items resident, as on a 96-EU Xe3 iGPU
    const std::vector<chunk_case> cases = {
        {1, 8192, 16, 960, 8},       // a single row is split to fill the device, down to 1024 values per chunk
        {1, 8192, 16, 0, 1},         // unknown device size: below the default chunking threshold
        {64, 8192, 16, 960, 1},      // the rows alone fill the device
        {1, 4000, 16, 960, 1},       // too short to split
        {1, 8192, 1024, 960, 1},     // chunks could not hold 8 * K values
        {1, 262144, 2048, 960, 8},   // the merge input of chunks * K candidates stays within half a chunk
        {1, 248320, 16, 0, 31},      // default chunks of about 8192 values
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(std::to_string(c.rows) + "x" + std::to_string(c.sort_size) + " k=" + std::to_string(c.top_k) +
                     " threads=" + std::to_string(c.threads));
        kernel_selector::arg_max_min_params params;
        params.inputs[0] = kernel_selector::DataTensor(std::vector<size_t>{c.sort_size, c.rows, 1, 1},
                                                       kernel_selector::Datatype::F32,
                                                       kernel_selector::DataLayout::bfyx);
        params.outputs[0] = kernel_selector::DataTensor(std::vector<size_t>{c.top_k, c.rows, 1, 1},
                                                        kernel_selector::Datatype::F32,
                                                        kernel_selector::DataLayout::bfyx);
        params.argMaxMinAxis = kernel_selector::ArgMaxMinAxis::X;
        params.argMaxMinOut = kernel_selector::ArgMaxMinOut::MAX;
        params.argMaxMinSortType = kernel_selector::ArgMaxMinSortType::VALUE;
        params.topK = static_cast<uint32_t>(c.top_k);
        params.outputs_num = 1;
        params.engineInfo.maxWorkGroupSize = 256;
        params.engineInfo.maxLocalMemSize = 65536;
        params.engineInfo.maxThreadsPerDevice = c.threads;

        kernel_selector::ArgMaxMinKernelTopKRadix kernel;
        auto kernels_data = kernel.GetKernelsData(params);
        ASSERT_EQ(kernels_data.size(), size_t(1));
        const auto& kd = kernels_data[0];
        ASSERT_EQ(kd.kernels.size(), c.chunks > 1 ? size_t(2) : size_t(1));
        if (c.chunks > 1) {
            ASSERT_EQ(kd.internalBuffers.size(), size_t(3));
            ASSERT_EQ(kd.internalBuffers[1].byte_count, sizeof(uint32_t) * c.chunks * c.top_k * c.rows);
        }
    }
}

// Verify that arg_max_min_axis is selected over radix for small sort_size and k=1
TEST(arg_max_gpu_topk_radix, fallback_to_axis_for_small_sort_size_and_topk1) {
    auto& engine = get_test_engine();

    // sort_size=8 (< 256) + topK=1 → both conditions trigger axis fallback
    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {1, 8, 1, 1}});
    set_values(input, {ov::float16(1.f), ov::float16(5.f), ov::float16(3.f), ov::float16(9.f),
                       ov::float16(2.f), ov::float16(7.f), ov::float16(4.f), ov::float16(8.f)});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", {input_info("input")},
                             ov::op::TopKMode::MAX, 1, 1,
                             ov::op::TopKSortType::SORT_VALUES, true, false, data_types::f16, 2));

    auto config = get_test_default_config(engine);
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto info = network.get_primitive_info("arg_max");
    ASSERT_TRUE(info.find("arg_max_min_axis") != std::string::npos)
        << "Expected arg_max_min_axis, got: " << info;

    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(output, get_test_stream());
    ASSERT_NEAR(static_cast<float>(out_ptr[0]), 9.0f, 0.01f);
}

TEST(arg_max_gpu_dynamic_large_input, f32) {
    const int b = 1;
    const int f = 1024;
    const int y = 2048;
    const int x = 1;
    const int top_k = 1;

    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{
        ov::PartialShape::dynamic(4),
        data_types::f32,
        format::bfyx
    };

    auto input_layout_static = layout{
        ov::PartialShape{b, f, y, x},
        data_types::f32,
        format::bfyx
    };

    auto input = engine.allocate_memory(input_layout_static);

    std::vector<float> input_vec(b * f * y * x);
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    for (auto& v : input_vec)
        v = dist(gen);

    set_values(input, input_vec);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));

    topology.add(arg_max_min("arg_max",
        { input_info("input") },
        ov::op::TopKMode::MAX,
        top_k,
        2,
        ov::op::TopKSortType::SORT_VALUES,
        false,
        false,
        data_types::i32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("arg_max");
    ASSERT_TRUE(inst->get_impl()->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<int32_t, mem_lock_type::read> out_ptr(output, get_test_stream());

    for (int fi = 0; fi < f; fi++) {
        std::vector<float> row(
            input_vec.begin() + fi * y,
            input_vec.begin() + (fi + 1) * y
        );
        int expected = std::distance(
            row.begin(),
            std::max_element(row.begin(), row.end())
        );
        ASSERT_EQ(out_ptr[fi], expected);
    }
}
