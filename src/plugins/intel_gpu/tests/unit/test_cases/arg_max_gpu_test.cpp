// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/arg_max_min.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <arg_max_min_inst.h>
#include "test_utils.h"

#include "program_wrapper.h"

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
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        ASSERT_EQ(out_buffer[i], ref_vec[i]);
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
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());
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

TEST(arg_max_min_gpu, dynamic) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx};
    auto input = engine.allocate_memory(input_layout_static);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(arg_max_min("arg_max", { input_info("input") }, ov::op::TopKMode::MIN, top_k, 0));

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

    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), out_size);
    for (uint32_t i = 0; i < out_size; i++) {
        ASSERT_FLOAT_EQ(output_ptr[i], i < (out_size / 2) ? 0 : 1);
    }
}

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
