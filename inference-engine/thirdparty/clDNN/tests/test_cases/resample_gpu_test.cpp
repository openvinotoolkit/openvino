// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/resample.hpp>
#include <cldnn/primitives/reorder.hpp>
#include <cldnn/primitives/data.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(resample_gpu, basic_in2x3x2x2_nearest) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 3, 2 } });

    auto output_size = tensor(batch(2), feature(2), spatial(6, 4));
    uint32_t num_filter = 0u;

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::nearest));

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, -17.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
        1.f, 1.f, 2.f,   2.f,   -10.f,  -10.f,
        1.f, 1.f, 2.f,   2.f,   -10.f,  -10.f,
        3.f, 3.f, 4.f,   4.f,   -14.f,  -14.f,
        3.f, 3.f, 4.f,   4.f,   -14.f,  -14.f,
        5.f, 5.f, 6.f,   6.f,   -12.f,  -12.f,
        5.f, 5.f, 6.f,   6.f,   -12.f,  -12.f,
        7.f, 7.f, 8.f,   8.f,   -16.f,  -16.f,
        7.f, 7.f, 8.f,   8.f,   -16.f,  -16.f,
        0.f, 0.f, 0.f,   0.f,   -11.f,  -11.f,
        0.f, 0.f, 0.f,   0.f,   -11.f,  -11.f,
        0.5f,0.5f, -0.5f, -0.5f, -15.f,  -15.f,
        0.5f,0.5f, -0.5f, -0.5f, -15.f,  -15.f,
        1.5f,1.5f, 5.2f,  5.2f,  -13.f,  -13.f,
        1.5f,1.5f, 5.2f,  5.2f,  -13.f,  -13.f,
        12.f,12.f, 9.f,   9.f,  -17.f,  -17.f,
        12.f,12.f, 9.f,   9.f,  -17.f,  -17.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    EXPECT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id]));
                }
            }
        }
    }
}

TEST(resample_gpu, basic_in2x3x2x2_bilinear) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x4x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));
    uint32_t num_filter = 1u;

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::caffe_bilinear));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.f, 1.25f, 1.75f, 2.f,
        1.5f, 1.75f, 2.25f, 2.5f,
        2.5f, 2.75f, 3.25f, 3.5f,
        3.f, 3.25f, 3.75f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 4; ++l) { // X
            auto linear_id = l + k * 4;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, basic_in1x1x2x2_interp) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x4x4
    //  Sample Type: Interp

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, resample_type::bilinear));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.0f, 1.5f, 2.0f, 2.0f,
        2.0f, 2.5f, 3.0f, 3.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 4; ++l) { // X
            auto linear_id = l + k * 4;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, basic_in1x1x2x2_interp_f16) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x4x4
    //  Sample Type: Interp

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("input_to_b_fs_yx_fsv16", "input", format::b_fs_yx_fsv16, data_types::f32));
    topology.add(resample("resample", "input_to_b_fs_yx_fsv16", output_size, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, resample_type::bilinear));
    topology.add(reorder("res_to_bfyx", "resample", format::bfyx, data_types::f32));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    build_options bo;
    bo.set_option(build_option::outputs({"resample", "res_to_bfyx"}));

    cldnn::network net{ engine, topology, bo };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto resample_out = outputs.at("resample").get_memory();
    ASSERT_EQ(resample_out->get_layout().format, format::b_fs_yx_fsv16);

    auto output = outputs.at("res_to_bfyx").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.0f, 1.5f, 2.0f, 2.0f,
        2.0f, 2.5f, 3.0f, 3.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 4; ++l) { // X
            auto linear_id = l + k * 4;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, basic_in1x1x2x2_interp_fsv32) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x4x4
    //  Sample Type: Interp

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("input_to_fs_b_yx_fsv32", "input", format::fs_b_yx_fsv32, data_types::f16));
    topology.add(resample("resample", "input_to_fs_b_yx_fsv32", output_size, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, resample_type::bilinear));
    topology.add(reorder("res_to_bfyx", "resample", format::bfyx, data_types::f32));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    build_options bo;
    bo.set_option(build_option::outputs({"resample", "res_to_bfyx"}));

    cldnn::network net{ engine, topology, bo };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto resample_out = outputs.at("resample").get_memory();
    ASSERT_EQ(resample_out->get_layout().format, format::fs_b_yx_fsv32);

    auto output = outputs.at("res_to_bfyx").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.0f, 1.5f, 2.0f, 2.0f,
        2.0f, 2.5f, 3.0f, 3.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 4; ++l) { // X
            auto linear_id = l + k * 4;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}


TEST(resample_gpu, basic_in1x1x2x2_interp_align_1) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x4x4
    //  Sample Type: Interp

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, {0, 0, 0, 0}, {0, 0, 0, 0}, 1, resample_type::bilinear));

    set_values(input, {
            1.f, 2.f,
            3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
            1.000000f, 1.333333f, 1.666667f, 2.000000f,
            1.666667f, 2.000000f, 2.333333f, 2.666667f,
            2.333333f, 2.666667f, 3.000000f, 3.333333f,
            3.000000f, 3.333333f, 3.666667f, 4.000000f
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 4; ++l) { // X
            auto linear_id = l + k * 4;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, nearest_asymmetric) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x5x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(5, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::nearest));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t)20);

    float answers[20] = {
        1.f, 1.f, 1.f, 2.f, 2.f,
        1.f, 1.f, 1.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 4.f, 4.f,
        3.f, 3.f, 3.f, 4.f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 5; ++l) { // X
            auto linear_id = l + k * 5;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, nearest_asymmetric_i8) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x5x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(5, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::nearest));

    set_values<int8_t>(input, {
            1, 2,
            3, 4,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t)20);

    int8_t answers[20] = {
            1, 1, 1, 2, 2,
            1, 1, 1, 2, 2,
            3, 3, 3, 4, 4,
            3, 3, 3, 4, 4,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 5; ++l) { // X
            auto linear_id = l + k * 5;
            EXPECT_EQ(answers[linear_id], output_ptr[linear_id]);
        }
    }
}

TEST(resample_gpu, bilinear_asymmetric) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x5x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(6, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::caffe_bilinear));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
               });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    EXPECT_EQ(output->get_layout().get_linear_size(), (size_t)24);

    float answers[24] = {
        1.f, 1.f, 1.33333f, 1.66667f, 2.f, 2.f,
        1.5f, 1.5f, 1.83333f, 2.16667f, 2.5f, 2.5f,
        2.5f, 2.5f, 2.83333f, 3.16667f, 3.5f, 3.5f,
        3.f, 3.f, 3.33333f, 3.66667f, 4.f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 6; ++l) { // X
            auto linear_id = l + k * 6;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 5e-03F) << l << " " << k;
        }
    }
}

struct resample_random_test_params {
    data_types input_type;
    tensor input_size;
    tensor output_size;
    uint32_t num_filter;
    resample_type operation_type;
    uint32_t align_corners;
    format::type in_format;
    format::type out_format;
};

struct resample_random_test : testing::TestWithParam<resample_random_test_params>{
    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto size = mem->get_layout().size;
        size_t b = size.batch[0];
        size_t f = size.feature[0];
        size_t x = size.spatial[0];
        size_t y = size.spatial[1];

        auto data = generate_random_4d<T>(b, f, y, x, min, max, k);
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<FLOAT16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_nearest_typed(const memory::ptr input, const memory::ptr output, uint32_t align_corners) {
        auto output_lay = output->get_layout();
        size_t b = output_lay.size.batch[0];
        size_t f = output_lay.size.feature[0];
        size_t x = output_lay.size.spatial[0];
        size_t y = output_lay.size.spatial[1];
        size_t in_x = input->get_layout().size.spatial[0];
        size_t in_y = input->get_layout().size.spatial[1];
        float x_ratio = x > align_corners ? static_cast<float>(in_x - align_corners) / static_cast<float>(x - align_corners) : 0.f;
        float y_ratio = y > align_corners ? static_cast<float>(in_y - align_corners) / static_cast<float>(y - align_corners) : 0.f;

        cldnn::mem_lock<T> in_ptr(input, get_test_stream());
        cldnn::mem_lock<T> out_ptr(output, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto in_xi = static_cast<size_t>(floor(x_ratio * xi));
                        auto in_yi = static_cast<size_t>(floor(y_ratio * yi));
                        auto in_coords = tensor(batch(bi), feature(fi), spatial(in_xi, in_yi, 0, 0));
                        auto in_offset = input->get_layout().get_linear_offset(in_coords);
                        auto in_val = in_ptr[in_offset];
                        auto out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto out_offset = output->get_layout().get_linear_offset(out_coords);
                        auto out_val = out_ptr[out_offset];
                        EXPECT_EQ(in_val, out_val) << " at bi=" << bi << ", fi=" << fi << ", xi=" << xi << ", yi=" << yi;
                    }
                }
            }
        }
    }

    template <typename InT, typename OutT>
    void compare_bilinear_typed(const memory::ptr input, const memory::ptr output, uint32_t align_corners) {
        auto output_lay = output->get_layout();
        size_t b = output_lay.size.batch[0];
        size_t f = output_lay.size.feature[0];
        size_t x = output_lay.size.spatial[0];
        size_t y = output_lay.size.spatial[1];
        auto input_lay = input->get_layout();
        size_t in_x = input_lay.size.spatial[0];
        size_t in_y = input_lay.size.spatial[1];
        float x_ratio = x > align_corners ? static_cast<float>(in_x - align_corners) / static_cast<float>(x - align_corners) : 0.f;
        float y_ratio = y > align_corners ? static_cast<float>(in_y - align_corners) / static_cast<float>(y - align_corners) : 0.f;

        cldnn::mem_lock<InT> in_ptr(input, get_test_stream());
        cldnn::mem_lock<OutT> out_ptr(output, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto low_in_xi = static_cast<size_t>(floor(x_ratio * xi));
                        auto low_in_yi = static_cast<size_t>(floor(y_ratio * yi));
                        auto high_in_xi = static_cast<size_t>(ceil(x_ratio * xi));
                        auto high_in_yi = static_cast<size_t>(ceil(y_ratio * yi));

                        high_in_xi = std::min(high_in_xi, static_cast<size_t>(in_x - 1));
                        high_in_yi = std::min(high_in_yi, static_cast<size_t>(in_y - 1));

                        auto dx = x_ratio * xi - static_cast<float>(low_in_xi);
                        auto dy = y_ratio * yi - static_cast<float>(low_in_yi);

                        auto top_left_coords = tensor(batch(bi), feature(fi), spatial(low_in_xi, low_in_yi, 0, 0));
                        auto top_right_coords = tensor(batch(bi), feature(fi), spatial(high_in_xi, low_in_yi, 0, 0));
                        auto bottom_left_coords = tensor(batch(bi), feature(fi), spatial(low_in_xi, high_in_yi, 0, 0));
                        auto bottom_right_coords = tensor(batch(bi), feature(fi), spatial(high_in_xi, high_in_yi, 0, 0));

                        auto top_left_val = in_ptr[input_lay.get_linear_offset(top_left_coords)];
                        auto top_right_val = in_ptr[input_lay.get_linear_offset(top_right_coords)];
                        auto bottom_left_val = in_ptr[input_lay.get_linear_offset(bottom_left_coords)];
                        auto bottom_right_val = in_ptr[input_lay.get_linear_offset(bottom_right_coords)];

                        auto top_val = static_cast<float>(top_left_val)
                            + (static_cast<float>(top_right_val) - static_cast<float>(top_left_val)) * dx;
                        auto bottom_val = static_cast<float>(bottom_left_val)
                            + (static_cast<float>(bottom_right_val) - static_cast<float>(bottom_left_val)) * dx;

                        auto final_val = top_val + (bottom_val - top_val) * dy;

                        auto output_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto output_val = out_ptr[output_lay.get_linear_offset(output_coords)];

                        EXPECT_NEAR(static_cast<float>(output_val), final_val, 1.e-1f)
                            << " at bi=" << bi << ", fi=" << fi << ", xi=" << xi << ", yi=" << yi;
                    }
                }
            }
        }
    }

    void compare(const memory::ptr input, const memory::ptr output, resample_type operation, uint32_t align_corners) {
        auto dt = input->get_layout().data_type;
        if (operation == resample_type::nearest) {
            // Nearest resampling implicitly ignores align_corners
            if (dt == data_types::f32) {
                compare_nearest_typed<float>(input, output, 0);
            } else if (dt == data_types::f16) {
                compare_nearest_typed<FLOAT16>(input, output, 0);
            } else if (dt == data_types::i8) {
                compare_nearest_typed<int8_t>(input, output, 0);
            } else if (dt == data_types::u8) {
                compare_nearest_typed<uint8_t>(input, output, 0);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(dt);
            }
        } else if (operation == resample_type::bilinear) {
            if (dt == data_types::f32) {
                compare_bilinear_typed<float, float>(input, output, align_corners);
            } else if (dt == data_types::f16) {
                compare_bilinear_typed<FLOAT16, FLOAT16>(input, output, align_corners);
            } else if (dt == data_types::i8) {
                compare_bilinear_typed<int8_t, float>(input, output, align_corners);
            } else if (dt == data_types::u8) {
                compare_bilinear_typed<uint8_t, float>(input, output, align_corners);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(dt);
            }
        } else {
            FAIL() << "Not supported resample_type: " << static_cast<int32_t>(operation);
        }
    }

    void execute(const resample_random_test_params& params) {
        auto& engine = get_test_engine();

        auto in_layout = layout(params.input_type, params.in_format, params.input_size);

        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        auto prim = resample("resample", "in", params.output_size, params.num_filter, params.operation_type);
        prim.align_corners = params.align_corners;
        topo.add(prim);

        auto build_opts = build_options(
            build_option::force_implementations({ {"resample", {params.out_format, ""}} })
        );
        auto net = network(engine, topo, build_opts);

        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);
        net.set_input_data("in", in_mem);

        auto result = net.execute();
        auto output = result.at("resample").get_memory();

        std::string kernel = "";
        for (auto& info : net.get_primitives_info()) {
            if (info.original_id == "resample")
                kernel = info.kernel_id;
        }

        compare(in_mem, output, params.operation_type, params.align_corners);
    }
};

TEST_P(resample_random_test, random) {
    execute(GetParam());
}

struct resample_random_test_param_generator : std::vector<resample_random_test_params> {
    resample_random_test_param_generator& add(resample_random_test_params params) {
        push_back(params);
        return *this;
    }

    resample_random_test_param_generator& smoke_params(data_types type, format::type input_format, format::type output_format) {
        push_back(resample_random_test_params{ type, {1, 17, 5, 9}, {1, 17, 15, 18}, 1, resample_type::nearest, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {2, 17, 5, 9}, {2, 17, 15, 18}, 1, resample_type::nearest, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {1, 7, 10, 17}, {1, 7, 21, 35}, 1, resample_type::nearest, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {2, 7, 10, 17}, {2, 7, 21, 35}, 1, resample_type::nearest, 1, input_format, output_format });

        push_back(resample_random_test_params{ type, {1, 17, 5, 9}, {1, 17, 15, 18}, 1, resample_type::bilinear, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {2, 17, 5, 9}, {2, 17, 15, 18}, 1, resample_type::bilinear, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {1, 7, 10, 17}, {1, 7, 21, 35}, 1, resample_type::bilinear, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {2, 7, 10, 17}, {2, 7, 21, 35}, 1, resample_type::bilinear, 1, input_format, output_format });

        return *this;
    }

};

INSTANTIATE_TEST_SUITE_P(smoke_resample,
                        resample_random_test,
                        testing::ValuesIn(
                            resample_random_test_param_generator()
                            .smoke_params(data_types::i8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .smoke_params(data_types::u8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .smoke_params(data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)

                            .smoke_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                        ));


/////////////////////////////////////////////////////////////////////////

struct caffe_resample_random_test_params {
    data_types input_type;
    tensor input_size;
    tensor output_size;
    uint32_t num_filter;
    resample_type operation_type;
    uint32_t align_corners;
    format::type in_format;
    format::type out_format;
    std::vector<int32_t> pads_begin;
    std::vector<int32_t> pads_end;
};

struct caffe_resample_random_test : testing::TestWithParam<caffe_resample_random_test_params>
{
    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto size = mem->get_layout().size;
        size_t b = size.batch[0];
        size_t f = size.feature[0];
        size_t x = size.spatial[0];
        size_t y = size.spatial[1];

        auto data = generate_random_4d<T>(b, f, y, x, min, max, k);
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<FLOAT16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    bool compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.size.batch[0];
        size_t f = output_lay.size.feature[0];
        size_t x = output_lay.size.spatial[0];
        size_t y = output_lay.size.spatial[1];
        cldnn::mem_lock<T> ref_ptr(out_ref, get_test_stream());
        cldnn::mem_lock<T> opt_ptr(out_opt, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                        auto ref_out_val = ref_ptr[ref_out_offset];

                        auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                        auto opt_out_val = opt_ptr[opt_out_offset];

                        EXPECT_EQ(ref_out_offset, opt_out_offset);
                        EXPECT_EQ(opt_out_val, ref_out_val);
                        // EXPECT_NEAR(static_cast<float>(opt_out_val), static_cast<float>(ref_out_val), 1.e-1f);
                    }
                }
            }
        }

        return true;
    }

    void execute_compare(const caffe_resample_random_test_params& params, bool check_result) {
        auto& engine = get_test_engine();

        auto in_layout = layout(params.input_type, params.in_format, params.input_size);
        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        auto prim = resample("resample", "in", params.output_size, params.num_filter, params.operation_type);
        prim.align_corners = params.align_corners;
        prim.pads_begin = params.pads_begin;
        prim.pads_end = params.pads_end;
        topo.add(prim);

        auto build_opts = build_options();
        build_opts.set_option(build_option::outputs({"resample"}));
        build_opts.set_option(build_option::force_implementations({ {"resample", {params.in_format, "resample_ref"}} }));

        auto net = network(engine, topo, build_opts);
        net.set_input_data("in", in_mem);

        auto result = net.execute();
        auto output = result.at("resample").get_memory();

        // Execute resample_opt
        cldnn::topology topo_opt;
        topo_opt.add(input_layout("in", in_layout));
        auto prim_opt = resample("resample_opt", "in", params.output_size, params.num_filter, params.operation_type);
        prim_opt.align_corners = params.align_corners;
        prim_opt.pads_begin = params.pads_begin;
        prim_opt.pads_end = params.pads_end;
        topo_opt.add(prim_opt);

        auto build_opts_opt = build_options();
        build_opts_opt.set_option(build_option::outputs({"resample_opt"}));
        build_opts.set_option(build_option::force_implementations({ {"resample_opt", {params.in_format, "resample_opt"}} }));

        auto net_opt = network(engine, topo_opt, build_opts_opt);

        // Use in_mem from ref network
        net_opt.set_input_data("in", in_mem);

        auto result_opt = net_opt.execute();
        auto output_opt = result_opt.at("resample_opt").get_memory();

        if (check_result == true) {
            // Check data_types
            if (params.input_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (params.input_type == data_types::f16) {
                compare_outputs<FLOAT16>(output, output_opt);
            } else if (params.input_type == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (params.input_type == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        }
    }
};

struct caffe_resample_random_test_param_generator : std::vector<caffe_resample_random_test_params> {
    caffe_resample_random_test_param_generator& add(caffe_resample_random_test_params params) {
        push_back(params);
        return *this;
    }

    caffe_resample_random_test_param_generator& smoke_params(data_types type, format::type input_format, format::type output_format) {
        push_back(caffe_resample_random_test_params{ type, {1, 512, 16, 16}, {1, 512, 32, 32}, 1, resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 512, 32, 32}, {1, 512, 16, 16}, 1, resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 24, 32, 32}, {1, 24, 64, 64}, 1,   resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 24, 96, 96}, {1, 24, 32, 32}, 1,   resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 8, 64, 64},  {1, 8, 32, 32},  1,   resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 20, 10, 10}, {1, 20, 20, 20}, 1,   resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 20, 20, 20}, {1, 20, 10, 10}, 1,   resample_type::caffe_bilinear, 1, input_format, output_format, {}, {}});
        // Padding applied
        push_back(caffe_resample_random_test_params{ type, {1, 96, 16, 16}, {1, 96, 32, 32}, 1, resample_type::caffe_bilinear, 1, input_format, output_format, {0, 0, 1, 1}, {0, 0, 1, 1}});
        push_back(caffe_resample_random_test_params{ type, {1, 96, 32, 32}, {1, 96, 16, 16}, 1, resample_type::caffe_bilinear, 1, input_format, output_format, {0, 0, 1, 1}, {0, 0, 1, 1}});
        return *this;
    }
};

TEST_P(caffe_resample_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true);
}

INSTANTIATE_TEST_SUITE_P(caffe_smoke_caffe_fsv16,
                        caffe_resample_random_test,
                        testing::ValuesIn(
                            caffe_resample_random_test_param_generator()
                            .smoke_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                        ));

INSTANTIATE_TEST_SUITE_P(caffe_smoke_caffe_fsv32,
                        caffe_resample_random_test,
                        testing::ValuesIn(
                            caffe_resample_random_test_param_generator()
                            .smoke_params(data_types::f16, format::fs_b_yx_fsv32, format::fs_b_yx_fsv32)
                        ));

TEST(resample_gpu, interpolate_in2x2x3x2_nearest1) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    auto output_size = tensor(batch(b), feature(f), spatial(x*2, y*2));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel;
    nearest_mode nm = nearest_mode::ceil;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  1.f,  1.f,  1.f,
         2.f,  3.f,  3.f,  3.f,
         2.f,  3.f,  3.f,  3.f,
         4.f,  5.f,  5.f,  5.f,
         4.f,  5.f,  5.f,  5.f,
         4.f,  5.f,  5.f,  5.f,

         6.f,  7.f,  7.f,  7.f,
         8.f,  9.f,  9.f,  9.f,
         8.f,  9.f,  9.f,  9.f,
        10.f, 11.f, 11.f, 11.f,
        10.f, 11.f, 11.f, 11.f,
        10.f, 11.f, 11.f, 11.f,

        12.f, 13.f, 13.f, 13.f,
        14.f, 15.f, 15.f, 15.f,
        14.f, 15.f, 15.f, 15.f,
        16.f, 17.f, 17.f, 17.f,
        16.f, 17.f, 17.f, 17.f,
        16.f, 17.f, 17.f, 17.f,

        18.f, 19.f, 19.f, 19.f,
        20.f, 21.f, 21.f, 21.f,
        20.f, 21.f, 21.f, 21.f,
        22.f, 23.f, 23.f, 23.f,
        22.f, 23.f, 23.f, 23.f,
        22.f, 23.f, 23.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    EXPECT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest2) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    auto output_size = tensor(batch(b), feature(f), spatial(x*2, y*2));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel;
    nearest_mode nm = nearest_mode::round_prefer_floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  1.f,  1.f,
         0.f,  0.f,  1.f,  1.f,
         2.f,  2.f,  3.f,  3.f,
         2.f,  2.f,  3.f,  3.f,
         4.f,  4.f,  5.f,  5.f,
         4.f,  4.f,  5.f,  5.f,

         6.f,  6.f,  7.f,  7.f,
         6.f,  6.f,  7.f,  7.f,
         8.f,  8.f,  9.f,  9.f,
         8.f,  8.f,  9.f,  9.f,
        10.f, 10.f, 11.f, 11.f,
        10.f, 10.f, 11.f, 11.f,

        12.f, 12.f, 13.f, 13.f,
        12.f, 12.f, 13.f, 13.f,
        14.f, 14.f, 15.f, 15.f,
        14.f, 14.f, 15.f, 15.f,
        16.f, 16.f, 17.f, 17.f,
        16.f, 16.f, 17.f, 17.f,

        18.f, 18.f, 19.f, 19.f,
        18.f, 18.f, 19.f, 19.f,
        20.f, 20.f, 21.f, 21.f,
        20.f, 20.f, 21.f, 21.f,
        22.f, 22.f, 23.f, 23.f,
        22.f, 22.f, 23.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    EXPECT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest3) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    auto output_size = tensor(batch(b), feature(f), spatial(x*2, y*2));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel;
    nearest_mode nm = nearest_mode::round_prefer_ceil;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  1.f,  1.f,
         0.f,  0.f,  1.f,  1.f,
         2.f,  2.f,  3.f,  3.f,
         2.f,  2.f,  3.f,  3.f,
         4.f,  4.f,  5.f,  5.f,
         4.f,  4.f,  5.f,  5.f,

         6.f,  6.f,  7.f,  7.f,
         6.f,  6.f,  7.f,  7.f,
         8.f,  8.f,  9.f,  9.f,
         8.f,  8.f,  9.f,  9.f,
        10.f, 10.f, 11.f, 11.f,
        10.f, 10.f, 11.f, 11.f,

        12.f, 12.f, 13.f, 13.f,
        12.f, 12.f, 13.f, 13.f,
        14.f, 14.f, 15.f, 15.f,
        14.f, 14.f, 15.f, 15.f,
        16.f, 16.f, 17.f, 17.f,
        16.f, 16.f, 17.f, 17.f,

        18.f, 18.f, 19.f, 19.f,
        18.f, 18.f, 19.f, 19.f,
        20.f, 20.f, 21.f, 21.f,
        20.f, 20.f, 21.f, 21.f,
        22.f, 22.f, 23.f, 23.f,
        22.f, 22.f, 23.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    EXPECT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest4) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    auto output_size = tensor(batch(b), feature(f), spatial(x*2, y*2));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel;
    nearest_mode nm = nearest_mode::floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         2.f,  2.f,  2.f,  3.f,
         2.f,  2.f,  2.f,  3.f,
         4.f,  4.f,  4.f,  5.f,

         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         8.f,  8.f,  8.f,  9.f,
         8.f,  8.f,  8.f,  9.f,
        10.f, 10.f, 10.f, 11.f,

        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        14.f, 14.f, 14.f, 15.f,
        14.f, 14.f, 14.f, 15.f,
        16.f, 16.f, 16.f, 17.f,

        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        20.f, 20.f, 20.f, 21.f,
        20.f, 20.f, 20.f, 21.f,
        22.f, 22.f, 22.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    EXPECT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest5) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    auto output_size = tensor(batch(b), feature(f), spatial(x*2, y*2));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel;
    nearest_mode nm = nearest_mode::simple;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         2.f,  2.f,  2.f,  3.f,
         2.f,  2.f,  2.f,  3.f,
         4.f,  4.f,  4.f,  5.f,

         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         8.f,  8.f,  8.f,  9.f,
         8.f,  8.f,  8.f,  9.f,
        10.f, 10.f, 10.f, 11.f,

        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        14.f, 14.f, 14.f, 15.f,
        14.f, 14.f, 14.f, 15.f,
        16.f, 16.f, 16.f, 17.f,

        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        20.f, 20.f, 20.f, 21.f,
        20.f, 20.f, 20.f, 21.f,
        22.f, 22.f, 22.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    EXPECT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode1) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel;
    nearest_mode nm = nearest_mode::round_prefer_floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  0.f,  1.f,
         4.f,  4.f,  5.f,

         6.f,  6.f,  7.f,
        10.f, 10.f, 11.f,

        12.f, 12.f, 13.f,
        16.f, 16.f, 17.f,

        18.f, 18.f, 19.f,
        22.f, 22.f, 23.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode2) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 1;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::pytorch_half_pixel;
    nearest_mode nm = nearest_mode::round_prefer_floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  0.f,  1.f,
         6.f,  6.f,  7.f,

        12.f, 12.f, 13.f,
        18.f, 18.f, 19.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode3) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::asymmetric;
    nearest_mode nm = nearest_mode::round_prefer_floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  1.f,  1.f,
         2.f,  3.f,  3.f,

         6.f,  7.f,  7.f,
         8.f,  9.f,  9.f,

        12.f, 13.f, 13.f,
        14.f, 15.f, 15.f,

        18.f, 19.f, 19.f,
        20.f, 21.f, 21.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode4) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::tf_half_pixel_for_nn;
    nearest_mode nm = nearest_mode::round_prefer_floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         2.f,  3.f,  3.f,
         4.f,  5.f,  5.f,

         8.f,  9.f,  9.f,
        10.f, 11.f, 11.f,

        14.f, 15.f, 15.f,
        16.f, 17.f, 17.f,

        20.f, 21.f, 21.f,
        22.f, 23.f, 23.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode5) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::nearest;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::align_corners;
    nearest_mode nm = nearest_mode::round_prefer_floor;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  0.f,  1.f,
         4.f,  4.f,  5.f,

         6.f,  6.f,  7.f,
        10.f, 10.f, 11.f,

        12.f, 12.f, 13.f,
        16.f, 16.f, 17.f,

        18.f, 18.f, 19.f,
        22.f, 22.f, 23.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_cubic) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::cubic;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.29600694f,  0.8828125f,  1.46961806f,
         3.53038194f,  4.1171875f,  4.70399306f,

         6.29600694f,  6.8828125f,  7.46961806f,
         9.53038194f, 10.1171875f, 10.70399306f,

        12.29600694f, 12.8828125f, 13.46961806f,
        15.53038194f, 16.1171875f, 16.70399306f,

        18.29600694f, 18.8828125f, 19.46961806f,
        21.53038194f, 22.1171875f, 22.70399306f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_cubic2) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 1;
    int f = 1;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::cubic;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        5.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
          5.34722222f,  3.f, 0.65277778f,
          1.91319444f, 2.5f, 3.08680556f,
          3.91319444f, 4.5f, 5.08680556f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_linear) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::caffe_bilinear;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.5f,  1.f,  1.5f,
         3.5f,  4.f,  4.5f,

         6.5f,  7.f,  7.5f,
         9.5f, 10.f, 10.5f,

        12.5f, 13.f, 13.5f,
        15.5f, 16.f, 16.5f,

        18.5f, 19.f, 19.5f,
        21.5f, 22.f, 22.5f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_linear_onnx) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();

    int b = 1;
    int f = 1;
    int y = 2;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 4;
    x = 4;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::bilinear;
    coordinate_transformation_mode ctm = coordinate_transformation_mode::asymmetric;
    resample::AxesAndScales axesAndScales;
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::sizes;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
             1.f, 1.33333f, 1.66667f,      2.f,
        1.66667f,      2.f, 2.33333f, 2.66667f,
        2.33333f, 2.66667f,      3.f, 3.33333f,
             3.f, 3.33333f, 3.66667f,      4.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in1x1x2x4_linear_scale) {
    //  Input  : 1x1x2x4
    //  Output : 1x1x1x2
    //  Sample Type: Linear

    auto& engine = get_test_engine();

    int b = 1;
    int f = 1;
    int y = 2;
    int x = 4;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 1;
    x = 2;
    auto output_size = tensor(batch(b), feature(f), spatial(x, y));

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    resample_type mode = resample_type::caffe_bilinear;
    resample::AxesAndScales axesAndScales = {
        {cldnn::resample::resample_axis::along_y, 0.6f},
        {cldnn::resample::resample_axis::along_x, 0.6f},
    };
    shape_calculation_mode shapeCalcMode = shape_calculation_mode::scales;
    topology.add(resample("interpolate", "input", output_size, axesAndScales, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
    });

    cldnn::network net {engine, topology };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         2.6666665f,  4.3333331f
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}
