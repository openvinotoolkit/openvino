/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/resample.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/reorder.hpp>
#include <api/data.hpp>

using namespace cldnn;
using namespace tests;

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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 2 } });

    auto output_size = tensor(batch(2), feature(2), spatial(6, 4));
    uint32_t num_filter = 0u;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
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
    auto output_ptr = output.pointer<float>();

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

    for (int i = 0; i < 2; ++i) { //B
        for (int j = 0; j < 2; ++j) { //F
            for (int k = 0; k < 4; ++k) { //Y
                for (int l = 0; l < 6; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));
    uint32_t num_filter = 1u;

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::caffe_bilinear));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.f, 1.25f, 1.75f, 2.f,
        1.5f, 1.75f, 2.25f, 2.5f,
        2.5f, 2.75f, 3.25f, 3.5f,
        3.f, 3.25f, 3.75f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 4; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(resample("upsampling", "input", output_size, 0, 0, 0, resample_type::bilinear));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.0f, 1.5f, 2.0f, 2.0f,
        2.0f, 2.5f, 3.0f, 3.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 4; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reorder("input_to_b_fs_yx_fsv16", "input", format::b_fs_yx_fsv16, data_types::f32));
    topology.add(resample("resample", "input_to_b_fs_yx_fsv16", output_size, 0, 0, 0, resample_type::bilinear));
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
    ASSERT_EQ(resample_out.get_layout().format, format::b_fs_yx_fsv16);

    auto output = outputs.at("res_to_bfyx").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.0f, 1.5f, 2.0f, 2.0f,
        2.0f, 2.5f, 3.0f, 3.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 4; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reorder("input_to_fs_b_yx_fsv32", "input", format::fs_b_yx_fsv32, data_types::f16));
    topology.add(resample("resample", "input_to_fs_b_yx_fsv32", output_size, 0, 0, 0, resample_type::bilinear));
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
    ASSERT_EQ(resample_out.get_layout().format, format::fs_b_yx_fsv32);

    auto output = outputs.at("res_to_bfyx").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.0f, 1.5f, 2.0f, 2.0f,
        2.0f, 2.5f, 3.0f, 3.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
        3.0f, 3.5f, 4.0f, 4.0f,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 4; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(resample("upsampling", "input", output_size, 0, 0, 1, resample_type::bilinear));

    set_values(input, {
            1.f, 2.f,
            3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
            1.000000f, 1.333333f, 1.666667f, 2.000000f,
            1.666667f, 2.000000f, 2.333333f, 2.666667f,
            2.333333f, 2.666667f, 3.000000f, 3.333333f,
            3.000000f, 3.333333f, 3.666667f, 4.000000f
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 4; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(5, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::nearest));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t)20);

    float answers[20] = {
        1.f, 1.f, 1.f, 2.f, 2.f,
        1.f, 1.f, 1.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 4.f, 4.f,
        3.f, 3.f, 3.f, 4.f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 5; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::i8, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(5, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::nearest));

    set_values<int8_t>(input, {
            1, 2,
            3, 4,
    });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    auto output_ptr = output.pointer<int8_t>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t)20);

    int8_t answers[20] = {
            1, 1, 1, 2, 2,
            1, 1, 1, 2, 2,
            3, 3, 3, 4, 4,
            3, 3, 3, 4, 4,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 5; ++l) { //X
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

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(6, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(resample("upsampling", "input", output_size, num_filter, resample_type::caffe_bilinear));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
               });

    cldnn::network net{ engine, topology };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    auto output_ptr = output.pointer<float>();

    EXPECT_EQ(output.get_layout().get_linear_size(), (size_t)24);

    float answers[24] = {
        1.f, 1.f, 1.33333f, 1.66667f, 2.f, 2.f,
        1.5f, 1.5f, 1.83333f, 2.16667f, 2.5f, 2.5f,
        2.5f, 2.5f, 2.83333f, 3.16667f, 3.5f, 3.5f,
        3.f, 3.f, 3.33333f, 3.66667f, 4.f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { //Y
        for (int l = 0; l < 6; ++l) { //X
            auto linear_id = l + k * 6;
            EXPECT_NEAR(answers[linear_id], output_ptr[linear_id], 5e-03F) << l << " " << k;
        }
    }
}
