/*
// Copyright (c) 2017 Intel Corporation
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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/scale.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "api/CPP/reorder.hpp"

#include <iostream>

using namespace cldnn;
using namespace tests;

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.3f, 1.1f, 1.3f,
        0.2f, 0.4f, 1.2f, 1.4f,
        0.25f, 0.5f, 1.25f, 1.5f,
        0.6f, 0.8f, 1.6f, 1.8f,
        0.7f, 0.9f, 1.7f, 1.9f,
        0.75f, 1.f, 1.75f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size_bfyx) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -10.f, 0.f, 0.f, -11.f,
        3.f, 4.f, -14.f, 0.5f, -0.5f, -15.f,
        5.f, 6.f, -12.f, 1.5f, 5.2f, -13.f,
        7.f, 8.f, -16.f, 12.f, 8.f, -17.f
    };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.75f, 0.8f, 0.9f, 1.f,
        1.1f, 1.2f, 1.25f, 1.3f, 1.4f, 1.5f,
        1.6f, 1.7f, 1.75f, 1.8f, 1.9f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size_scale_bfyx) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.75f, 0.8f, 0.9f, 1.f,
        1.1f, 1.2f, 1.25f, 1.3f, 1.4f, 1.5f,
        1.6f, 1.7f, 1.75f, 1.8f, 1.9f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = l + x_size * (k + y_size * (j + i * feature_num));
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_same_size_bias_term) {
    //  Scale  : 2x3x2x2
    //  Bias   : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25   b1:   0.3   0.4   0.5
    //  f0: b0:  0.6    0.7  0.75   b1:   0.8   0.9   1  
    //  f1: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5     
    //  f1: b0:  1.6    1.7  1.75   b1:   1.8   1.9   2
    //
    //  Bias:
    //  f0: b0:  1.1    1.2  1.25   b1:   1.3   1.4   1.5
    //  f0: b0:  2.6    2.7  2.75   b1:   2.8   2.9   2  
    //  f1: b0:  3.1    3.2  3.25   b1:   3.3   3.4   3.5     
    //  f1: b0:  4.6    4.7  4.75   b1:   4.8   4.9   4

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 2 } });
    auto bias = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(input_layout("bias", bias.get_layout()));
    topology.add(scale("scale", "input", "scale_input", "bias"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.3f, 1.1f, 1.3f,
        0.2f, 0.4f, 1.2f, 1.4f,
        0.25f, 0.5f, 1.25f, 1.5f,
        0.6f, 0.8f, 1.6f, 1.8f,
        0.7f, 0.9f, 1.7f, 1.9f,
        0.75f, 1.f, 1.75f, 2.f
    };
    set_values(scale_input, scale_input_vec);

    std::vector<float> bias_vec = {
        1.1f, 2.3f, 3.1f, 4.3f,
        1.2f, 2.4f, 3.2f, 4.4f,
        1.25f, 2.5f, 3.25f, 4.5f,
        1.6f, 2.8f, 3.6f, 4.8f,
        1.7f, 2.9f, 3.7f, 4.9f,
        1.75f, 2.f, 3.75f, 4.f
    };
    set_values(bias, bias_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);
    network.set_input_data("bias", bias);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_input_vec[i] + bias_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_scalar) {
    //  Scale  : 1
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  0.1    0.2

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = 0;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_y) {
    //  Scale  : 2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  0.1    0.2

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1,1,1,y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
        0.2f,
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = k;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_fb) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale: per feature per batch
    //  f0b0: 0.1   f0b1: 0.2
    //  f1b0: 0.5   f1b1: 2.0

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.2f, 0.5f, 2.0f,
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = i + feature_num * j;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_f) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale: per feature
    //  f0bx: 0.1   f1bx: 0.2

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, feature_num, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        //f0bx  //f1bx
        0.1f,   0.2f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = j;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_x) {
    //  Scale  : 3
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  0.1    0.2  0.25

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, x_size, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
        0.2f,
        0.25f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = l;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_xy) {
    //  Scale  : 2x3x1x1
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0:  0.1    0.2  0.25
    //  f0:  0.6    0.7  0.75

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f,
        0.2f,
        0.25f,
        0.6f,
        0.7f,
        0.75f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = l + x_size * k;
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_batch1) {
    //  Scale  : 2x3x2x1
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Scale:
    //  f0: b0:  0.1    0.2  0.25
    //  f0: b0:  0.6    0.7  0.75
    //  f1: b0:  1.1    1.2  1.25    
    //  f1: b0:  1.6    1.7  1.75

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ batch_num, feature_num, x_size, y_size } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 1.1f,
        0.2f, 1.2f,
        0.25f, 1.25f,
        0.6f, 1.6f,
        0.7f, 1.7f,
        0.75f, 1.75f
    };
    set_values(scale_input, scale_input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int j = 0; j < feature_num; ++j) { //F
        for (int i = 0; i < batch_num; ++i) { //B
            for (int k = 0; k < y_size; ++k) { //Y
                for (int l = 0; l < x_size; ++l) { //X
                    int linear_id = i + batch_num * (j + feature_num * (l + x_size * k));
                    int linear_id_scale = j + feature_num * (l + x_size * k);
                    EXPECT_NEAR(output_ptr[linear_id], input_vec[linear_id] * scale_input_vec[linear_id_scale], 1e-05F);
                }
            }
        }
    }
}

TEST(scale_gpu, basic_in2x3_scale_same_size_bx) {
    //  Scale  : 2x3
    //  Bias   : 2x3
    //  Input  : 2x3
    //  Output : 2x3

    //  Input:
    //  b0: 1  2  -0.75
    //  b1: 0 -1.5  -3
    //
    //  Scale:
    //  b0: 3.1  0.2  0.17
    //  b1:  10   -3     1

    //  Bias:
    //  b0: -0.1 3.2  7
    //  b1: 0    1   -1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 1 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 1 } });
    auto bias_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(input_layout("bias_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", "bias_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -0.75f,
        0.f, -1.5f, -3.f,
    };
    set_values(input, input_vec);

    std::vector<float> scale_vec = {
        3.1f, 0.2f, 0.17f,
        10.f, -3.f, 1.f,
    };
    set_values(scale_input, scale_vec);

    std::vector<float> bias_vec = {
        -0.1f, 3.2f, 7.f,
        0.f, 1.f, -1.f,
    };
    set_values(bias_input, bias_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);
    network.set_input_data("bias_input", bias_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_vec[i] + bias_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3_scale_same_size_xb) {
    //  Scale  : 2x3
    //  Bias   : 2x3
    //  Input  : 2x3
    //  Output : 2x3

    //  Input:
    //  x0: 1     2  -0.75
    //  x1: 0  -1.5     -3
    //
    //  Scale:
    //  x0: 3.1   0.2  0.17
    //  x1: 10     -3     1

    //  Bias:
    //  x0: -0.1  3.2   7
    //  x1: 0       1  -1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 3, 1, 2, 1 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 3, 1, 2, 1 } });
    auto bias_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 3, 1, 2, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(input_layout("bias_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", "bias_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -0.75f,
        0.f, -1.5f, -3.f,
    };
    set_values(input, input_vec);

    std::vector<float> scale_vec = {
        3.1f, 0.2f, 0.17f,
        10.f, -3.f, 1.f,
    };
    set_values(scale_input, scale_vec);

    std::vector<float> bias_vec = {
        -0.1f, 3.2f, 7.f,
        0.f, 1.f, -1.f,
    };
    set_values(bias_input, bias_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);
    network.set_input_data("bias_input", bias_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_vec[i] + bias_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3_scale_single_value_bx) {
    //  Scale  : 1x1
    //  Bias   : 1x1
    //  Input  : 2x3
    //  Output : 2x3

    //  Input:
    //  b0: 1    2 -0.75
    //  b1: 0 -1.5    -3
    //
    //  Scale:
    //  3.1

    //  Bias:
    //  -0.1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 1 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto bias_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(input_layout("bias_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", "bias_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -0.75f,
        0.f, -1.5f, -3.f,
    };
    set_values(input, input_vec);

    std::vector<float> scale_vec = {
        3.1f,
    };
    set_values(scale_input, scale_vec);

    std::vector<float> bias_vec = {
        -0.1f,
    };
    set_values(bias_input, bias_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);
    network.set_input_data("bias_input", bias_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_vec[0] + bias_vec[0], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3_scale_single_value_xb) {
    //  Scale  : 1x1
    //  Bias   : 1x1
    //  Input  : 2x3
    //  Output : 2x3

    //  Input:
    //  x0: 1     2 -0.75
    //  x1: 0  -1.5    -3
    //
    //  Scale:
    //  3.1

    //  Bias:
    //  -0.1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 3, 1, 2, 1 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 1, 1 } });
    auto bias_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(input_layout("bias_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input", "bias_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -0.75f,
        0.f, -1.5f, -3.f,
    };
    set_values(input, input_vec);

    std::vector<float> scale_vec = {
        3.1f,
    };
    set_values(scale_input, scale_vec);

    std::vector<float> bias_vec = {
        -0.1f,
    };
    set_values(bias_input, bias_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);
    network.set_input_data("bias_input", bias_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_vec[0] + bias_vec[0], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3_scale_same_size_no_bias_bx) {
    //  Scale  : 2x3
    //  Input  : 2x3
    //  Output : 2x3

    //  Input:
    //  b0: 1    2 -0.75
    //  b1: 0 -1.5    -3
    //
    //  Scale:
    //  b0: 3.1   0.2   0.17
    //  b1: 10     -3      1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 1 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 3, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -0.75f,
        0.f, -1.5f, -3.f,
    };
    set_values(input, input_vec);

    std::vector<float> scale_vec = {
        3.1f, 0.2f, 0.17f,
        10.f, -3.f, 1.f,
    };
    set_values(scale_input, scale_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3_scale_same_size_no_bias_xb) {
    //  Scale  : 2x3
    //  Input  : 2x3
    //  Output : 2x3

    //  Input:
    //  x0: 1     2  -0.75
    //  x1: 0  -1.5     -3
    //
    //  Scale:
    //  x0: 3.1    0.2   0.17
    //  x1: 10      -3      1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 3, 1, 2, 1 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::yxfb, { 3, 1, 2, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(input_layout("scale_input", scale_input.get_layout()));
    topology.add(scale("scale", "input", "scale_input"));

    std::vector<float> input_vec = {
        1.f, 2.f, -0.75f,
        0.f, -1.5f, -3.f,
    };
    set_values(input, input_vec);

    std::vector<float> scale_vec = {
        3.1f, 0.2f, 0.17f,
        10.f, -3.f, 1.f,
    };
    set_values(scale_input, scale_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("scale_input", scale_input);

    auto outputs = network.execute();

    auto output = outputs.at("scale").get_memory();
    auto output_ptr = output.pointer<float>();

    for (unsigned int i = 0; i < input_vec.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], input_vec[i] * scale_vec[i], 1e-05F);
    }
}

TEST(scale_gpu, basic_in2x3x2x2_scale_yxfb_bfyx_same_size_padding) {
    //  Scale  : 2x2x1x1
    //  Input  : 2x2x1x1
    //  Output : 2x2x1x1
    //  Output Padding: 2x2
    //  Input Padding: 2x1 (with reorder)

    //  Input:
    //  1    2
    //  3    4

    //
    //  Scale:
    //  0.1    0.2
    //  0.6    0.5
     
    const auto& engine = get_test_engine();
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor(1, 1, 2, 2);

        auto input = memory::allocate(engine, { data_types::f32, *it, input_tensor });
        auto scale_input = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input", input.get_layout()));
        topology.add(reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 1, 2 }, 0 })));
        topology.add(input_layout("scale_input", scale_input.get_layout()));
        topology.add(scale("scale", "reorder", "scale_input", padding( { 0, 0, 2, 2 }, 0 )));

        std::vector<float> input_vec = { 1.f, 2.f, 3.f, 4.f };
        set_values(input, input_vec);

        std::vector<float> scale_input_vec = { 0.1f, 0.2f, 0.6f, 0.5f };
        set_values(scale_input, scale_input_vec);

        std::vector<float> expected = {
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.1f, 0.4f, 0.f, 0.f,
            0.f, 0.f, 1.8f, 2.0f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        };

        network network(engine, topology);

        network.set_input_data("input", input);
        network.set_input_data("scale_input", scale_input);

        auto outputs = network.execute();

        auto output = outputs.at("scale").get_memory();
        auto output_ptr = output.pointer<float>();

        for (unsigned int i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(output_ptr[i], expected[i], 1e-05F);
        }
    }
}
//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO: this should be done using TEST_P or some equivallent construct
static network setup_scale_network(
    const data_types dt,
    const tensor input_tensor,
    const tensor scale_tensor,
    const tensor bias_tensor,
    const format f,
    const format of,
    bool pass_bias          //TODO: a WA for lack of std::optional<tensor> bias
)
{
    const auto& engine = get_test_engine();
    topology topology;

    auto input_mem = memory::allocate(engine, { dt, f, input_tensor });
    auto scale_mem = memory::allocate(engine, { dt, of, scale_tensor });
    topology.add(input_layout("input", input_mem.get_layout()));
    topology.add(input_layout("scale_input", scale_mem.get_layout()));

    if (pass_bias)
    {
        auto bias_mem = memory::allocate(engine, { dt, f, bias_tensor });
        topology.add(input_layout("bias_input", bias_mem.get_layout()));

        topology.add(scale("scale", "input", "scale_input", "bias_input" ));
    }
    else
    {
        topology.add(scale("scale", "input", "scale_input"));
    }
//TODO: this will be supported after the API change
//    else
//    {
//        assert(!pass_bias);
//
//        topology.add(scale("scale", "input", "scale_input"));
//    }

    return network(engine, topology);
}

TEST(NegativeScaleTest, TestAll) {
    auto d = data_types::f32;
    auto f = format::bfyx;
    auto of = format::yxfb;

    std::vector<int> t { 3, 4, 5, 6 };
    std::vector<int> t2 { 5, 6, 4, 3 };

    // broadcast rules mean that either the dim size is equal to input dim or is 1
    std::vector<std::vector<int>> good_ts =
    {
        { 1, 4, 5, 6 }, { 3, 1, 5, 6 }, { 3, 4, 1, 6 }, { 3, 4, 5, 1 },
        { 1, 1, 5, 6 }, { 1, 4, 1, 6 }, { 1, 4, 5, 1 }, { 3, 1, 1, 6 }, { 3, 1, 5, 1 }, { 3, 4, 1, 1 },
        { 1, 1, 1, 6 }, { 1, 1, 5, 1 }, { 1, 4, 1, 1 }, { 3, 1, 1, 1 }
    };
    std::vector<std::vector<int>> bad_ts = { { 2, 4, 5, 6 }, { 3, 2, 5, 6 }, { 3, 4, 2, 6 }, { 3, 4, 5, 2 } };

    //TODO: should be ASSERT_THROW(statement, exception_type) - but what exception type?
    ASSERT_ANY_THROW(setup_scale_network(d, { }, { }, { }, f, of, false));
    ASSERT_ANY_THROW(setup_scale_network(d, { }, { }, { }, f, of, true));

    ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(t2), tensor(t), f, of, true));
    ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(t2), tensor(t), f, of, false));

    // make sure that it's the input that's masked in the scale/bias with a "1", not ther other way around
    for (const auto & good : good_ts)
    {
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(good), tensor(t), tensor(t), f, of, true));
    }

    // sizes must either be equal to input or at most have 
    for (const auto & bad : bad_ts)
    {
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(bad), tensor(t), f, of, true));
        ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(t), tensor(bad), f, of, true));

        for (const auto & good : good_ts)
        {
            ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(bad), tensor(good), f, of, true));
            ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(good), tensor(bad), f, of, true));
        }
    }

    // we expect the broadcast mask to be identical for scale and bias, when present
    for (unsigned i = 0; i < good_ts.size(); ++i)
    for (unsigned j = 0; j < good_ts.size(); ++j)
        if (i != j)
        {
            ASSERT_ANY_THROW(setup_scale_network(d, tensor(t), tensor(good_ts[i]), tensor(good_ts[j]), f, of, true));
        }

}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class scale_test : public tests::generic_test
{
public:
    static void TearDownTestCase()
    {
        all_generic_params.clear();
        all_layer_params.clear();
    }

    //TODO: use an enum instead of int i
    static std::vector<cldnn::primitive*> generate_specific_test_params(int variant)
    {
        std::vector<cldnn::primitive*> all_layer_params;

        switch(variant)
        {
            case 0: all_layer_params.push_back(new scale("scale", "input0", "input1")); break;
            case 1: all_layer_params.push_back(new scale("scale", "input0", "input1", "input2")); break;
                    //    case 3: all_layer_params.push_back(new scale("scale", "input0", "input1", true));    // This case should be checked by negative_scale_test
                    //    case 4: all_layer_params.push_back(new scale("scale", "input0", "input1", false));    // This case should be checked by negative_scale_test
            default: assert(0);
        }

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params(int variant)
    {
        assert(!variant || variant == 1);

        std::vector<tests::test_params*> all_generic_params;

        auto data_types = test_data_types();

        for (cldnn::data_types dt : data_types)
        for (tensor & t : test_input_sizes)
        {
            std::vector<std::vector<int>> attempted_dims;

            for (int32_t b : test_batch_sizes)
            for (auto f : test_feature_sizes)
            for (int mask = 0; mask < 16; ++mask)    //TODO: do we want to restrict it to some smaller subset like for (auto mask : { 0, 1, 3, 7, 15, 5, 10})? the problem is that because of the layout we might miss some interesting combinations since this is effectively hardcoded int he kernels
            {
                const int w = t.spatial[0];
                const int h = t.spatial[1];

                const auto mb = mask & 0x8 ? b : 1;
                const auto mf = mask & 0x4 ? f : 1;
                const auto mh = mask & 0x2 ? h : 1;
                const auto mw = mask & 0x1 ? w : 1;

                // avoid adding test cases with different masks leading to the same dimensions
                if(attempted_dims.end() == std::find_if(attempted_dims.begin(), attempted_dims.end(), [=](const std::vector<int> & arr) { return arr[0] == mb && arr[1] == mf && arr[2] == mh && arr[3] == mw; }))
                {
                    std::vector<int> tmp { mb, mf, mh, mw };
                    attempted_dims.push_back(tmp);

                    test_params * tp = new test_params();
                    tp->data_type = dt;

                    tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(  b, f, w, h  )));
                    tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(  mb, mf, mw, mh  )));
                    if (variant)
                            tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(  mb, mf, mw, mh  )));

                    all_generic_params.emplace_back(tp);
                }
            }
        }

        return all_generic_params;
    }

    static std::vector<std::tuple<test_params*, cldnn::primitive*>> generate_all_test_params()
    {
        std::vector<std::tuple<test_params*, cldnn::primitive*>> res;

        for (int variant = 0; variant <= 1; ++variant)
        {
            auto tpv = generate_generic_test_params(variant); 
            auto pv = generate_specific_test_params(variant);

            for (auto & tp : tpv)
                all_generic_params.emplace_back(tp);

            for (auto & p : pv)
                all_layer_params.emplace_back(p);

            for (auto & tp : tpv)
            for (auto & p: pv)
                res.emplace_back(tp, p);
        }

        return res;
    }

    virtual bool is_format_supported(cldnn::format format) override
    {
        return format == cldnn_format_type::cldnn_format_bfyx;
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<memory> & inputs)
    {
        assert(inputs.size() == 3 || inputs.size() == 2);
        const bool bias_input_present = inputs.size() == 3;

        const memory & input = inputs[0];
        const memory & scale = inputs[1];
        const memory * bias = bias_input_present ? &inputs[2] : nullptr;
        assert(!bias_input_present || bias);

        //Output is bfyx
        auto output = memory::allocate(engine, cldnn::layout(input.get_layout().data_type, cldnn::format::bfyx, input.get_layout().size ));

        const auto in0_mem = input.pointer<Type>();
        const auto in1_mem = scale.pointer<Type>();
        const auto in2_mem_ptr = bias ? std::make_shared<pointer<Type>>(*bias) : nullptr;
        const Type * const in2_mem = in2_mem_ptr ? in2_mem_ptr->data() : nullptr; //TODO: is the condition needed or is it nullptr anyway?
        auto out_mem = output.pointer<Type>();

        const auto input_sizes = input.get_layout().size.sizes(cldnn::format::bfyx);

        const int in0_b = input_sizes[0];
        const int in0_f = input_sizes[1];
        const int in0_h = input_sizes[2];
        const int in0_w = input_sizes[3];

        { // asserting dims
            const auto output_sizes = output.get_layout().size.sizes(cldnn::format::bfyx);
            const int out_b = output_sizes[0]; (void) out_b;
            const int out_f = output_sizes[1]; (void) out_f;
            const int out_h = output_sizes[2]; (void) out_h;
            const int out_w = output_sizes[3]; (void) out_w;

            const auto scale_sizes = scale.get_layout().size.sizes(cldnn::format::bfyx);
            const int in1_b = scale_sizes[0]; (void) in1_b;
            const int in1_f = scale_sizes[1]; (void) in1_f;
            const int in1_h = scale_sizes[2]; (void) in1_h;
            const int in1_w = scale_sizes[3]; (void) in1_w;
            // input and output dims must match
            assert(in0_b == out_b && in0_f == out_f && in0_h == out_h && in0_w == out_w);

            // scale/bias dims must be equal to in/out or be 1 for broadcast
            assert(in1_b == 1 || in1_b == in0_b);
            assert(in1_f == 1 || in1_f == in0_f);
            assert(in1_h == 1 || in1_h == in0_h);
            assert(in1_w == 1 || in1_w == in0_w);

            if (bias)
            {
                const auto bias_sizes = bias->get_layout().size.sizes(cldnn::format::bfyx);
                const int in2_b = bias_sizes[0]; (void) in2_b;
                const int in2_f = bias_sizes[1]; (void) in2_f;
                const int in2_h = bias_sizes[2]; (void) in2_h;
                const int in2_w = bias_sizes[3]; (void) in2_w;

                // scale/bias dims must be equal to in/out or be 1 for broadcast
                assert(in2_b == 1 || in2_b == in1_b);
                assert(in2_b == 1 || in2_f == in1_f);
                assert(in2_b == 1 || in2_h == in1_h);
                assert(in2_b == 1 || in2_w == in1_w);
            }
        }

        const auto input_desc = get_linear_memory_desc(input.get_layout());
        const auto output_desc = get_linear_memory_desc(output.get_layout());
        const auto scale_desc = get_linear_memory_desc(scale.get_layout());
        const auto bias_desc =
            bias ?
            get_linear_memory_desc(bias->get_layout()) :
            memory_desc();

        for (int n = 0; n < in0_b; ++n)
        for (int c = 0; c < in0_f; ++c)
        for (int y = 0; y < in0_h; ++y)
        for (int x = 0; x < in0_w; ++x)
        {
            const size_t in0_idx = get_linear_index(input.get_layout(), n, c, y, x, input_desc);
            const size_t in1_idx = get_linear_index_with_broadcast(scale.get_layout(), n, c, y, x, scale_desc);
            const size_t out_idx = get_linear_index(output.get_layout(), n, c, y, x, output_desc);

            out_mem[out_idx] = in0_mem[in0_idx] * in1_mem[in1_idx];

            if (bias)
            {
                const size_t in2_idx = get_linear_index_with_broadcast(bias->get_layout(), n, c, y, x, bias_desc);
                out_mem[out_idx] += in2_mem[in2_idx];
            }
        }
        return output;
    }

    virtual memory generate_reference(const std::vector<memory> & inputs) override
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info)
    {
        std::stringstream res;

        const auto & p = std::get<0>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i)
        {
            if (i == 0) res << "_Input";
            if (i == 1) res << "_ScaleInput";
            if (i == 2) res << "_BiasInput";

            const auto chans = format::traits(p->fmt).order;

            for (unsigned int j = 0; j < p->input_layouts[i].size.sizes(p->fmt).size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].size.sizes(p->fmt)[j];
            }
        }
        return res.str();
    }

private:
    static std::vector<std::unique_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::unique_ptr<cldnn::primitive>> all_layer_params;
};

std::vector<std::unique_ptr<cldnn::primitive>> scale_test::all_layer_params = {};
std::vector<std::unique_ptr<tests::test_params>> scale_test::all_generic_params = {};

TEST_P(scale_test, SCALE)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(DISABLED_SCALE,
    scale_test,
    ::testing::ValuesIn(scale_test::generate_all_test_params()),
    scale_test::custom_param_name);
