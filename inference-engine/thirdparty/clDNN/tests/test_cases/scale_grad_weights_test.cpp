/*
// Copyright (c) 2018 Intel Corporation
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
#include <api/CPP/data.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/mutable_data.hpp>
#include "api/CPP/scale_grad_weights.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

#include <iostream>

using namespace cldnn;
using namespace tests;

TEST(scale_grad_weights_gpu, basic_in2x3x2x2) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Input grad:
    //  f0: b0:  1    2   3   b1:   0    0    -11
    //  f0: b0:  4    5  -6   b1:   0.5 -0.5  -15  
    //  f1: b0: -7    8  -9   b1:   1.5  5.2  -13     
    //  f1: b0:  12  11  10   b1:   12   8    -17
    //
    //  Scale:
    //  f0: 0.1
    //  f1: 0.6  

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 2 } });
    auto grad_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("scale_input", scale_input));
    topology.add(data("grad_input", grad_input));
    topology.add(scale_grad_weights("scale_grad", "input", "grad_input", "scale_input"));

    std::vector<float> input_vec = { 
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 8.f, -17.f
    };
    set_values(input, input_vec);

    std::vector<float> grad_vec = {
        1.f, 2.f, 3.f,
        4.f, 5.f, -6.f,
        -7.f, 8.f, -9.f,
        12.f, 11.f, 10.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 8.f, -17.f
    };
    set_values(grad_input, grad_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.6f
    };
    set_values(scale_input, scale_input_vec);

    build_options options;
    network network(engine, topology);
    
    network.set_learning_rate(0.0001f);
    network.set_input_data("input", input);

    std::vector<float> expected_out = {
        0.05625f, 0.517171f
    };

    auto outputs = network.execute();

    auto output_ptr = scale_input.pointer<float>();

    for (unsigned int i = 0; i < expected_out.size(); ++i) {
        EXPECT_NEAR(output_ptr[i], expected_out[i], 1e-04F);
    }
}

TEST(scale_grad_weights_gpu, basic_in2x3x2x2_bias) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Input grad:
    //  f0: b0:  1    2   3   b1:   0    0    -11
    //  f0: b0:  4    5  -6   b1:   0.5 -0.5  -15  
    //  f1: b0: -7    8  -9   b1:   1.5  5.2  -13     
    //  f1: b0:  12  11  10   b1:   12   8    -17
    //
    //  Scale:
    //  f0: 0.1
    //  f1: 0.6  
    //
    //  Bias:
    //  f0: 1
    //  f1: 0.5

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto grad_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("scale_input", scale_input));
    topology.add(data("grad_input", grad_input));
    topology.add(mutable_data("bias", bias));
    topology.add(scale_grad_weights("scale_grad", "input", "grad_input", "scale_input", "bias", ""));

    std::vector<float> input_vec = {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 8.f, -17.f
    };
    set_values(input, input_vec);

    std::vector<float> grad_vec = {
        1.f, 2.f, 3.f,
        4.f, 5.f, -6.f,
        -7.f, 8.f, -9.f,
        12.f, 11.f, 10.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 8.f, -17.f
    };
    set_values(grad_input, grad_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.6f
    };
    set_values(scale_input, scale_input_vec);

    std::vector<float> bias_vec = {
        1.f, 0.5f  
    };
    set_values(bias, bias_vec);

    build_options options;
    network network(engine, topology);

    network.set_learning_rate(0.0001f);
    network.set_input_data("input", input);

    std::vector<float> expected_scale = {
        0.05625f, 0.517171f
    };

    std::vector<float> expected_bias = {
        1.0017f, 0.4978f
    };

    auto outputs = network.execute();

    auto scale_ptr = scale_input.pointer<float>();
    auto bias_ptr = bias.pointer<float>();

    for (unsigned int i = 0; i < expected_scale.size(); ++i) {
        EXPECT_NEAR(scale_ptr[i], expected_scale[i], 1e-04F);
    }
    for (unsigned int i = 0; i < expected_bias.size(); ++i) {
        EXPECT_NEAR(bias_ptr[i], expected_bias[i], 1e-04F);
    }
}

TEST(scale_grad_weights_gpu, basic_in2x3x2x2_bias_momentum) {
    //  Scale  : 2x3x2x2
    //  Input  : 2x3x2x2
    //  Output : 2x3x2x2

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17
    //
    //  Input grad:
    //  f0: b0:  1    2   3   b1:   0    0    -11
    //  f0: b0:  4    5  -6   b1:   0.5 -0.5  -15  
    //  f1: b0: -7    8  -9   b1:   1.5  5.2  -13     
    //  f1: b0:  12  11  10   b1:   12   8    -17
    //
    //  Scale:
    //  f0: 0.1
    //  f1: 0.6  
    //
    //  Bias:
    //  f0: 1
    //  f1: 0.5

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto grad_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });
    auto scale_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto prev_scale = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto prev_bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mutable_data("scale_input", scale_input));
    topology.add(data("grad_input", grad_input));
    topology.add(mutable_data("bias", bias));
    topology.add(mutable_data("prev_scale", prev_scale));
    topology.add(mutable_data("prev_bias", prev_bias));
    topology.add(scale_grad_weights("scale_grad", "input", "grad_input", "scale_input", "bias", "prev_scale", "prev_bias"));

    std::vector<float> input_vec = {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 8.f, -17.f
    };
    set_values(input, input_vec);

    std::vector<float> grad_vec = {
        1.f, 2.f, 3.f,
        4.f, 5.f, -6.f,
        -7.f, 8.f, -9.f,
        12.f, 11.f, 10.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 8.f, -17.f
    };
    set_values(grad_input, grad_vec);

    std::vector<float> scale_input_vec = {
        0.1f, 0.6f
    };
    set_values(scale_input, scale_input_vec);

    std::vector<float> bias_vec = {
        1.f, 0.5f
    };
    set_values(bias, bias_vec);

    build_options options;
    network network(engine, topology);

    network.set_learning_rate(0.0001f);
    network.set_input_data("input", input);

    std::vector<float> expected_scale = {
        0.05625f, 0.517171f
    };

    std::vector<float> expected_bias = {
        1.0017f, 0.4978f
    };

    auto outputs = network.execute();

    auto scale_ptr = scale_input.pointer<float>();
    auto bias_ptr = bias.pointer<float>();
    auto mom_scale_ptr = prev_scale.pointer<float>();
    auto mom_bias_ptr = prev_bias.pointer<float>();

    for (unsigned int i = 0; i < expected_scale.size(); ++i) {
        EXPECT_NEAR(scale_ptr[i], expected_scale[i], 1e-04F);
    }
    for (unsigned int i = 0; i < expected_bias.size(); ++i) {
        EXPECT_NEAR(bias_ptr[i], expected_bias[i], 1e-04F);
    }
    for (unsigned int i = 0; i < mom_scale_ptr.size(); ++i) {
        EXPECT_NEAR(mom_scale_ptr[i], scale_input_vec[i] - expected_scale[i], 1e-04F);
    }
    for (unsigned int i = 0; i < mom_bias_ptr.size(); ++i) {
        EXPECT_NEAR(mom_bias_ptr[i], bias_vec[i] - expected_bias[i], 1e-04F);
    }
}