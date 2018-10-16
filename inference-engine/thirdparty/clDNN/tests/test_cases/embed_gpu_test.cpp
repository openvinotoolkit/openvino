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
#include <api/CPP/input_layout.hpp>
#include "api/CPP/embed.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/tensor.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include <api/CPP/data.hpp>
#include <boost/filesystem.hpp>
#include "test_utils/test_utils.h"


#include <cmath>

using namespace cldnn;
using namespace tests;


TEST(embed_gpu, seq3num4) {
    //  Input  : 1x1x1x3
    //  Weights: 4x1x3x1
    //  Bias   : 1x1x1x4
    //  Output : 1x3x4x1
    //  Input:
    //   1.0    2.0   0.0
    //
    //  Weights:
    //   1.0    1.0   1.0    1.0
    //   2.0    2.0   2.0    2.0
    //   3.0    3.0   3.0    3.0
    //  Biases:
    //   1.0    2.0   3.0    4.0
    //
    //  Output:
    //   2.0    4.0   6.0    8.0
    //   0.0    0.0   0.0    0.0
    //   6.0    8.0  -2.0   -2.0

    engine engine;
    auto batch = 1;
    auto sequence_length = 3;
    auto num_output_size = 4;
    auto vocab_size = 3;
    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch, 1, 1, sequence_length } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ num_output_size, 1, vocab_size, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch, 1, 1, num_output_size } });
    auto output_ref = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch, sequence_length, num_output_size, 1 } });

    set_values(input_prim, { 1.0f, 2.0f, 0.0f });
    set_values(weights_prim, { 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });
    set_values(output_ref, { 3.0f, 4.0f, 5.0f, 6.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        2.0f, 3.0f, 4.0f, 5.0f });

    auto input = input_layout("input", input_prim.get_layout());
    auto w_data = data("weights", weights_prim);
    auto b_data = data("bias", bias_prim);

    auto embed_test = embed("embed_prim", "input", "weights", "bias");
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(b_data);
    topology.add(embed_test);

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "embed_prim");

    auto output_prim = outputs.begin()->second.get_memory();
    auto ref = output_ref.pointer<float>();
    auto output_ptr = output_prim.pointer<float>();
    for (auto i = 0; i < batch * sequence_length * num_output_size; i++) {
        EXPECT_EQ(ref[i], output_ptr[i]);
    }

}

TEST(embed_gpu, b2seq2num3) {
    //  Input  : 2x1x1x2
    //  Weights: 3x1x3x1
    //  Bias   : 1x1x1x4
    //  Output : 1x3x4x1
    //  Input:
    //   0.0    1.0
    //   2.0    0.0
    //
    //  Weights:
    //  -1.0   -2.0  -3.0 
    //  -1.0    2.0   0.0 
    //   10.0   16.0  15.0 
    //  Biases:
    //   0.0    2.0   4.0
    //
    //  Output:
    //   -1.0   0.0   1.0   -1.0   4.0   4.0
    //    10.0  18.0  19.0  -1.0   0.0   1.0

    engine engine;
    auto batch = 2;
    auto sequence_length = 2;
    auto num_output_size = 3;
    auto vocab_size = 3;
    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch, 1, 1, sequence_length } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ num_output_size, 1, vocab_size, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 1, num_output_size } });
    auto output_ref = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch, sequence_length, num_output_size, 1 } });

    set_values(input_prim, { 0.0f, 1.0f, 2.0f, 0.0f });
    set_values(weights_prim, { -1.0f, -2.0f, -3.0f,
        -1.0f,  2.0f,  0.0f,
        10.0f, 16.0f, 15.0f });
    set_values(bias_prim, { 0.0f, 2.0f, 4.0f });
    set_values(output_ref, { -1.0f, 0.0f, 1.0f, -1.0f, 4.0f, 4.0f,
        10.0f, 18.0f, 19.0f, -1.0f, 0.0f, 1.0f });

    auto input = input_layout("input", input_prim.get_layout());
    auto w_data = data("weights", weights_prim);
    auto b_data = data("bias", bias_prim);

    auto embed_test = embed("embed_prim", "input", "weights", "bias");
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(b_data);
    topology.add(embed_test);

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "embed_prim");

    auto output_prim = outputs.begin()->second.get_memory();
    auto ref = output_ref.pointer<float>();
    auto output_ptr = output_prim.pointer<float>();
    for (auto i = 0; i < batch * sequence_length * num_output_size; i++) {
        EXPECT_EQ(ref[i], output_ptr[i]);
    }

}

