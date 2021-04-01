// Copyright (c) 2020 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include <api/normalize.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/data.hpp>

#include <vector>
#include <iostream>

using namespace cldnn;
using namespace ::tests;

TEST(normalizel2_f32_gpu, basic) {
    //  Input  : 1x2x3x3
    //  Output : 1x2x3x3

    const auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {b, f, y, x}});
    auto weights = memory::allocate(engine, {data_types::f32, format::bfyx, {1, f, 1, 1}});

    std::vector<float> inputVals(b * f * y * x);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 0;
        return n++;
    });
    std::vector<float> weightVals(f);
    for (auto& it : weightVals) {
        it = 1.f;
    }

    set_values(input, inputVals);
    set_values(weights, weightVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", false));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {0.f,
                                           0.0995037f,
                                           0.178885f,
                                           0.242536f,
                                           0.294086f,
                                           0.336336f,
                                           0.371391f,
                                           0.400819f,
                                           0.425797f,
                                           1.f,
                                           0.995037f,
                                           0.98387f,
                                           0.970143f,
                                           0.955779f,
                                           0.941742f,
                                           0.928477f,
                                           0.916157f,
                                           0.904819f};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(normalizel2_f32_gpu, basic2) {
    //  Input  : 1x2x3x3
    //  Output : 1x2x3x3

    const auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {b, f, y, x}});
    auto weights = memory::allocate(engine, {data_types::f32, format::bfyx, {1, f, 1, 1}});

    std::vector<float> inputVals(b * f * y * x);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 0;
        return n++;
    });
    std::vector<float> weightVals(f);
    for (auto& it : weightVals) {
        it = 1.f;
    }

    set_values(input, inputVals);
    set_values(weights, weightVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", true));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {0.f,
                                           0.0236691f,
                                           0.0473381f,
                                           0.0710072f,
                                           0.0946762f,
                                           0.118345f,
                                           0.142014f,
                                           0.165683f,
                                           0.189352f,
                                           0.213021f,
                                           0.236691f,
                                           0.26036f,
                                           0.284029f,
                                           0.307698f,
                                           0.331367f,
                                           0.355036f,
                                           0.378705f,
                                           0.402374f};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(normalizel2_int8_gpu, basic) {
    //  Input  : 1x2x3x3
    //  Output : 1x2x3x3

    const auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = memory::allocate(engine, {data_types::i8, format::bfyx, {b, f, y, x}});
    auto weights = memory::allocate(engine, {data_types::f32, format::bfyx, {1, f, 1, 1}});

    std::vector<int8_t> inputVals(b * f * y * x);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static int8_t n = 0;
        return n++;
    });
    std::vector<float> weightVals(f);
    for (auto& it : weightVals) {
        it = 1;
    }

    set_values(input, inputVals);
    set_values(weights, weightVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", false));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {0.f,
                                           0.0995037f,
                                           0.178885f,
                                           0.242536f,
                                           0.294086f,
                                           0.336336f,
                                           0.371391f,
                                           0.400819f,
                                           0.425797f,
                                           1.f,
                                           0.995037f,
                                           0.98387f,
                                           0.970143f,
                                           0.955779f,
                                           0.941742f,
                                           0.928477f,
                                           0.916157f,
                                           0.904819f};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}

TEST(normalizel2_int8_gpu, basic2) {
    //  Input  : 1x2x3x3
    //  Output : 1x2x3x3

    const auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = memory::allocate(engine, {data_types::i8, format::bfyx, {b, f, y, x}});
    auto weights = memory::allocate(engine, {data_types::f32, format::bfyx, {1, f, 1, 1}});

    std::vector<int8_t> inputVals(b * f * y * x);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static int8_t n = 0;
        return n++;
    });
    std::vector<float> weightVals(f);
    for (auto& it : weightVals) {
        it = 1.f;
    }

    set_values(input, inputVals);
    set_values(weights, weightVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", true));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {0.f,
                                           0.0236691f,
                                           0.0473381f,
                                           0.0710072f,
                                           0.0946762f,
                                           0.118345f,
                                           0.142014f,
                                           0.165683f,
                                           0.189352f,
                                           0.213021f,
                                           0.236691f,
                                           0.26036f,
                                           0.284029f,
                                           0.307698f,
                                           0.331367f,
                                           0.355036f,
                                           0.378705f,
                                           0.402374f};

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_TRUE(are_equal(expected_results[i], output_ptr[i]));
    }
}
