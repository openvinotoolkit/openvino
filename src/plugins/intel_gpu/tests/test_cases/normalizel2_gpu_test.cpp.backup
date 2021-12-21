// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/normalize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <vector>
#include <iostream>

using namespace cldnn;
using namespace ::tests;

TEST(normalizel2_f32_gpu, basic) {
    //  Input  : 1x2x3x3
    //  Output : 1x2x3x3

    auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {b, f, y, x}});
    auto weights = engine.allocate_memory({data_types::f32, format::bfyx, {1, f, 1, 1}});

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
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", false));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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

    auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {b, f, y, x}});
    auto weights = engine.allocate_memory({data_types::f32, format::bfyx, {1, f, 1, 1}});

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
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", true));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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

    auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = engine.allocate_memory({data_types::i8, format::bfyx, {b, f, y, x}});
    auto weights = engine.allocate_memory({data_types::f32, format::bfyx, {1, f, 1, 1}});

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
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", false));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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

    auto& engine = get_test_engine();

    const unsigned b = 1;
    const unsigned f = 2;
    const unsigned y = 3;
    const unsigned x = 3;

    auto input = engine.allocate_memory({data_types::i8, format::bfyx, {b, f, y, x}});
    auto weights = engine.allocate_memory({data_types::f32, format::bfyx, {1, f, 1, 1}});

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
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(data("Input1", weights));
    topology.add(normalize("normalizel2", "Input0", "Input1", true));

    network network(engine, topology);

    network.set_input_data("Input0", input);

    auto outputs = network.execute();

    auto output = outputs.at("normalizel2").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

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
