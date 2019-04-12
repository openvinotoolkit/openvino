/*
// Copyright (c) 2019 Intel Corporation
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
#include "api/CPP/fully_connected.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/tensor.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>
#include "instrumentation.h"

#include <cmath>

namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;

cldnn::format::type layout_4d(cldnn::format f) {
    switch (f.value) {
    case cldnn::format::bfyx:
        return cldnn::format::bfyx;
    case cldnn::format::yxfb:
        return cldnn::format::yxfb;
    default:
        return f.value;
    }
}

template <typename T>
VVVVF<T> fully_connected_reference(VVVVF<T> &input, VVVVF<T> &weights, VF<T> &bias, bool relu = false, T slope = 0.0f) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();        // input is assumed to be bfyx
    size_t output_f = weights.size();    // weights is assumed to be bfyx
    VVVVF<T> output(output_b, VVVF<T>(1, VVF<T>(1, VF<T>(output_f))));
    float res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t n = 0; n < output_f; ++n) {
            res = bias[n];
            for (size_t f = 0; f < input_f; ++f) {
                for (size_t y = 0; y < input_y; ++y) {
                    for (size_t x = 0; x < input_x; ++x) {
                        res += (float)input[b][f][y][x] * (float)weights[n][f][y][x];
                    }
                }
            }
            if (relu && res < (float)0)
                res *= (float)slope;
            output[b][0][0][n] = (T)res;
        }
    }
    return output;
}

template <typename T>
void generic_fully_connected_test(cldnn::format test_input_fmt, cldnn::format test_weights_fmt, int input_b, int f, int y, int x, int output_f, bool relu, T slope = 0) {
    int min_random = -2, max_random = 2;
    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, f, y, x, min_random, max_random);
    VVVVF<T> weights_rnd = generate_random_4d<T>(output_f, f, y, x, min_random, max_random);
    VF<T> bias_rnd_vec = generate_random_1d<T>(output_f, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
    VF<T> weights_rnd_vec = flatten_4d<T>(test_weights_fmt, weights_rnd);

    const auto& engine = get_test_engine();
    tensor input_tensor(input_b, f, x, y);
    tensor weights_tensor(output_f, f, x, y);
    auto input = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    auto weights = memory::allocate(engine, { type_to_data_type<T>::value, test_weights_fmt, weights_tensor });
    auto bias = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,1,output_f,1 } });
    set_values(input, input_rnd_vec);
    set_values(weights, weights_rnd_vec);
    set_values(bias, bias_rnd_vec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("bias", bias),
        fully_connected("fully_connected", "input", "weights", "bias", relu, slope)
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "fully_connected");

    auto output_memory = outputs.at("fully_connected").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    //EXPECT_EQ(output_layout.format.value, test_input_fmt);
    tensor output_tensor = output_layout.size;
    int b_size = output_tensor.batch[0];
    int x_size = output_tensor.feature[0];
    EXPECT_EQ(b_size, input_b);
    EXPECT_EQ(x_size, output_f);
    unsigned num_of_operations = f * x * y * 2;
    float ulp = (1.0f / 1024.0f) * num_of_operations;
    bool test_is_correct = true;
    VVVVF<T> output_cpu = fully_connected_reference<T>(input_rnd, weights_rnd, bias_rnd_vec, relu, slope);
    VF<T> output_cpu_vec = flatten_4d<T>(layout_4d(output_layout.format), output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (std::abs(float(output_cpu_vec[i]) - float(output_ptr[i])) > ulp) {
            EXPECT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]); // to print the problematic values
            test_is_correct = false;
            break;
        }
    }

    EXPECT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "test_input_fmt = " << format::traits(test_input_fmt).order << std::endl
        << "test_weights_fmt = " << format::traits(test_weights_fmt).order << std::endl
        << "input_b = " << input_b << std::endl
        << "f = " << f << std::endl
        << "y = " << y << std::endl
        << "x = " << x << std::endl
        << "output_f = " << output_f << std::endl
        << "relu = " << relu << std::endl
        << "slope = " << (float)slope << std::endl
        << "type = " << (sizeof(T) == 2 ? "float16" : "float32") << std::endl;
}

TEST(DISABLED_fully_connected_gpu, generic_random_short) {
    VF<cldnn::format> test_input_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
    VF<cldnn::format> test_weights_fmts = { cldnn::format::yxfb };
    VF<bool> relu = { true, false };
    std::vector<int> batches = { 1, 2, 4, 8, 16 };
    std::vector<int> features = { 1, 2 };
    std::vector<std::pair<int, int>> input_sizes = { {28, 28}, {64, 64}, {100, 100}, {227, 227}, {1000, 1}, {1, 4096} };
    VF<int> outputs_x = { 5, 16 };

    const auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    for (cldnn::format test_input_fmt : test_input_fmts) {
        for (cldnn::format test_weights_fmt : test_weights_fmts) {
            for (const auto& b : batches) {
                for(const auto& f : features) {
                    for (const auto& sizes : input_sizes) {
                        for (int output_f : outputs_x) {
                            for (bool relu_activated : relu) {
                                    generic_fully_connected_test<float>(test_input_fmt, test_weights_fmt, b, f, sizes.second, sizes.first, output_f, relu_activated);
                                    if (!f16_supported) continue;
                                    generic_fully_connected_test<FLOAT16>(test_input_fmt, test_weights_fmt, b, f, sizes.second, sizes.first, output_f, relu_activated);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(fully_connected_gpu, no_biases) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   no biases
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b, 1, input_x, 1} });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    auto input = input_layout("input", input_prim.get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("full_con_prim", "input", "weights");
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(1.5f, output_ptr[0]);
    EXPECT_EQ(0.75f, output_ptr[1]);
    EXPECT_EQ(-2.25f, output_ptr[2]);
    EXPECT_EQ(3.0f, output_ptr[3]);
}


TEST(fully_connected_gpu, no_biases_int8) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  8.0f, 2.0f, -4.0f
    //
    //  Weights:
    //   2.0f    1.0f  0.0f
    //  -3.0f   -2.0f  1.0f
    //   0.0f   -2.0f -4.0f
    //  -5.0f   10.0f  8.0f
    //
    //
    //  Biases:
    //   no biases
    //
    //  Output:
    //  18    -32    12   -52

    const int32_t input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ input_b, 1, input_x, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::i8,format::bfyx,{ weight_b, 1, weight_x, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.99f });
    set_values<char>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim.get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", "input", { data_types::i8,format::bfyx,{ input_b, 1, input_x, 1 } });
    auto fc = fully_connected("full_con_prim", "reorder_to_int", "weights");
    auto rf = reorder("reorder_to_float", "full_con_prim", { data_types::f32,format::bfyx,{ input_b, 1, 4, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);
    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(18.0f, output_ptr[0]);
    EXPECT_EQ(-32.0f, output_ptr[1]);
    EXPECT_EQ(12.0f, output_ptr[2]);
    EXPECT_EQ(-52.0f, output_ptr[3]);
}


TEST(fully_connected_gpu, xb_f32_batch_1) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_f = 4,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate( engine, { data_types::f32, format::yxfb, { input_b, 1, input_x, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx, { 1,1,output_f, 1} });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.5f, output_ptr[0]);
    EXPECT_EQ(2.75f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(7.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75     0.75   7
    //   4      1        2.75   5

    const int32_t output_f = 4,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b,1,input_x, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(2.75f, output_ptr[2]);
    EXPECT_EQ(1.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(7.00f, output_ptr[6]);
    EXPECT_EQ(5.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_f = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx, { 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_f } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(2.75f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(7.00f, output_ptr[3]);
}


TEST(fully_connected_gpu, yxfn_f32) {
    //  Input  : 1x2x1x2 - 1 batch 2 feature maps of size 2x1
    //  Output : 2x1 - 2 batches 1 neuron each
    //  Weights: 2x2x1x2 - 2 neurons with weights of 2 feature maps of size 2x1
    //
    //  Input:
    //   1  -2      f0: b0
    //   3  -4      f1: b0

    //  Weights:
    //   1  -1      n0: fm0
    //   2   0      n0: fm1
    //   3   4      n1: fm0
    //   0.5 5      n1: fm1
    //
    //  Biases:
    //   1.0 -5
    //
    //  Output:
    //   10  -28.5

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 2, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ 2 ,{ { 1 } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input_prim, { 1.f, 3.f, -2.f, -4.f });
    set_values(weights_prim, { 1.f, -1.f, 2.0f, 0.f, 3.0f, 4.0f, 0.5f, 5.0f });
    set_values(bias_prim, { 1.0f, -5.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(10, output_ptr[0]);
    EXPECT_EQ(-28.5, output_ptr[1]);
}

TEST(fully_connected_gpu, xb_f32_batch_1_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0,  -2.0,  3.0,  -4.0
    //
    //  Output:
    //   2.5   0      0.75  0

    const int32_t output_f = 4,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_f } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2_relu) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //
    //  Output:
    //   2.5    0   0.75   0
    //   4      0   2.75   0

    const int32_t output_f = 4,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_f } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(0.00f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(0.00f, output_ptr[6]);
    EXPECT_EQ(0.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   0    0.75  0

    const int32_t output_f = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_f } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, x_f32_relu_with_negative_slope) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //  Negative Slope: 0.1
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   -0.125    0.75  -0.1

    const int32_t output_f = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    const auto& engine = get_test_engine();

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_f } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0.1f)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(-0.125f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(-0.1f, output_ptr[3]);
}

TEST(fully_connected_gpu, b_fs_yx_fsv4)
{
    const auto& engine = get_test_engine();

    const int in_B = 2;
    const int in_F = 2048;
    const int in_Y = 1;
    const int in_X = 1;

    const int W_B = 1000;
    const int W_F = in_F;
    const int W_Y = in_Y;
    const int W_X = in_X;

    // Input data
    std::vector<char> Data(in_F * in_B); // in_X=in_Y=1
    int i = 0;
    std::generate(Data.begin(), Data.end(), [i]() mutable { return i++ % 9; });
    auto input = memory::allocate(engine, {data_types::i8, format::bfyx, {in_B, in_F, in_X, in_Y}});
    set_values(input, std::move(Data));

    // Create a topology
    topology topology(input_layout("input", input.get_layout()));

    // Reorder
    topology.add(reorder("reorder_in",
                         "input",
                         layout(data_types::i8, format::b_fs_yx_fsv4, {in_B, in_F, in_X, in_Y})));

    // Weights
    std::vector<char> Weights(W_B * W_F);
    i = 0;
    std::generate(Weights.begin(), Weights.end(), [W_F, i]() mutable {
        return i % 2 ? -(i++) / W_F - 1 : (i++) / W_F + 1;
    });
    auto weights_gold =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    auto weights_imad =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology.add(data("weights_gold", weights_gold), data("weights_imad", weights_imad));

    // Bias, Callibraiton, Quantization
    std::vector<float> vB(in_F), vC(in_F), vQ(in_F);
    float x = 0.1f;
    std::generate(vB.begin(), vB.end(), [x]() mutable {
        x += 0.01f;
        if (x >= 0.9f)
            x = 0.1f;
        return x;
    });
    x = 0.2f;
    std::generate(vC.begin(), vC.end(), [x]() mutable {
        x += 0.01f;
        if (x >= 0.9f)
            x = 0.2f;
        return x;
    });
    x = 0.3f;
    std::generate(vQ.begin(), vQ.end(), [x]() mutable {
        x += 0.01f;
        if (x >= 0.9f)
            x = 0.3f;
        return x;
    });
    auto bias_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, in_F, 1}});
    auto bias_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, in_F, 1}});
    auto callib_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, in_F, 1}});
    auto callib_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, in_F, 1}});
    auto quant_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, in_F, 1}});
    auto quant_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, in_F, 1}});
    set_values(bias_gold, vB);
    set_values(bias_imad, std::move(vB));
    set_values(callib_gold, vC);
    set_values(callib_imad, std::move(vC));
    set_values(quant_gold, vQ);
    set_values(quant_imad, std::move(vQ));
    topology.add(data("bias_gold", bias_gold),
                 data("callib_gold", callib_gold),
                 data("quant_gold", quant_gold));
    topology.add(data("bias_imad", bias_imad),
                 data("callib_imad", callib_imad),
                 data("quant_imad", quant_imad));

    // Fully connected
    fully_connected fullc_gold(
        "fullc_gold", "input", "weights_gold", {"bias_gold"}, {"quant_gold"}, {"callib_gold"}, 1.0f);
    fully_connected fullc_imad(
        "fullc_imad", "reorder_in", "weights_imad", {"bias_imad"}, {"quant_imad"}, {"callib_imad"}, 1.0f);
    topology.add(fullc_gold, fullc_imad);

    // Output reorder
    auto reorder_gold =
        reorder("reorder_gold", fullc_gold, layout(data_types::i8, format::bfyx, {in_B, W_B, 1, 1}));
    auto reorder_imad =
        reorder("reorder_imad", fullc_imad, layout(data_types::i8, format::bfyx, {in_B, W_B, 1, 1}));
    topology.add(reorder_gold, reorder_imad);

    // Network build
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, build_opt);

    // Network execuiton
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto out_gold = outputs.find("reorder_gold");
    auto out_test = outputs.find("reorder_imad");

    ASSERT_NE(out_gold, outputs.end());
    ASSERT_NE(out_test, outputs.end());
    auto gold_ptr = out_gold->second.get_memory().pointer<char>();
    auto test_ptr = out_test->second.get_memory().pointer<char>();

    ASSERT_EQ(gold_ptr.size(), test_ptr.size());
    for (size_t i = 0; i < gold_ptr.size(); i++)
    {
        ASSERT_EQ(gold_ptr[i], test_ptr[i]);
    }
}
