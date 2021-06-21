// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"
#include "network_test.h"

#include <cldnn/primitives/input_layout.hpp>
#include "cldnn/primitives/fully_connected.hpp"
#include <cldnn/primitives/quantize.hpp>
#include <cldnn/primitives/data.hpp>

#include <cmath>

namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace ::tests;

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

    auto& engine = get_test_engine();
    tensor input_tensor(input_b, f, x, y);
    tensor weights_tensor(output_f, f, x, y);
    auto input = engine.allocate_memory({ type_to_data_type<T>::value, test_input_fmt, input_tensor });
    auto weights = engine.allocate_memory({ type_to_data_type<T>::value, test_weights_fmt, weights_tensor });
    auto bias = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx, { 1,1,output_f,1 } });
    set_values(input, input_rnd_vec);
    set_values(weights, weights_rnd_vec);
    set_values(bias, bias_rnd_vec);

    primitive_id out_id = "fully_connected";
    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("bias", bias),
        fully_connected(out_id, "input", "weights", "bias")
    );
    if (relu)
    {
        topology.add(activation("out", out_id, activation_func::relu, { slope, 0.0f }));
        out_id = "out";
    }
    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

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

    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::yxfb,{ input_b, 1, input_x, 1} });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    auto input = input_layout("input", input_prim->get_layout());
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

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ input_b, 1, input_x, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::i8,format::bfyx,{ weight_b, 1, weight_x, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f });
    set_values<char>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
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

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, 1, input_x, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx, { 1,1,output_f, 1} });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
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

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::yxfb,{ input_b,1,input_x, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
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

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx, { 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_f } },{ 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
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

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 2, 2, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ 2 ,{ { 1 } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input_prim, { 1.f, 3.f, -2.f, -4.f });
    set_values(weights_prim, { 1.f, -1.f, 2.0f, 0.f, 3.0f, 4.0f, 0.5f, 5.0f });
    set_values(bias_prim, { 1.0f, -5.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
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

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::yxfb,{ input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_f } },{ 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias"),
        activation("out", "full_con_prim", activation_func::relu)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::yxfb,{ input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_f } },{ 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias"),
        activation("out", "full_con_prim", activation_func::relu)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_f } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias"),
        activation("out", "full_con_prim", activation_func::relu)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

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

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_f } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1,1,output_f,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias"),
        activation("out", "full_con_prim", activation_func::relu_negative_slope, { 0.1f })
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(-0.125f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(-0.1f, output_ptr[3]);
}

TEST(fully_connected_gpu, b_fs_yx_fsv4)
{
    auto& engine = get_test_engine();

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
    auto input = engine.allocate_memory({data_types::i8, format::bfyx, {in_B, in_F, in_X, in_Y}});
    set_values(input, std::move(Data));

    // Create a topology
    topology topology(input_layout("input", input->get_layout()));

    // Reorder
    topology.add(reorder("reorder_in",
                         "input",
                         layout(data_types::i8, format::b_fs_yx_fsv4, {in_B, in_F, in_X, in_Y})));

    // Weights
    std::vector<char> Weights(W_B * W_F);
    i = 0;
    std::generate(Weights.begin(), Weights.end(), [=]() mutable {
        return i % 2 ? -(i++) / W_F - 1 : (i++) / W_F + 1;
    });
    auto weights_gold =
        engine.allocate_memory({data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    auto weights_imad =
        engine.allocate_memory({data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology.add(data("weights_gold", weights_gold), data("weights_imad", weights_imad));

    auto bias_gold = engine.allocate_memory({data_types::f32, format::bfyx, {1, in_F, 1, 1}});
    auto bias_imad = engine.allocate_memory({data_types::f32, format::bfyx, {1, in_F, 1, 1}});

    std::vector<float> bias_data(in_F, 0);
    set_values(bias_gold, bias_data);
    set_values(bias_imad, bias_data);

    topology.add(data("bias_gold", bias_gold));
    topology.add(data("bias_imad", bias_imad));

    // Fully connected
    fully_connected fullc_gold(
        "fullc_gold", "input", "weights_gold", "bias_gold");
    fully_connected fullc_imad(
        "fullc_imad", "reorder_in", "weights_imad", "bias_imad");
    topology.add(fullc_gold, fullc_imad);


    auto input_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, {1, W_B, 1, 1} });
    auto input_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, {1, W_B, 1, 1} });
    auto output_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, {1, 1, 1, 1} });
    auto output_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, {1, 1, 1, 1} });
    set_values(input_low_mem,  generate_random_1d<float>(W_B, -200, 0));
    set_values(input_high_mem, generate_random_1d<float>(W_B, 1, 200));
    set_values(output_low_mem,  {-127.0f});
    set_values(output_high_mem, {127.0f});

    topology.add(data("in_lo", input_low_mem),
        data("in_hi", input_high_mem),
        data("out_lo", output_low_mem),
        data("out_hi", output_high_mem),
        quantize("quant_gold", "fullc_gold", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        quantize("quant_imad", "fullc_imad", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8)
    );

    // Output reorder
    auto reorder_gold =
        reorder("reorder_gold", "quant_gold", layout(data_types::i8, format::bfyx, {in_B, W_B, 1, 1}));
    auto reorder_imad =
        reorder("reorder_imad", "quant_imad", layout(data_types::i8, format::bfyx, {in_B, W_B, 1, 1}));
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
    cldnn::mem_lock<char> gold_ptr(out_gold->second.get_memory(), get_test_stream());
    cldnn::mem_lock<char> test_ptr(out_test->second.get_memory(), get_test_stream());

    ASSERT_EQ(gold_ptr.size(), test_ptr.size());
    for (size_t i = 0; i < gold_ptr.size(); i++) {
        ASSERT_EQ(gold_ptr[i], test_ptr[i]);
    }
}

TEST(fully_connected_gpu, DISABLED_fs_byx_fsv32_b12) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }
    // Test parameters
    const int batch_num = 12;
    const int output_f = 40;
    const int input_x = 3;
    const int input_y = 3;
    const int input_f = 64;

    // Allocate memory
    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { batch_num, input_f, input_y, input_x } });
    auto weights_prim = engine.allocate_memory({ data_types::f16,format::bfyx,{ output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16,format::bfyx,{ 1, 1, output_f, 1 } });

    // Generate random input data and set values
    auto input_data = generate_random_4d<FLOAT16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto weights_data = generate_random_4d<FLOAT16>(output_f, input_f, input_y, input_x, -1, 1);
    auto bias_data = generate_random_1d<FLOAT16>(output_f, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    set_values(input_prim, input_data_bfyx);
    set_values(weights_prim, weights_data_bfyx);
    set_values(bias_prim, bias_data);

    // Calculate CPU reference
    auto reference_output = fully_connected_reference(input_data, weights_data, bias_data, true);

    // Create topology to test
    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        reorder("input_fsv", "input", {data_types::f16, format::fs_b_yx_fsv32, { batch_num, input_f, input_y, input_x } }),
        fully_connected("fc", "input_fsv", "weights", "bias"),
        activation("out", "fc", activation_func::relu)
    );

    // Set data optimization to allow weights reordering to optimal format
    build_options opts;
    opts.set_option(build_option::optimize_data(true));

    network network(engine, topology, opts);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi)
    {
        for (size_t fi = 0; fi < output_f; ++fi)
        {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            EXPECT_TRUE(equal);
            if (!equal)
            {
                std::cout << "At b = " << bi << ", f = " << fi << std::endl;
            }
        }
    }
}

TEST(fully_connected_gpu, DISABLED_fs_byx_fsv32_b34)
{
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }
    // Test parameters
    const int batch_num = 34;
    const int output_f = 40;
    const int input_x = 3;
    const int input_y = 3;
    const int input_f = 64;

    // Allocate memory
    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { batch_num, input_f, input_y, input_x } });
    auto weights_prim = engine.allocate_memory({ data_types::f16,format::bfyx,{ output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16,format::bfyx,{ 1, 1, output_f, 1 } });

    // Generate random input data and set values
    auto input_data = generate_random_4d<FLOAT16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto weights_data = generate_random_4d<FLOAT16>(output_f, input_f, input_y, input_x, -1, 1);
    auto bias_data = generate_random_1d<FLOAT16>(output_f, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    set_values(input_prim, input_data_bfyx);
    set_values(weights_prim, weights_data_bfyx);
    set_values(bias_prim, bias_data);

    // Calculate CPU reference
    auto reference_output = fully_connected_reference(input_data, weights_data, bias_data, true);

    // Create topology to test
    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        reorder("input_fsv", "input", { data_types::f16, format::fs_b_yx_fsv32, { batch_num, input_f, input_y, input_x } }),
        fully_connected("fc", "input_fsv", "weights", "bias"),
        activation("out", "fc", activation_func::relu)
    );

    // Set data optimization to allow weights reordering to optimal format
    build_options opts;
    opts.set_option(build_option::optimize_data(true));

    network network(engine, topology, opts);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi)
    {
        for (size_t fi = 0; fi < output_f; ++fi)
        {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            EXPECT_TRUE(equal);
            if (!equal)
            {
                std::cout << "At b = " << bi << ", f = " << fi << std::endl;
            }
        }
    }
}

using fully_connected_test_params = std::tuple<
    size_t,        // batch_num
    size_t,        // input_f
    size_t,        // input_x
    size_t,        // input_y
    size_t,        // output_f
    format::type,  // input format
    format::type,  // output format
    std::string    // kernel
>;

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct fully_connected_random_test : ::testing::TestWithParam<fully_connected_test_params> {
    void run_test() {
        size_t batch, input_f, input_x, input_y, output_f;
        format::type input_format, output_format;
        std::string kernel;

        std::tie(batch, input_f, input_x, input_y, output_f, input_format, output_format, kernel) = GetParam();

        auto input_data = generate_smart_random_4d<InputT>(batch, input_f, input_y, input_x);
        auto weights_data = generate_smart_random_4d<WeightsT>(output_f, input_f, input_y, input_x);
        auto bias_data = generate_smart_random_2d<BiasT>(1, output_f);

        auto& eng = get_test_engine();
        auto net = network_test(eng);
        auto input = net.add_input_layout<InputT, 4>("input", input_format, std::move(input_data));
        auto weights = net.add_data<WeightsT, 4>("weights", format::oiyx, std::move(weights_data));
        auto bias = net.add_data<BiasT, 2>("bias", format::bfyx, std::move(bias_data));
        auto fc = net.add_fully_connected<OutputT>("fc", input, weights, bias, implementation_desc{ output_format, kernel });

        net.run(build_options(build_option::optimize_data(true)));
    }
};

using fully_connected_random_test_f32 = fully_connected_random_test<float, float, float, float>;
using fully_connected_random_test_f16 = fully_connected_random_test<FLOAT16, FLOAT16, FLOAT16, FLOAT16>;

TEST_P(fully_connected_random_test_f32, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        fully_connected_random_test_f32,
                        ::testing::Combine(
                            ::testing::Values(1, 2),
                            ::testing::Values(3, 32),
                            ::testing::Values(1, 3),
                            ::testing::Values(1, 3),
                            ::testing::Values(3, 32),
                            ::testing::Values(format::bfyx, format::yxfb),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

INSTANTIATE_TEST_SUITE_P(smoke_bfyx_batched,
                        fully_connected_random_test_f32,
                        ::testing::Combine(
                            ::testing::Values(2, 8),
                            ::testing::Values(3, 32),
                            ::testing::Values(1, 3),
                            ::testing::Values(1, 3),
                            ::testing::Values(3, 32),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::bfyx),
                            ::testing::Values("")));

TEST_P(fully_connected_random_test_f16, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(smoke_bfyx_b2,
                        fully_connected_random_test_f16,
                        ::testing::Combine(
                            // Batch 1 is disabled due to sporadic failures in `fully_connected_gpu_bs_f_bsv16_b1`
                            // - there are nans in output.
                            ::testing::Values(2),
                            ::testing::Values(3, 32),
                            ::testing::Values(1, 3),
                            ::testing::Values(1, 3),
                            ::testing::Values(3, 32),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

INSTANTIATE_TEST_SUITE_P(smoke,
                        fully_connected_random_test_f16,
                        ::testing::Combine(
                            ::testing::Values(1, 2),
                            ::testing::Values(3, 32),
                            ::testing::Values(1, 3),
                            ::testing::Values(1, 3),
                            ::testing::Values(3, 32),
                            ::testing::Values(format::yxfb),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

INSTANTIATE_TEST_SUITE_P(smoke_bfyx_batched,
                        fully_connected_random_test_f16,
                        ::testing::Combine(
                            ::testing::Values(2, 8),
                            ::testing::Values(3, 32),
                            ::testing::Values(1, 3),
                            ::testing::Values(1, 3),
                            ::testing::Values(3, 32),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::bfyx),
                            ::testing::Values("")));


template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct fully_connected_random_test_3d : ::testing::TestWithParam<fully_connected_test_params> {
    void run_test() {
        size_t batch, input_f, input_x, input_y, output_y;
        format::type input_format, output_format;
        std::string kernel;

        std::tie(batch, input_f, input_x, input_y, output_y, input_format, output_format, kernel) = GetParam();

        auto input_data = generate_smart_random_4d<InputT>(batch, input_f, input_y, input_x);
        auto weights_data = generate_smart_random_4d<WeightsT>(output_y, input_y, 1, 1);
        auto bias_data = generate_smart_random_2d<BiasT>(1, output_y);

        auto& eng = get_test_engine();
        auto net = network_test(eng);
        auto input = net.add_input_layout<InputT, 4>("input", input_format, std::move(input_data));
        auto weights = net.add_data<WeightsT, 4>("weights", format::oiyx, std::move(weights_data));
        auto bias = net.add_data<BiasT, 2>("bias", format::bfyx, std::move(bias_data));
        auto fc = net.add_fully_connected_3d<OutputT>("fc", input, weights, bias, implementation_desc{ output_format, kernel }, 3);

        net.run(build_options(build_option::optimize_data(true)));
    }
};


using fully_connected_random_test_f32_3d = fully_connected_random_test_3d<float, float, float, float>;
using fully_connected_random_test_f16_3d = fully_connected_random_test_3d<FLOAT16, FLOAT16, FLOAT16, FLOAT16>;
using fully_connected_random_test_i8_3d = fully_connected_random_test_3d<int8_t, int8_t, int8_t, float>;

TEST_P(fully_connected_random_test_f32_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        fully_connected_random_test_f32_3d,
                        ::testing::Combine(
                            ::testing::Values(1,3),
                            ::testing::Values(1,3),
                            ::testing::Values(1),
                            ::testing::Values(1,3,16),
                            ::testing::Values(1,3,16),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

INSTANTIATE_TEST_SUITE_P(smoke_big,
                        fully_connected_random_test_f32_3d,
                        ::testing::Combine(
                            ::testing::Values(3),
                            ::testing::Values(16, 17, 32),
                            ::testing::Values(1),
                            ::testing::Values(17, 32),
                            ::testing::Values(17, 32),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

TEST_P(fully_connected_random_test_f16_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        fully_connected_random_test_f16_3d,
                        ::testing::Combine(
                            ::testing::Values(1,3),
                            ::testing::Values(1,3),
                            ::testing::Values(1),
                            ::testing::Values(1,3,16),
                            ::testing::Values(1,3,16),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

TEST_P(fully_connected_random_test_i8_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                        fully_connected_random_test_i8_3d,
                        ::testing::Combine(
                            ::testing::Values(1,3),
                            ::testing::Values(1,3),
                            ::testing::Values(1),
                            ::testing::Values(1,3,16),
                            ::testing::Values(1,3,16),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::any),
                            ::testing::Values("")));

INSTANTIATE_TEST_SUITE_P(smoke_big,
                        fully_connected_random_test_i8_3d,
                        ::testing::Combine(
                            ::testing::Values(1,3),
                            ::testing::Values(16,17),
                            ::testing::Values(1),
                            ::testing::Values(17, 32),
                            ::testing::Values(17, 32),
                            ::testing::Values(format::bfyx),
                            ::testing::Values(format::any),
                            ::testing::Values("")));


struct quantization_t {
    VF<float> input_low;
    VF<float> input_high;
    float output_low;
    float output_high;
    int levels;
};

using fully_connected_quantized_test_params = std::tuple<
        size_t,  // batch_num
        size_t,  // input_f
        size_t,  // input_x
        size_t,  // input_y
        size_t,  // output_f
        format::type  // format
>;

template <typename InputT, typename OutputT>
class fully_connected_quantized_test : public ::testing::Test {
private:
    size_t batch_num() { return _input.size(); }
    size_t input_f() { return _input[0].size(); }
    size_t input_y() { return _input[0][0].size(); }
    size_t input_x() { return _input[0][0][0].size(); }
    size_t output_f() { return _weights.size(); }

    data_types input_data_type() {
        return type_to_data_type<InputT>::value;
    }

    data_types output_data_type() {
        return type_to_data_type<OutputT>::value;
    }

    bool has_bias() { return _bias.size() > 0; }

public:
    static std::string PrintToStringParamName(testing::TestParamInfo<fully_connected_quantized_test_params> param_info) {
        // construct a readable name
        return std::to_string(param_info.index) + "_in_" + std::to_string(testing::get<0>(param_info.param))
               + "x" + std::to_string(testing::get<1>(param_info.param))
               + "x" + std::to_string(testing::get<2>(param_info.param))
               + "x" + std::to_string(testing::get<3>(param_info.param))
               + "_of_" + std::to_string(testing::get<4>(param_info.param))
               + "_" + fmt_to_str(testing::get<5>(param_info.param));
    }

    void set_input(VVVVF<InputT> _data) {
        _input = std::move(_data);
    }

    void set_weights(VVVVF<int8_t> _data) {
        _weights = std::move(_data);
    }

    void set_bias(VF<int> _data) {
        _bias = std::move(_data);
    }

    void set_quantization(quantization_t quant_data) {
        _quantization = std::move(quant_data);
    }

    void set_input_format(format::type fmt) {
        _fmt = fmt;
    }

    void run_test(VVF<OutputT> expected) {
        auto& engine = get_test_engine();

        auto input_size = tensor(TensorValue(batch_num()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));
        auto weights_size = tensor(TensorValue(output_f()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));

        auto input_prim = engine.allocate_memory({ input_data_type(), _fmt, input_size });
        auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, weights_size });
        auto quantization_input_low = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(output_f())) });
        auto quantization_input_high = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(output_f())) });
        auto quantization_output_low = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(1)) });
        auto quantization_output_high = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(1)) });

        VF<InputT> input_flattened(input_prim->get_layout().get_linear_size());
        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < input_f(); ++fi)
                for (size_t yi = 0; yi < input_y(); ++yi)
                    for (size_t xi = 0; xi < input_x(); ++xi) {
                        auto idx = tensor((int32_t)bi, (int32_t)fi, (int32_t)xi, (int32_t)yi);
                        auto offset = input_size.get_linear_offset(idx, _fmt);
                        input_flattened[offset] = _input[bi][fi][yi][xi];
                    }

        set_values(input_prim, input_flattened);
        set_values(weights_prim, flatten_4d(format::bfyx, _weights));
        set_values(quantization_input_low, _quantization.input_low);
        set_values(quantization_input_high, _quantization.input_high);
        set_values(quantization_output_low, { _quantization.output_low });
        set_values(quantization_output_high, { _quantization.output_high });

        auto bias_prim = engine.allocate_memory({ data_types::i32, format::bfyx, tensor(feature(output_f())) });
        set_values(bias_prim, _bias);

        topology topo;
        topo.add(data("weights", weights_prim));
        topo.add(data("bias", bias_prim));

        topo.add(input_layout("input", input_prim->get_layout()));
        auto fc_prim = fully_connected("fc", "input", "weights", "bias");
        fc_prim.output_data_type = type_to_data_type<OutputT>::value;
        topo.add(fc_prim);

        topo.add(data("quant_input_low", quantization_input_low));
        topo.add(data("quant_input_high", quantization_input_high));
        topo.add(data("quant_output_low", quantization_output_low));
        topo.add(data("quant_output_high", quantization_output_high));
        topo.add(quantize("quantization_prim",
            "fc",
            "quant_input_low",
            "quant_input_high",
            "quant_output_low",
            "quant_output_high",
            _quantization.levels,
            output_data_type()
            ));

        topo.add(reorder("output", "quantization_prim", format::bfyx, output_data_type()));

        build_options build_opts;
        build_opts.set_option(build_option::optimize_data(true));

        network net(engine, topo, build_opts);
        net.set_input_data("input", input_prim);

        auto output = net.execute();
        auto out_mem = output.at("output").get_memory();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num(); ++bi) {
            for (size_t fi = 0; fi < output_f(); ++fi) {
                EXPECT_NEAR(out_ptr[bi * output_f() + fi], expected[bi][fi], 1) << "at b = " << bi << ", fi = " << fi;
            }
        }
    }

private:
    VVVVF<InputT> _input;
    VVVVF<int8_t> _weights;
    VF<int> _bias;
    quantization_t _quantization;
    format::type _fmt;
};


template <typename OutputT, typename AccT, typename InputT, typename WeightsT, typename BiasT>
VVF<OutputT> ref_fully_connected(
    const VVVVF<InputT>& input,
    const VVVVF<WeightsT>& weights,
    const VF<BiasT>& bias,
    const quantization_t& quantization) {

    auto batch_num = input.size();
    auto input_f = input[0].size();
    auto input_y = input[0][0].size();
    auto input_x = input[0][0][0].size();

    auto output_f = weights.size();

    auto output = VVF<OutputT>(batch_num, VF<OutputT>(output_f));

    for (size_t bi = 0; bi < batch_num; ++bi)
    for (size_t fi = 0; fi < output_f; ++fi) {
        AccT acc = static_cast<AccT>(0);
        for (size_t ifi = 0; ifi < input_f; ++ifi)
        for (size_t iyi = 0; iyi < input_y; ++iyi)
        for (size_t ixi = 0; ixi < input_x; ++ixi) {
            auto input_val = static_cast<AccT>(input[bi][ifi][iyi][ixi]);
            auto weights_val = static_cast<AccT>(weights[fi][ifi][iyi][ixi]);
            acc += input_val * weights_val;
        }
        acc += static_cast<AccT>(bias[fi]);

        //quantization
        auto input_low = quantization.input_low[fi];
        auto input_high = quantization.input_high[fi];
        auto output_low = quantization.output_low;
        auto output_high = quantization.output_high;
        float levels = static_cast<float>(quantization.levels); // just to get correct output values
        if (acc <= input_low)
            output[bi][fi] = static_cast<OutputT>(output_low);
        else if (acc > input_high)
            output[bi][fi] = static_cast<OutputT>(output_high);
        else {
            if (std::is_same<OutputT, float>::value) {
                output[bi][fi] = static_cast<OutputT>(
                    std::round((acc - input_low) / (input_high - input_low) * (levels - 1))
                        *  (1 / (levels - 1) * (output_high - output_low))
                        + output_low);
            }
            else {
                output[bi][fi] = static_cast<OutputT>(std::round(
                    std::round((acc - input_low) / (input_high - input_low) * (levels - 1))
                        *  (1 / (levels - 1) * (output_high - output_low))
                        + output_low));
            }
        }
    }
    return output;
}

namespace {
    template<typename T>
    std::vector<T> generate_random_values(size_t a, int min, int max) {
    static std::default_random_engine generator(random_seed);
    // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int> distribution(min, max);
    std::vector<T> v(a);

    for (size_t i = 0; i < a; ++i) {
        v[i] = (T)distribution(generator);
    }
    return v;
    }
} // namespace

template <typename InputT, typename OutputT>
class fc_quantized_random_test
    : public fully_connected_quantized_test<InputT, OutputT>
    , public ::testing::WithParamInterface< fully_connected_quantized_test_params> {
public:
    void run_random_test() {
        size_t b, in_f, in_x, in_y, out_f;
        format::type in_fmt;

        std::tie(b, in_f, in_x, in_y, out_f, in_fmt) = GetParam();

        VVVVF<InputT> input_data = generate_random_4d<InputT>(b, in_f, in_y, in_x, -127, 127);
        VVVVF<int8_t> weights_data = generate_random_4d<int8_t>(out_f, in_f, in_y, in_x, -127, 127);
        VF<int> bias_data = generate_random_1d<int>(out_f, -127, 127);
        bool is_unsigned = std::is_same<OutputT, uint8_t>::value;
        quantization_t quant_data;
        quant_data.input_low   = generate_random_values<float>(out_f, -200, 0);
        quant_data.input_high  = generate_random_values<float>(out_f, 1, 200);
        quant_data.output_low  = is_unsigned ? 0.0f   : -127.0f;
        quant_data.output_high = is_unsigned ? 255.0f : 127.0f;
        quant_data.levels      = is_unsigned ? 256    : 255;

        this->set_input(input_data);
        this->set_weights(weights_data);
        this->set_bias(bias_data);
        this->set_quantization(quant_data);
        this->set_input_format(in_fmt);

        this->run_test(ref_fully_connected<OutputT, float>(input_data, weights_data, bias_data, quant_data));
    }
};

using fully_connected_i8_i8_test = fc_quantized_random_test<int8_t, int8_t>;
using fully_connected_i8_u8_test = fc_quantized_random_test<int8_t, uint8_t>;
using fully_connected_i8_f32_test = fc_quantized_random_test<int8_t, float>;

using fully_connected_u8_i8_test = fc_quantized_random_test<uint8_t, int8_t>;
using fully_connected_u8_u8_test = fc_quantized_random_test<uint8_t, uint8_t>;
using fully_connected_u8_f32_test = fc_quantized_random_test<uint8_t, float>;

TEST_P(fully_connected_i8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_i8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_i8_f32_test, random) {
    run_random_test();
}

TEST_P(fully_connected_u8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_u8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_u8_f32_test, random) {
    run_random_test();
}

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1, 3),
        testing::Values(1, 3),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1, 3),
        testing::Values(1, 3),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1, 3),
        testing::Values(1, 3),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_f32_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_u8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1, 3),
        testing::Values(1, 3),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_u8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1, 3),
        testing::Values(1, 3),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_u8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1, 3),
        testing::Values(1, 3),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_f32_test::PrintToStringParamName
);
