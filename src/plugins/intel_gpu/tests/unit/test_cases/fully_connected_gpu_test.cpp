// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "test_utils.h"
#include "random_generator.hpp"
#include "network_test.h"
#include <intel_gpu/runtime/utils.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/fully_connected.hpp"
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "intel_gpu/runtime/compilation_context.hpp"
#include "fully_connected_inst.h"

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
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

template <typename OutputT, typename InputT, typename WeightsT, typename BiasT, typename AccT = OutputT>
VF<OutputT> dynamic_fully_connected_reference_calc(ov::Dimension::value_type batch,
                                                    ov::Dimension::value_type input_f,
                                                    ov::Dimension::value_type output_f,
                                                    VF<InputT>& input,
                                                    VF<WeightsT>& weights,
                                                    VF<BiasT>& bias) {
    VF<OutputT> result(batch * output_f);
    for (int b = 0; b < batch; b++) {
        for (int ofm = 0; ofm < output_f; ofm++) {
            AccT acc = static_cast<AccT>(bias[ofm]);
            for (int ifm = 0; ifm < input_f; ifm++) {
                acc += weights[ofm * input_f + ifm] * input[b * input_f + ifm];
            }
            result[b * output_f + ofm] = acc;
        }
    }

    return result;
}

template <typename T>
VVVVF<T> fully_connected_reference(VVVVF<T> &input, VVVVF<T> &weights, VF<T> &bias, bool relu = false, T slope = 0.0f) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();         // input is assumed to be bfyx
    size_t output_f = weights.size();       // weights are assumed to be bfyx
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
void generic_fully_connected_test(cldnn::format test_input_fmt, cldnn::format test_weights_fmt,
                                  int input_b, int f, int y, int x, int output_f, bool relu, T slope = 0) {
    tests::random_generator rg(GET_SUITE_NAME);
    int min_random = -2, max_random = 2;
    VVVVF<T> input_rnd = rg.generate_random_4d<T>(input_b, f, y, x, min_random, max_random);
    VVVVF<T> weights_rnd = rg.generate_random_4d<T>(output_f, f, y, x, min_random, max_random);
    VF<T> bias_rnd_vec = rg.generate_random_1d<T>(output_f, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
    VF<T> weights_rnd_vec = flatten_4d<T>(test_weights_fmt, weights_rnd);

    auto& engine = get_test_engine();
    tensor input_tensor(input_b, f, x, y);
    tensor weights_tensor(output_f, f, x, y);
    auto input = engine.allocate_memory({ ov::element::from<T>(), test_input_fmt, input_tensor });
    auto weights = engine.allocate_memory({ ov::element::from<T>(), test_weights_fmt, weights_tensor });
    auto bias = engine.allocate_memory({ ov::element::from<T>(), format::bfyx, { 1, 1, output_f, 1 } });
    set_values(input, input_rnd_vec);
    set_values(weights, weights_rnd_vec);
    set_values(bias, bias_rnd_vec);

    primitive_id out_id = "fully_connected";
    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("bias", bias),
        fully_connected(out_id, input_info("input"), "weights", "bias")
    );
    if (relu)
    {
        topology.add(activation("out", input_info(out_id), activation_func::relu, { slope, 0.0f }));
        out_id = "out";
    }
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    //ASSERT_EQ(output_layout.format.value, test_input_fmt);
    tensor output_tensor = output_layout.get_tensor();
    int b_size = output_tensor.batch[0];
    int x_size = output_tensor.feature[0];
    ASSERT_EQ(b_size, input_b);
    ASSERT_EQ(x_size, output_f);
    unsigned num_of_operations = f * x * y * 2;
    float ulp = (1.0f / 1024.0f) * num_of_operations;
    bool test_is_correct = true;
    VVVVF<T> output_cpu = fully_connected_reference<T>(input_rnd, weights_rnd, bias_rnd_vec, relu, slope);
    VF<T> output_cpu_vec = flatten_4d<T>(layout_4d(output_layout.format), output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (std::abs(float(output_cpu_vec[i]) - float(output_ptr[i])) > ulp) {
            ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]); // to print the problematic values
            test_is_correct = false;
            break;
        }
    }

    ASSERT_EQ(test_is_correct, true) << std::endl
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
}  // namespace

TEST(DISABLED_fully_connected_gpu, generic_random_short) {
    VF<cldnn::format> test_input_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
    VF<cldnn::format> test_weights_fmts = { cldnn::format::yxfb };
    VF<bool> relu = { true, false };
    std::vector<int> batches = { 1, 2, 4, 8, 16 };
    std::vector<int> features = { 1, 2 };
    std::vector<std::pair<int, int>> input_sizes = { { 28, 28 }, { 64, 64 }, { 100, 100 }, { 227, 227 }, { 1000, 1 }, { 1, 4096 } };
    VF<int> outputs_x = { 5, 16 };

    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    for (cldnn::format test_input_fmt : test_input_fmts) {
        for (cldnn::format test_weights_fmt : test_weights_fmts) {
            for (const auto& b : batches) {
                for(const auto& f : features) {
                    for (const auto& sizes : input_sizes) {
                        for (int output_f : outputs_x) {
                            for (bool relu_activated : relu) {
                                generic_fully_connected_test<float>(test_input_fmt, test_weights_fmt,
                                                                    b, f, sizes.second, sizes.first, output_f, relu_activated);
                                if (!f16_supported) continue;
                                generic_fully_connected_test<ov::float16>(test_input_fmt, test_weights_fmt,
                                                                      b, f, sizes.second, sizes.first, output_f, relu_activated);
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

    const int32_t input_f = 3, input_b = 1,    // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights");
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(1.5f, output_ptr[0]);
    ASSERT_EQ(0.75f, output_ptr[1]);
    ASSERT_EQ(-2.25f, output_ptr[2]);
    ASSERT_EQ(3.0f, output_ptr[3]);
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

    const int32_t input_f = 3, input_b = 1,    // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f });
    set_values<int8_t>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", input_info("input"), { data_types::i8, format::bfyx, { input_b, input_f, 1, 1 } });
    auto fc = fully_connected("fc_prim", input_info("reorder_to_int"), "weights");
    auto rf = reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { input_b, weight_b, 1, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(18.0f, output_ptr[0]);
    ASSERT_EQ(-32.0f, output_ptr[1]);
    ASSERT_EQ(12.0f, output_ptr[2]);
    ASSERT_EQ(-52.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, no_biases_fc_i32) {
    const int32_t input_b = 1, input_f = 3,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::i32, format::bfyx, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::i32, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values<int32_t>(input_prim, {
        8, 2, -4
    });

    set_values<int32_t>(weights_prim, {
        2, 1, 0,
        -3, -2, 1,
        0, -2, -4,
        -5, 10, 8
    });

    cldnn::topology topology{
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        reorder("reorder_to_i32", input_info("input"), { data_types::i32, format::bfyx, { input_b, input_f, 1, 1 } }),
        fully_connected("fc_prim", input_info("reorder_to_i32"), "weights"),
        reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { input_b, weight_b, 1, 1 } })
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network(engine, topology, config);

    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_memory = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr (output_memory, get_test_stream());
    auto output_layout = output_memory->get_layout();

    ASSERT_EQ(output_layout.format, format::bfyx);

    VVF<float> output_ref = {
        { 18, -32, 12, -52 },
    };

    for (int i = 0; i< weight_b; ++i) {
        ASSERT_EQ(output_ref[0][i], output_ptr[i]);
    }
}

TEST(fully_connected_gpu, no_biases_4d_input) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;

    //  Input  : 1x8x8x12
    //  Weights: 48x12x1x1
    //  Output : 1x8x8x48

    const int32_t input_b = 1, input_f = 8, input_y = 8, input_x = 12,          // size of the whole input buffer
                  weight_b = 48, weight_f = 12, weight_y = 1, weight_x = 1;     // size of the whole weights buffer

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, input_x, input_y } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, weight_x, weight_y } });

    std::vector<float> input_data(input_b * input_f * input_y * input_x, 0);
    std::vector<float> weights_data(weight_b * weight_f * weight_y * weight_x, 0);

    set_values(input_prim, std::move(input_data));
    set_values(weights_prim, std::move(weights_data));

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 4, 2);
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->second.get_layout().batch(), input_b);
    ASSERT_EQ(outputs.begin()->second.get_layout().feature(), input_f);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(1), input_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(0), weight_b);
}

TEST(fully_connected_gpu, no_biases_4d_input_immad) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    //  Input  : 1x8x8x12
    //  Weights: 48x12x1x1
    //  Output : 64x48x1x1

    const int32_t input_b = 1, input_f = 8, input_y = 8, input_x = 12,          // size of the whole input buffer
                  weight_b = 48, weight_f = 12, weight_y = 1, weight_x = 1;     // size of the whole weights buffer

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, input_x, input_y } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, weight_x, weight_y } });

    std::vector<float> input_data(input_b * input_f * input_y * input_x, 0);
    std::vector<float> weights_data(weight_b * weight_f * weight_y * weight_x, 0);

    set_values(input_prim, std::move(input_data));
    set_values(weights_prim, std::move(weights_data));

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 4, 2);
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->second.get_layout().batch(), input_f*input_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().feature(), weight_b);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(1), weight_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(0), weight_x);
}

TEST(fully_connected_gpu, no_biases_5d_input) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;

    //  Input  : 1x8x8x8x12
    //  Weights: 48x12
    //  Output : 1x8x8x8x48

    const int32_t input_b = 1, input_f = 8, input_z = 8, input_y = 8, input_x = 12, // size of the whole input buffer
                  weight_b = 48, weight_f = 12;                                     // size of the whole weights buffer

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfzyx, { input_b, input_f, input_x, input_y, input_z } });
    auto weights_prim = engine.allocate_memory({ { weight_b, weight_f }, data_types::f32, format::bfyx });

    std::vector<float> input_data(input_b * input_f * input_z * input_y * input_x, 0);
    std::vector<float> weights_data(weight_b * weight_f, 0);

    set_values(input_prim, std::move(input_data));
    set_values(weights_prim, std::move(weights_data));

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 5, 2);
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->second.get_layout().batch(), input_b);
    ASSERT_EQ(outputs.begin()->second.get_layout().feature(), input_f);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(2), input_z);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(1), input_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(0), weight_b);
}

TEST(fully_connected_gpu, no_biases_5d_input_immad) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    //  Input  : 1x8x8x8x12
    //  Weights: 48x12
    //  Output : 512x48x1x1

    const int32_t input_b = 1, input_f = 8, input_z = 8, input_y = 8, input_x = 12, // size of the whole input buffer
                  weight_b = 48, weight_f = 12;                                     // size of the whole weights buffer

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfzyx, { input_b, input_f, input_x, input_y, input_z } });
    auto weights_prim = engine.allocate_memory({ { weight_b, weight_f }, data_types::f32, format::bfyx });

    std::vector<float> input_data(input_b * input_f * input_z * input_y * input_x, 0);
    std::vector<float> weights_data(weight_b * weight_f, 0);

    set_values(input_prim, std::move(input_data));
    set_values(weights_prim, std::move(weights_data));

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 5, 2);
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->second.get_layout().batch(), input_f*input_z*input_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().feature(), weight_b);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(1), 1);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(0), 1);
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

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 1,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1} });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.5f, output_ptr[0]);
    ASSERT_EQ(2.75f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(7.0f, output_ptr[3]);
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

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 2,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(4.00f, output_ptr[1]);
    ASSERT_EQ(2.75f, output_ptr[2]);
    ASSERT_EQ(1.00f, output_ptr[3]);
    ASSERT_EQ(0.75f, output_ptr[4]);
    ASSERT_EQ(2.75f, output_ptr[5]);
    ASSERT_EQ(7.00f, output_ptr[6]);
    ASSERT_EQ(5.00f, output_ptr[7]);
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

    const int32_t output_f = 4,                // size of the whole output buffer
                  input_f = 3,                 // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32, { output_b, { { output_f } }, { 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(2.75f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(7.00f, output_ptr[3]);
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

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 1,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32, { output_b, { { output_f } }, { 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(0.00f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(0.00f, output_ptr[3]);
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

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 2,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32, { output_b, { { output_f } }, { 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(4.00f, output_ptr[1]);
    ASSERT_EQ(0.00f, output_ptr[2]);
    ASSERT_EQ(0.00f, output_ptr[3]);
    ASSERT_EQ(0.75f, output_ptr[4]);
    ASSERT_EQ(2.75f, output_ptr[5]);
    ASSERT_EQ(0.00f, output_ptr[6]);
    ASSERT_EQ(0.00f, output_ptr[7]);
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

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3,                  // size of the whole input buffer
                  weight_b = 4, weight_y = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32, { 1       , { { output_f } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_y, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(0.00f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(0.00f, output_ptr[3]);
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

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3,                  // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32, { 1       , { { output_f } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu_negative_slope, { 0.1f })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(-0.125f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(-0.1f, output_ptr[3]);
}

TEST(fully_connected_gpu, b_fs_yx_fsv4)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int in_B = 2;
    const int in_F = 2048;
    const int in_Y = 1;
    const int in_X = 1;

    const int W_B = 1000;
    const int W_F = in_F;
    const int W_Y = in_Y;
    const int W_X = in_X;

    const int O_F = W_B;

    // Input data
    std::vector<char> Data(in_F * in_B); // in_X = in_Y = 1
    int i = 0;
    std::generate(Data.begin(), Data.end(), [i]() mutable { return i++ % 9; });
    auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
    set_values(input, std::move(Data));

    // Create a topology
    topology topology(input_layout("input", input->get_layout()));

    // Reorder
    topology.add(reorder("reorder_in",
                         input_info("input"),
                         layout(data_types::i8, format::b_fs_yx_fsv4, { in_B, in_F, in_X, in_Y })));

    // Weights
    std::vector<char> Weights(W_B * W_F);
    i = 0;
    std::generate(Weights.begin(), Weights.end(), [=]() mutable {
        return i % 2 ? -(i++) / W_F - 1 : (i++) / W_F + 1;
    });
    auto weights_gold =
        engine.allocate_memory({ data_types::i8, format::bfyx, { W_B, W_F, W_X, W_Y } });
    auto weights_imad =
        engine.allocate_memory({ data_types::i8, format::bfyx, { W_B, W_F, W_X, W_Y } });
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology.add(data("weights_gold", weights_gold), data("weights_imad", weights_imad));

    auto bias_gold = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, O_F, 1, 1 } });
    auto bias_imad = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, O_F, 1, 1 } });

    std::vector<float> bias_data(O_F, 0);
    set_values(bias_gold, bias_data);
    set_values(bias_imad, bias_data);

    topology.add(data("bias_gold", bias_gold));
    topology.add(data("bias_imad", bias_imad));

    // Fully connected
    fully_connected fullc_gold(
        "fullc_gold", input_info("input"), "weights_gold", "bias_gold");
    fully_connected fullc_imad(
        "fullc_imad", input_info("reorder_in"), "weights_imad", "bias_imad");
    topology.add(fullc_gold, fullc_imad);


    auto input_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, W_B, 1, 1 } });
    auto input_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, W_B, 1, 1 } });
    auto output_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto output_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    set_values(input_low_mem,  rg.generate_random_1d<float>(W_B, -200, 0));
    set_values(input_high_mem, rg.generate_random_1d<float>(W_B, 1, 200));
    set_values(output_low_mem, { -127.0f });
    set_values(output_high_mem, { 127.0f });

    topology.add(data("in_lo", input_low_mem),
        data("in_hi", input_high_mem),
        data("out_lo", output_low_mem),
        data("out_hi", output_high_mem),
        quantize("quant_gold", input_info("fullc_gold"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        quantize("quant_imad", input_info("fullc_imad"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8)
    );

    // Output reorder
    auto reorder_gold =
        reorder("reorder_gold", input_info("quant_gold"), layout(data_types::i8, format::bfyx, { in_B, W_B, 1, 1 }));
    auto reorder_imad =
        reorder("reorder_imad", input_info("quant_imad"), layout(data_types::i8, format::bfyx, { in_B, W_B, 1, 1 }));
    topology.add(reorder_gold, reorder_imad);

    // Network build
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

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
        ASSERT_EQ(1, 1);
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
    auto weights_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, output_f, 1 } });

    // Generate random input data and set values
    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, input_y, input_x, -1, 1);
    auto bias_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);

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
        reorder("input_fsv", input_info("input"), { data_types::f16, format::fs_b_yx_fsv32, { batch_num, input_f, input_y, input_x } }),
        fully_connected("fc_prim", input_info("input_fsv"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi)
    {
        for (size_t fi = 0; fi < output_f; ++fi)
        {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            ASSERT_TRUE(equal);
            if (!equal)
            {
                std::cout << "At b = " << bi << ", f = " << fi << std::endl;
            }
        }
    }
}


TEST(fully_connected_gpu, bf_tiled_with_pad) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    // Test parameters
    const int batch_num = 2;
    const int feature_num = 128;
    const int input_x = 1;
    const int input_y = 36;
    const int output_y = input_y;

    // Allocate memory
    auto input_mem = engine.allocate_memory({ { batch_num, feature_num, input_y, input_x }, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ { output_y, input_y }, data_types::f16, format::bfyx });
    auto eltw_mem = engine.allocate_memory({ { 1, feature_num*2, 1, 1 }, data_types::f16, format::bfyx });

    // Generate random input data and set values
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, feature_num, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_y, input_y, 1, 1, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    std::vector<ov::float16> empty_bias(output_y, 0);
    std::vector<ov::float16> eltwise_input(feature_num*2, 0);
    for (size_t i = 0; i < feature_num; i++) {
        eltwise_input[i] = 100.f;
    }

    set_values(input_mem, input_data_bfyx);
    set_values(weights_mem, weights_data_bfyx);
    set_values(eltw_mem, eltwise_input);

    // Calculate CPU reference
    auto reference_output = dynamic_fully_connected_reference_calc<ov::float16>(batch_num * feature_num,
                                                                                input_y,
                                                                                output_y,
                                                                                input_data_bfyx,
                                                                                weights_data_bfyx,
                                                                                empty_bias);

    // Create topology to test
    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights", weights_mem),
        data("eltw_input", eltw_mem),
        tile("tile", input_info("input"), std::vector<int64_t>{1, 2, 1, 1}),
        eltwise("eltw", { input_info("tile"), input_info("eltw_input") }, eltwise_mode::sum),
        crop("crop", input_info("eltw"), tensor{batch_num, feature_num, input_x, input_y}, tensor{0, feature_num, 0, 0}),
        fully_connected("fc_prim", input_info("crop"), "weights", "", 3, 3)
    );

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));


    ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));


    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    auto output_mem = outputs.at("fc_prim").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), batch_num * feature_num * output_y);

    for (size_t i = 0; i < batch_num * feature_num * output_y; ++i) {
        ASSERT_FLOAT_EQ(reference_output[i], output_ptr[i]) << " i = " << i;
    }
}

TEST(fully_connected_gpu, bf_tiled_with_unaligned_batch) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    // Test parameters with unaligned batch size.
    const int batch_num = 17;
    const int feature_num = 1;
    const int input_x = 1;
    const int input_y = 64;
    const int output_y = input_y;

    // Allocate memory
    auto input_mem =
        engine.allocate_memory({{batch_num, feature_num, input_y, input_x}, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{output_y, input_y}, data_types::f16, format::bfyx});

    // Generate random input data and set values
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, feature_num, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_y, input_y, 1, 1, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    std::vector<ov::float16> empty_bias(output_y, 0);
    set_values(input_mem, input_data_bfyx);
    set_values(weights_mem, weights_data_bfyx);
    auto reference_output = dynamic_fully_connected_reference_calc<ov::float16>(batch_num * feature_num,
                                                                                input_y,
                                                                                output_y,
                                                                                input_data_bfyx,
                                                                                weights_data_bfyx,
                                                                                empty_bias);
    topology topology(input_layout("input", input_mem->get_layout()),
                      data("weights", weights_mem),
                      fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3));

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    auto output_mem = outputs.at("fc_prim").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), batch_num * feature_num * output_y);

    for (size_t i = 0; i < batch_num * feature_num * output_y; ++i) {
        ASSERT_FLOAT_EQ(reference_output[i], output_ptr[i]) << " i = " << i;
    }
}

TEST(fully_connected_gpu, DISABLED_fs_byx_fsv32_b34)
{
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
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
    auto weights_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, output_f, 1 } });

    // Generate random input data and set values
    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, input_y, input_x, -1, 1);
    auto bias_data = rg.generate_random_1d<ov::float16>(output_f, -1, 1);

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
        reorder("input_fsv", input_info("input"), { data_types::f16, format::fs_b_yx_fsv32, { batch_num, input_f, input_y, input_x } }),
        fully_connected("fc_prim", input_info("input_fsv"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi)
    {
        for (size_t fi = 0; fi < output_f; ++fi)
        {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            ASSERT_TRUE(equal);
            if (!equal)
            {
                std::cout << "At b = " << bi << ", f = " << fi << std::endl;
            }
        }
    }
}

TEST(fully_connected_gpu, fully_connected_gpu_fb_io_block_fp16) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    // Test parameters
    const int batch_num = 6528;
    const int in_feature_num = 69;
    const int out_feature_num = 136;

    // Allocate memory
    auto input_mem = engine.allocate_memory({ { batch_num, in_feature_num }, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ { out_feature_num, in_feature_num }, data_types::f16, format::bfyx });

    // Generate random input data and set values
    auto input_data = rg.generate_random_1d<ov::float16>(batch_num * in_feature_num, -1, 1);
    auto weights_data = rg.generate_random_1d<ov::float16>(out_feature_num * in_feature_num, -1, 1);

    std::vector<ov::float16> empty_bias(out_feature_num, 0);

    set_values(input_mem, input_data);
    set_values(weights_mem, weights_data);

    // Calculate CPU reference
    auto reference_output = dynamic_fully_connected_reference_calc<ov::float16>(batch_num,
                                                                                in_feature_num,
                                                                                out_feature_num,
                                                                                input_data,
                                                                                weights_data,
                                                                                empty_bias);
    // Create topology to test
    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights", weights_mem),
        fully_connected("fc_prim", input_info("input"), "weights", "")
    );

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_fb_io_block_fp16", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));


    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    auto output_mem = outputs.at("fc_prim").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), batch_num * out_feature_num);

    for (int b = 0; b < batch_num; b++) {
        for (int ofm = 0; ofm < out_feature_num; ofm++) {
            // fully_connected_gpu_fb_io_block_fp16 output format is yxfb while reference output format is bfyx
            ASSERT_FLOAT_EQ(reference_output[b * out_feature_num + ofm], output_ptr[ofm * batch_num + b]);
        }
    }
}

class fully_connected_gpu_tests: public ::testing::Test {
public:
    void test_compressed_scale_zp_bias(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
        auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
        auto zp_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });

        set_values(input_mem, { -0.5f, 2.0f, 0.5f, 1.0f,
                                0.5f, -2.0f, -0.5f, -1.0f });
        set_values<uint8_t>(weights_mem, { 1, 2, 3, 4,
                                        5, 6, 7, 8,
                                        9, 10, 11, 12,
                                        13, 14, 15, 0,
                                        15, 14, 13, 12,
                                        11, 10, 9, 8,
                                        7, 6, 5, 4,
                                        3, 2, 1, 0});


        set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f  });
        set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f  });
        set_values(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f });

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            data("bias", bias_mem),
            data("scale", scale_mem),
            data("zp", zp_mem),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", "scale", "zp", data_types::f32, 3, 2)
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr (output_mem, get_test_stream());

        ov::PartialShape expected_shape{1, 2, 8};
        ASSERT_EQ(expected_shape, output_mem->get_layout().get_partial_shape());

        std::vector<float> expected_result = {13.f, 58.f, -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, -11.f, -62.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f, };

        for (size_t i = 0; i < expected_result.size(); i++) {
            ASSERT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_compressed_int4_scale_dyn_quan(bool is_caching_test, bool is_dynamic, int batch = 1, bool is_wei_dyn = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int batch_num = batch;
        long int ifm_num = 1024;
        long int ofm_num = 4096;
        long int scales_group_size = 32;

        bool is_3d = true;

        auto input_ps = is_3d ?  ov::PartialShape{ batch_num, 1, ifm_num } : ov::PartialShape{ batch_num, ifm_num};
        auto dyn_input_ps = is_3d ?  ov::PartialShape{ -1, 1, ifm_num } : ov::PartialShape{ -1, ifm_num};
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });

        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -16.0f, 16.0f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weigths_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);

        if (is_wei_dyn) {
            // ifm_num is dynamic
            dyn_input_ps = is_3d ?  ov::PartialShape{ -1, -1, -1 } : ov::PartialShape{ -1, -1};
        }

        auto in_layout = is_dynamic ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, is_3d ? 3 : 2, 2);
        fc_prim.decompression_zero_point_scalar = 0;

        // Implemented dynamic quantize kernel
        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout),
                data("weights", weights_mem),
                data("scale", scale_mem),
                fc_prim
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(32));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            ASSERT_EQ(impl->get_kernels().size(), size_t((is_dynamic ? 3 : 2))); // shape-agnostic kernels
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
            auto abs_diff = std::abs((float)output_ptr_ref[i] - (float)output_ptr[i]);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
            OPENVINO_ASSERT(abs_diff < 256);
        }
        std::cout << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
    }

    void test_compressed_int4_scale(bool is_caching_test, bool is_dynamic, long int batch_num, long int scales_group_size = 128, bool is_wei_dyn = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        long int ifm_num = 256;
        long int ofm_num = 512;

        auto input_mem = engine.allocate_memory({ { batch_num, ifm_num}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {1, 1, 1, 1}, data_types::u8, format::bfyx });

        set_values<int8_t>(dcomp_zp_mem, {8});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weigths_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ {-1, ifm_num}, data_types::f16, format::bfyx }
                                    : layout{ {batch_num, ifm_num}, data_types::f16, format::bfyx };

        if (is_dynamic && is_wei_dyn) {
            // ifm_num is dynamic
            in_layout = layout{ {-1, -1}, data_types::f16, format::bfyx };
        }

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, 2, 2);

        fc_prim.decompression_zero_point_scalar = 8;

        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout),
                data("weights", weights_mem),
                data("scale", scale_mem),
                data("dcomp_zp", dcomp_zp_mem),
                fc_prim
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("dcomp_zp", dcomp_zp_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // Impl is selected only when it is running from cldnn
        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            ASSERT_EQ(impl->get_kernels().size(), 2);
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        for (size_t i = 0; i < output_ptr_ref.size(); i++)
            ASSERT_NEAR(output_ptr_ref[i], output_ptr[i], 9.0) << "i = " << i;
    }

    void test_compressed_int4_scale_large_n(bool is_caching_test, bool is_dynamic, long int batch_num, bool is_dyn_quan = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        auto supports_immad = engine.get_device_info().supports_immad;

        long int ifm_num = 4096;
        long int ofm_num = 14336;
        long int scales_group_size = 32;

        auto input_mem = engine.allocate_memory({ { batch_num, 1, ifm_num}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {1, 1, 1, 1}, data_types::u8, format::bfyx });

        set_values<int8_t>(dcomp_zp_mem, {8});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weigths_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -1.0f, 1.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ {-1, 1, ifm_num}, data_types::f16, format::bfyx }
                                    : layout{ {batch_num, 1, ifm_num}, data_types::f16, format::bfyx };

        if (is_dynamic) {
            // ifm_num is dynamic
            in_layout = layout{ {-1, -1, -1}, data_types::f16, format::bfyx };
        }

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, 3, 2);

        fc_prim.decompression_zero_point_scalar = 8;

        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout),
                data("weights", weights_mem),
                data("scale", scale_mem),
                data("dcomp_zp", dcomp_zp_mem),
                fc_prim
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("dcomp_zp", dcomp_zp_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        if (is_dyn_quan) {
            config.set_user_property(ov::hint::dynamic_quantization_group_size(32));
        } else {
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
        }

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // Impl is selected only when it is running from cldnn
        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            ASSERT_EQ(impl->get_kernels().size(), 2);
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        for (size_t i = 0; i < output_ptr_ref.size(); i++)
            ASSERT_NEAR(output_ptr_ref[i], output_ptr[i], 12.0) << "i = " << i;
    }

    void test_compressed_int4_accumulation(bool is_caching_test, bool is_dynamic, long int batch_num) {
        auto& engine = get_test_engine();

        long int ifm_num = 4096;
        long int ofm_num = 4;

        auto input_mem      = engine.allocate_memory({ { batch_num, ifm_num}, data_types::f16, format::bfyx });
        auto weights_mem    = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem      = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::f16, format::bfyx });

        auto input_data = std::vector<ov::float16>(input_mem->count());
        std::fill(input_data.begin(), input_data.end(), 1);
        set_values(input_mem, input_data);

        auto weigths_data = std::vector<uint8_t>(weights_mem->count() / 2);
        std::fill(weigths_data.begin(), weigths_data.end(), 0x11);
        set_values(weights_mem, weigths_data);

        auto scale_data = std::vector<ov::float16>(scale_mem->count());
        std::fill(scale_data.begin(), scale_data.end(), 1);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ {-1, ifm_num}, data_types::f16, format::bfyx }
                                    : layout{ {batch_num, ifm_num}, data_types::f16, format::bfyx };
        primitive_id empty_id = "";

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", empty_id, "scale", empty_id, data_types::f16);

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // Impl is selected only when it is running from cldnn
        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            ASSERT_EQ(impl->get_kernels().size(), 1);
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++)
            ASSERT_NEAR(ov::float16(ifm_num), output_ptr[i], 9.0) << "i = " << i;
    }

    void test_compressed_int4_scale_reuse(bool is_caching_test, bool is_dynamic, long int batch_num, long int scales_group_size = 128) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        long int ifm_num = 256;
        long int ofm_num = 256;

        auto input_mem = engine.allocate_memory({ { batch_num, ifm_num}, data_types::f16, format::bfyx });
        auto weights_mem1 = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto weights_mem2 = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {1, 1, 1, 1}, data_types::u8, format::bfyx });

        set_values<int8_t>(dcomp_zp_mem, {8});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem1, weigths_data);
        set_values(weights_mem2, weigths_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ {-1, ifm_num}, data_types::f16, format::bfyx }
                                    : layout{ {batch_num, ifm_num}, data_types::f16, format::bfyx };

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim1 = fully_connected("fc_prim1", input_info("input"), "weights1", "", "scale", dcomp_zp_name, data_types::f16, 2, 2);
        auto fc_prim2 = fully_connected("fc_prim2", input_info("input"), "weights2", "", "scale", dcomp_zp_name, data_types::f16, 2, 2);

        fc_prim1.decompression_zero_point_scalar = 8;
        fc_prim2.decompression_zero_point_scalar = 8;

        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout),
                data("weights1", weights_mem1),
                data("weights2", weights_mem2),
                data("scale", scale_mem),
                data("dcomp_zp", dcomp_zp_mem),
                fc_prim1,
                fc_prim2
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl = { in_layout.format, "", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "fc_prim1", fc_impl }, { "fc_prim2", fc_impl }  }));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 2);

            std::vector<cldnn::memory::ptr> res {
                engine.reinterpret_buffer(*outputs.at("fc_prim1").get_memory(), outputs.at("fc_prim1").get_layout()),
                engine.reinterpret_buffer(*outputs.at("fc_prim2").get_memory(), outputs.at("fc_prim2").get_layout()),
            };

            return res;
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights1", weights_mem1),
            data("weights2", weights_mem2),
            data("scale", scale_mem),
            data("dcomp_zp", dcomp_zp_mem),
            fc_prim1,
            fc_prim2
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // Impl is selected only when it is running from cldnn
        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim1");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            ASSERT_EQ(impl->get_kernels().size(), 2);
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(2));

        std::vector<cldnn::memory::ptr> output_mem {
            engine.reinterpret_buffer(*outputs.at("fc_prim1").get_memory(), outputs.at("fc_prim1").get_layout()),
            engine.reinterpret_buffer(*outputs.at("fc_prim2").get_memory(), outputs.at("fc_prim2").get_layout()),
        };
        auto ref_output_mem = get_ref_results();

        for (size_t i = 0; i < 2; i++) {
            cldnn::mem_lock<ov::float16> output_ptr (output_mem[i], get_test_stream());
            cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem[i], get_test_stream());

            for (size_t i = 0; i < output_ptr_ref.size(); i++)
                ASSERT_NEAR(output_ptr_ref[i], output_ptr[i], 9.0) << "i = " << i;
        }
    }

    void test_compressed_int8_scale_zp_bias(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({ {1, 1, 2}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {32, 2}, data_types::u8, format::bfyx });
        auto bias_mem = engine.allocate_memory({ {1, 1, 32}, data_types::f16, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {32, 1}, data_types::f32, format::bfyx });
        auto zp_mem = engine.allocate_memory({ {32, 1}, data_types::f32, format::bfyx });

        set_values<ov::float16>(input_mem, { -0.5f, 2.0f});
        set_values<uint8_t>(weights_mem, { 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 0,
                                           15, 14, 13, 12, 11, 10, 9, 8,
                                           7, 6, 5, 4, 3, 2, 1, 0,
                                           1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 0,
                                           15, 14, 13, 12, 11, 10, 9, 8,
                                           7, 6, 5, 4, 3, 2, 1, 0 });

        set_values<ov::float16>(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f,
                                            2.0f, -1.0f, 4.0f, -3.0f, 6.0f, -5.0f, 8.0f, 3.0f,
                                            0.0f, -3.0f, 2.0f, -5.0f, 4.0f, -5.0f, 6.0f, 1.0f,
                                            1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f });
        set_values<float>(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f,
                                             3.0f, 5.0f, -1.0f, -3.0f, 1.5f, 0.5f, 3.0f, 3.0f,
                                             1.0f, 5.0f, -1.0f, -3.0f, -0.5f, -1.5f, 1.0f, 1.0f,
                                             2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });
        set_values<float>(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f,
                                    2.0f, 3.0f, 1.0f, 0.0f, 3.0f, 1.0f, 2.0f, 2.0f,
                                    1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 6.0f, 1.0f,
                                    1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 5.0f, 2.0f });

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            data("bias", bias_mem),
            data("scale", scale_mem),
            data("zp", zp_mem),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", "scale", "zp", data_types::f16, 3, 2)
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();

        cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

        ov::PartialShape expected_shape{1, 1, 32};
        ASSERT_EQ(expected_shape, output_mem->get_layout().get_partial_shape());

        std::vector<ov::float16> expected_result = { 5.f, 12.f, -10.f, -48.f, 9.75f, -14.5f, 32.f, -19.f,
                                                     54.5f, 64.f, -9.f, -37.5f, 12.f, -3.f, 6.5f, -7.5f,
                                                     2.f, 14.5f, -4.5f, -38.f, -3.f, -30.5f, 18.5f, -8.f,
                                                     39.f, 62.f, -20.f, -32.f, 6.25f, -8.f, -3.f, -5.f, };

        for (size_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_compressed_int8_scale(bool is_caching_test, bool is_dynamic, int64_t batch_num, bool use_bias = false, bool use_zp = false, bool is_3d = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        int64_t ifm_num = 33;
        int64_t ofm_num = 65;

        auto in_shape = is_3d ? ov::PartialShape({batch_num, 1, ifm_num}) : ov::PartialShape({batch_num, ifm_num});
        auto bias_shape = is_3d ? ov::PartialShape({1, 1, ofm_num}) : ov::PartialShape({1, ofm_num});
        auto input_mem = engine.allocate_memory({ in_shape, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u8, format::bfyx });
        auto bias_mem = engine.allocate_memory({ bias_shape, data_types::f16, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, 1}, data_types::f16, format::bfyx });
        auto zp_mem = engine.allocate_memory({ {ofm_num, 1}, data_types::u8, format::bfyx });

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num, 0, 10);
        set_values(weights_mem, weigths_data);

        auto bias_data = rg.generate_random_1d<ov::float16>(ofm_num, -2.0f, 2.0f);;
        set_values(bias_mem, bias_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num, -1.0f, 1.0f);
        set_values(scale_mem, scale_data);

        auto zp_data = rg.generate_random_1d<uint8_t>(ofm_num, 0, 4);
        set_values(zp_mem, zp_data);

        auto in_partial_shape = is_3d ? ov::PartialShape({-1, -1, ifm_num}) : ov::PartialShape({-1, ifm_num});
        auto in_layout = is_dynamic ? layout{ in_partial_shape, data_types::f16, format::bfyx }
                                    : layout{ {batch_num, ifm_num}, data_types::f16, format::bfyx };

        auto bias_id = use_bias ? "bias" : "";
        auto zp_id = use_zp ? "zp" : "";

        auto fc_prim = fully_connected("fc_prim", input_info("input"),
                                        "weights", bias_id,
                                        "scale", zp_id,
                                        data_types::f16,
                                                                                in_shape.size(), 2);

        auto get_ref_results = [&]() {
            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

            topology topology(
                input_layout("input", in_layout),
                data("weights", weights_mem),
                data("scale", scale_mem),
                data("zp", zp_mem),
                data("bias", bias_mem),
                fc_prim
            );

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("zp", zp_mem),
            data("bias", bias_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        const float threshold_fp16 = 1e-1;
        for (size_t i = 0; i < output_ptr_ref.size(); i++) {
            ASSERT_NEAR(output_ptr_ref[i], output_ptr[i], threshold_fp16) << "i = " << i;
        }
    }

    void test_compressed_scale_bias(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
        auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });

        set_values(input_mem, { -0.5f, 2.0f, 0.5f, 1.0f,
                                0.5f, -2.0f, -0.5f, -1.0f });
        set_values<uint8_t>(weights_mem, { 1, 2, 3, 4,
                                        5, 6, 7, 8,
                                        9, 10, 11, 12,
                                        13, 14, 15, 0,
                                        15, 14, 13, 12,
                                        11, 10, 9, 8,
                                        7, 6, 5, 4,
                                        3, 2, 1, 0});

        set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f });
        set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 1.0f });

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            data("bias", bias_mem),
            data("scale", scale_mem),
            fully_connected("fc_prim", input_info("input"), "weights", "bias", "scale", "", data_types::f32, 3, 2)
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr (output_mem, get_test_stream());

        ov::PartialShape expected_shape{1, 2, 8};
        ASSERT_EQ(expected_shape, output_mem->get_layout().get_partial_shape());

        std::vector<float> expected_result = {19.f, 82.f, -63.f, -120.f, 24.5f, -19.5f, 37.f, -5.f, -17.f, -86.f, 69.f, 112.f, -14.5f, 7.5f, -23.f, -11.f };

        for (size_t i = 0; i < expected_result.size(); i++) {
            ASSERT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_compressed_scale_fp16(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({ { 2, 4}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f16, format::bfyx });

        set_values<ov::float16>(input_mem, { ov::float16(-0.5f), ov::float16(2.0f),  ov::float16(0.5f),  ov::float16(1.0f),
                                        ov::float16(0.5f),  ov::float16(-2.0f), ov::float16(-0.5f), ov::float16(-1.0f) });
        set_values<uint8_t>(weights_mem, { 1, 2, 3, 4,
                                        5, 6, 7, 8,
                                        9, 10, 11, 12,
                                        13, 14, 15, 0,
                                        15, 14, 13, 12,
                                        11, 10, 9, 8,
                                        7, 6, 5, 4,
                                        3, 2, 1, 0});

        set_values<ov::float16>(scale_mem, {ov::float16(2.0f), ov::float16(4.0f), ov::float16(-2.0f), ov::float16(-4.0f), ov::float16(0.5f), ov::float16(-0.5f), ov::float16(2.0f), ov::float16(2.0f)});

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            data("scale", scale_mem),
            fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 2, 2)
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();

        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        ov::PartialShape expected_shape{2, 8};
        ASSERT_EQ(expected_shape, output_mem->get_layout().get_partial_shape());

        std::vector<ov::float16> expected_result = {
        ov::float16(18), ov::float16(84), ov::float16(-66), ov::float16(-116), ov::float16(19.5), ov::float16(-13.5), ov::float16(30), ov::float16(6),
        ov::float16(-18), ov::float16(-84), ov::float16(66), ov::float16(116), ov::float16(-19.5), ov::float16(13.5), ov::float16(-30), ov::float16(-6) };

        for (size_t i = 0; i < expected_result.size(); i++) {
            ASSERT_FLOAT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_compressed_int8_scale_zp_scalar(bool is_caching_test) {
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        long ifm_num = 6;
        long ofm_num = 8;

        auto input_mem = engine.allocate_memory({ { 1, ifm_num }, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ { ofm_num, ifm_num }, data_types::u8, format::bfyx });
        auto scale_mem = engine.allocate_memory({ { ofm_num, 1 }, data_types::f16, format::bfyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {1, 1, 1, 1}, data_types::u8, format::bfyx });

        set_values<int8_t>(dcomp_zp_mem, {8});

        set_values<ov::float16>(input_mem, { -0.5f, 2.0f, 0.5f, 1.0f, 0.5f, 2.0f });
        set_values<uint8_t>(weights_mem, { 0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5,
                                           0, 1, 2, 3, 4, 5 });
        set_values<ov::float16>(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16);

        fc_prim.decompression_zero_point_scalar = 8;

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("dcomp_zp", dcomp_zp_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

        ov::PartialShape expected_shape{1, 8};
        ASSERT_EQ(expected_shape, output_mem->get_layout().get_partial_shape());

        std::vector<ov::float16> expected_result = { -52.f, -104.f, 52.f, 104.f, -13.f, 13.f, -52.f, -52.f };
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_dynamic(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, input_b = 1, weight_b = 4;

        auto fake_alignment_size = engine.get_device_info().supports_immad ? 8 : 16;
        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
        auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx });
        auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });

        set_values(input_data, { -0.5f, 2.0f, 0.5f });
        set_values(weights_data, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_data);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network->get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, fake_alignment_size)); // fake_alignment
        ASSERT_EQ(out_l.batch(), input_b);
        ASSERT_EQ(out_l.feature(), weight_b);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(1.5f, output_ptr[0]);
        ASSERT_EQ(0.75f, output_ptr[1]);
        ASSERT_EQ(-2.25f, output_ptr[2]);
        ASSERT_EQ(3.0f, output_ptr[3]);
    }

    void test_dynamic_6d_input(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_b = 1, input_f = 3, input_w = 2, input_z = 1, input_y = 1, input_x = 4;
        const int32_t weight_b = 2;

        auto input_dyn_layout = layout{ov::PartialShape{ov::Dimension(-1), input_f, input_w, input_z, input_y, input_x}, data_types::f32, format::bfwzyx};
        auto input_data = engine.allocate_memory(layout{ov::PartialShape{input_b, input_f, input_w, input_z, input_y, input_x}, data_types::f32, format::bfwzyx});
        auto weights_data = engine.allocate_memory({ov::PartialShape{weight_b, input_x}, data_types::f32, format::bfyx });

        set_values(input_data, {-0.5f, 2.0f, 0.5f, 1.f,  -1.5f, 2.0f, 0.5f, 1.f,
                                -0.5f, 2.5f, 0.5f, 1.f,  -0.5f, 3.0f, 0.5f, 1.f,
                                -0.5f, 2.0f, 0.5f, 1.f,  -0.5f, 2.0f, 2.5f, 1.f});
        set_values(weights_data, {1.5f, 1.0f, -1.0f, 0.0f,
                                0.5f, -0.5f, -0.5f, 1.0f, });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights", "", input_dyn_layout.get_rank())
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_data);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network->get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), 1);
        ASSERT_EQ(out_l.batch(), 1);
        ASSERT_EQ(out_l.feature(), 3);
        ASSERT_EQ(out_l.spatial(0), 2);
        ASSERT_EQ(out_l.spatial(1), 1);
        ASSERT_EQ(out_l.spatial(2), 1);
        ASSERT_EQ(out_l.spatial(3), 2);

        std::vector<float> expected_output = {
            0.75, -0.5, -0.75, -1, 1.25, -0.75, 1.75, -1, 0.75, -0.5, -1.25, -1.5
        };

        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        for (size_t i = 0 ; i < out_l.get_linear_size(); i++) {
            ASSERT_EQ(expected_output[i], output_ptr[i]);
        }
    }

    void test_static_6d_input(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_b = 1, input_f = 3, input_w = 2, input_z = 1, input_y = 1, input_x = 4;
        const int32_t weight_b = 2;

        auto input_dyn_layout = layout{ov::PartialShape{input_b, input_f, input_w, input_z, input_y, input_x}, data_types::f32, format::bfwzyx};
        auto input_data = engine.allocate_memory(input_dyn_layout);
        auto weights_data = engine.allocate_memory({ov::PartialShape{weight_b, input_x}, data_types::f32, format::bfyx });

        set_values(input_data, {-0.5f, 2.0f, 0.5f, 1.f,  -1.5f, 2.0f, 0.5f, 1.f,
                                -0.5f, 2.5f, 0.5f, 1.f,  -0.5f, 3.0f, 0.5f, 1.f,
                                -0.5f, 2.0f, 0.5f, 1.f,  -0.5f, 2.0f, 2.5f, 1.f});
        set_values(weights_data, {1.5f, 1.0f, -1.0f, 0.0f,
                                0.5f, -0.5f, -0.5f, 1.0f, });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights", "", input_dyn_layout.get_rank()),
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_data);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network->get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), 6);
        ASSERT_EQ(out_l.batch(), 6);
        ASSERT_EQ(out_l.feature(), 2);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        std::vector<float> expected_output = {
            0.75, -0.5, -0.75, -1, 1.25, -0.75, 1.75, -1, 0.75, -0.5, -1.25, -1.5
        };

        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        for (size_t i = 0 ; i < out_l.get_linear_size(); i++) {
            ASSERT_EQ(expected_output[i], output_ptr[i]);
        }
    }

    void test_dynamic_multi_inference_same_shape(bool is_caching_test) {
        auto& engine = get_test_engine();
        const int32_t input_f = 3, input_b = 1, weight_b = 4;

        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
        auto input_actual_layout = layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx };
        auto input_data1 = engine.allocate_memory(input_actual_layout);
        auto input_data2 = engine.allocate_memory(input_actual_layout);
        auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });
        auto fake_alignment_size = engine.get_device_info().supports_immad ? 8 : 16;
        set_values(input_data1, { 0.5f, -2.0f, -0.5f });
        set_values(input_data2, { -0.5f, 2.0f, 0.5f });
        set_values(weights_data, { 1.5f, 1.0f, 0.5f,
                                -1.0f, 0.0f, 0.5f,
                                0.5f, -0.5f, -2.0f,
                                -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        {
            network->set_input_data("input", input_data1);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network->get_output_layout(outputs.begin()->first);
            ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, fake_alignment_size)); // fake_alignment
            ASSERT_EQ(out_l.batch(), input_b);
            ASSERT_EQ(out_l.feature(), weight_b);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);

            cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

            ASSERT_EQ(-1.5f, output_ptr[0]);
            ASSERT_EQ(-0.75f, output_ptr[1]);
            ASSERT_EQ(2.25f, output_ptr[2]);
            ASSERT_EQ(-3.0f, output_ptr[3]);
        }

        {
            network->set_input_data("input", input_data2);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network->get_output_layout(outputs.begin()->first);
            ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, fake_alignment_size)); // fake_alignment
            ASSERT_EQ(out_l.batch(), input_b);
            ASSERT_EQ(out_l.feature(), weight_b);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);

            cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

            ASSERT_EQ(1.5f, output_ptr[0]);
            ASSERT_EQ(0.75f, output_ptr[1]);
            ASSERT_EQ(-2.25f, output_ptr[2]);
            ASSERT_EQ(3.0f, output_ptr[3]);
        }
    }

    void test_dynamic_multi_inference_different_shape(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, weight_b = 4;

        auto fake_alignment_size = engine.get_device_info().supports_immad ? 8 : 16;
        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
        auto input_actual_layout1 = layout{ ov::PartialShape{ 2, input_f }, data_types::f32,format::bfyx};
        auto input_actual_layout2 = layout{ ov::PartialShape{ 1, input_f }, data_types::f32,format::bfyx};
        auto input_data1 = engine.allocate_memory(input_actual_layout1);
        auto input_data2 = engine.allocate_memory(input_actual_layout2);
        auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});

        set_values(input_data1, { 0.5f, -2.0f, -0.5f,
                                -0.5f, 2.0f, 0.5f });
        set_values(input_data2, { -0.5f, 2.0f, 0.5f });
        set_values(weights_data, { 1.5f, 1.0f, 0.5f,
                                -1.0f, 0.0f, 0.5f,
                                0.5f, -0.5f, -2.0f,
                                -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto inst = network->get_primitive("fc");
        ASSERT_TRUE(inst->is_dynamic());

        {
            network->set_input_data("input", input_data1);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network->get_output_layout(outputs.begin()->first);
            ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(2, fake_alignment_size)); // fake_alignment
            ASSERT_EQ(out_l.batch(), 2);
            ASSERT_EQ(out_l.feature(), weight_b);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);

            cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

            ASSERT_EQ(-1.5f, output_ptr[0]);
            ASSERT_EQ(-0.75f, output_ptr[1]);
            ASSERT_EQ(2.25f, output_ptr[2]);
            ASSERT_EQ(-3.0f, output_ptr[3]);

            ASSERT_EQ(1.5f, output_ptr[4]);
            ASSERT_EQ(0.75f, output_ptr[5]);
            ASSERT_EQ(-2.25f, output_ptr[6]);
            ASSERT_EQ(3.0f, output_ptr[7]);
        }

        {
            network->set_input_data("input", input_data2);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network->get_output_layout(outputs.begin()->first);
            ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(1, fake_alignment_size)); // fake_alignment
            ASSERT_EQ(out_l.batch(), 1);
            ASSERT_EQ(out_l.feature(), weight_b);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);

            cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

            ASSERT_EQ(1.5f, output_ptr[0]);
            ASSERT_EQ(0.75f, output_ptr[1]);
            ASSERT_EQ(-2.25f, output_ptr[2]);
            ASSERT_EQ(3.0f, output_ptr[3]);
        }
    }

    void test_dynamic_multi_inference_multiple_shapes(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, weight_b = 4;

        auto fake_alignment_size = engine.get_device_info().supports_immad ? 8 : 16;
        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
        auto input_actual_layout1 = layout{ ov::PartialShape{ 2, input_f }, data_types::f32,format::bfyx};
        auto input_actual_layout2 = layout{ ov::PartialShape{ 1, input_f }, data_types::f32,format::bfyx};
        auto input_data1 = engine.allocate_memory(input_actual_layout1);
        auto input_data2 = engine.allocate_memory(input_actual_layout2);
        auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});

        set_values(input_data1, { 0.5f, -2.0f, -0.5f,
                                -0.5f, 2.0f, 0.5f });
        set_values(input_data2, { -0.5f, 2.0f, 0.5f });
        set_values(weights_data, { 1.5f, 1.0f, 0.5f,
                                -1.0f, 0.0f, 0.5f,
                                0.5f, -0.5f, -2.0f,
                                -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // Call different shape multiple times to ensure caching works fine
        for (size_t i = 0; i < 2; i++) {
            {
                network->set_input_data("input", input_data1);

                auto outputs = network->execute();
                ASSERT_EQ(outputs.size(), size_t(1));
                ASSERT_EQ(outputs.begin()->first, "fc");

                auto output_prim_mem = outputs.begin()->second.get_memory();

                auto out_l = network->get_output_layout(outputs.begin()->first);
                ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(2, fake_alignment_size)); // fake_alignment
                ASSERT_EQ(out_l.batch(), 2); // fake_alignment
                ASSERT_EQ(out_l.feature(), weight_b);
                ASSERT_EQ(out_l.spatial(0), 1);
                ASSERT_EQ(out_l.spatial(1), 1);

                cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

                ASSERT_EQ(-1.5f, output_ptr[0]);
                ASSERT_EQ(-0.75f, output_ptr[1]);
                ASSERT_EQ(2.25f, output_ptr[2]);
                ASSERT_EQ(-3.0f, output_ptr[3]);

                ASSERT_EQ(1.5f, output_ptr[4]);
                ASSERT_EQ(0.75f, output_ptr[5]);
                ASSERT_EQ(-2.25f, output_ptr[6]);
                ASSERT_EQ(3.0f, output_ptr[7]);
            }

            {
                network->set_input_data("input", input_data2);

                auto outputs = network->execute();
                ASSERT_EQ(outputs.size(), size_t(1));
                ASSERT_EQ(outputs.begin()->first, "fc");

                auto output_prim_mem = outputs.begin()->second.get_memory();

                auto out_l = network->get_output_layout(outputs.begin()->first);
                ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(1, fake_alignment_size)); // fake_alignment
                ASSERT_EQ(out_l.batch(), 1); // fake_alignment
                ASSERT_EQ(out_l.feature(), weight_b);
                ASSERT_EQ(out_l.spatial(0), 1);
                ASSERT_EQ(out_l.spatial(1), 1);

                cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

                ASSERT_EQ(1.5f, output_ptr[0]);
                ASSERT_EQ(0.75f, output_ptr[1]);
                ASSERT_EQ(-2.25f, output_ptr[2]);
                ASSERT_EQ(3.0f, output_ptr[3]);
            }
        }
    }

    void test_has_cached_weights_reorder(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, input_b = 1, weight_b = 4;

        auto fake_alignment_size = engine.get_device_info().supports_immad ? 8 : 16;
        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
        auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx });
        auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });

        set_values(input_data, { -0.5f, 2.0f, 0.5f });
        set_values(weights_data, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights")
        };

        ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl };
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_impl_desc} })),
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_data);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto inst = network->get_primitive("fc");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        // Disable for now as current impl selection logic unexpectedly process impl forcing
        // In shape agnostic FC impl we check that onednn impl exists (which returns true regardless of forcing options)
        // Can be enabled back once implementation manager checks global model settings and forcing map too.
        // ASSERT_TRUE(impl->is_dynamic());

        auto reorder_kernel_params = impl->get_weights_reorder_kernel_params();
        ASSERT_TRUE(reorder_kernel_params != nullptr);
        auto reorder_impl = network->get_program()->get_implementations_cache().get(*reorder_kernel_params);
        // cldnn shape agnostic kernel reorder is done in build time
        // therefore the reorder is no longer in cache, but the program_node of weight data is in the preferred format
        ASSERT_TRUE(reorder_impl == nullptr);

        auto out_l = network->get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, fake_alignment_size)); // fake_alignment
        ASSERT_EQ(out_l.batch(), input_b);
        ASSERT_EQ(out_l.feature(), weight_b);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(1.5f, output_ptr[0]);
        ASSERT_EQ(0.75f, output_ptr[1]);
        ASSERT_EQ(-2.25f, output_ptr[2]);
        ASSERT_EQ(3.0f, output_ptr[3]);
    }

    void test_weights_reorder_shapes_update(bool is_caching_test) {
        auto& engine = get_test_engine();

        // This test is skipped for immad case because shape-agnostic kernel is not used.
        if (engine.get_device_info().supports_immad)
            return;

        const int32_t input_f = 3, input_b = 1, weight_b = 4;

        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32, format::bfyx };
        auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f }, data_types::f32, format::bfyx });
        auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32, format::bfyx });

        set_values(input_data, { -0.5f, 2.0f, 0.5f });
        set_values(weights_data, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_data),
            fully_connected("fc", input_info("input"), "weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_data);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto inst = network->get_primitive("fc");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        ASSERT_TRUE(impl->need_weights_reorder());
        auto weights_reorder_params = impl->get_weights_reorder_params();
        auto out_weights_reorder_layout = weights_reorder_params->get_output_layout();
        auto out_weights_reorder_pshape = out_weights_reorder_layout.get_partial_shape();
        ASSERT_EQ(weights_data->get_layout().get_partial_shape(), out_weights_reorder_pshape);

        auto output_prim_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(1.5f, output_ptr[0]);
        ASSERT_EQ(0.75f, output_ptr[1]);
        ASSERT_EQ(-2.25f, output_ptr[2]);
        ASSERT_EQ(3.0f, output_ptr[3]);
    }

    void test_compressed_int4_scale_dyn_quan_weight_i4(bool is_dynamic, int batch = 1, int ifm = 512, int ofm = 2048,
                                                        size_t quantize_group_size = 32, int scales_group_size = 128,
                                                        bool is_wzp_test = false, bool is_wzp_scalar = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int batch_num = batch;
        long int ifm_num = ifm;
        long int ofm_num = ofm;
        long int wzp_num = is_wzp_scalar ? 1 : ofm_num;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });

        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::i4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::fbyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {wzp_num, 1}, data_types::u8, format::bfyx });


        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.f, 2.f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 4);
        set_values(weights_mem, weigths_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -2.f, 2.f);
        set_values(scale_mem, scale_data);

        if (is_wzp_test) {
            auto zp_data = rg.generate_random_1d<uint8_t>(wzp_num, 0, 2);
            set_values(dcomp_zp_mem, zp_data);
        }

        auto in_layout = is_dynamic ? layout{ ov::PartialShape{ -1, -1, -1 }, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto dcomp_zp_name = is_wzp_test ? "wzp" : "";
        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, 3, 2);

        if (is_wzp_test) {
            fc_prim.compressed_weights = true;
            fc_prim.decompression_zero_point = is_wzp_test ? "wzp" : "";
        }

        // Implemented dynamic quantize kernel
        auto get_ref_results = [&]() {
            topology topo;
            topo.add(input_layout("input", in_layout));
            topo.add(data("weights", weights_mem));
            topo.add(data("scale", scale_mem));
            topo.add(data("wzp", dcomp_zp_mem));
            topo.add(fc_prim);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topo, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("wzp", dcomp_zp_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(quantize_group_size));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            auto kernel_num = (is_dynamic) ? 3 : 2;
            kernel_num = (quantize_group_size < 32) ? 2 : kernel_num;
            ASSERT_EQ(impl->get_kernels().size(), size_t(kernel_num));
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
            auto abs_diff = std::abs((float)output_ptr_ref[i] - (float)output_ptr[i]);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
            OPENVINO_ASSERT(abs_diff < 6);
        }
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
        OPENVINO_ASSERT((avg/count) < 0.5);
    }

    void test_compressed_int8_scale_dyn_quan_weight_u8(bool is_dynamic, int batch = 1, int ifm = 512, int ofm = 2048,
                                                        size_t quantize_group_size = 32, int scales_group_size = 128,
                                                        bool is_wzp_test = false, bool is_wzp_scalar = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int batch_num = batch;
        long int ifm_num = ifm;
        long int ofm_num = ofm;
        long int wzp_num = is_wzp_scalar ? 1 : ofm_num;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });

        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u8, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::fbyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {wzp_num, 1}, data_types::u8, format::bfyx });


        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.f, 2.f);
        set_values(input_mem, input_data);

        auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num, 0, 4);
        set_values(weights_mem, weigths_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -2.f, 2.f);
        set_values(scale_mem, scale_data);

        if (is_wzp_test) {
            auto zp_data = rg.generate_random_1d<uint8_t>(wzp_num, 0, 2);
            set_values(dcomp_zp_mem, zp_data);
        }

        auto in_layout = is_dynamic ? layout{ ov::PartialShape{ -1, -1, -1 }, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto dcomp_zp_name = is_wzp_test ? "wzp" : "";
        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, 3, 2);

        if (is_wzp_test) {
            fc_prim.compressed_weights = true;
            fc_prim.decompression_zero_point = is_wzp_test ? "wzp" : "";
        }

        // Implemented dynamic quantize kernel
        auto get_ref_results = [&]() {
            topology topo;
            topo.add(input_layout("input", in_layout));
            topo.add(data("weights", weights_mem));
            topo.add(data("scale", scale_mem));
            topo.add(data("wzp", dcomp_zp_mem));
            topo.add(fc_prim);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topo, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("wzp", dcomp_zp_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(quantize_group_size));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

        if (is_dynamic && !engine.get_device_info().supports_immad) {
            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            // For UINT8 weight, SLM kernel (no dyn-quan) would not be selected
            auto kernel_num = (is_dynamic) ? 3 : 1;
            kernel_num = (quantize_group_size < 32) ? 1 : kernel_num;
            ASSERT_EQ(impl->get_kernels().size(), size_t(kernel_num));
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        const int batch_alignment = 64;
        if ((batch > batch_alignment) && (batch % batch_alignment != 0)) {
            ASSERT_EQ(output_mem->get_layout().batch(), align_to(batch, batch_alignment));
        }
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
            auto abs_diff = std::abs((float)output_ptr_ref[i] - (float)output_ptr[i]);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
            OPENVINO_ASSERT(abs_diff < 8);
        }
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
        OPENVINO_ASSERT((avg/count) < 0.8);
    }
};

using shared_dims = std::tuple<size_t, size_t, size_t>;
using fully_connected_test_params = std::tuple<
    size_t,        // batch_num
    shared_dims,   // input_f input_x input_y
    size_t,        // output_f
    format::type,  // input format
    format::type,  // output format
    std::string   // kernel
>;

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct fully_connected_random_test : ::testing::TestWithParam<fully_connected_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void run_test(bool is_caching_test = false) {
        shared_dims dims;
        size_t batch, input_f, input_x, input_y, output_f;
        format::type input_format, output_format;
        std::string kernel;

        std::tie(batch, dims, output_f, input_format, output_format, kernel) = GetParam();
        std::tie(input_f, input_x, input_y) = dims;

        auto input_data = rg.generate_random_4d<InputT>(batch, input_f, input_y, input_x, type_test_ranges<InputT>::min, type_test_ranges<InputT>::max, type_test_ranges<InputT>::k);
        auto weights_data = rg.generate_random_4d<WeightsT>(output_f, input_f, input_y, input_x, type_test_ranges<WeightsT>::min, type_test_ranges<WeightsT>::max, type_test_ranges<WeightsT>::k);
        auto bias_data = rg.generate_random_2d<BiasT>(1, output_f, type_test_ranges<BiasT>::min, type_test_ranges<BiasT>::max, type_test_ranges<BiasT>::k);

        auto& eng = get_test_engine();
        auto net = network_test(eng);
        auto input = net.add_input_layout<InputT, 4>("input", input_format, std::move(input_data));
        auto weights = net.add_data<WeightsT, 4>("weights", format::oiyx, std::move(weights_data));
        auto bias = net.add_data<BiasT, 2>("bias", format::bfyx, std::move(bias_data));
        auto fc = net.add_fully_connected<OutputT>("fc_prim", input, weights, bias, ov::intel_gpu::ImplementationDesc{ output_format, kernel });

        net.run(get_test_default_config(eng, ov::intel_gpu::optimize_data(true)), is_caching_test);
    }
};

using fully_connected_random_test_f32 = fully_connected_random_test<float, float, float, float>;
using fully_connected_random_test_f16 = fully_connected_random_test<ov::float16, ov::float16, ov::float16, ov::float16>;

TEST_P(fully_connected_random_test_f32, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f32,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx, format::yxfb),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_bfyx_batched,
    fully_connected_random_test_f32,
    ::testing::Combine(
        ::testing::Values(2, 8),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::bfyx),
        ::testing::Values(""))
);

TEST_P(fully_connected_random_test_f16, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_bfyx_b2,
    fully_connected_random_test_f16,
    ::testing::Combine(
        // Batch 1 is disabled due to sporadic failures in `fully_connected_gpu_bs_f_bsv16_b1`
        // - there are nans in output.
        ::testing::Values(2),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_fully_connected_gpu_bs_f_bsv16_af8_vload,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(16),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{17, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values("fully_connected_gpu_bs_f_bsv16_af8_vload"))
);

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::yxfb),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_bfyx_batched,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(2, 8),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::bfyx),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    export_import,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(2),
        ::testing::Values(shared_dims{32, 1, 1}),
        ::testing::Values(32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::bfyx),
        ::testing::Values(""))
);

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct fully_connected_random_test_3d : ::testing::TestWithParam<fully_connected_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void run_test(bool is_caching_test = false) {
        shared_dims dims;
        size_t batch, input_f, input_x, input_y, output_y;
        format::type input_format, output_format;
        std::string kernel;

        std::tie(batch, dims, output_y, input_format, output_format, kernel) = GetParam();
        std::tie(input_f, input_x, input_y) = dims;

        auto input_data = rg.generate_random_4d<InputT>(batch, input_f, input_y, input_x, type_test_ranges<InputT>::min, type_test_ranges<InputT>::max, type_test_ranges<InputT>::k);
        auto weights_data = rg.generate_random_4d<WeightsT>(output_y, input_y, 1, 1, type_test_ranges<WeightsT>::min, type_test_ranges<WeightsT>::max, type_test_ranges<WeightsT>::k);
        auto bias_data = rg.generate_random_2d<BiasT>(1, output_y, type_test_ranges<BiasT>::min, type_test_ranges<BiasT>::max, type_test_ranges<BiasT>::k);

        auto& eng = get_test_engine();
        auto net = network_test(eng);
        auto input = net.add_input_layout<InputT, 4>("input", input_format, std::move(input_data));
        auto weights = net.add_data<WeightsT, 4>("weights", format::oiyx, std::move(weights_data));
        auto bias = net.add_data<BiasT, 2>("bias", format::bfyx, std::move(bias_data));
        auto fc = net.add_fully_connected_3d<OutputT>("fc_prim", input, weights, bias, ov::intel_gpu::ImplementationDesc{ output_format, kernel }, 3);

        ExecutionConfig config = get_test_default_config(eng);
        config.set_property(ov::intel_gpu::optimize_data(true));
        net.run(config, is_caching_test);
    }
};


using fully_connected_random_test_f32_3d = fully_connected_random_test_3d<float, float, float, float>;
using fully_connected_random_test_f16_3d = fully_connected_random_test_3d<ov::float16, ov::float16, ov::float16, ov::float16>;
using fully_connected_random_test_i8_3d = fully_connected_random_test_3d<int8_t, int8_t, int8_t, float>;


TEST_P(fully_connected_random_test_f32_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f32_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{1, 1, 2},
                          shared_dims{1, 1, 3},
                          shared_dims{3, 1, 2},
                          shared_dims{3, 1, 3}),
        ::testing::Values(1, 3, 16),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_tiled,
    fully_connected_random_test_f32_3d,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{64, 1, 65},
                          shared_dims{64, 1, 128},
                          shared_dims{65, 1, 65},
                          shared_dims{65, 1, 128}),
        ::testing::Values(1, 32, 64),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_big,
    fully_connected_random_test_f32_3d,
    ::testing::Combine(
        ::testing::Values(3),
        ::testing::Values(shared_dims{16, 1, 17},
                          shared_dims{16, 1, 32},
                          shared_dims{32, 1, 17},
                          shared_dims{32, 1, 32}),
        ::testing::Values(17, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

TEST_P(fully_connected_random_test_f16_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f16_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{1, 1, 2},
                          shared_dims{1, 1, 16},
                          shared_dims{3, 1, 2},
                          shared_dims{3, 1, 16}),
        ::testing::Values(1, 3, 16),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

TEST_P(fully_connected_random_test_i8_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_i8_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{1, 1, 2},
                          shared_dims{1, 1, 16},
                          shared_dims{3, 1, 2},
                          shared_dims{3, 1, 16}),
        ::testing::Values(1, 3, 16),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_imad,
    fully_connected_random_test_i8_3d,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{64, 1, 65},
                          shared_dims{64, 1, 128},
                          shared_dims{65, 1, 65},
                          shared_dims{65, 1, 128}),
        ::testing::Values(1, 32, 64),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_big,
    fully_connected_random_test_i8_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{16, 1, 17},
                          shared_dims{16, 1, 32},
                          shared_dims{32, 1, 17},
                          shared_dims{32, 1, 32}),
        ::testing::Values(17, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""))
);


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
        return ov::element::from<InputT>();
    }

    data_types output_data_type() {
        return ov::element::from<OutputT>();
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

        auto input_sizes = input_size.sizes();
        auto last_dim = std::find_if(input_sizes.rbegin(), input_sizes.rend(),
                                     [](tensor::value_type x) { return x != 1l; });
        size_t input_rank = std::distance(input_sizes.begin(), last_dim.base());
        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "bias", input_rank);
        fc_prim.output_data_types = {static_cast<ov::element::Type_t>(ov::element::from<OutputT>())};
        topo.add(fc_prim);

        topo.add(data("quant_input_low", quantization_input_low));
        topo.add(data("quant_input_high", quantization_input_high));
        topo.add(data("quant_output_low", quantization_output_low));
        topo.add(data("quant_output_high", quantization_output_high));
        topo.add(quantize("quantization_prim",
            input_info("fc_prim"),
            input_info("quant_input_low"),
            input_info("quant_input_high"),
            input_info("quant_output_low"),
            input_info("quant_output_high"),
            _quantization.levels,
            output_data_type()
            ));

        topo.add(reorder("output", input_info("quantization_prim"), format::bfyx, output_data_type()));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network net(engine, topo, config);
        net.set_input_data("input", input_prim);

        auto output = net.execute();
        auto out_mem = output.at("output").get_memory();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num(); ++bi) {
            for (size_t fi = 0; fi < output_f(); ++fi) {
                ASSERT_NEAR(out_ptr[bi * output_f() + fi], expected[bi][fi], 1) << "at b = " << bi << ", fi = " << fi;
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
                        * (1 / (levels - 1) * (output_high - output_low))
                        + output_low);
            }
            else {
                output[bi][fi] = static_cast<OutputT>(std::round(
                    std::round((acc - input_low) / (input_high - input_low) * (levels - 1))
                        * (1 / (levels - 1) * (output_high - output_low))
                        + output_low));
            }
        }
    }
    return output;
}


template <typename InputT, typename OutputT>
class fc_quantized_random_test
    : public fully_connected_quantized_test<InputT, OutputT>
    , public ::testing::WithParamInterface< fully_connected_quantized_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

public:
    void run_random_test() {
        size_t b, in_f, in_x, in_y, out_f;
        format::type in_fmt;

        std::tie(b, in_f, in_x, in_y, out_f, in_fmt) = GetParam();

        VVVVF<InputT> input_data = rg.generate_random_4d<InputT>(b, in_f, in_y, in_x, -127, 127);
        VVVVF<int8_t> weights_data = rg.generate_random_4d<int8_t>(out_f, in_f, in_y, in_x, -127, 127);
        VF<int> bias_data = rg.generate_random_1d<int>(out_f, -127, 127);
        bool is_unsigned = std::is_same<OutputT, uint8_t>::value;
        quantization_t quant_data;
        quant_data.input_low   = rg.generate_random_1d<float>(out_f, -200, 0);
        quant_data.input_high  = rg.generate_random_1d<float>(out_f, 1, 200);
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
        testing::Values(16, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(16, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    imad,
    fully_connected_i8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(64, 128),
        testing::Values(1),
        testing::Values(1),
        testing::Values(1, 31, 64, 65),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    mmad,
    fully_connected_i8_i8_test,
    testing::Combine(
        testing::Values(1),
        testing::Values(16, 43, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(16, 32, 64),
        testing::Values(format::bfyx, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(16, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(16, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1),
        testing::Values(1),
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
        testing::Values(1),
        testing::Values(1),
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
        testing::Values(1),
        testing::Values(1),
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
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_f32_test::PrintToStringParamName
);

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(fully_connected_onednn_gpu, no_biases_int8) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3

    const int32_t input_f = 3, input_b = 1,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    // Change input data of fully-connected node from bx to bf
    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f });
    set_values<int8_t>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", input_info("input"), { data_types::i8, format::bfyx, { input_b, input_f, 1, 1 } });
    auto fc = fully_connected("fc_prim", input_info("reorder_to_int"), "weights");
    auto rf = reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { input_b, 4, 1, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl} }));
    network network(engine, topology, cfg);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(18.0f, output_ptr[0]);
    ASSERT_EQ(-32.0f, output_ptr[1]);
    ASSERT_EQ(12.0f, output_ptr[2]);
    ASSERT_EQ(-52.0f, output_ptr[3]);
}

TEST(fully_connected_3d_onednn_gpu, no_biases_int8) {
    //  Input  : 1x2x3x1 (3D FC case)
    //  Output : 2x4
    //  Weights: 4x3

    const int32_t input_y = 3, input_f = 2, input_b = 1,  // size of the whole input buffer
                  weight_o = 4, weight_i = 3,             // size of the whole weights buffer
                  output_b = 2, output_f = 4;

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, 1, input_y } });
    auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, { weight_o, weight_i, 1, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f, 8.4f, 2.3f, -4.49f });
    set_values<int8_t>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", input_info("input"), { data_types::i8, format::bfyx, { input_b, input_f, 1, input_y } });
    auto fc = fully_connected("fc_prim", input_info("reorder_to_int"), "weights", "", 3);
    auto rf = reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { output_b, output_f, 1, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl} }));

    network network(engine, topology, cfg);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    for (int b = 0; b < output_b; b++) {
        ASSERT_EQ(18.0f, output_ptr[b * output_f + 0]);
        ASSERT_EQ(-32.0f, output_ptr[b * output_f + 1]);
        ASSERT_EQ(12.0f, output_ptr[b * output_f + 2]);
        ASSERT_EQ(-52.0f, output_ptr[b * output_f + 3]);
    }
}

TEST(fully_connected_3d_onednn_gpu, compressed_int4_scale_static) {
    tests::random_generator rg(GET_SUITE_NAME);

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    long int batch_num = 2;
    long int ifm_num = 64;
    long int ofm_num = 4;
    long int scales_group_size = 32;

    auto input_mem = engine.allocate_memory({ { 1, batch_num, ifm_num, 1}, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num, 1, 1}, data_types::u4, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size, 1, 1}, data_types::f16, format::bfyx });
    auto dcomp_zp_mem = engine.allocate_memory({ {1, 1, 1, 1}, data_types::u8, format::bfyx });

    set_values<int8_t>(dcomp_zp_mem, {8});

    auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
    set_values(input_mem, input_data);

    auto weigths_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
    set_values(weights_mem, weigths_data);

    auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
    set_values(scale_mem, scale_data);

    auto in_layout = layout{ {1, batch_num, ifm_num, 1}, data_types::f16, format::bfyx };

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "dcomp_zp", data_types::f16, 3, 4);

    fc_prim.decompression_zero_point_scalar = 8;

    auto get_ref_results = [&]() {
        topology topology(
            input_layout("input", in_layout),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("dcomp_zp", dcomp_zp_mem),
            fc_prim
        );

        auto config = get_test_default_config(engine);
        ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));

        network network(engine, topology, config);
        network.set_input_data("input", input_mem);

        auto outputs = network.execute();
        OPENVINO_ASSERT(outputs.size() == 1);
        OPENVINO_ASSERT(outputs.begin()->first == "fc_prim");

        auto output_layout = outputs.begin()->second.get_layout();
        auto output_mem = outputs.begin()->second.get_memory();

        return engine.reinterpret_buffer(*output_mem, output_layout);
    };

    topology topology(
        input_layout("input", in_layout),
        data("weights", weights_mem),
        data("scale", scale_mem),
        data("dcomp_zp", dcomp_zp_mem),
        fc_prim
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    network->set_input_data("input", input_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

    auto ref_output_mem = get_ref_results();
    cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

    for (size_t i = 0; i < output_ptr_ref.size(); i++) {
        ASSERT_NEAR(output_ptr_ref[i], output_ptr[i], 9.0) << "i = " << i << std::endl;
    }
}
#endif

TEST_F(fully_connected_gpu_tests, compressed_scale_zp_bias) {
    this->test_compressed_scale_zp_bias(false);
}

TEST_F(fully_connected_gpu_tests, compressed_scale_zp_bias_cached) {
    this->test_compressed_scale_zp_bias(true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale) {
    this->test_compressed_int4_scale(false, false, 256);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_large_n) {
    this->test_compressed_int4_scale_large_n(false, false, 1);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_large_n_cached) {
    this->test_compressed_int4_scale_large_n(true, false, 1);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_large_n_dynamic) {
    this->test_compressed_int4_scale_large_n(false, true, 1);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_large_n_dynamic_cached) {
    this->test_compressed_int4_scale_large_n(true, true, 1);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_large_n_dyn_quan) {
    this->test_compressed_int4_scale_large_n(false, false, 1, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_large_n_dyn_quan_dynamic) {
    this->test_compressed_int4_scale_large_n(true, false, 1, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_reuse_scale) {
    this->test_compressed_int4_scale_reuse(false, true, 256);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_cached) {
    this->test_compressed_int4_scale(true, false, 256);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic) {
    this->test_compressed_int4_scale(false, true, 260);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_dynamic_acc) {
    this->test_compressed_int4_accumulation(false, true, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_cached) {
    this->test_compressed_int4_scale(true, true, 260);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_b1g32) {
    this->test_compressed_int4_scale(false, true, 1, 32);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_b48g32) {
    this->test_compressed_int4_scale(false, true, 48, 32);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_b1g64) {
    this->test_compressed_int4_scale(false, true, 1, 64);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_b1g128) {
    this->test_compressed_int4_scale(false, true, 1, 128);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_b1g32) {
    this->test_compressed_int4_scale(false, false, 1, 32);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_b1g64) {
    this->test_compressed_int4_scale(false, false, 1, 64);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_b1g128) {
    this->test_compressed_int4_scale(false, false, 1, 128);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_single_batch) {
    this->test_compressed_int4_scale_dyn_quan(false, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan) {
    this->test_compressed_int4_scale_dyn_quan(false, false, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_unaligned) {
    this->test_compressed_int4_scale_dyn_quan(false, false, 511);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_dynamic_single_batch) {
    this->test_compressed_int4_scale_dyn_quan(false, true, 1);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_dynamic) {
    this->test_compressed_int4_scale_dyn_quan(false, true, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_dynamic_unaligned) {
    this->test_compressed_int4_scale_dyn_quan(false, true, 511);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_cache) {
    this->test_compressed_int4_scale_dyn_quan(true, false, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_cache_dynamic) {
    this->test_compressed_int4_scale_dyn_quan(true, true, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_f_input) {
    this->test_compressed_int4_scale(false, true, 256, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_f_input_cached) {
    this->test_compressed_int4_scale(true, true, 260, true);
}
TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_f_input_b1g64) {
    this->test_compressed_int4_scale(false, true, 1, 64, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_f_input_b1g128) {
    this->test_compressed_int4_scale(false, true, 1, 128, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_dynamic_f_input_single_batch) {
    this->test_compressed_int4_scale_dyn_quan(false, true, 1, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_dynamic_f_input) {
    this->test_compressed_int4_scale_dyn_quan(false, true, 512, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dyn_quan_dynamic_f_input_unaligned) {
    this->test_compressed_int4_scale_dyn_quan(false, true, 511, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_batch_1) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 1, 2048, 3072);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_12_groupsize) {
    // Expect no dynamic-quantized FC
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 269, 512, 1024, 12);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_34_groupsize) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560, 34);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_64_groupsize) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560, 64);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_148_groupsize) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560, 148);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_128_groupsize) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560, 128);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_128_groupsize_32_scale) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560, 128, 32);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_128_groupsize_64_scale) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 359, 1536, 2560, 128, 64);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_no_wzp) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 320, 1024, 1024, 32, 32, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 320, 1024, 1024, 32, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp_scalar) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 320, 1024, 1024, 32, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp_128) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 320, 1024, 1024, 128, 128, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp_static) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(false, 320, 1024, 1024, 32, 32, true);
}

// Test weight zp for INT8 ASYM
TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_large_input_1025) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 1025, 3584, 4608, 128, 128, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_large) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 320, 4096, 4096, 128, 128, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_32_ifm_1024) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 320, 1024, 1024, 32, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_32_ifm_2048) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 320, 2048, 2048, 32, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_32_ifm_4096) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 320, 4096, 4096, 32, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_small) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 16, 1024, 1024, 128, 128, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_single) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 1, 1024, 1024, 128, 128, true);
}

// Test per-token dyn-quan
TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_test_fake_per_token) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 600, 1024, 2048, -1, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_test_per_token) {
    this->test_compressed_int4_scale_dyn_quan_weight_i4(true, 600, 1024, 2048, -1, 1024, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_test_per_token_small_scale) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 600, 1024, 2048, -1, 32, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_test_per_token_full_scale) {
    this->test_compressed_int8_scale_dyn_quan_weight_u8(true, 600, 1024, 2048, -1, 1024, true);
}

TEST_F(fully_connected_gpu_tests, compressed_scale_bias) {
    this->test_compressed_scale_bias(false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_zp_bias) {
    this->test_compressed_int8_scale_zp_bias(false);
}

TEST_F(fully_connected_gpu_tests, compressed_scale_bias_cached) {
    this->test_compressed_scale_bias(true);
}

TEST_F(fully_connected_gpu_tests, compressed_scale_fp16) {
    this->test_compressed_scale_fp16(false);
}

TEST_F(fully_connected_gpu_tests, compressed_scale_fp16_cached) {
    this->test_compressed_scale_fp16(false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_zp_scalar) {
    // Testing support for decompression zero points with group size that is not a power of two
    this->test_compressed_int8_scale_zp_scalar(false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_b1) {
    this->test_compressed_int8_scale(false, true, 1, false, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_b1_bias) {
    this->test_compressed_int8_scale(false, true, 1, true, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_b1_bias_zp_3d) {
    this->test_compressed_int8_scale(false, true, 1, true, true, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_cached) {
    this->test_compressed_int8_scale(true, false, 1, true, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_zp_b1) {
    this->test_compressed_int8_scale(false, true, 1, false, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_zp_b13) {
    this->test_compressed_int8_scale(false, true, 13, false, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_zp_b12_3d) {
    this->test_compressed_int8_scale(false, true, 12, false, true, true);
}

TEST_F(fully_connected_gpu_tests, dynamic) {
    this->test_dynamic(false);
}

TEST_F(fully_connected_gpu_tests, dynamic_cached) {
    this->test_dynamic(true);
}

TEST_F(fully_connected_gpu_tests, dynamic_6d_input) {
    this->test_dynamic_6d_input(false);
}

TEST_F(fully_connected_gpu_tests, dynamic_6d_input_cached) {
    this->test_dynamic_6d_input(true);
}

TEST_F(fully_connected_gpu_tests, static_6d_input) {
    this->test_static_6d_input(false);
}

TEST_F(fully_connected_gpu_tests, static_6d_input_cached) {
    this->test_static_6d_input(true);
}

TEST_F(fully_connected_gpu_tests, dynamic_multi_inference_same_shape) {
    this->test_dynamic_multi_inference_same_shape(false);
}

TEST_F(fully_connected_gpu_tests, dynamic_multi_inference_same_shape_cached) {
    this->test_dynamic_multi_inference_same_shape(true);
}

TEST_F(fully_connected_gpu_tests, dynamic_multi_inference_different_shape) {
    this->test_dynamic_multi_inference_different_shape(false);
}

TEST_F(fully_connected_gpu_tests, dynamic_multi_inference_different_shape_cached) {
    this->test_dynamic_multi_inference_different_shape(true);
}

TEST_F(fully_connected_gpu_tests, dynamic_multi_inference_multiple_shapes) {
    this->test_dynamic_multi_inference_multiple_shapes(false);
}

TEST_F(fully_connected_gpu_tests, dynamic_multi_inference_multiple_shapes_cached) {
    this->test_dynamic_multi_inference_multiple_shapes(true);
}


using fully_connected_dynamic_test_params = std::tuple<
    std::vector<ov::Dimension::value_type>, // batch_sizes
    ov::Dimension::value_type,              // input_f
    ov::Dimension::value_type,              // output_f
    bool                                    // 3D case
>;

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct dynamic_fully_connected_gpu : ::testing::TestWithParam<fully_connected_dynamic_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void run_test(bool is_caching_test = false) {
        std::vector<ov::Dimension::value_type> batch_sizes;
        ov::Dimension::value_type input_f;
        ov::Dimension::value_type output_f;
        bool fc_3d = false;

        std::tie(batch_sizes, input_f, output_f, fc_3d) = GetParam();

        auto input_dt = ov::element::from<InputT>();
        auto weights_dt = ov::element::from<WeightsT>();
        auto output_dt = ov::element::from<OutputT>();

        auto& engine = get_test_engine();
        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), input_f }, input_dt, format::bfyx };
        if (fc_3d)
            input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), ov::Dimension(), input_f }, input_dt, format::bfyx };

        auto weights_mem = engine.allocate_memory({ ov::PartialShape{ output_f, input_f }, weights_dt, format::bfyx });
        auto weights_data_vec = rg.generate_random_1d<WeightsT>(output_f * input_f, -1, 1);

        auto bias_mem = engine.allocate_memory({ ov::PartialShape{ output_f }, output_dt, format::bfyx });
        auto bias_data_vec = rg.generate_random_1d<OutputT>(output_f, 0, 1);

        set_values(weights_mem, weights_data_vec);
        set_values(bias_mem, bias_data_vec);

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_mem),
            data("bias", bias_mem),
        };

        if (fc_3d)
            topology.add(fully_connected("fc", input_info("input"), "weights", "bias", output_dt, 3));
        else
            topology.add(fully_connected("fc", input_info("input"), "weights", "bias", output_dt));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        for (const auto& batch_size : batch_sizes) {
            auto input_actual_layout = layout{ ov::PartialShape{ batch_size, input_f }, input_dt, format::bfyx };
            if (fc_3d)
                input_actual_layout = layout{ ov::PartialShape{ 1, batch_size, input_f }, input_dt, format::bfyx };
            cldnn::memory_ptr input_mem = engine.allocate_memory(input_actual_layout);
            std::vector<InputT> input_data_vec = rg.generate_random_1d<InputT>(batch_size * input_f, 0, 1);
            set_values(input_mem, input_data_vec);
            network->set_input_data("input", input_mem);

            auto outputs = network->execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network->get_output_layout(outputs.begin()->first);
            ASSERT_EQ(out_l.batch(), fc_3d ? 1 : batch_size);
            ASSERT_EQ(out_l.feature(), fc_3d ? batch_size : output_f);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), fc_3d ? output_f : 1);

            cldnn::mem_lock<OutputT> output_ptr(output_prim_mem, get_test_stream());

            auto ref_result = dynamic_fully_connected_reference_calc<OutputT>(batch_size,
                                                                              input_f,
                                                                              output_f,
                                                                              input_data_vec,
                                                                              weights_data_vec,
                                                                              bias_data_vec);

            if (engine.get_device_info().supports_immad) {
                for (int b = 0; b < batch_size; b++) {
                    for (int ofm = 0; ofm < output_f; ofm++) {
                        EXPECT_NEAR(ref_result[b * output_f + ofm], output_ptr[b * output_f + ofm],
                                    default_tolerance(input_dt));
                    }
                }
            } else {
                for (int b = 0; b < batch_size; b++) {
                    for (int ofm = 0; ofm < output_f; ofm++) {
                        ASSERT_EQ(ref_result[b * output_f + ofm], output_ptr[b * output_f + ofm]);
                    }
                }
            }
        }
    }
};

using dynamic_fully_connected_gpu_f32_3d = dynamic_fully_connected_gpu<float, float, float, float>;
using dynamic_fully_connected_gpu_f16_3d = dynamic_fully_connected_gpu<ov::float16, ov::float16, ov::float16, ov::float16>;
using dynamic_fully_connected_gpu_i8_3d = dynamic_fully_connected_gpu<int8_t, int8_t, int8_t, float>;

static const std::vector<ov::Dimension::value_type>
    dyn_batches_full = {1, 2, 4, 7, 8, 9, 15, 16, 31, 32, 33, 47, 48, 49, 58, 63, 64};
static const std::vector<ov::Dimension::value_type>
    dyn_batches_smoke = {1, 2, 7, 8, 9, 16, 32, 33, 47, 48, 58};

TEST_P(dynamic_fully_connected_gpu_f32_3d, basic) {
    run_test();
}

TEST_P(dynamic_fully_connected_gpu_f16_3d, basic) {
    run_test();
}

TEST_P(dynamic_fully_connected_gpu_i8_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    dynamic_fully_connected_gpu_f32_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_smoke),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    smoke,
    dynamic_fully_connected_gpu_f16_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_smoke),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    smoke,
    dynamic_fully_connected_gpu_i8_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_smoke),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    full,
    dynamic_fully_connected_gpu_f32_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_full),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 16, 32, 64, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    full,
    dynamic_fully_connected_gpu_f16_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_full),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 16, 32, 64, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    full,
    dynamic_fully_connected_gpu_i8_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_full),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 16, 32, 64, 128),
        ::testing::Values(false, true))
);

TEST_F(fully_connected_gpu_tests, has_cached_weights_reorder) {
    this->test_has_cached_weights_reorder(false);
}

TEST_F(fully_connected_gpu_tests, has_cached_weights_reorder_cached) {
    this->test_has_cached_weights_reorder(true);
}

template <typename InputT, typename T>
VVF<T> fully_connected_types_reference(VVVVF<InputT> &input, VVVVF<T> &weights, VF<T> &bias, const quantization_t& quantization, bool relu = false, T slope = 0.0f) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();         // input is assumed to be bfyx
    size_t output_f = weights.size();       // weights are assumed to be bfyx
    VVF<T> output(output_b, VF<T>(output_f));
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
            if (res > quantization.output_high)
                output[b][n] = quantization.output_high;
            else {
                if (res < quantization.output_low)
                    output[b][n] = quantization.output_low;
                else
                    output[b][n] = (T)res;
            }
        }
    }
    return output;
}

using fully_connected_types_test_params = std::tuple<
        size_t,  // batch_num
        size_t,  // input_f
        size_t,  // input_x
        size_t,  // input_y
        size_t,  // output_f
        format::type  // format
>;

template <typename InputT, typename WeightsT>
class fully_connected_types_test : public ::testing::Test {
private:
    size_t batch_num() { return _input.size(); }
    size_t input_f() { return _input[0].size(); }
    size_t input_y() { return _input[0][0].size(); }
    size_t input_x() { return _input[0][0][0].size(); }
    size_t output_f() { return _weights.size(); }

    data_types input_data_type() {
        return ov::element::from<InputT>();
    }

    data_types weights_data_type() {
        return ov::element::from<WeightsT>();
    }

    bool has_bias() { return _bias.size() > 0; }

public:
    static std::string PrintToStringParamName(testing::TestParamInfo<fully_connected_types_test_params> param_info) {
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

    void set_weights(VVVVF<WeightsT> _data) {
        _weights = std::move(_data);
    }

    void set_bias(VF<WeightsT> _data) {
        _bias = std::move(_data);
    }

    void set_input_format(format::type fmt) {
        _fmt = fmt;
    }

    void run_test(VVF<WeightsT> expected) {
        auto& engine = get_test_engine();

        auto input_size = tensor(TensorValue(batch_num()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));
        auto weights_size = tensor(TensorValue(output_f()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));

        auto input_prim = engine.allocate_memory({ input_data_type(), _fmt, input_size });
        auto weights_prim = engine.allocate_memory({ weights_data_type(), format::bfyx, weights_size });

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

        auto bias_prim = engine.allocate_memory({ weights_data_type(), format::bfyx, tensor(feature(output_f()))});
        set_values(bias_prim, _bias);

        topology topo;
        topo.add(data("weights", weights_prim));
        topo.add(data("bias", bias_prim));

        topo.add(input_layout("input", input_prim->get_layout()));

        auto input_sizes = input_size.sizes();
        auto last_dim = std::find_if(input_sizes.rbegin(), input_sizes.rend(),
                                     [](tensor::value_type x) { return x != 1l; });
        size_t input_rank = std::distance(input_sizes.begin(), last_dim.base());
        auto fc_prim = fully_connected("output", input_info("input"), "weights", "bias", input_rank);
        fc_prim.output_data_types = { static_cast<ov::element::Type_t>(ov::element::from<WeightsT>()) };
        topo.add(fc_prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network net(engine, topo, config);
        net.set_input_data("input", input_prim);

        auto output = net.execute();
        auto out_mem = output.at("output").get_memory();
        cldnn::mem_lock<WeightsT> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num(); ++bi) {
            for (size_t fi = 0; fi < output_f(); ++fi) {
                ASSERT_NEAR(out_ptr[bi * output_f() + fi], expected[bi][fi], 1) << "at b = " << bi << ", fi = " << fi << ", output_f() = " << output_f();
            }
        }
    }

private:
    VVVVF<InputT> _input;
    VVVVF<WeightsT> _weights;
    VF<WeightsT> _bias;
    format::type _fmt;
};

template <typename InputT, typename WeightsT>
class fc_random_types_test
    : public fully_connected_types_test<InputT, WeightsT>
    , public ::testing::WithParamInterface< fully_connected_types_test_params> {
public:
    void run_random_test() {
        tests::random_generator rg(GET_SUITE_NAME);
        size_t b, in_f, in_x, in_y, out_f;
        format::type in_fmt;

        std::tie(b, in_f, in_x, in_y, out_f, in_fmt) = GetParam();

        quantization_t quant_data;
        quant_data.output_low  = std::numeric_limits<WeightsT>::lowest();
        quant_data.output_high = std::numeric_limits<WeightsT>::max();

        int min = -10;
        int max = 10;

        if (!std::numeric_limits<WeightsT>::is_signed) {
            min = 0;
            max = 20;
        }

        VVVVF<InputT> input_data = rg.template generate_random_4d<InputT>(b, in_f, in_y, in_x, 0, 127);
        VVVVF<WeightsT> weights_data = rg.template generate_random_4d<WeightsT>(out_f, in_f, in_y, in_x, min, max);
        VF<WeightsT> bias_data = rg.template generate_random_1d<WeightsT>(out_f, min, max);

        this->set_input(input_data);
        this->set_weights(weights_data);
        this->set_bias(bias_data);
        this->set_input_format(in_fmt);

        //this->run_test(ref_fully_connected<WeightsT, float,  InputT, WeightsT>(input_data, weights_data, bias_data, quant_data));
        this->run_test(fully_connected_types_reference<InputT, WeightsT>(input_data, weights_data, bias_data, quant_data));
    }
};

using fully_connected_types_i8_i8_test = fc_random_types_test<int8_t, int8_t>;
using fully_connected_types_i8_u8_test = fc_random_types_test<int8_t, uint8_t>;
using fully_connected_types_i8_f32_test = fc_random_types_test<int8_t, float>;

using fully_connected_types_u8_i8_test = fc_random_types_test<uint8_t, int8_t>;
using fully_connected_types_u8_u8_test = fc_random_types_test<uint8_t, uint8_t>;
using fully_connected_types_u8_f32_test = fc_random_types_test<uint8_t, float>;

TEST_P(fully_connected_types_i8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_i8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_i8_f32_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_u8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_u8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_u8_f32_test, random) {
    run_random_test();
}

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_i8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_i8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_i8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_i8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_i8_f32_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_u8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_u8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_u8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_u8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_u8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_u8_f32_test::PrintToStringParamName
);

TEST_F(fully_connected_gpu_tests, weights_reorder_shapes_update) {
    this->test_weights_reorder_shapes_update(false);
}

TEST_F(fully_connected_gpu_tests, weights_reorder_shapes_update_cached) {
    this->test_weights_reorder_shapes_update(true);
}

TEST(fully_connected_gpu, cm) {
    int min_random = -2, max_random = 2;
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);

    if (!cldnn::check_cm_jit_support(engine, config)) {
        GTEST_SKIP();
    }

    // Test parameters
    const int batch_num = 2;
    const int output_f = 4;
    const int input_x = 1;
    const int input_y = 1;
    const int input_f = 3;

    // Allocate memory
    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { batch_num, input_f, input_y, input_x } });
    auto weights_prim = engine.allocate_memory({ data_types::f16, format::oiyx, { output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, output_f, 1 } });

    // Generate random input data and set values
    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, input_f, input_y, input_x, min_random, max_random);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_f, input_f, input_y, input_x, min_random, max_random);
    auto bias_data = rg.generate_random_1d<ov::float16>(output_f, min_random, max_random);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    set_values(input_prim, input_data_bfyx);
    set_values(weights_prim, weights_data_bfyx);
    set_values(bias_prim, bias_data);
    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );
    ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "", impl_types::cm };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
    network network(engine, topology, config);
    network.set_input_data("input", input_prim);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    // Do not validate output for CM
}
