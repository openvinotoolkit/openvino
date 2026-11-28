// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/runtime/utils.hpp>

#include "fully_connected_inst.h"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "network_test.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/runtime/core.hpp"
#include "primitive_inst_test_helper.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {
void test_runtime_weights_lru_buffer(bool large_buffer, bool is_caching_test) {
    auto& engine = get_test_engine();
    const layout input_layout_value{ov::PartialShape{1, 32}, data_types::f32, format::bfyx};
    const layout weights_layout{ov::PartialShape{64, 32}, data_types::f32, format::bfyx};
    auto input = engine.allocate_memory(input_layout_value);
    auto weights = engine.allocate_memory(weights_layout);
    set_values(input, std::vector<float>(32, 1.f));
    set_values(weights, std::vector<float>(64 * 32, 1.f));
    topology topology{input_layout("input", {ov::PartialShape{ov::Dimension(1, 10), 32}, data_types::f32, format::bfyx}),
                      input_layout("weights", weights_layout),
                      fully_connected("fc", input_info("input"), "weights")};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(
        ov::intel_gpu::allow_new_shape_infer(true),
        ov::intel_gpu::use_onednn(false),
        ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", {format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl}}}));
    // An internal program keeps weights reordering in FC instead of inserting a preceding Reorder.
    auto prog = program::build_program(engine, topology, config, true);
    if (is_caching_test) {
        membuf buffer;
        {
            std::ostream output(&buffer);
            BinaryOutputBuffer ob(output);
            ob.set_stream(get_test_stream_ptr().get());
            prog->save(ob);
        }
        {
            std::istream input(&buffer);
            BinaryInputBuffer ib(input, engine);
            prog = std::make_shared<program>(engine, config);
            prog->load(ib);
        }
    }
    auto network = std::make_shared<cldnn::network>(prog, true, true);
    network->set_input_data("input", input);
    network->set_input_data("weights", weights);
    network->execute().at("fc").get_memory();
    auto inst = network->get_primitive("fc");
    ASSERT_TRUE(inst->get_impl()->is_dynamic());
    ASSERT_NE(inst->get_impl()->get_weights_reorder_kernel_params(), nullptr);
    const auto expected_layout = inst->get_impl_params()->weights_layout.value();
    ASSERT_EQ(inst->dep_memory_ptr(1)->buffer_ptr(), weights->buffer_ptr());
    ASSERT_FALSE(weights->get_layout().compatible(expected_layout));
    ASSERT_NE(PrimitiveInstTestHelper::get_cached_weights(inst, expected_layout), nullptr);

    const auto storage_layout = expected_layout.clone_with_other_shape(ov::PartialShape{128, 32});
    auto storage = engine.allocate_memory(storage_layout);
    ASSERT_GE(storage->size(), expected_layout.bytes_count());
    const auto cached_layout = large_buffer ? storage_layout : expected_layout.clone_with_other_shape(ov::PartialShape{1, 1});
    // Keep backing storage large enough to run the old condition safely; the cached view exposes the tested capacity.
    auto lru_memory = engine.reinterpret_buffer(*storage, cached_layout);
    if (large_buffer) {
        ASSERT_GE(lru_memory->size(), expected_layout.bytes_count());
    } else {
        ASSERT_LT(lru_memory->size(), expected_layout.bytes_count());
    }
    // Runtime-only weights give this network a one-entry cache. Evict the reordered layout.
    PrimitiveInstTestHelper::cache_weights(inst, lru_memory);
    ASSERT_EQ(PrimitiveInstTestHelper::get_cached_weights(inst, expected_layout), nullptr);
    set_values(weights, std::vector<float>(64 * 32, 2.f));
    // Check the allocation before execution can switch to an asynchronously compiled implementation.
    PrimitiveInstTestHelper::update_weights(inst);
    auto reordered = PrimitiveInstTestHelper::get_cached_weights(inst, expected_layout);
    ASSERT_NE(reordered, nullptr);
    EXPECT_GE(reordered->size(), expected_layout.bytes_count());
    EXPECT_EQ(reordered->buffer_ptr() == lru_memory->buffer_ptr(), large_buffer);
    auto output = network->execute().at("fc").get_memory();
    mem_lock<float, mem_lock_type::read> result(output, get_test_stream());
    ASSERT_EQ(result.size(), 64u);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 64.f);
    }
}
}  // namespace

TEST(fully_connected_gpu, runtime_weights_lru_undersized_view) {
    test_runtime_weights_lru_buffer(false, false);
}

TEST(fully_connected_gpu, runtime_weights_lru_large_buffer) {
    test_runtime_weights_lru_buffer(true, false);
}

TEST(fully_connected_gpu, runtime_weights_lru_undersized_view_cached) {
    test_runtime_weights_lru_buffer(false, true);
}

TEST(fully_connected_gpu, runtime_weights_lru_large_buffer_cached) {
    test_runtime_weights_lru_buffer(true, true);
}

TEST(fully_connected_gpu, dynamic_byxf) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ov::PartialShape{2, 2, 2}, data_types::f32, format::byxf});
    auto weights = engine.allocate_memory({ov::PartialShape{2, 2}, data_types::f32, format::bfyx});
    set_values(input, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    set_values(weights, {1.f, 1.f, 1.f, -1.f});

    topology topology(input_layout("input", {ov::PartialShape::dynamic(3), data_types::f32, format::byxf}),
                      data("weights", weights),
                      fully_connected("fc", input_info("input"), "weights", "", 3, 2));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    ASSERT_TRUE(network.get_primitive("fc")->get_impl()->is_dynamic());
    auto output = network.execute().at("fc").get_memory();
    ASSERT_EQ(output->get_layout().format, format::byxf);
    mem_lock<float, mem_lock_type::read> result(output, get_test_stream());
    const std::vector<float> expected = {4.f, 6.f, -2.f, -2.f, 12.f, 14.f, -2.f, -2.f};
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(result[i], expected[i]) << i;
}

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
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output_memory, get_test_stream());

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
        GTEST_SKIP() << "float16 combinations are skipped (cl_khr_fp16 is not supported).";
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

    // Use rank-2 layouts for new shape infer
    auto input_prim = engine.allocate_memory({ { input_b, input_f}, data_types::i32, format::bfyx});
    auto weights_prim = engine.allocate_memory({ { weight_b, weight_f} , data_types::i32, format::bfyx});

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
        reorder("reorder_to_i32", input_info("input"), { { input_b, input_f }, data_types::i32, format::bfyx}),
        fully_connected("fc_prim", input_info("reorder_to_i32"), "weights"),
        reorder("reorder_to_float", input_info("fc_prim"), { { input_b, weight_b }, data_types::f32, format::bfyx})
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

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->second.get_layout().batch(), input_b);
    ASSERT_EQ(outputs.begin()->second.get_layout().feature(), input_f);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(1), input_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(0), weight_b);
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

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.begin()->second.get_layout().batch(), input_b);
    ASSERT_EQ(outputs.begin()->second.get_layout().feature(), input_f);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(2), input_z);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(1), input_y);
    ASSERT_EQ(outputs.begin()->second.get_layout().spatial(0), weight_b);
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
        GTEST_SKIP() << "The test is skipped (cl_khr_fp16 is not supported).";
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
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

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
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), batch_num * feature_num * output_y);

    for (size_t i = 0; i < batch_num * feature_num * output_y; ++i) {
        ASSERT_FLOAT_EQ(reference_output[i], output_ptr[i]) << " i = " << i;
    }
}

namespace {

ExecutionConfig make_forced_fc_config(cldnn::engine& engine,
                                      const std::string& impl_name,
                                      bool optimize_data,
                                      bool allow_new_shape_infer = false,
                                      int64_t dyn_quant_gs = -1) {
    ExecutionConfig config = get_test_default_config(engine);
    if (allow_new_shape_infer) {
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    }
    config.set_property(ov::intel_gpu::optimize_data(optimize_data));
    if (dyn_quant_gs >= 0) {
        config.set_user_property(ov::hint::dynamic_quantization_group_size(static_cast<uint64_t>(dyn_quant_gs)));
    }
    ov::intel_gpu::ImplementationDesc impl_desc = {format::bfyx, impl_name, impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", impl_desc}}));
    return config;
}

bool kernel_source_has_all_defines(const primitive_inst& inst, const std::vector<std::string>& defines) {
    const auto& node = inst.get_node();
    auto selected_impl = node.type()->create_impl(node);
    for (const auto& s : selected_impl->get_kernels_source()) {
        if (s == nullptr) {
            continue;
        }
        const auto src = s->get_str();
        bool all_found = true;
        for (const auto& def : defines) {
            if (src.find(def) == std::string::npos) {
                all_found = false;
                break;
            }
        }
        if (all_found) {
            return true;
        }
    }
    return false;
}

template <typename T>
void compare_fc_memory_near(cldnn::memory::ptr test_out, cldnn::memory::ptr ref_out, size_t expected_count, float tol, const std::string& ctx = "") {
    ASSERT_EQ(test_out->count(), expected_count) << ctx;
    cldnn::mem_lock<T, mem_lock_type::read> out_ptr(test_out, get_test_stream());
    cldnn::mem_lock<T, mem_lock_type::read> ref_ptr(ref_out, get_test_stream());
    for (size_t i = 0; i < expected_count; ++i) {
        ASSERT_NEAR(static_cast<float>(ref_ptr[i]), static_cast<float>(out_ptr[i]), tol) << ctx << " i=" << i;
    }
}

void fill_random_fc_tensor(cldnn::memory::ptr mem, tests::random_generator& rg, data_types dt, size_t count, float lo, float hi) {
    if (dt == data_types::f16) {
        set_values(mem, rg.generate_random_1d<ov::float16>(count, lo, hi));
    } else {
        set_values(mem, rg.generate_random_1d<float>(count, lo, hi));
    }
}

}  // namespace

TEST(fully_connected_gpu, bf_tiled_3d_crop_feature_pad_cross_batch) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "cl_khr_fp16 is not supported.";
    }

    // Reproduces 3D FC [B=2, F=2, K=64] preceded by an in-place feature-axis crop
    // (offset=[0, 3, 0, 0] from [B=2, F_total=5, K=64, 1]) where TILE_B=4 > OUTPUT_FEATURE_NUM=2
    // wraps across batch b=0 -> b=1 inside a single work-item.
    const int batch_num = 2;
    const int in_features_total = 5;
    const int feature_offset = 3;
    const int feature_num = 2;
    const int input_x = 1;
    const int input_y = 64;
    const int output_y = 64;

    auto input_mem = engine.allocate_memory({{batch_num, in_features_total, input_y, input_x}, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{output_y, input_y}, data_types::f16, format::bfyx});
    auto eltw_mem = engine.allocate_memory({{1, in_features_total, 1, 1}, data_types::f16, format::bfyx});

    auto full_input_data = rg.generate_random_4d<ov::float16>(batch_num, in_features_total, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_y, input_y, 1, 1, -1, 1);
    auto full_input_bfyx = flatten_4d(format::bfyx, full_input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    // Put a large 100.f sentinel in cropped-out features f=0..2 so reading b=1,f=0..1 instead of b=1,f=3..4 is caught.
    std::vector<ov::float16> eltwise_input(in_features_total, ov::float16(0.f));
    for (int f = 0; f < feature_offset; ++f) {
        eltwise_input[f] = ov::float16(100.f);
    }

    std::vector<ov::float16> cropped_input_bfyx;
    cropped_input_bfyx.reserve(batch_num * feature_num * input_y);
    for (int b = 0; b < batch_num; ++b) {
        for (int f = feature_offset; f < feature_offset + feature_num; ++f) {
            for (int y = 0; y < input_y; ++y) {
                cropped_input_bfyx.push_back(full_input_data[b][f][y][0]);
            }
        }
    }
    std::vector<ov::float16> empty_bias(output_y, ov::float16(0.f));
    auto reference_output = dynamic_fully_connected_reference_calc<ov::float16, ov::float16, ov::float16, ov::float16, float>(batch_num * feature_num,
                                                                                                                              input_y,
                                                                                                                              output_y,
                                                                                                                              cropped_input_bfyx,
                                                                                                                              weights_data_bfyx,
                                                                                                                              empty_bias);

    set_values(input_mem, full_input_bfyx);
    set_values(weights_mem, weights_data_bfyx);
    set_values(eltw_mem, eltwise_input);

    topology topology(input_layout("input", input_mem->get_layout()),
                      data("weights", weights_mem),
                      data("eltw_input", eltw_mem),
                      eltwise("eltw", {input_info("input"), input_info("eltw_input")}, eltwise_mode::sum),
                      crop("crop", input_info("eltw"), tensor{batch_num, feature_num, input_x, input_y}, tensor{0, feature_offset, 0, 0}),
                      fully_connected("fc_prim", input_info("crop"), "weights", "", 3, 3));

    network network(engine, topology, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true));
    ASSERT_TRUE(network.get_primitive("crop")->can_be_optimized());
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto fc_inst = network.get_primitive("fc_prim");
    ASSERT_NE(fc_inst, nullptr);
    const auto kernels = fc_inst->get_impl()->get_kernels_dump_info(*fc_inst->get_impl_params()).get_entries();
    ASSERT_NE(kernels.find("fully_connected_gpu_bf_tiled"), std::string::npos) << "Executed kernels: " << kernels;

    auto output_mem = outputs.at("fc_prim").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), static_cast<size_t>(batch_num * feature_num * output_y));

    for (size_t i = 0; i < reference_output.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(reference_output[i]), static_cast<float>(output_ptr[i]), 1e-2f) << " i = " << i;
    }
}

TEST(fully_connected_gpu, bf_tiled_fp16_large_k_accumulation_accuracy) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "cl_khr_fp16 is not supported.";
    }

    // Reproduces FP16 large-K running-sum stagnation/drift in fully_connected_gpu_bf_tiled.
    // First 16 products sum to +/-64.0 (where FP16 1 ULP = 0.0625); each subsequent 16-element tile sums to
    // +/-0.030 (< 0.5 ULP in FP16), which stagnates completely if accumulated in half instead of float.
    // Exercises batch=1 (TILE_B=1), prime multi-batch=17 (TILE_B=8, dispatch_bsv=3, BATCH_LEFTOVER),
    // and K=3074 (with K-leftover).
    const int feature_num = 1;
    const int input_x = 1;
    const int output_y = 64;

    struct AccCase {
        int batch_num;
        int input_y;
    };
    for (const auto& tc : std::vector<AccCase>{{1, 3072}, {17, 3072}, {17, 3074}}) {
        const int batch_num = tc.batch_num;
        const int input_y = tc.input_y;

        auto weights_mem = engine.allocate_memory({{output_y, input_y}, data_types::f16, format::bfyx});
        std::vector<ov::float16> weights_data(static_cast<size_t>(output_y) * input_y);
        for (int ofm = 0; ofm < output_y; ++ofm) {
            const float sign = (ofm % 2 == 0) ? -1.0f : 1.0f;
            for (int k = 0; k < input_y; ++k) {
                weights_data[static_cast<size_t>(ofm) * input_y + k] = (k < 16) ? ov::float16(sign * 1.0f) : ov::float16(sign * 0.015f);
            }
        }
        set_values(weights_mem, weights_data);

        auto input_mem = engine.allocate_memory({{batch_num, feature_num, input_y, input_x}, data_types::f16, format::bfyx});
        std::vector<ov::float16> input_data(static_cast<size_t>(batch_num) * input_y);
        for (int b = 0; b < batch_num; ++b) {
            const float b_sign = (b % 2 == 0) ? 1.0f : -1.0f;
            for (int k = 0; k < input_y; ++k) {
                input_data[static_cast<size_t>(b) * input_y + k] = (k < 16) ? ov::float16(b_sign * 4.0f) : ov::float16(b_sign * 0.125f);
            }
        }
        set_values(input_mem, input_data);

        std::vector<float> expected_f32(static_cast<size_t>(batch_num) * output_y, 0.0f);
        for (int b = 0; b < batch_num; ++b) {
            for (int ofm = 0; ofm < output_y; ++ofm) {
                float acc = 0.0f;
                for (int k = 0; k < input_y; ++k) {
                    acc += static_cast<float>(input_data[static_cast<size_t>(b) * input_y + k]) *
                           static_cast<float>(weights_data[static_cast<size_t>(ofm) * input_y + k]);
                }
                expected_f32[static_cast<size_t>(b) * output_y + ofm] = acc;
            }
        }

        topology topology(input_layout("input", input_mem->get_layout()),
                          data("weights", weights_mem),
                          fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3));

        network network(engine, topology, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true));
        network.set_input_data("input", input_mem);
        auto outputs = network.execute();

        auto fc_inst = network.get_primitive("fc_prim");
        ASSERT_NE(fc_inst, nullptr);
        const auto kernels = fc_inst->get_impl()->get_kernels_dump_info(*fc_inst->get_impl_params()).get_entries();
        ASSERT_NE(kernels.find("fully_connected_gpu_bf_tiled"), std::string::npos) << "Executed kernels: " << kernels;

        auto output_mem = outputs.at("fc_prim").get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
        ASSERT_EQ(output_mem->count(), static_cast<size_t>(batch_num) * output_y);

        for (int b = 0; b < batch_num; ++b) {
            for (int ofm = 0; ofm < output_y; ++ofm) {
                const size_t idx = static_cast<size_t>(b) * output_y + ofm;
                const float ref_f16 = static_cast<float>(ov::float16(expected_f32[idx]));
                ASSERT_NEAR(ref_f16, static_cast<float>(output_ptr[idx]), 0.0625f)
                    << "batch_num=" << batch_num << " K=" << input_y << " b=" << b << " ofm=" << ofm;
            }
        }
    }

    // Verify each 16-element K-tile is promoted to float individually rather than summed across
    // multiple K-tiles in half: with rows [-1024 x 32, +2048 x 32] and unit weights, each 16-element
    // tile is finite (+/-16384 or +32768), whereas merging two tiles in half hits +65536 -> +Inf.
    {
        const int batch_num = 8;
        const int input_y = 64;
        auto weights_mem = engine.allocate_memory({{output_y, input_y}, data_types::f16, format::bfyx});
        set_values(weights_mem, std::vector<ov::float16>(static_cast<size_t>(output_y) * input_y, ov::float16(1.0f)));

        auto input_mem = engine.allocate_memory({{batch_num, feature_num, input_y, input_x}, data_types::f16, format::bfyx});
        std::vector<ov::float16> input_data(static_cast<size_t>(batch_num) * input_y);
        for (int b = 0; b < batch_num; ++b) {
            for (int k = 0; k < input_y; ++k) {
                input_data[static_cast<size_t>(b) * input_y + k] = (k < 32) ? ov::float16(-1024.0f) : ov::float16(2048.0f);
            }
        }
        set_values(input_mem, input_data);

        topology topo(input_layout("input", input_mem->get_layout()),
                      data("weights", weights_mem),
                      fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3));
        network net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true));
        net.set_input_data("input", input_mem);
        auto out_mem = net.execute().at("fc_prim").get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
        ASSERT_EQ(out_mem->count(), static_cast<size_t>(batch_num) * output_y);
        for (size_t i = 0; i < out_mem->count(); ++i) {
            const float val = static_cast<float>(out_ptr[i]);
            ASSERT_TRUE(std::isfinite(val)) << "Overflowed to non-finite value at index " << i;
            ASSERT_NEAR(32768.0f, val, 1.0f) << "index=" << i;
        }
    }
}

TEST(fully_connected_gpu, bf_tiled_fp32_tile_partial_sum_cancellation) {
    auto& engine = get_test_engine();

    // Preserving per-tile partial sums (acc_tmp) avoids FP32 cancellation when large and small tiles mix:
    // tile 0 sums to +2^24 (where FP32 1 ULP = 2.0f, so element-by-element +1.0f would round away),
    // tiles 1 and 2 each sum to +16.0f (exact when added to 2^24), and tile 3 cancels -2^24 -> 32.0f.
    const int input_y = 64;
    const int output_y = 32;

    auto input_mem = engine.allocate_memory({{1, 1, input_y, 1}, data_types::f32, format::bfyx});
    auto weights_mem = engine.allocate_memory({{output_y, input_y}, data_types::f32, format::bfyx});

    std::vector<float> input_data(input_y, 1.0f);
    for (int k = 0; k < 16; ++k) {
        input_data[k] = 1048576.0f;
        input_data[48 + k] = -1048576.0f;
    }
    std::vector<float> weights_data(output_y * input_y, 1.0f);
    set_values(input_mem, input_data);
    set_values(weights_mem, weights_data);

    topology topology(input_layout("input", input_mem->get_layout()),
                      data("weights", weights_mem),
                      fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3));

    network network(engine, topology, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true));
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto fc_inst = network.get_primitive("fc_prim");
    ASSERT_NE(fc_inst, nullptr);
    const auto kernels = fc_inst->get_impl()->get_kernels_dump_info(*fc_inst->get_impl_params()).get_entries();
    ASSERT_NE(kernels.find("fully_connected_gpu_bf_tiled"), std::string::npos) << "Executed kernels: " << kernels;

    auto output_mem = outputs.at("fc_prim").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), static_cast<size_t>(output_y));
    for (int ofm = 0; ofm < output_y; ++ofm) {
        ASSERT_FLOAT_EQ(32.0f, output_ptr[ofm]) << " ofm = " << ofm;
    }
}

TEST(fully_connected_gpu, bf_tiled_dynamic_int4_3d_crop_forced_tile_and_dyn_b) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "cl_khr_fp16 is not supported.";
    }

    // Exercises shape-agnostic INT4 3D FC after an in-place feature-axis variadic_split crop
    // (has_dynamic_pad() == true at compile time, offset=[0, 3, 0, 0] at runtime, F_total=5 -> F=2).
    // - Square weights (IFM=64, OFM=64) with B=2 (flattened batch=4) run fc_bf_tiled_kernel_forced_tile_b4
    //   (include/fully_connected_gpu_bf_tiled_common.cl), and B=256 (flattened batch=512 > 256) verifies
    //   dynamic-pad rejection prevents tile-8/tile-16 SLM dispatch mismatch.
    // - Imbalanced weights (IFM=64, OFM=256) with B=2 (flattened batch=4) run fully_connected_gpu_bf_tiled_dyn_b
    //   (include/fully_connected_gpu_bf_tiled_dyn_b_core.cl).
    const int in_features_total = 5;
    const int feature_offset = 3;
    const int feature_num = 2;
    const int ifm_num = 64;
    const int scales_group_size = 32;

    auto axis_mem = engine.allocate_memory({ov::PartialShape{1}, data_types::i64, format::bfyx});
    auto splits_mem = engine.allocate_memory({ov::PartialShape{2}, data_types::i64, format::bfyx});
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_mem, {feature_offset, feature_num});

    const bool is_igpu = engine.get_device_info().dev_type == device_type::integrated_gpu;
    // On integrated GPUs, bf_tiled attaches dyn_b as a sub-kernel alongside its iGPU-only SLM kernel.
    // Test standalone fully_connected_gpu_bf_tiled_dyn_b on all GPUs (including dGPUs such as DG2)
    // and additionally verify bf_tiled's integrated dyn_b dispatch on integrated GPUs.
    struct CaseCfg {
        int ofm_num;
        std::string forced_impl;
        bool expect_dyn_b;
    };
    std::vector<CaseCfg> cases = {
        {64, "fully_connected_gpu_bf_tiled", false},
        {256, "fully_connected_gpu_bf_tiled_dyn_b", true},
    };
    if (is_igpu) {
        cases.push_back({256, "fully_connected_gpu_bf_tiled", true});
    }

    for (const auto& tc : cases) {
        const int ofm_num = tc.ofm_num;
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u4, format::bfyx});
        auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        set_values(weights_mem, rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10));
        set_values(scale_mem, rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -0.5f, 0.5f));

        const layout dyn_in_layout(ov::PartialShape{-1, in_features_total, ifm_num}, data_types::f16, format::bfyx);
        auto fc_prim = fully_connected("fc_prim", input_info("crop"), "weights", "", "scale", "", data_types::f16, 3, 2);
        fc_prim.decompression_zero_point_scalar = 0;

        topology test_topo(
            input_layout("input", dyn_in_layout),
            data("axis", axis_mem),
            data("splits", splits_mem),
            data("weights", weights_mem),
            data("scale", scale_mem),
            crop("crop", {input_info("input"), input_info("axis"), input_info("splits")}, tensor(1), tensor(0), crop_ngraph_op_mode::variadic_split, 1, 1),
            fc_prim);

        network test_net(engine, test_topo, make_forced_fc_config(engine, tc.forced_impl, true, true, 32));
        network ref_net(engine, test_topo, make_forced_fc_config(engine, "fully_connected_gpu_bfyx_ref", false, true));

        ASSERT_TRUE(test_net.get_primitive("crop")->can_be_optimized());

        for (int batch_num : {2, 256}) {
            if (ofm_num == 256 && batch_num == 256) {
                continue;
            }
            const layout actual_in_layout(ov::PartialShape{batch_num, in_features_total, ifm_num}, data_types::f16, format::bfyx);
            auto input_mem = engine.allocate_memory(actual_in_layout);
            std::vector<ov::float16> input_data(batch_num * in_features_total * ifm_num);
            auto rand_vals = rg.generate_random_1d<ov::float16>(batch_num * feature_num * ifm_num, -1.0f, 1.0f);
            size_t r_idx = 0;
            for (int b = 0; b < batch_num; ++b) {
                for (int f = 0; f < in_features_total; ++f) {
                    for (int k = 0; k < ifm_num; ++k) {
                        const size_t idx = (static_cast<size_t>(b) * in_features_total + f) * ifm_num + k;
                        input_data[idx] = (f < feature_offset) ? ov::float16(100.0f) : rand_vals[r_idx++];
                    }
                }
            }
            set_values(input_mem, input_data);

            test_net.set_input_data("input", input_mem);
            ref_net.set_input_data("input", input_mem);
            auto test_out = test_net.execute().at("fc_prim").get_memory();
            auto ref_out = ref_net.execute().at("fc_prim").get_memory();

            auto fc_inst = test_net.get_primitive("fc_prim");
            EXPECT_EQ(fc_inst->get_intermediates_memories().size(), 0U) << "Dynamic quantize must be disabled on dynamically padded input";
            const auto kernels = fc_inst->get_impl()->get_kernels_dump_info(*fc_inst->get_impl_params()).get_entries();
            if (tc.expect_dyn_b && batch_num == 2) {
                EXPECT_NE(kernels.find("fully_connected_gpu_bf_tiled_dyn_b"), std::string::npos) << "Expected dyn_b kernel, got: " << kernels;
            } else {
                EXPECT_NE(kernels.find("fully_connected_gpu_bf_tiled"), std::string::npos) << "Expected bf_tiled kernel, got: " << kernels;
            }

            compare_fc_memory_near<ov::float16>(test_out,
                                                ref_out,
                                                static_cast<size_t>(batch_num) * feature_num * ofm_num,
                                                5e-2f,
                                                "ofm_num=" + std::to_string(ofm_num) + " batch_num=" + std::to_string(batch_num));
        }
    }
}

TEST(fully_connected_gpu, bf_tiled_zero_offset_upper_feature_pad_disables_dyn_quant) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16 || engine.get_device_info().dev_type != device_type::integrated_gpu) {
        GTEST_SKIP() << "Requires integrated GPU with cl_khr_fp16 support.";
    }

    // Zero-offset crop (offset=[0, 0, 0, 0], F_total=5 -> F=2) leaves GetFirstElementOffset() == 0 while
    // INPUT0_BATCH_PITCH != OUTPUT_FEATURE_NUM * INPUT0_FEATURE_PITCH (PitchesDifferFromLogicalDims() == true).
    // With flattened batch=512 > 256 and u4 weights, quantize_input must be omitted for the padded case
    // while enabled for the contiguous control.
    const int batch_num = 256;
    const int in_features_total = 5;
    const int feature_num = 2;
    const int ifm_num = 64;
    const int ofm_num = 64;
    const int scales_group_size = 32;

    auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u4, format::bfyx});
    auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
    set_values(weights_mem, rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10));
    set_values(scale_mem, rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -0.5f, 0.5f));

    auto full_input_mem = engine.allocate_memory({{batch_num, in_features_total, ifm_num, 1}, data_types::f16, format::bfyx});
    auto contig_input_mem = engine.allocate_memory({{batch_num, feature_num, ifm_num, 1}, data_types::f16, format::bfyx});
    auto eltw_mem = engine.allocate_memory({{1, in_features_total, 1, 1}, data_types::f16, format::bfyx});

    auto active_vals = rg.generate_random_1d<ov::float16>(batch_num * feature_num * ifm_num, -1.0f, 1.0f);
    std::vector<ov::float16> full_input_data(batch_num * in_features_total * ifm_num);
    size_t r_idx = 0;
    for (int b = 0; b < batch_num; ++b) {
        for (int f = 0; f < in_features_total; ++f) {
            for (int k = 0; k < ifm_num; ++k) {
                const size_t idx = (static_cast<size_t>(b) * in_features_total + f) * ifm_num + k;
                full_input_data[idx] = (f < feature_num) ? active_vals[r_idx++] : ov::float16(100.0f);
            }
        }
    }
    set_values(full_input_mem, full_input_data);
    set_values(contig_input_mem, active_vals);
    set_values(eltw_mem, std::vector<ov::float16>(in_features_total, ov::float16(0.0f)));

    auto fc_padded = fully_connected("fc_prim", input_info("crop"), "weights", "", "scale", "", data_types::f16, 3, 2);
    fc_padded.decompression_zero_point_scalar = 0;
    topology padded_topo(input_layout("input", full_input_mem->get_layout()),
                         data("eltw_in", eltw_mem),
                         data("weights", weights_mem),
                         data("scale", scale_mem),
                         eltwise("eltw", {input_info("input"), input_info("eltw_in")}, eltwise_mode::sum),
                         crop("crop", input_info("eltw"), tensor{batch_num, feature_num, 1, ifm_num}, tensor{0, 0, 0, 0}),
                         fc_padded);

    auto fc_contig = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);
    fc_contig.decompression_zero_point_scalar = 0;
    topology contig_topo(input_layout("input", contig_input_mem->get_layout()), data("weights", weights_mem), data("scale", scale_mem), fc_contig);

    auto config = make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true, false, 32);
    auto ref_config = make_forced_fc_config(engine, "fully_connected_gpu_bfyx_ref", false);

    network padded_net(engine, padded_topo, config);
    network contig_net(engine, contig_topo, config);
    network ref_net(engine, contig_topo, ref_config);

    ASSERT_TRUE(padded_net.get_primitive("crop")->can_be_optimized());

    padded_net.set_input_data("input", full_input_mem);
    contig_net.set_input_data("input", contig_input_mem);
    ref_net.set_input_data("input", contig_input_mem);

    auto padded_out = padded_net.execute().at("fc_prim").get_memory();
    contig_net.execute();
    auto ref_out = ref_net.execute().at("fc_prim").get_memory();

    auto padded_fc_inst = padded_net.get_primitive("fc_prim");
    auto contig_fc_inst = contig_net.get_primitive("fc_prim");
    EXPECT_EQ(padded_fc_inst->get_impl()->get_kernels().size(), 1U) << "Padded static input must omit quantize_input kernel";
    EXPECT_EQ(padded_fc_inst->get_intermediates_memories().size(), 0U) << "Padded input must not allocate dynamic quantize buffers";
    EXPECT_EQ(contig_fc_inst->get_impl()->get_kernels().size(), 2U) << "Contiguous static control must include quantize_input + FC kernels";
    EXPECT_EQ(contig_fc_inst->get_intermediates_memories().size(), 2U) << "Contiguous control must allocate dynamic quantize buffers";

    compare_fc_memory_near<ov::float16>(padded_out, ref_out, static_cast<size_t>(batch_num) * feature_num * ofm_num, 5e-2f);
}

TEST(fully_connected_gpu, bf_tiled_dynamic_k_static_upper_feature_pad_disables_dyn_quant) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "cl_khr_fp16 is not supported.";
    }

    // Dynamic K ([-1, 2, -1]) with static upper feature padding (pad.after = 3, pad.before = 0,
    // pad.is_dynamic = false) zeroes Feature/Batch pitches at compile time so PitchesDifferFromLogicalDims()
    // is false while d.pad.Total() != 0. should_dynamic_quantize() must still reject it at compile time
    // to prevent a compile-time TILE_B=8 vs runtime TILE_B=16 SLM dispatch mismatch at B=256 (512 rows).
    const int batch_num = 256;
    const int feature_num = 2;
    const int feature_pad_after = 3;
    const int in_features_total = feature_num + feature_pad_after;
    const int ifm_num = 64;
    const int ofm_num = 64;
    const int scales_group_size = 32;

    auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u4, format::bfyx});
    auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
    set_values(weights_mem, rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10));
    set_values(scale_mem, rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -0.5f, 0.5f));

    const padding static_upper_f_pad({0, 0, 0, 0}, {0, feature_pad_after, 0, 0});
    const layout dyn_in_layout(ov::PartialShape{-1, feature_num, -1}, data_types::f16, format::bfyx, static_upper_f_pad);
    const layout actual_in_layout(ov::PartialShape{batch_num, feature_num, ifm_num}, data_types::f16, format::bfyx, static_upper_f_pad);

    auto input_mem = engine.allocate_memory(actual_in_layout);
    auto active_vals = rg.generate_random_1d<ov::float16>(static_cast<size_t>(batch_num) * feature_num * ifm_num, -1.0f, 1.0f);
    std::vector<ov::float16> padded_input_data(static_cast<size_t>(batch_num) * in_features_total * ifm_num);
    size_t r_idx = 0;
    for (int b = 0; b < batch_num; ++b) {
        for (int f = 0; f < in_features_total; ++f) {
            for (int k = 0; k < ifm_num; ++k) {
                const size_t idx = (static_cast<size_t>(b) * in_features_total + f) * ifm_num + k;
                padded_input_data[idx] = (f < feature_num) ? active_vals[r_idx++] : ov::float16(100.0f);
            }
        }
    }
    set_values(input_mem, padded_input_data);

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);
    fc_prim.decompression_zero_point_scalar = 0;
    topology topo(input_layout("input", dyn_in_layout), data("weights", weights_mem), data("scale", scale_mem), fc_prim);

    network test_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true, true, 32));
    network ref_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bfyx_ref", false, true));

    test_net.set_input_data("input", input_mem);
    ref_net.set_input_data("input", input_mem);
    auto test_out = test_net.execute().at("fc_prim").get_memory();
    auto ref_out = ref_net.execute().at("fc_prim").get_memory();

    EXPECT_EQ(test_net.get_primitive("fc_prim")->get_intermediates_memories().size(), 0U)
        << "Dynamic quantize must be disabled for dynamic-K statically padded input";
    compare_fc_memory_near<ov::float16>(test_out, ref_out, static_cast<size_t>(batch_num) * feature_num * ofm_num, 5e-2f);
}

TEST(fully_connected_gpu, bf_tiled_dyn_b_non_post_op_scale_group_no_tile_overflow) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "cl_khr_fp16 is not supported.";
    }

    // With scale_group_size=8 (not divisible by SIMD=16), DECOMPRESSION_SCALE_POST_OP is disabled in dyn_b.
    // Input [-2048 x 16, +4096 x 16] with unit weights/scales has tile 1 partial sum +65536 (> FP16 max 65504),
    // which overflows to +Inf if accumulated into half acc_tmp instead of directly into float acc (+32768).
    const int batch_num = 2;
    const int ifm_num = 32;
    const int ofm_num = 256;
    const int scales_group_size = 8;

    auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u4, format::bfyx});
    auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
    set_values(weights_mem, std::vector<uint8_t>(static_cast<size_t>(ofm_num) * ifm_num / 2, 0x11));
    set_values(scale_mem, std::vector<ov::float16>(static_cast<size_t>(ofm_num) * (ifm_num / scales_group_size), ov::float16(1.0f)));

    const layout dyn_in_layout(ov::PartialShape{-1, ifm_num}, data_types::f16, format::bfyx);
    auto input_mem = engine.allocate_memory(layout(ov::PartialShape{batch_num, ifm_num}, data_types::f16, format::bfyx));
    std::vector<ov::float16> input_data(static_cast<size_t>(batch_num) * ifm_num);
    for (int b = 0; b < batch_num; ++b) {
        for (int k = 0; k < ifm_num; ++k) {
            input_data[static_cast<size_t>(b) * ifm_num + k] = (k < 16) ? ov::float16(-2048.0f) : ov::float16(4096.0f);
        }
    }
    set_values(input_mem, input_data);

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 2, 2);
    fc_prim.decompression_zero_point_scalar = 0;
    topology topo(input_layout("input", dyn_in_layout), data("weights", weights_mem), data("scale", scale_mem), fc_prim);

    network test_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled_dyn_b", true, true, 0));
    test_net.set_input_data("input", input_mem);
    auto test_out = test_net.execute().at("fc_prim").get_memory();

    auto fc_inst = test_net.get_primitive("fc_prim");
    const auto kernels = fc_inst->get_impl()->get_kernels_dump_info(*fc_inst->get_impl_params()).get_entries();
    ASSERT_NE(kernels.find("fully_connected_gpu_bf_tiled_dyn_b"), std::string::npos) << "Executed kernels: " << kernels;

    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(test_out, get_test_stream());
    const size_t expected_count = static_cast<size_t>(batch_num) * ofm_num;
    ASSERT_EQ(test_out->count(), expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        ASSERT_FLOAT_EQ(32768.0f, static_cast<float>(out_ptr[i])) << " i = " << i;
    }
}

TEST(fully_connected_gpu, bf_tiled_unaligned_batch_padded_dispatch_bsv) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int ifm_num = 64;
    const int ofm_num = 64;

    for (int batch_num : {688, 690}) {
        for (data_types dt : {data_types::f16, data_types::f32}) {
            if (dt == data_types::f16 && !engine.get_device_info().supports_fp16) {
                continue;
            }
            const float tol = (dt == data_types::f16) ? 5e-2f : 1e-3f;
            auto input_mem = engine.allocate_memory({{1, batch_num, ifm_num, 1}, dt, format::bfyx});
            auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, dt, format::bfyx});
            fill_random_fc_tensor(input_mem, rg, dt, static_cast<size_t>(batch_num) * ifm_num, -1.0f, 1.0f);
            fill_random_fc_tensor(weights_mem, rg, dt, static_cast<size_t>(ofm_num) * ifm_num, -0.5f, 0.5f);

            topology topo(input_layout("input", input_mem->get_layout()),
                          data("weights", weights_mem),
                          fully_connected("fc_prim", input_info("input"), "weights", "", 3, 2));

            network test_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true));
            network ref_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bfyx_ref", false));

            test_net.set_input_data("input", input_mem);
            ref_net.set_input_data("input", input_mem);
            auto test_out = test_net.execute().at("fc_prim").get_memory();
            auto ref_out = ref_net.execute().at("fc_prim").get_memory();

            const std::string expected_bl = (batch_num % 8 == 0) ? "#define BATCH_LEFTOVER 0" : "#define BATCH_LEFTOVER 1";
            EXPECT_TRUE(kernel_source_has_all_defines(
                *test_net.get_primitive("fc_prim"),
                {"#define TILE_B 8", "#define DISPATCH_BSV 16", "#define DISPATCH_FSV 2", "#define DISPATCH_BATCH_LEFTOVER 1", expected_bl}))
                << "dt=" << dt << " batch=" << batch_num;

            const size_t expected_count = static_cast<size_t>(batch_num) * ofm_num;
            const std::string ctx = "batch=" + std::to_string(batch_num);
            if (dt == data_types::f16) {
                compare_fc_memory_near<ov::float16>(test_out, ref_out, expected_count, tol, "f16 " + ctx);
            } else {
                compare_fc_memory_near<float>(test_out, ref_out, expected_count, tol, "f32 " + ctx);
            }
        }
    }
}

TEST(fully_connected_gpu, bf_tiled_unaligned_batch_fused_binary_add_oob_safety) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "cl_khr_fp16 is not supported.";
    }

    const int ifm_num = 64;
    const int ofm_num = 64;

    // Test both dynamic batch (fuses binary Add into bf_tiled on all GPUs including non-XMX,
    // exercising IS_DYNAMIC + HAS_FUSED_OPS partial-tile bounds guard) and static batch
    // (exercises padded DISPATCH_BSV + BATCH_LEFTOVER, and fuses binary Add on XMX/Xe2+ GPUs).
    for (bool is_dynamic : {true, false}) {
        for (int batch_num : {17, 690}) {
            auto input_mem = engine.allocate_memory({ov::PartialShape{1, batch_num, ifm_num}, data_types::f16, format::bfyx});
            auto residual_mem = engine.allocate_memory({ov::PartialShape{1, batch_num, ofm_num}, data_types::f16, format::bfyx});
            auto weights_mem = engine.allocate_memory({ov::PartialShape{ofm_num, ifm_num}, data_types::f16, format::bfyx});

            fill_random_fc_tensor(input_mem, rg, data_types::f16, static_cast<size_t>(batch_num) * ifm_num, -1.0f, 1.0f);
            fill_random_fc_tensor(residual_mem, rg, data_types::f16, static_cast<size_t>(batch_num) * ofm_num, -2.0f, 2.0f);
            fill_random_fc_tensor(weights_mem, rg, data_types::f16, static_cast<size_t>(ofm_num) * ifm_num, -0.5f, 0.5f);

            const layout in_layout = is_dynamic ? layout{ov::PartialShape{1, -1, ifm_num}, data_types::f16, format::bfyx} : input_mem->get_layout();
            const layout res_layout = is_dynamic ? layout{ov::PartialShape{1, -1, ofm_num}, data_types::f16, format::bfyx} : residual_mem->get_layout();

            topology topo(input_layout("input", in_layout),
                          input_layout("residual", res_layout),
                          data("weights", weights_mem),
                          fully_connected("fc_prim", input_info("input"), "weights", "", 3, 2),
                          eltwise("add_prim", {input_info("fc_prim"), input_info("residual")}, eltwise_mode::sum),
                          reorder("reorder_out", input_info("add_prim"), format::bfyx, data_types::f32));

            network test_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true, true));
            network ref_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bfyx_ref", false, true));

            auto fc_inst = test_net.get_primitive("fc_prim");
            ASSERT_NE(fc_inst, nullptr);
            if (is_dynamic) {
                EXPECT_FALSE(fc_inst->get_node().get_fused_primitives().empty())
                    << "Binary Add should be fused into dynamic fully_connected_gpu_bf_tiled for batch=" << batch_num;
            }

            test_net.set_input_data("input", input_mem);
            test_net.set_input_data("residual", residual_mem);
            ref_net.set_input_data("input", input_mem);
            ref_net.set_input_data("residual", residual_mem);

            auto test_out = test_net.execute().at("reorder_out").get_memory();
            auto ref_out = ref_net.execute().at("reorder_out").get_memory();
            compare_fc_memory_near<float>(test_out,
                                          ref_out,
                                          static_cast<size_t>(batch_num) * ofm_num,
                                          5e-2f,
                                          "is_dynamic=" + std::to_string(is_dynamic) + " batch=" + std::to_string(batch_num));
        }
    }
}

TEST(fully_connected_gpu, bf_tiled_small_batch_exact_tiles) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int ifm_num = 64;
    const int ofm_num = 64;

    for (data_types dt : {data_types::f16, data_types::f32}) {
        if (dt == data_types::f16 && !engine.get_device_info().supports_fp16) {
            continue;
        }
        const float tol = (dt == data_types::f16) ? 5e-2f : 1e-3f;
        for (int batch_num : {1, 2, 7, 30, 196}) {
            auto input_mem = engine.allocate_memory({{1, batch_num, ifm_num, 1}, dt, format::bfyx});
            auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, dt, format::bfyx});
            fill_random_fc_tensor(input_mem, rg, dt, static_cast<size_t>(batch_num) * ifm_num, -1.0f, 1.0f);
            fill_random_fc_tensor(weights_mem, rg, dt, static_cast<size_t>(ofm_num) * ifm_num, -0.5f, 0.5f);

            topology topo(input_layout("input", input_mem->get_layout()),
                          data("weights", weights_mem),
                          fully_connected("fc_prim", input_info("input"), "weights", "", 3, 2));

            network test_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bf_tiled", true));
            network ref_net(engine, topo, make_forced_fc_config(engine, "fully_connected_gpu_bfyx_ref", false));

            test_net.set_input_data("input", input_mem);
            ref_net.set_input_data("input", input_mem);
            auto test_out = test_net.execute().at("fc_prim").get_memory();
            auto ref_out = ref_net.execute().at("fc_prim").get_memory();

            if (batch_num == 30 || batch_num == 196) {
                const std::string expected_tb = (batch_num == 30) ? "#define TILE_B 6" : "#define TILE_B 7";
                const std::string expected_bsv = (batch_num == 30) ? "#define DISPATCH_BSV 5" : "#define DISPATCH_BSV 14";
                EXPECT_TRUE(kernel_source_has_all_defines(*test_net.get_primitive("fc_prim"), {expected_tb, expected_bsv, "#define BATCH_LEFTOVER 0"}))
                    << "dt=" << dt << " batch=" << batch_num << " should keep exact " << expected_tb << ", " << expected_bsv;
            }

            const size_t expected_count = static_cast<size_t>(batch_num) * ofm_num;
            const std::string ctx = "batch=" + std::to_string(batch_num);
            if (dt == data_types::f16) {
                compare_fc_memory_near<ov::float16>(test_out, ref_out, expected_count, tol, "f16 " + ctx);
            } else {
                compare_fc_memory_near<float>(test_out, ref_out, expected_count, tol, "f32 " + ctx);
            }
        }
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
    auto input_mem = engine.allocate_memory({{batch_num, feature_num, input_y, input_x}, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{output_y, input_y}, data_types::f16, format::bfyx});

    // Generate random input data and set values
    auto input_data = rg.generate_random_4d<ov::float16>(batch_num, feature_num, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(output_y, input_y, 1, 1, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    std::vector<ov::float16> empty_bias(output_y, 0);
    set_values(input_mem, input_data_bfyx);
    set_values(weights_mem, weights_data_bfyx);
    auto reference_output =
        dynamic_fully_connected_reference_calc<ov::float16>(batch_num * feature_num, input_y, output_y, input_data_bfyx, weights_data_bfyx, empty_bias);
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
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_mem->count(), batch_num * feature_num * output_y);

    for (size_t i = 0; i < batch_num * feature_num * output_y; ++i) {
        ASSERT_FLOAT_EQ(reference_output[i], output_ptr[i]) << " i = " << i;
    }
}

TEST(fully_connected_gpu, DISABLED_fs_byx_fsv32_b34) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16) {
        GTEST_SKIP() << "The test is skipped (cl_khr_fp16 is not supported).";
    }
    // Test parameters
    const int batch_num = 34;
    const int output_f = 40;
    const int input_x = 3;
    const int input_y = 3;
    const int input_f = 64;

    // Allocate memory
    auto input_prim = engine.allocate_memory({data_types::f16, format::bfyx, {batch_num, input_f, input_y, input_x}});
    auto weights_prim = engine.allocate_memory({data_types::f16, format::bfyx, {output_f, input_f, input_y, input_x}});
    auto bias_prim = engine.allocate_memory({data_types::f16, format::bfyx, {1, 1, output_f, 1}});

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
    topology topology(input_layout("input", input_prim->get_layout()),
                      data("weights", weights_prim),
                      data("bias", bias_prim),
                      reorder("input_fsv", input_info("input"), {data_types::f16, format::fs_b_yx_fsv32, {batch_num, input_f, input_y, input_x}}),
                      fully_connected("fc_prim", input_info("input_fsv"), "weights", "bias"),
                      activation("out", input_info("fc_prim"), activation_func::relu));

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi) {
        for (size_t fi = 0; fi < output_f; ++fi) {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            ASSERT_TRUE(equal);
            if (!equal) {
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
    auto input_mem = engine.allocate_memory({{batch_num, in_feature_num}, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{out_feature_num, in_feature_num}, data_types::f16, format::bfyx});

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
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
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
    enum class TestMode { Static, Dynamic };
    enum class WzpMode { Symmetric, AsymmetricScalar, AsymmetricNonScalar };
    enum class WeightMode { Bit8, Bit4 };
    enum class TargetDevice { SkipDgpu, RunOnDgpu};

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

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weights_data);

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
            ASSERT_GE(impl->get_kernels().size(), size_t((is_dynamic ? 3 : 2))); // shape-agnostic kernels
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
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
    }

    template <typename T>
    void compare_outputs(memory::ptr output_mem,
                         memory::ptr ref_output_mem,
                         float tolerance = 9.0f) {
        cldnn::mem_lock<T> output_ptr (output_mem, get_test_stream());
        cldnn::mem_lock<T> output_ptr_ref (ref_output_mem, get_test_stream());

        for (size_t i = 0; i < output_ptr_ref.size(); i++) {
            ASSERT_NEAR(output_ptr_ref[i], output_ptr[i], tolerance) << "i = " << i;
        }
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

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weights_data);

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
        auto ref_output_mem = get_ref_results();
        compare_outputs<ov::float16>(output_mem, ref_output_mem, 12.0f);
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

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weights_data);

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
            ASSERT_GE(impl->get_kernels().size(), 2u);
        }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        compare_outputs<ov::float16>(output_mem, ref_output_mem, 14.0f);
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

        auto weights_data = std::vector<uint8_t>(weights_mem->count() / 2);
        std::fill(weights_data.begin(), weights_data.end(), 0x11);
        set_values(weights_mem, weights_data);

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

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem1, weights_data);
        set_values(weights_mem2, weights_data);

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
            compare_outputs<ov::float16>(output_mem[i], ref_output_mem[i], 9.0f);
        }
    }

    void test_compressed_int4_scale_gemv(bool is_caching_test,
                                         bool is_dynamic,
                                         long int batch_num,
                                         long int scales_group_size = 128,
                                         bool is_uint4 = false,
                                         bool is_wei_dyn = false,
                                         bool is_output_fp16 = true) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        if (engine.get_device_info().dev_type == device_type::discrete_gpu || supports_immad)
            GTEST_SKIP();

        long int ifm_num = 1024;
        long int ofm_num = 14336;
        auto wei_type = is_uint4 ? data_types::u4 : data_types::i4;

        auto input_mem = engine.allocate_memory({{batch_num, ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, wei_type, format::bfyx});
        auto scale_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem = engine.allocate_memory({{1, 1, 1, 1}, data_types::u8, format::bfyx});
        auto bias_mem = engine.allocate_memory({{batch_num, ofm_num}, data_types::f16, format::bfyx});

        set_values<int8_t>(dcomp_zp_mem, {1});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 255);
        set_values(weights_mem, weights_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -2.0f, 2.0f);
        set_values(scale_mem, scale_data);

        auto bias_data = rg.generate_random_1d<ov::float16>(batch_num * ofm_num, -2.0f, 2.0f);
        set_values(bias_mem, bias_data);

        auto in_layout = is_dynamic ? layout{{-1, ifm_num}, data_types::f16, format::bfyx}
                                    : layout{{batch_num, ifm_num}, data_types::f16, format::bfyx};

        if (is_dynamic && is_wei_dyn) {
            // ifm_num is dynamic
            in_layout = layout{{-1, -1}, data_types::f16, format::bfyx};
        }

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "bias",
                                       "scale",
                                       dcomp_zp_name,
                                       is_output_fp16 ? data_types::f16 : data_types::f32,
                                       2,
                                       2);

        fc_prim.decompression_zero_point_scalar = 1;

        auto get_ref_results = [&]() {
            topology topology(input_layout("input", in_layout),
                              data("weights", weights_mem),
                              data("bias", bias_mem),
                              data("scale", scale_mem),
                              data("dcomp_zp", dcomp_zp_mem),
                              fc_prim);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx,
                                                              "fully_connected_gpu_bfyx_ref",
                                                              impl_types::ocl};
            config.set_property(
                ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
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

        topology topology(input_layout("input", in_layout),
                          data("weights", weights_mem),
                          data("bias", bias_mem),
                          data("scale", scale_mem),
                          data("dcomp_zp", dcomp_zp_mem),
                          fc_prim);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        if (batch_num == 1) {
            // Wait for switching to gemv kernel due to it is async switch for dynamic shape
            network->get_program()->get_compilation_context().wait_all();
            outputs = network->execute();
        }
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto inst = network->get_primitive("fc_prim");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != NULL);
        auto fc_kernels = impl->get_kernels();

        for (auto& it : fc_kernels) {
            auto kernel_name = it->get_id();
            if (batch_num == 1) {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") != kernel_name.npos);
            } else {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") == kernel_name.npos);
            }
        }
        if (batch_num == 1) {
            ASSERT_EQ(fc_kernels.size(), 1);
        } else if (batch_num > 1) {
            ASSERT_GE(fc_kernels.size(), 2u);
        } else {
            ASSERT_TRUE(false);
        }

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        if (is_output_fp16) {
            compare_outputs<ov::float16>(output_mem, ref_output_mem, 30.0f);
        } else {
            compare_outputs<float>(output_mem, ref_output_mem, 30.0f);
        }
    }

    void test_compressed_int4_scale_zp_gemv(bool is_caching_test,
                                            bool is_dynamic,
                                            long int batch_num,
                                            long int scales_group_size = 128,
                                            bool is_uint4 = false,
                                            bool is_wei_dyn = false,
                                            bool is_output_fp16 = true) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        if (engine.get_device_info().dev_type == device_type::discrete_gpu || supports_immad)
            GTEST_SKIP();

        long int ifm_num = 256;
        long int ofm_num = 512;
        auto wei_type = is_uint4 ? data_types::u4 : data_types::i4;

        auto input_mem = engine.allocate_memory({{batch_num, ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, wei_type, format::bfyx});
        auto scale_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto bias_mem = engine.allocate_memory({{batch_num, ofm_num}, data_types::f16, format::bfyx});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 255);
        set_values(weights_mem, weights_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -2.0f, 2.0f);
        set_values(scale_mem, scale_data);

        auto dcomp_zp_mem_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 2.0f, 4.0f);
        set_values(dcomp_zp_mem, dcomp_zp_mem_data);

        auto bias_data = rg.generate_random_1d<ov::float16>(batch_num * ofm_num, -1.0f, 1.0f);
        set_values(bias_mem, bias_data);

        auto in_layout = is_dynamic ? layout{{-1, ifm_num}, data_types::f16, format::bfyx}
                                    : layout{{batch_num, ifm_num}, data_types::f16, format::bfyx};

        if (is_dynamic && is_wei_dyn) {
            // ifm_num is dynamic
            in_layout = layout{{-1, -1}, data_types::f16, format::bfyx};
        }

        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "",
                                       "scale",
                                       "dcomp_zp",
                                       is_output_fp16 ? data_types::f16 : data_types::f32,
                                       2,
                                       2);

        auto get_ref_results = [&]() {
            topology topology(input_layout("input", in_layout),
                              data("weights", weights_mem),
                              data("bias", bias_mem),
                              data("scale", scale_mem),
                              data("dcomp_zp", dcomp_zp_mem),
                              fc_prim);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx,
                                                              "fully_connected_gpu_bfyx_ref",
                                                              impl_types::ocl};
            config.set_property(
                ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
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

        topology topology(input_layout("input", in_layout),
                          data("weights", weights_mem),
                          data("bias", bias_mem),
                          data("scale", scale_mem),
                          data("dcomp_zp", dcomp_zp_mem),
                          fc_prim);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        if (batch_num == 1) {
            // Wait for switching to gemv kernel due to it is async switch for dynamic shape
            network->get_program()->get_compilation_context().wait_all();
            outputs = network->execute();
        }

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto inst = network->get_primitive("fc_prim");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != NULL);
        auto fc_kernels = impl->get_kernels();

        for (auto& it : fc_kernels) {
            auto kernel_name = it->get_id();
            if (batch_num == 1) {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") != kernel_name.npos);
            } else {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") == kernel_name.npos);
            }
        }
        if (batch_num == 1) {
            ASSERT_EQ(fc_kernels.size(), 1);
        } else if (batch_num > 1) {
            ASSERT_EQ(fc_kernels.size(), 2);
        } else {
            ASSERT_TRUE(false);
        }

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        if (is_output_fp16) {
            compare_outputs<ov::float16>(output_mem, ref_output_mem, 10.0f);
        } else {
            compare_outputs<float>(output_mem, ref_output_mem, 10.0f);
        }
    }

    void test_compressed_int4_scale_activation_gemv(bool is_caching_test,
                                                    bool is_dynamic,
                                                    long int batch_num,
                                                    long int scales_group_size = 128,
                                                    bool is_wei_dyn = false,
                                                    bool is_output_fp16 = true) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        if (engine.get_device_info().dev_type == device_type::discrete_gpu || supports_immad)
            GTEST_SKIP();

        long int ifm_num = 256;
        long int ofm_num = 512;

        auto input_mem = engine.allocate_memory({{batch_num, ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::i4, format::bfyx});
        auto scale_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem = engine.allocate_memory({{1, 1, 1, 1}, data_types::u8, format::bfyx});
        auto bias_mem = engine.allocate_memory({{batch_num, ofm_num}, data_types::f16, format::bfyx});

        set_values<int8_t>(dcomp_zp_mem, {8});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weights_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);

        auto bias_data = rg.generate_random_1d<ov::float16>(batch_num * ofm_num, -2.0f, 2.0f);
        set_values(bias_mem, bias_data);

        auto in_layout = is_dynamic ? layout{{-1, ifm_num}, data_types::f16, format::bfyx}
                                    : layout{{batch_num, ifm_num}, data_types::f16, format::bfyx};

        if (is_dynamic && is_wei_dyn) {
            // ifm_num is dynamic
            in_layout = layout{{-1, -1}, data_types::f16, format::bfyx};
        }

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "bias",
                                       "scale",
                                       dcomp_zp_name,
                                       is_output_fp16? data_types::f16 : data_types::f32,
                                       2,
                                       2);

        fc_prim.decompression_zero_point_scalar = 8;

        auto get_ref_results = [&]() {
            topology topology(input_layout("input", in_layout),
                              data("weights", weights_mem),
                              data("bias", bias_mem),
                              data("scale", scale_mem),
                              data("dcomp_zp", dcomp_zp_mem),
                              fc_prim,
                              activation("out", input_info("fc_prim"), activation_func::relu));

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx,
                                                              "fully_connected_gpu_bfyx_ref",
                                                              impl_types::ocl};
            config.set_property(
                ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 1);
            OPENVINO_ASSERT(outputs.begin()->first == "out");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(input_layout("input", in_layout),
                          data("weights", weights_mem),
                          data("bias", bias_mem),
                          data("scale", scale_mem),
                          data("dcomp_zp", dcomp_zp_mem),
                          fc_prim,
                          activation("out", input_info("fc_prim"), activation_func::relu));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        if (batch_num == 1) {
            // Wait for switching to gemv kernel due to it is async switch for dynamic shape
            network->get_program()->get_compilation_context().wait_all();
            outputs = network->execute();
        }

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "out");
        auto inst = network->get_primitive("fc_prim");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != NULL);
        auto fc_kernels = impl->get_kernels();

        for (auto& it : fc_kernels) {
            auto kernel_name = it->get_id();
            if (batch_num == 1) {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") != kernel_name.npos);
            } else {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") == kernel_name.npos);
            }
        }

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        if (is_output_fp16) {
            compare_outputs<ov::float16>(output_mem, ref_output_mem, 9.0f);
        } else {
            compare_outputs<float>(output_mem, ref_output_mem, 9.0f);
        }
    }

    void test_compressed_int4_scale_large_n_gemv(bool is_caching_test,
                                                 bool is_dynamic,
                                                 long int batch_num,
                                                 bool is_dyn_quan = false,
                                                 bool is_output_fp16 = true) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        auto supports_immad = engine.get_device_info().supports_immad;
        if (engine.get_device_info().dev_type == device_type::discrete_gpu || supports_immad)
            GTEST_SKIP();

        long int ifm_num = 4096;
        long int ofm_num = 14336;
        long int scales_group_size = 128;

        auto input_mem = engine.allocate_memory({{batch_num, 1, ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::i4, format::bfyx});
        auto scale_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem = engine.allocate_memory({{1, 1, 1, 1}, data_types::u8, format::bfyx});

        set_values<int8_t>(dcomp_zp_mem, {2});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 5);
        set_values(weights_mem, weights_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -1.0f, 1.0f);
        set_values(scale_mem, scale_data);


        auto in_layout = is_dynamic ? layout{{-1, 1, ifm_num}, data_types::f16, format::bfyx}
                                    : layout{{batch_num, 1, ifm_num}, data_types::f16, format::bfyx};

        if (is_dynamic) {
            // ifm_num is dynamic
            in_layout = layout{{-1, -1, -1}, data_types::f16, format::bfyx};
        }

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "",
                                       "scale",
                                       dcomp_zp_name,
                                       is_output_fp16? data_types::f16 : data_types::f32,
                                       3,
                                       2);

        fc_prim.decompression_zero_point_scalar = 2;

        auto get_ref_results = [&]() {
            topology topology(input_layout("input", in_layout),
                              data("weights", weights_mem),
                              data("scale", scale_mem),
                              data("dcomp_zp", dcomp_zp_mem),
                              fc_prim);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx,
                                                              "fully_connected_gpu_bfyx_ref",
                                                              impl_types::ocl};
            config.set_property(
                ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
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

        topology topology(input_layout("input", in_layout),
                          data("weights", weights_mem),
                          data("scale", scale_mem),
                          data("dcomp_zp", dcomp_zp_mem),
                          fc_prim);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        if (is_dyn_quan) {
            config.set_user_property(ov::hint::dynamic_quantization_group_size(32));
        } else {
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
        }

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        if (batch_num == 1) {
            // Wait for switching to gemv kernel due to it is async switch for dynamic shape
            network->get_program()->get_compilation_context().wait_all();
            outputs = network->execute();
        }

        auto inst = network->get_primitive("fc_prim");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != NULL);
        auto fc_kernels = impl->get_kernels();

        for (auto& it : fc_kernels) {
            auto kernel_name = it->get_id();
            if (batch_num == 1) {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") != kernel_name.npos);
            } else {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") == kernel_name.npos);
            }
        }
        if (batch_num == 1) {
            ASSERT_EQ(fc_kernels.size(), 1);

        } else if (batch_num > 1) {
            ASSERT_GE(fc_kernels.size(), 2u);
        } else {
            ASSERT_TRUE(false);
        }

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        if (is_output_fp16) {
            compare_outputs<ov::float16>(output_mem, ref_output_mem, 9.0f);
        } else {
            compare_outputs<float>(output_mem, ref_output_mem, 9.0f);
        }
    }

    void test_compressed_int4_scale_reuse_gemv(bool is_caching_test,
                                               bool is_dynamic,
                                               long int batch_num,
                                               long int scales_group_size = 128) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        if (engine.get_device_info().dev_type == device_type::discrete_gpu || supports_immad)
            GTEST_SKIP();

        long int ifm_num = 1024;
        long int ofm_num = 256;

        auto input_mem = engine.allocate_memory({{batch_num, ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem1 = engine.allocate_memory({{ofm_num, ifm_num}, data_types::i4, format::bfyx});
        auto weights_mem2 = engine.allocate_memory({{ofm_num, ifm_num}, data_types::i4, format::bfyx});
        auto scale_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem = engine.allocate_memory({{1, 1, 1, 1}, data_types::u8, format::bfyx});

        set_values<int8_t>(dcomp_zp_mem, {8});

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem1, weights_data);
        set_values(weights_mem2, weights_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{{-1, ifm_num}, data_types::f16, format::bfyx}
                                    : layout{{batch_num, ifm_num}, data_types::f16, format::bfyx};

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim1 = fully_connected("fc_prim1",
                                        input_info("input"),
                                        "weights1",
                                        "",
                                        "scale",
                                        dcomp_zp_name,
                                        data_types::f16,
                                        2,
                                        2);
        auto fc_prim2 = fully_connected("fc_prim2",
                                        input_info("input"),
                                        "weights2",
                                        "",
                                        "scale",
                                        dcomp_zp_name,
                                        data_types::f16,
                                        2,
                                        2);

        fc_prim1.decompression_zero_point_scalar = 8;
        fc_prim2.decompression_zero_point_scalar = 8;

        auto get_ref_results = [&]() {
            topology topology(input_layout("input", in_layout),
                              data("weights1", weights_mem1),
                              data("weights2", weights_mem2),
                              data("scale", scale_mem),
                              data("dcomp_zp", dcomp_zp_mem),
                              fc_prim1,
                              fc_prim2);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl = {in_layout.format, "", impl_types::ocl};
            config.set_property(ov::intel_gpu::force_implementations(
                ov::intel_gpu::ImplForcingMap{{"fc_prim1", fc_impl}, {"fc_prim2", fc_impl}}));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.size() == 2);

            std::vector<cldnn::memory::ptr> res{
                engine.reinterpret_buffer(*outputs.at("fc_prim1").get_memory(), outputs.at("fc_prim1").get_layout()),
                engine.reinterpret_buffer(*outputs.at("fc_prim2").get_memory(), outputs.at("fc_prim2").get_layout()),
            };

            return res;
        };

        topology topology(input_layout("input", in_layout),
                          data("weights1", weights_mem1),
                          data("weights2", weights_mem2),
                          data("scale", scale_mem),
                          data("dcomp_zp", dcomp_zp_mem),
                          fc_prim1,
                          fc_prim2);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        if (batch_num == 1) {
            // Wait for switching to gemv kernel due to it is async switch for dynamic shape
            network->get_program()->get_compilation_context().wait_all();
            outputs = network->execute();
        }
        ASSERT_EQ(outputs.size(), size_t(2));

        auto inst = network->get_primitive("fc_prim1");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != NULL);
        auto fc_kernels = impl->get_kernels();

        for (auto& it : fc_kernels) {
            auto kernel_name = it->get_id();
            if (batch_num == 1) {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") != kernel_name.npos);
            } else {
                ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") == kernel_name.npos);
            }
        }
        if (batch_num == 1) {
            ASSERT_EQ(fc_kernels.size(), 1);

        } else if (batch_num > 1) {
            ASSERT_GE(fc_kernels.size(), 2u);
        } else {
            ASSERT_TRUE(false);
        }

        std::vector<cldnn::memory::ptr> output_mem{
            engine.reinterpret_buffer(*outputs.at("fc_prim1").get_memory(), outputs.at("fc_prim1").get_layout()),
            engine.reinterpret_buffer(*outputs.at("fc_prim2").get_memory(), outputs.at("fc_prim2").get_layout()),
        };
        auto ref_output_mem = get_ref_results();

        for (size_t i = 0; i < 2; i++) {
            compare_outputs<ov::float16>(output_mem[i], ref_output_mem[i], 9.0f);
        }
    }

void test_compressed_int4_scale_dynamic_batch_gemv(bool is_caching_test,
                                         long int scales_group_size = 128,
                                         bool is_wei_dyn = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();
        auto supports_immad = engine.get_device_info().supports_immad;

        if (engine.get_device_info().dev_type == device_type::discrete_gpu || supports_immad)
            GTEST_SKIP();

        long int ifm_num = 1024;
        long int ofm_num = 1024;
        long int batch_num_list[] = {4, 1, 2, 1};

        auto input_mem = engine.allocate_memory({{batch_num_list[0], ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::i4, format::bfyx});
        auto scale_mem =
            engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem = engine.allocate_memory({{1, 1, 1, 1}, data_types::u8, format::bfyx});

        set_values<int8_t>(dcomp_zp_mem, {8});

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 5);
        set_values(weights_mem, weights_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -2.0f, 2.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = layout{{-1, ifm_num}, data_types::f16, format::bfyx};
        if (is_wei_dyn) {
            // ifm_num is dynamic
            in_layout = layout{{-1, -1}, data_types::f16, format::bfyx};
        }

        auto dcomp_zp_name = supports_immad ? "dcomp_zp" : "";

        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "",
                                       "scale",
                                       dcomp_zp_name,
                                       data_types::f16,
                                       2,
                                       2);

        fc_prim.decompression_zero_point_scalar = 8;

        auto get_ref_results = [&]() {
            topology topology(input_layout("input", in_layout),
                              data("weights", weights_mem),
                              data("scale", scale_mem),
                              data("dcomp_zp", dcomp_zp_mem),
                              fc_prim);

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx,
                                                              "fully_connected_gpu_bfyx_ref",
                                                              impl_types::ocl};
            config.set_property(
                ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
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

        topology topology(input_layout("input", in_layout),
                          data("weights", weights_mem),
                          data("scale", scale_mem),
                          data("dcomp_zp", dcomp_zp_mem),
                          fc_prim);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        for(const auto& batch_num : batch_num_list) {
            input_mem = engine.allocate_memory({{batch_num, ifm_num}, data_types::f16, format::bfyx});
            auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
            set_values(input_mem, input_data);

            network->set_input_data("input", input_mem);

            auto outputs = network->execute();
            if (batch_num == 1) {
                // Wait for switching to gemv kernel due to it is async switch for dynamic shape
                network->get_program()->get_compilation_context().wait_all();
                outputs = network->execute();
            }

            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc_prim");

            auto inst = network->get_primitive("fc_prim");
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != NULL);
            auto fc_kernels = impl->get_kernels();
            for (auto& it : fc_kernels) {
                auto kernel_name = it->get_id();
                if (batch_num == 1) {
                    ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") != kernel_name.npos);
                } else {
                    ASSERT_TRUE(kernel_name.find("fully_connected_gpu_gemv") == kernel_name.npos);
                }
            }
            if (batch_num == 1) {
                ASSERT_EQ(fc_kernels.size(), 1);

            } else if (batch_num > 1) {
                ASSERT_EQ(fc_kernels.size(), 2);
            } else {
                ASSERT_TRUE(false);
            }

            auto output_mem = outputs.begin()->second.get_memory();
            auto ref_output_mem = get_ref_results();
            compare_outputs<ov::float16>(output_mem, ref_output_mem, 9.0f);
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

        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());

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

    void test_compressed_int8_scale(bool is_caching_test,
                                    bool is_dynamic,
                                    int64_t batch_num,
                                    bool use_bias = false,
                                    bool use_zp = false,
                                    bool is_3d = false,
                                    bool common_scale = false) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        int64_t ifm_num = 33;
        int64_t ofm_num = 65;
        int64_t scales_num = common_scale ? 1 : ofm_num;

        auto in_shape = is_3d ? ov::PartialShape({batch_num, 1, ifm_num}) : ov::PartialShape({batch_num, ifm_num});
        auto bias_shape = is_3d ? ov::PartialShape({1, 1, ofm_num}) : ov::PartialShape({1, ofm_num});
        auto input_mem = engine.allocate_memory({ in_shape, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u8, format::bfyx });
        auto bias_mem = engine.allocate_memory({ bias_shape, data_types::f16, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {scales_num, 1}, data_types::f16, format::bfyx });
        auto zp_mem = engine.allocate_memory({ {ofm_num, 1}, data_types::u8, format::bfyx });

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num, 0, 10);
        set_values(weights_mem, weights_data);

        auto bias_data = rg.generate_random_1d<ov::float16>(ofm_num, -2.0f, 2.0f);;
        set_values(bias_mem, bias_data);

        auto scale_data = rg.generate_random_1d<ov::float16>(scales_num, -1.0f, 1.0f);
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
        auto ref_output_mem = get_ref_results();
        const float threshold_fp16 = 1e-1;
        compare_outputs<ov::float16>(output_mem, ref_output_mem, threshold_fp16);
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
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());

        ov::PartialShape expected_shape{1, 8};
        ASSERT_EQ(expected_shape, output_mem->get_layout().get_partial_shape());

        std::vector<ov::float16> expected_result = { -52.f, -104.f, 52.f, 104.f, -13.f, 13.f, -52.f, -52.f };
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_compressed_int8_transpose_scale_zp(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int64_t ofm = 3;
        const int64_t ifm = 32;
        constexpr int64_t group_size = 16;

        auto input_mem = engine.allocate_memory({ {1, 2, ifm}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm, ifm}, data_types::u8, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ifm / group_size, ofm}, data_types::f16, format::bfyx });
        auto zp_mem = engine.allocate_memory({ {ifm / group_size, ofm}, data_types::u8, format::bfyx });

        set_values<ov::float16>(input_mem,
                                std::vector<ov::float16>(1 * 2 * ifm, ov::float16(1.0f)));
        set_values<ov::float16>(scale_mem, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        set_values<uint8_t>(zp_mem, {1, 2, 3, 4, 5, 6});

        std::vector<uint8_t> weights_data;
        weights_data.reserve(ifm * ofm);
        for (int64_t output = 0; output < ofm; ++output) {
            for (int64_t group = 0; group < ifm / group_size; ++group) {
                const auto zp = static_cast<uint8_t>(group * ofm + output + 1);
                weights_data.insert(weights_data.end(), static_cast<size_t>(group_size), static_cast<uint8_t>(zp + 1));
            }
        }
        set_values<uint8_t>(weights_mem, weights_data);

        auto fc_prim = fully_connected("fc_prim",
                                       input_info("input"),
                                       "weights",
                                       "",
                                       "scale",
                                       "zp",
                                       data_types::f16,
                                       3,
                                       2,
                                       true, 0, 0);

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            data("scale", scale_mem),
            data("zp", zp_mem),
            fc_prim);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());

        ASSERT_EQ((ov::PartialShape{1, 2, ofm}), output_mem->get_layout().get_partial_shape());

        const std::vector<ov::float16> expected_result = {80.0f, 112.0f, 144.0f, 80.0f, 112.0f, 144.0f};
        ASSERT_EQ(expected_result.size(), output_ptr.size());
        for (size_t i = 0; i < expected_result.size(); ++i) {
            ASSERT_EQ(expected_result[i], output_ptr[i]) << "i = " << i;
        }
    }

    void test_compressed_int8_parameter_scale_zp(bool is_per_group, bool transpose_scale_zp) {
        auto& engine = get_test_engine();

        constexpr int64_t ofm = 4;
        constexpr int64_t ifm = 32;
        const int64_t groups = is_per_group ? 2 : 1;
        const int64_t group_size = ifm / groups;

        // Non-transposed scale/zp are [groups, N]; transposed scale/zp are [N, groups].
        const ov::PartialShape param_shape = transpose_scale_zp ? ov::PartialShape{ofm, groups} : ov::PartialShape{groups, ofm};
        auto param_offset = [&](int64_t group, int64_t output) {
            return static_cast<size_t>(transpose_scale_zp ? output * groups + group : group * ofm + output);
        };

        auto input_mem = engine.allocate_memory({ {1, 2, ifm}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm, ifm}, data_types::u8, format::bfyx });
        auto scale_mem = engine.allocate_memory({ param_shape, data_types::f16, format::bfyx });
        auto zp_mem = engine.allocate_memory({ param_shape, data_types::u8, format::bfyx });

        std::vector<float> input_data(2 * ifm);
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>(static_cast<int64_t>(i % 3) - 1);
        }
        std::vector<uint8_t> weights_data(ofm * ifm);
        for (int64_t output = 0; output < ofm; ++output) {
            for (int64_t k = 0; k < ifm; ++k) {
                weights_data[output * ifm + k] = static_cast<uint8_t>((output * 3 + k) % 8);
            }
        }
        std::vector<float> scale_data(ofm * groups);
        std::vector<uint8_t> zp_data(ofm * groups);
        for (int64_t group = 0; group < groups; ++group) {
            for (int64_t output = 0; output < ofm; ++output) {
                scale_data[param_offset(group, output)] = 0.5f * static_cast<float>((group + output) % 2 + 1);
                zp_data[param_offset(group, output)] = static_cast<uint8_t>((group + output) % 3);
            }
        }

        std::vector<ov::float16> input_f16(input_data.begin(), input_data.end());
        std::vector<ov::float16> scale_f16(scale_data.begin(), scale_data.end());
        set_values(input_mem, input_f16);
        set_values(weights_mem, weights_data);
        set_values(scale_mem, scale_f16);
        set_values(zp_mem, zp_data);

        topology topology(
            input_layout("input", input_mem->get_layout()),
            data("weights", weights_mem),
            input_layout("scale", scale_mem->get_layout()),
            input_layout("zp", zp_mem->get_layout()),
            fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "zp", data_types::f16, 3, 2, true,
                            transpose_scale_zp, transpose_scale_zp));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        network network(engine, topology, config);
        network.set_input_data("input", input_mem);
        network.set_input_data("scale", scale_mem);
        network.set_input_data("zp", zp_mem);

        // Grouped transposed scale/zp [N, groups] must be reordered from bfyx to fbyx to make N innermost.
        // Per-channel [N, 1] already has N innermost, so it stays as the bfyx parameter.
        const bool expect_reorder = transpose_scale_zp && is_per_group;
        auto fc_inst = network.get_primitive("fc_prim");
        for (size_t idx : {size_t{2}, size_t{3}}) {
            const auto& dep = fc_inst->get_node().get_dependency(idx);
            if (expect_reorder) {
                EXPECT_TRUE(dep.is_type<reorder>()) << "input " << idx;
                EXPECT_EQ(fc_inst->get_input_layout(idx).format, format::fbyx) << "input " << idx;
            } else {
                EXPECT_TRUE(dep.is_type<input_layout>()) << "input " << idx;
                EXPECT_EQ(fc_inst->get_input_layout(idx).format, format::bfyx) << "input " << idx;
            }
        }

        auto outputs = network.execute();
        auto output_mem = outputs.at("fc_prim").get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
        ASSERT_EQ((ov::PartialShape{1, 2, ofm}), output_mem->get_layout().get_partial_shape());

        for (int64_t token = 0; token < 2; ++token) {
            for (int64_t output = 0; output < ofm; ++output) {
                float expected = 0.0f;
                for (int64_t k = 0; k < ifm; ++k) {
                    const auto offset = param_offset(k / group_size, output);
                    const float weight = (static_cast<float>(weights_data[output * ifm + k]) - zp_data[offset]) * scale_data[offset];
                    expected += input_data[token * ifm + k] * weight;
                }
                EXPECT_NEAR(static_cast<float>(output_ptr[token * ofm + output]), expected, 1e-2f)
                    << "token = " << token << ", output = " << output;
            }
        }
    }

    void test_dynamic(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, input_b = 1, weight_b = 4;

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
        ASSERT_EQ(output_prim_mem->get_layout().batch(), input_b);
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
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(output_prim_mem->get_layout().batch(), 1);
            ASSERT_EQ(out_l.batch(), 1);
            ASSERT_EQ(out_l.feature(), 3);
            ASSERT_EQ(out_l.spatial(0), 2);
            ASSERT_EQ(out_l.spatial(1), 1);
            ASSERT_EQ(out_l.spatial(2), 1);
            ASSERT_EQ(out_l.spatial(3), 2);
        } else {
            ASSERT_EQ(output_prim_mem->get_layout().batch(), 6);
            ASSERT_EQ(out_l.batch(), 6);
            ASSERT_EQ(out_l.feature(), 2);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);
        }

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
            ASSERT_EQ(output_prim_mem->get_layout().batch(), input_b);
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
            ASSERT_EQ(output_prim_mem->get_layout().batch(), input_b);
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
            ASSERT_EQ(output_prim_mem->get_layout().batch(), 2);
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
            ASSERT_EQ(output_prim_mem->get_layout().batch(), 1);
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
                ASSERT_EQ(output_prim_mem->get_layout().batch(), 2);
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
                ASSERT_EQ(output_prim_mem->get_layout().batch(), 1);
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
    }

    void test_has_cached_weights_reorder(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, input_b = 1, weight_b = 4;

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
        ASSERT_EQ(output_prim_mem->get_layout().batch(), input_b);
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

    void test_comp_weight_scale_zp(bool is_dynamic, int batch = 1, int ifm = 1024, int ofm = 1024,
                                        size_t quantize_group_size = 32, int scales_group_size = 32, int wzp_group_size = 32,
                                        WzpMode wzp_mode = WzpMode::Symmetric, WeightMode weight_mode = WeightMode::Bit8,
                                        TargetDevice dgpu = TargetDevice::RunOnDgpu) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (dgpu == TargetDevice::SkipDgpu && engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int batch_num = batch;
        long int ifm_num = ifm;
        long int ofm_num = ofm;
        long int scale_num = ifm_num / scales_group_size;
        long int wzp_num = (wzp_mode == WzpMode::AsymmetricNonScalar) ? ofm_num : 1;
        long int wzp_ifm_num = (wzp_mode == WzpMode::AsymmetricNonScalar) ? (ifm_num / wzp_group_size) : 1;
        auto weight_type = (weight_mode == WeightMode::Bit4) ? data_types::i4 : data_types::u8;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });

        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, weight_type, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, scale_num}, data_types::f16, format::fbyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {wzp_num, wzp_ifm_num}, data_types::u8, format::fbyx });

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.f, 2.f);
        set_values(input_mem, input_data);

        if (weight_mode == WeightMode::Bit4) {
            auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 4);
            set_values(weights_mem, weights_data);
        } else {
            auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num, 0, 4);
            set_values(weights_mem, weights_data);
        }

        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * scale_num, -2.f, 2.f);
        set_values(scale_mem, scale_data);

        bool need_wzp_test = (wzp_mode != WzpMode::Symmetric);
        if (need_wzp_test) {
            auto zp_data = rg.generate_random_1d<uint8_t>(wzp_num * wzp_ifm_num, 0, 4);
            set_values(dcomp_zp_mem, zp_data);
        }

        auto in_layout = is_dynamic ? layout{ ov::PartialShape{ -1, -1, -1 }, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto dcomp_zp_name = need_wzp_test ? "wzp" : "";
        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", dcomp_zp_name, data_types::f16, 3, 2);

        if (need_wzp_test) {
            fc_prim.compressed_weights = true;
            fc_prim.decompression_zero_point = need_wzp_test ? input_info("wzp") : input_info();
        }

        // Implemented reference kernel
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
            config.set_user_property(ov::hint::dynamic_quantization_group_size(quantize_group_size));

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

        ExecutionConfig config = get_test_default_config(engine);
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
            OPENVINO_ASSERT(abs_diff < 10);
        }
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
        OPENVINO_ASSERT((avg/count) < 1);
    }

    // Test for FP16 overflow prevention in dynamic quantization scale multiplication.
    // When convert_half(acc_tmp) * ds exceeds FP16 max (65504), the intermediate overflows to INF.
    // The fix reorders multiplication to: convert_half(acc_tmp) * de_quantize_scale * ds,
    // where de_quantize_scale is small (< 0.1), preventing intermediate overflow.
    // This test uses large batch (> 256) to trigger SLM/dyn_quan path and large decompression
    // scales to reproduce the overflow condition.
    void test_compressed_int4_dyn_quan_large_scale_no_overflow(bool is_dynamic, long int batch_num) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        // Use small IFM with only 2 scale groups so that the total output stays within
        // FP16 range, while the intermediate convert_half(acc_tmp) * ds can still overflow.
        long int ifm_num = 256;
        long int ofm_num = 64;
        long int scales_group_size = 128;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto dyn_input_ps = ov::PartialShape{ -1, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });

        // Input [-2, 2]: after dynamic quantization (dq_scale = max_abs/127 ~ 0.016),
        // INT8 values span [-127, 127]. Accumulating 128 INT8 x INT4 products yields
        // |acc_tmp| up to ~1000-2000 in realistic scenarios.
        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -2.0f, 2.0f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 15);
        set_values(weights_mem, weights_data);

        // Decompression scales up to 60. This triggers intermediate overflow:
        //   convert_half(1500) * 60 = 90,000 > 65,504 -> INF in old code
        // But the actual output value is small:
        //   1500 * 0.016 * 60 = 1,440 -> well within FP16 range
        // With only 2 scale groups and random cancellation, total output stays within FP16.
        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -60.0f, 60.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);
        fc_prim.decompression_zero_point_scalar = 0;

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

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();

        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
        cldnn::mem_lock<ov::float16> ref_ptr(ref_output_mem, get_test_stream());

        // Count INF/NaN in both test and reference outputs
        size_t test_inf = 0, test_nan = 0, ref_inf = 0;
        for (size_t i = 0; i < output_ptr.size(); i++) {
            if (std::isinf(static_cast<float>(output_ptr[i]))) test_inf++;
            if (std::isnan(static_cast<float>(output_ptr[i]))) test_nan++;
            if (std::isinf(static_cast<float>(ref_ptr[i]))) ref_inf++;
        }
        // The test output should not have more INF than the reference.
        // Any extra INF indicates intermediate FP16 overflow in the dyn_quan kernel.
        ASSERT_EQ(test_nan, 0u) << "Output contains " << test_nan << " NaN values";
        ASSERT_LE(test_inf, ref_inf) << "Dyn_quan output has " << test_inf
            << " INF values vs reference " << ref_inf << " (intermediate FP16 overflow)";

        // Compare non-INF outputs against reference
        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < ref_ptr.size(); ++i) {
            float ref_val = static_cast<float>(ref_ptr[i]);
            float test_val = static_cast<float>(output_ptr[i]);
            if (std::isinf(ref_val) || std::isinf(test_val))
                continue;
            auto abs_diff = std::abs(ref_val - test_val);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
        }
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff
                      << ", avg_diff: " << (count > 0 ? avg / count : 0.f) << std::endl;
        ASSERT_LT(max_diff, 512) << "max_diff = " << max_diff;
    }

    void test_compressed_int4_dyn_quan_large_activation_no_overflow(bool is_dynamic, long int batch_num) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int ifm_num = 256;
        long int ofm_num = 64;
        long int scales_group_size = 128;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto dyn_input_ps = ov::PartialShape{ -1, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });

        // Use positive gate-like activations to avoid cancellation while keeping the final
        // dequantized value well inside the fp16 range.
        // dq_scale ~= 0.75 / 127 = 0.0059, so the true result stays finite:
        //   243840 (scales_group_size: 128 * int8: 127 * u4: 15) * 0.0059 * 20 ~= 28.8K < 65504.
        // The old code still overflows because it first converts the large int accumulator to fp16.
        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, 0.25f, 0.75f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 15);
        set_values(weights_mem, weights_data);

        // Large positive decompression scales keep the old fp16 intermediate path unstable
        // without pushing the true fp32 result outside the fp16 output range.
        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 12.0f, 20.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);
        fc_prim.decompression_zero_point_scalar = 0;

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

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();

        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
        cldnn::mem_lock<ov::float16> ref_ptr(ref_output_mem, get_test_stream());

        size_t test_inf = 0, test_nan = 0, ref_inf = 0;
        for (size_t i = 0; i < output_ptr.size(); i++) {
            if (std::isinf(static_cast<float>(output_ptr[i]))) test_inf++;
            if (std::isnan(static_cast<float>(output_ptr[i]))) test_nan++;
            if (std::isinf(static_cast<float>(ref_ptr[i]))) ref_inf++;
        }
        ASSERT_EQ(test_nan, 0u) << "Output contains " << test_nan << " NaN values";
        ASSERT_LE(test_inf, ref_inf) << "Dyn_quan output has " << test_inf
            << " INF values vs reference " << ref_inf << " (intermediate FP16 overflow)";

        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < ref_ptr.size(); ++i) {
            float ref_val = static_cast<float>(ref_ptr[i]);
            float test_val = static_cast<float>(output_ptr[i]);
            if (std::isinf(ref_val) || std::isinf(test_val))
                continue;
            auto abs_diff = std::abs(ref_val - test_val);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
        }
        GPU_DEBUG_LOG << "---> count: " << count << ", max_diff:" << max_diff
                      << ", avg_diff: " << (count > 0 ? avg / count : 0.f) << std::endl;
        ASSERT_LT(max_diff, 512) << "max_diff = " << max_diff;
    }

    // Stricter variant: asserts zero INF in output regardless of reference.
    // The existing test allows INF when reference also has INF, which means a kernel
    // that produces intermediate overflow can still pass. This test enforces that no
    // INF appears at all, so any FP16 intermediate overflow is caught unconditionally.
    //
    // Parameter design:
    //   input [0.25, 0.75] -> dq_scale ~ 0.0059 -> high positive int_acc (reduced cancellation)
    //   ds [12, 20]        -> TRUE result still far below 65504 (FP16 safe)
    //   FP16 intermediate: convert_half(243840) * 0.0059 * 20 -> INF in old code
    //   FP32 path: 243840 * 0.0059 * 20 ~= 28.8K -> finite
    //
    // Covers empty-output failures caused by gate projection overflow.
    void test_compressed_int4_dyn_quan_large_activation_strict_no_inf(bool is_dynamic, long int batch_num) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int ifm_num = 256;
        long int ofm_num = 64;
        long int scales_group_size = 128;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto dyn_input_ps = ov::PartialShape{ -1, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });

        // Positive activations keep int accumulators large while the final fp32 result stays finite.
        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, 0.25f, 0.75f);
        set_values(input_mem, input_data);

        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 15);
        set_values(weights_mem, weights_data);

        // Large positive decompression scales still trigger the old intermediate overflow path,
        // while keeping the mathematically correct fp32 result finite.
        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 12.0f, 20.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);
        fc_prim.decompression_zero_point_scalar = 0;

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
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));

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

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
        cldnn::mem_lock<ov::float16> ref_ptr(ref_output_mem, get_test_stream());

        size_t inf_count = 0, nan_count = 0;
        for (size_t i = 0; i < output_ptr.size(); i++) {
            float v = static_cast<float>(output_ptr[i]);
            if (std::isinf(v)) inf_count++;
            if (std::isnan(v)) nan_count++;
        }
        ASSERT_EQ(nan_count, 0u) << "Output contains NaN (possible cascade from INF)";
        ASSERT_EQ(inf_count, 0u) << "Output contains " << inf_count
            << " INF values - FP16 intermediate overflow detected";

        size_t ref_inf_count = 0, ref_nan_count = 0;
        for (size_t i = 0; i < ref_ptr.size(); i++) {
            float ref_v = static_cast<float>(ref_ptr[i]);
            if (std::isinf(ref_v))
                ref_inf_count++;
            if (std::isnan(ref_v))
                ref_nan_count++;
        }
        ASSERT_EQ(ref_nan_count, 0u) << "Reference output contains NaN";
        ASSERT_EQ(ref_inf_count, 0u) << "Reference output contains INF";

        float max_diff = 0.f;
        float avg = 0.f;
        size_t count = 0;
        for (size_t i = 0; i < ref_ptr.size(); ++i) {
            float ref_val = static_cast<float>(ref_ptr[i]);
            float test_val = static_cast<float>(output_ptr[i]);
            auto abs_diff = std::abs(ref_val - test_val);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
        }
        ASSERT_GT(count, 0u) << "No finite elements were compared";
        GPU_DEBUG_LOG << "---> strict_int4 count: " << count << ", max_diff:" << max_diff
                      << ", avg_diff: " << (count > 0 ? avg / count : 0.f) << std::endl;
        ASSERT_LT(max_diff, 512) << "max_diff = " << max_diff;
    }

    // Test INT8 PER_TOKEN DQ path for FP16 overflow.
    // The PER_TOKEN INT8 path had a bug where modified_calc_buff was down-cast to FP16
    // before multiplication: (convert_half)(modified_calc_buff) * dq * ds.
    // With large modified_calc_buff, this intermediate exceeds FP16 max (65504) → INF.
    //
    // Parameter design:
    //   weights [0,255] UINT8, zp=0 -> effective range [0,255], large int_acc
    //   input [0.25, 0.5] -> dq_scale ~ 0.004 -> high positive int_acc >> 65504 (OLD: convert_half -> INF)
    //   ds [2, 4]          -> TRUE result stays << 65504 (FIXED: FP32, safe)
    //   FP16 intermediate (old): convert_half(690K) = INF → INF * ds = INF
    //   FP32 path (fixed):  float(690K) * 0.004 * 4 = 10K — well within FP16 range
    void test_compressed_int8_per_token_dyn_quan_strict_no_inf(bool is_dynamic, long int batch_num) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        long int ifm_num = 256;
        long int ofm_num = 64;
        long int scales_group_size = 128;

        auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
        auto dyn_input_ps = ov::PartialShape{ -1, 1, ifm_num };
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u8, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });

        // Positive activations: dq_scale = 0.5/127 ~ 0.004.
        // INT8 weights [0,255] produce large int_acc ~690K >> 65504 per group.
        // True result = 690K * 0.004 * 4 ~ 10K << 65504: always FP16-safe.
        // Old code: convert_half(690K) = INF → NaN cascade.
        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, 0.25f, 0.5f);
        set_values(input_mem, input_data);

        // UINT8 weights [0, 255]: large range ensures int_acc >> 65504.
        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num, 0, 255);
        set_values(weights_mem, weights_data);

        // Positive ds [2, 4]: keeps true result safe while avoiding cancellation.
        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 2.0f, 4.0f);
        set_values(scale_mem, scale_data);

        auto in_layout = is_dynamic ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);
        fc_prim.decompression_zero_point_scalar = 0;

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
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));

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
        // PER_TOKEN: quantize_group_size == ifm_num activates PER_TOKEN_SIZE_DYN_QUANTIZE path
        config.set_user_property(ov::hint::dynamic_quantization_group_size(ifm_num));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));

        auto output_mem = outputs.begin()->second.get_memory();
        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
        cldnn::mem_lock<ov::float16> ref_ptr(ref_output_mem, get_test_stream());

        size_t inf_count = 0, nan_count = 0;
        for (size_t i = 0; i < output_ptr.size(); i++) {
            float v = static_cast<float>(output_ptr[i]);
            if (std::isinf(v)) inf_count++;
            if (std::isnan(v)) nan_count++;
        }
        ASSERT_EQ(nan_count, 0u) << "Output contains NaN (INT8 PER_TOKEN path overflow cascade)";
        ASSERT_EQ(inf_count, 0u) << "Output contains " << inf_count
            << " INF values — INT8 PER_TOKEN FP16 intermediate overflow detected";

        size_t ref_inf_count = 0, ref_nan_count = 0;
        for (size_t i = 0; i < ref_ptr.size(); i++) {
            float ref_v = static_cast<float>(ref_ptr[i]);
            if (std::isinf(ref_v))
                ref_inf_count++;
            if (std::isnan(ref_v))
                ref_nan_count++;
        }
        ASSERT_EQ(ref_nan_count, 0u) << "Reference output contains NaN";
        ASSERT_EQ(ref_inf_count, 0u) << "Reference output contains INF";

        float max_diff = 0.f;
        float avg = 0.f;
        size_t count = 0;
        for (size_t i = 0; i < ref_ptr.size(); ++i) {
            float ref_val = static_cast<float>(ref_ptr[i]);
            float test_val = static_cast<float>(output_ptr[i]);
            auto abs_diff = std::abs(ref_val - test_val);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg += abs_diff;
            count++;
        }
        ASSERT_GT(count, 0u) << "No finite elements were compared";
        GPU_DEBUG_LOG << "---> strict_int8_per_token count: " << count << ", max_diff:" << max_diff
                      << ", avg_diff: " << (count > 0 ? avg / count : 0.f) << std::endl;
        ASSERT_LT(max_diff, 512) << "max_diff = " << max_diff;
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
        const auto& [batch, dims, output_f, input_format, output_format, kernel] = GetParam();
        const auto [input_f, input_x, input_y] = dims;

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
        const auto& [batch, dims, output_y, input_format, output_format, kernel] = GetParam();
        const auto [input_f, input_x, input_y] = dims;

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
        mem_lock<OutputT, mem_lock_type::read> out_ptr(out_mem, get_test_stream());

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
        const auto& [b, in_f, in_x, in_y, out_f, in_fmt] = GetParam();

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

class fully_connected_onednn_expert_zero_points_test : public testing::TestWithParam<std::tuple<int64_t, bool, bool, std::pair<int64_t, int64_t>>> {};

TEST_P(fully_connected_onednn_expert_zero_points_test, per_expert_per_channel) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN compressed FC requires IMMAD support.";

    const auto [experts, transpose_b_param, is_caching_test, decompression_groups] = GetParam();
    const auto scale_groups = decompression_groups.first;
    const auto zero_point_groups = decompression_groups.second;
    constexpr int64_t tokens = 2;
    constexpr int64_t ofm = 16;
    constexpr int64_t ifm = 64;
    const auto zero_point_group_size = ifm / zero_point_groups;
    const size_t weights_rank = experts == 1 ? 2 : 3;
    const auto weights_shape = experts == 1 ? ov::PartialShape{ofm, ifm}
                                           : ov::PartialShape{experts, ofm, ifm};
    const auto zp_shape = experts == 1
                              ? (transpose_b_param ? ov::PartialShape{ofm, zero_point_groups} : ov::PartialShape{zero_point_groups, ofm})
                              : (transpose_b_param ? ov::PartialShape{experts, ofm, zero_point_groups} : ov::PartialShape{experts, zero_point_groups, ofm});
    const auto scale_shape = experts == 1
                                 ? (transpose_b_param ? ov::PartialShape{ofm, scale_groups}
                                                      : ov::PartialShape{scale_groups, ofm})
                                 : (transpose_b_param ? ov::PartialShape{experts, ofm, scale_groups}
                                                      : ov::PartialShape{experts, scale_groups, ofm});

    auto input = engine.allocate_memory({ov::PartialShape{experts, tokens, ifm}, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({weights_shape, data_types::u8, format::bfyx});
    auto scale = engine.allocate_memory({scale_shape, data_types::f16, format::bfyx});
    auto zero_points = engine.allocate_memory({zp_shape, data_types::u8, format::bfyx});
    set_values<ov::float16>(input, std::vector<ov::float16>(experts * tokens * ifm, ov::float16(1.0f)));
    set_values<uint8_t>(weights, std::vector<uint8_t>(experts * ofm * ifm, 9));
    set_values<ov::float16>(scale, std::vector<ov::float16>(experts * ofm * scale_groups, ov::float16(1.0f)));
    std::vector<uint8_t> zp_values(experts * ofm * zero_point_groups);
    const bool transpose_zp_layout = transpose_b_param;
    const auto zp_idx = [=](int64_t expert, int64_t group, int64_t oc) {
        return transpose_zp_layout ? static_cast<size_t>(expert * ofm * zero_point_groups + oc * zero_point_groups + group)
                                   : static_cast<size_t>(expert * zero_point_groups * ofm + group * ofm + oc);
    };
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t group = 0; group < zero_point_groups; ++group) {
            for (int64_t oc = 0; oc < ofm; ++oc) {
                zp_values[zp_idx(expert, group, oc)] = static_cast<uint8_t>((expert + group + oc) % 8);
            }
        }
    }
    set_values<uint8_t>(zero_points, zp_values);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("scale", scale),
        data("zero_points", zero_points),
        fully_connected("fc", input_info("input"), "weights", "", "scale", "zero_points",
                        data_types::f16, 3, weights_rank, true, transpose_b_param, transpose_b_param));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    const ov::intel_gpu::ImplementationDesc impl = {format::bfyx, "", impl_types::onednn};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", impl}}));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    ASSERT_TRUE(network->get_primitive("fc")->get_impl()->is_onednn());
    network->set_input_data("input", input);
    auto output = network->execute().at("fc").get_memory();
    ASSERT_EQ(output->get_layout().get_partial_shape(), (ov::PartialShape{experts, tokens, ofm}));
    mem_lock<ov::float16, mem_lock_type::read> result(output, get_test_stream());
    ASSERT_EQ(result.size(), static_cast<size_t>(experts * tokens * ofm));
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t token = 0; token < tokens; ++token) {
            for (int64_t oc = 0; oc < ofm; ++oc) {
                const auto idx = (expert * tokens + token) * ofm + oc;
                float expected = 0.0f;
                for (int64_t group = 0; group < zero_point_groups; ++group) {
                    expected += static_cast<float>(zero_point_group_size * (9 - zp_values[zp_idx(expert, group, oc)]));
                }
                EXPECT_FLOAT_EQ(static_cast<float>(result[idx]), expected)
                    << "expert=" << expert << ", token=" << token << ", oc=" << oc;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_onednn_expert_zero_points_test,
                         testing::Combine(testing::Values(int64_t{1}, int64_t{64}),
                                          testing::Bool(),
                                          testing::Bool(),
                                          testing::ValuesIn(std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {2, 2}, {1, 4}, {4, 1}})));

// Rounds the values to what the given data type can hold, so that the reference uses the same inputs as the GPU.
static std::vector<float> round_to_data_type(std::vector<float> values, data_types dt) {
    for (auto& v : values) {
        if (dt == data_types::f16) {
            v = static_cast<float>(ov::float16(v));
        } else if (dt == data_types::bf16) {
            v = static_cast<float>(ov::bfloat16(v));
        } else if (dt == data_types::i8 || dt == data_types::u8) {
            v = std::round(v);
        }
    }
    return values;
}

static void set_values_as(memory::ptr mem, const std::vector<float>& values) {
    switch (mem->get_layout().data_type) {
    case data_types::f16:
        set_values(mem, std::vector<ov::float16>(values.begin(), values.end()));
        break;
    case data_types::bf16:
        set_values(mem, std::vector<ov::bfloat16>(values.begin(), values.end()));
        break;
    case data_types::f32:
        set_values(mem, values);
        break;
    case data_types::i8:
        set_values(mem, std::vector<int8_t>(values.begin(), values.end()));
        break;
    case data_types::u8:
        set_values(mem, std::vector<uint8_t>(values.begin(), values.end()));
        break;
    default:
        OPENVINO_THROW("Unexpected data type");
    }
}

static void test_dynamic_shape_agnostic_onednn_fc(size_t input_rank,
                                                  bool is_caching_test,
                                                  bool is_dynamic_ifm = false,
                                                  data_types in_dt = data_types::f16,
                                                  data_types wei_dt = data_types::f16,
                                                  data_types out_dt = data_types::f16,
                                                  bool weights_transposed = true) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP();

    const int64_t ifm = 32, ofm = 16;
    // Several shapes executed on the same network, including a repeated one
    const std::vector<std::vector<int64_t>> shapes = input_rank == 2 ? std::vector<std::vector<int64_t>>{{1}, {7}, {16}, {3}, {7}}
                                                                     : std::vector<std::vector<int64_t>>{{1, 1}, {2, 5}, {1, 16}, {3, 4}, {2, 5}};

    const bool is_int8 = in_dt == data_types::i8 || in_dt == data_types::u8;
    const auto bias_dt = is_int8 ? data_types::f32 : in_dt;
    auto gen = [](tests::random_generator& rg, size_t count, data_types dt) {
        const float min = dt == data_types::u8 ? 0.f : (dt == data_types::i8 ? -4.f : -1.f);
        const float max = (dt == data_types::u8 || dt == data_types::i8) ? 4.f : 1.f;
        return round_to_data_type(rg.generate_random_1d<float>(count, min, max), dt);
    };

    tests::random_generator rg(GET_SUITE_NAME);
    // Weights are [ofm, ifm] when transposed and [ifm, ofm] otherwise
    auto weights_data = gen(rg, ofm * ifm, wei_dt);
    auto bias_data = gen(rg, ofm, bias_dt);

    const auto weights_shape = weights_transposed ? ov::PartialShape{ofm, ifm} : ov::PartialShape{ifm, ofm};
    auto weights_mem = engine.allocate_memory({weights_shape, wei_dt, format::bfyx});
    auto bias_shape = input_rank == 2 ? ov::PartialShape{1, ofm} : ov::PartialShape{1, 1, ofm};
    auto bias_mem = engine.allocate_memory({bias_shape, bias_dt, format::bfyx});
    set_values_as(weights_mem, weights_data);
    set_values_as(bias_mem, bias_data);

    const auto in_dyn_ifm = is_dynamic_ifm ? ov::Dimension::dynamic() : ov::Dimension(ifm);
    auto in_dyn_shape = input_rank == 2 ? ov::PartialShape{ov::Dimension::dynamic(), in_dyn_ifm}
                                        : ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), in_dyn_ifm};
    topology topology(input_layout("input", layout{in_dyn_shape, in_dt, format::bfyx}),
                      data("weights", weights_mem),
                      data("bias", bias_mem),
                      fully_connected("fc", input_info("input"), "weights", "bias", out_dt, input_rank, 2, weights_transposed));

    ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx, "", impl_types::onednn};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", fc_impl_desc}}));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    // Stop asynchronous compilation of shape-specific impls so that every execution goes through the dynamic impl
    network->get_program()->get_compilation_context().cancel();

    const float tolerance = out_dt == data_types::bf16 ? 1e-1f : (out_dt == data_types::f16 ? 2e-2f : 1e-3f);
    for (const auto& outer_dims : shapes) {
        ov::PartialShape in_shape;
        for (auto d : outer_dims)
            in_shape.push_back(d);
        in_shape.push_back(ifm);
        const int64_t rows = std::accumulate(outer_dims.begin(), outer_dims.end(), int64_t{1}, std::multiplies<int64_t>());

        auto input_data = gen(rg, rows * ifm, in_dt);
        auto input_mem = engine.allocate_memory({in_shape, in_dt, format::bfyx});
        set_values_as(input_mem, input_data);

        network->set_input_data("input", input_mem);
        auto outputs = network->execute();

        auto* impl = network->get_primitive("fc")->get_impl();
        ASSERT_NE(impl, nullptr);
        ASSERT_TRUE(impl->is_onednn()) << in_shape;
        ASSERT_TRUE(impl->is_dynamic()) << in_shape;

        auto output_mem = outputs.at("fc").get_memory();
        ov::PartialShape expected_out_shape = in_shape;
        expected_out_shape[expected_out_shape.size() - 1] = ofm;
        ASSERT_EQ(network->get_output_layout("fc").get_partial_shape(), expected_out_shape);
        ASSERT_EQ(output_mem->get_layout().data_type, out_dt);

        const auto output = get_output_values_to_float(*network, outputs.at("fc"));
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t o = 0; o < ofm; ++o) {
                float ref = bias_data[o];
                for (int64_t i = 0; i < ifm; ++i)
                    ref += input_data[r * ifm + i] * weights_data[weights_transposed ? o * ifm + i : i * ofm + o];
                ASSERT_NEAR(ref, output[r * ofm + o], tolerance) << "shape " << in_shape << " row " << r << " ofm " << o;
            }
        }
    }
}

// In-place crop + unsqueeze propagates a dynamic padding to the FC input. The shape-agnostic impl flattens the
// tensors and drops the padding, so it must not be used for such a node.
TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_dynamic_padding) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP();

    const int64_t ifm = 32, ofm = 16;
    tests::random_generator rg(GET_SUITE_NAME);
    auto weights_mem = engine.allocate_memory({ov::PartialShape{ofm, ifm}, data_types::f16, format::bfyx});
    set_values(weights_mem, rg.generate_random_1d<ov::float16>(ofm * ifm, -1, 1));
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{2}, data_types::i64, format::bfyx});
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_length_mem, {ifm, ifm});

    const auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), 2 * ifm}, data_types::f16, format::bfyx}),
        data("split_axis", axis_mem),
        data("splits_length", splits_length_mem),
        data("weights", weights_mem),
        crop("split.out0", {input_info("input"), input_info("split_axis"), input_info("splits_length")}, tensor(1), tensor(0), op_mode, 0, 1),
        crop("split.out1", {input_info("input"), input_info("split_axis"), input_info("splits_length")}, tensor(1), tensor(0), op_mode, 1, 1),
        reshape("unsqueeze", input_info("split.out1"), false, std::vector<int64_t>{0}, ov::PartialShape{1, -1, ifm}, reshape::reshape_mode::unsqueeze),
        fully_connected("fc", input_info("unsqueeze"), "weights", "", 3, 2),
        reorder("output", input_info("split.out0"), format::bfyx, data_types::f32));

    ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx, "", impl_types::onednn};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", fc_impl_desc}}));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->get_program()->get_compilation_context().cancel();

    for (int64_t rows : {1, 7, 3}) {
        auto input_mem = engine.allocate_memory({ov::PartialShape{rows, 2 * ifm}, data_types::f16, format::bfyx});
        set_values(input_mem, rg.generate_random_1d<ov::float16>(rows * 2 * ifm, -1, 1));
        network->set_input_data("input", input_mem);
        network->execute();

        auto fc_inst = network->get_primitive("fc");
        ASSERT_TRUE(fc_inst->get_impl_params()->get_input_layout(0).data_padding.is_dynamic()) << "rows " << rows;
        ASSERT_TRUE(fc_inst->get_impl()->is_onednn()) << "rows " << rows;
        ASSERT_FALSE(fc_inst->get_impl()->is_dynamic()) << "rows " << rows;
    }
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_2d) {
    test_dynamic_shape_agnostic_onednn_fc(2, false);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_3d) {
    test_dynamic_shape_agnostic_onednn_fc(3, false);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_2d_cached) {
    test_dynamic_shape_agnostic_onednn_fc(2, true);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_3d_cached) {
    test_dynamic_shape_agnostic_onednn_fc(3, true);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_2d_dynamic_ifm) {
    test_dynamic_shape_agnostic_onednn_fc(2, false, true);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_3d_dynamic_ifm) {
    test_dynamic_shape_agnostic_onednn_fc(3, false, true);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_2d_not_transposed_weights) {
    test_dynamic_shape_agnostic_onednn_fc(2, false, false, data_types::f16, data_types::f16, data_types::f16, false);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_3d_not_transposed_weights_dynamic_ifm) {
    test_dynamic_shape_agnostic_onednn_fc(3, false, true, data_types::f16, data_types::f16, data_types::f16, false);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_2d_f32) {
    test_dynamic_shape_agnostic_onednn_fc(2, false, false, data_types::f32, data_types::f32, data_types::f32);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_3d_bf16) {
    test_dynamic_shape_agnostic_onednn_fc(3, false, false, data_types::bf16, data_types::bf16, data_types::bf16);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_2d_u8s8) {
    test_dynamic_shape_agnostic_onednn_fc(2, false, false, data_types::u8, data_types::i8, data_types::f32);
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_3d_u8s8_dynamic_ifm) {
    test_dynamic_shape_agnostic_onednn_fc(3, false, true, data_types::u8, data_types::i8, data_types::f32);
}

// Fused ops are not supported by the shape-agnostic impl. With forcing, validate_impl() is skipped and create_impl()
// returns nullptr for the dynamic impl, so a static oneDNN impl with the fused op must be used instead.
TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_fused_op_rejected) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP();

    const int64_t ifm = 32, ofm = 16;
    tests::random_generator rg(GET_SUITE_NAME);
    auto weights_data = round_to_data_type(rg.generate_random_1d<float>(ofm * ifm, -1, 1), data_types::f16);
    auto weights_mem = engine.allocate_memory({ov::PartialShape{ofm, ifm}, data_types::f16, format::bfyx});
    set_values_as(weights_mem, weights_data);

    // The output node is not fused, so softmax consumes the activation to let it be fused into the FC
    topology topology(input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), ifm}, data_types::f16, format::bfyx}),
                      data("weights", weights_mem),
                      fully_connected("fc", input_info("input"), "weights", "", 2, 2),
                      activation("act", input_info("fc"), activation_func::relu),
                      softmax("softmax", input_info("act"), 1));

    ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx, "", impl_types::onednn};
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", fc_impl_desc}}));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->get_program()->get_compilation_context().cancel();

    for (int64_t rows : {1, 7}) {
        auto input_data = round_to_data_type(rg.generate_random_1d<float>(rows * ifm, -1, 1), data_types::f16);
        auto input_mem = engine.allocate_memory({ov::PartialShape{rows, ifm}, data_types::f16, format::bfyx});
        set_values_as(input_mem, input_data);
        network->set_input_data("input", input_mem);
        auto outputs = network->execute();

        auto fc_inst = network->get_primitive("fc");
        ASSERT_FALSE(fc_inst->get_node().get_fused_primitives().empty()) << "rows " << rows;
        ASSERT_TRUE(fc_inst->get_impl()->is_onednn()) << "rows " << rows;
        ASSERT_FALSE(fc_inst->get_impl()->is_dynamic()) << "rows " << rows;

        const auto output = get_output_values_to_float(*network, outputs.at("softmax"));
        for (int64_t r = 0; r < rows; ++r) {
            std::vector<float> ref(ofm);
            for (int64_t o = 0; o < ofm; ++o) {
                float acc = 0.f;
                for (int64_t i = 0; i < ifm; ++i)
                    acc += input_data[r * ifm + i] * weights_data[o * ifm + i];
                ref[o] = std::exp(std::max(acc, 0.f));
            }
            const float sum = std::accumulate(ref.begin(), ref.end(), 0.f);
            for (int64_t o = 0; o < ofm; ++o)
                ASSERT_NEAR(ref[o] / sum, output[r * ofm + o], 1e-2f) << "rows " << rows << " row " << r << " ofm " << o;
        }
    }
}

TEST(smoke_fully_connected_onednn_gpu, dynamic_shape_agnostic_not_selected_on_intel) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad || engine.get_device_info().vendor_id != INTEL_VENDOR_ID)
        GTEST_SKIP();

    const int64_t ifm = 32, ofm = 16;
    tests::random_generator rg(GET_SUITE_NAME);
    auto weights_mem = engine.allocate_memory({ov::PartialShape{ofm, ifm}, data_types::f16, format::bfyx});
    set_values(weights_mem, rg.generate_random_1d<ov::float16>(ofm * ifm, -1, 1));

    topology topology(input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), ifm}, data_types::f16, format::bfyx}),
                      data("weights", weights_mem),
                      fully_connected("fc", input_info("input"), "weights", "", 2, 2));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->get_program()->get_compilation_context().cancel();

    auto input_mem = engine.allocate_memory({ov::PartialShape{7, ifm}, data_types::f16, format::bfyx});
    set_values(input_mem, rg.generate_random_1d<ov::float16>(7 * ifm, -1, 1));
    network->set_input_data("input", input_mem);
    network->execute();

    auto* impl = network->get_primitive("fc")->get_impl();
    ASSERT_NE(impl, nullptr);
    ASSERT_FALSE(impl->is_onednn() && impl->is_dynamic());
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

    auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
    set_values(weights_mem, weights_data);

    auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -4.0f, 4.0f);
    set_values(scale_mem, scale_data);

    auto in_layout = layout{ {1, batch_num, ifm_num, 1}, data_types::f16, format::bfyx };

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "dcomp_zp", data_types::f16, 3, 2);

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

TEST_F(fully_connected_gpu_tests, compressed_int4_dyn_quan_large_scale_no_overflow_unaligned) {
    this->test_compressed_int4_dyn_quan_large_scale_no_overflow(false, 566);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_dyn_quan_large_activation_no_overflow) {
    this->test_compressed_int4_dyn_quan_large_activation_no_overflow(false, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_dyn_quan_large_activation_no_overflow_dynamic) {
    this->test_compressed_int4_dyn_quan_large_activation_no_overflow(true, 512);
}

// Strict variant: ASSERT_EQ(inf_count, 0) - no INF allowed regardless of reference.
// Catches FP16 intermediate overflow that the non-strict test may miss when
// reference also produces INF after an intermediate-overflow pattern is reintroduced.
TEST_F(fully_connected_gpu_tests, compressed_int4_dyn_quan_large_activation_strict_no_inf) {
    this->test_compressed_int4_dyn_quan_large_activation_strict_no_inf(false, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_dyn_quan_large_activation_strict_no_inf_dynamic) {
    this->test_compressed_int4_dyn_quan_large_activation_strict_no_inf(true, 512);
}

// INT8 PER_TOKEN path: verifies that the fixed FP32 multiplication prevents overflow.
// Old code: (convert_half)(modified_calc_buff) * dq * ds → FP16 intermediate overflow.
// Fixed:    (convert_half)(modified_calc_buff * (float)dq * (float)ds) → FP32, no overflow.
TEST_F(fully_connected_gpu_tests, compressed_int8_per_token_dyn_quan_strict_no_inf) {
    this->test_compressed_int8_per_token_dyn_quan_strict_no_inf(false, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_per_token_dyn_quan_strict_no_inf_dynamic) {
    this->test_compressed_int8_per_token_dyn_quan_strict_no_inf(true, 512);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_batch_1) {
    this->test_comp_weight_scale_zp(true, 1, 2048, 3072, 32, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 32, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_12_groupsize) {
    // Expect no dynamic-quantized FC
    this->test_comp_weight_scale_zp(true, 269, 512, 1024, 12, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_34_groupsize) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 34, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_64_groupsize) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 64, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_148_groupsize) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 148, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_128_groupsize) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 128, 128, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_128_groupsize_32_scale) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 128, 32, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_edge_case_128_groupsize_64_scale) {
    this->test_comp_weight_scale_zp(true, 359, 1536, 2560, 128, 64, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_no_wzp) {
    this->test_comp_weight_scale_zp(true, 320, 1024, 1024, 32, 32, 1, WzpMode::Symmetric, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp) {
    this->test_comp_weight_scale_zp(true, 320, 1024, 1024, 32, 32, 1024, WzpMode::AsymmetricNonScalar, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp_scalar) {
    this->test_comp_weight_scale_zp(true, 320, 1024, 1024, 32, 32, 1, WzpMode::AsymmetricScalar, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_wzp_128) {
    this->test_comp_weight_scale_zp(true, 320, 1024, 1024, 128, 128, 128, WzpMode::AsymmetricNonScalar, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

// Test gemv for INT4 weight compression
TEST_F(fully_connected_gpu_tests, gemv_compressed_uint4_scale_dynamic_cached_b1) {
    this->test_compressed_int4_scale_gemv(true, true, 1, 128, true);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_cached_b1) {
    this->test_compressed_int4_scale_gemv(true, true, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_cached_b10) {
    this->test_compressed_int4_scale_gemv(true, true, 10);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_b1g32) {
    this->test_compressed_int4_scale_gemv(false, true, 1, 32);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_b48g32) {
    this->test_compressed_int4_scale_gemv(false, true, 48, 32);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_b1g64) {
    this->test_compressed_int4_scale_gemv(false, true, 1, 64);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_b1g128) {
    this->test_compressed_int4_scale_gemv(false, true, 1, 128);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_b1g32) {
    this->test_compressed_int4_scale_gemv(false, false, 1, 32);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_b1g64) {
    this->test_compressed_int4_scale_gemv(false, false, 1, 64);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_b1g128) {
    this->test_compressed_int4_scale_gemv(false, false, 1, 128);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_uint4_scale_zp_dynamic_cached_b1) {
    this->test_compressed_int4_scale_zp_gemv(true, true, 1, 128, true);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_zp_dynamic_cached_b1) {
    this->test_compressed_int4_scale_zp_gemv(true, true, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_zp_dynamic_b1g128) {
    this->test_compressed_int4_scale_zp_gemv(false, true, 1, 128);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_relu_b1g128) {
    this->test_compressed_int4_scale_activation_gemv(false, false, 1, 128);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_relu_b2g128) {
    this->test_compressed_int4_scale_activation_gemv(false, false, 2, 128);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_relu_dynamic_b1g128) {
    this->test_compressed_int4_scale_activation_gemv(false, true, 1, 128);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_b1) {
    this->test_compressed_int4_scale_large_n_gemv(false, false, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_cached_b1) {
    this->test_compressed_int4_scale_large_n_gemv(true, false, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_dynamic_b1) {
    this->test_compressed_int4_scale_large_n_gemv(false, true, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_dynamic_b6) {
    this->test_compressed_int4_scale_large_n_gemv(false, true, 6);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_dynamic_cached_b1) {
    this->test_compressed_int4_scale_large_n_gemv(true, true, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_dyn_quan) {
    this->test_compressed_int4_scale_large_n_gemv(false, false, 1, true);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_dyn_quan_dynamic) {
    this->test_compressed_int4_scale_large_n_gemv(true, false, 1, true);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_reuse_scale_b1) {
    this->test_compressed_int4_scale_reuse_gemv(false, true, 1);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_reuse_scale_b12) {
    this->test_compressed_int4_scale_reuse_gemv(false, true, 12);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_dynamic_batch) {
    this->test_compressed_int4_scale_dynamic_batch_gemv(false, 128, false);
}

// Test for fp32 output
TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_dynamic_b1g32_f32) {
    this->test_compressed_int4_scale_gemv(false, true, 1, 32, false, false, false);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_b1g32_f32) {
    this->test_compressed_int4_scale_gemv(false, false, 1, 32, false, false, false);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_relu_b1g128_f32) {
    this->test_compressed_int4_scale_activation_gemv(false, false, 1, 128, false, false);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_b1_f32) {
    this->test_compressed_int4_scale_large_n_gemv(false, false, 1, false, false);
}

TEST_F(fully_connected_gpu_tests, gemv_compressed_int4_scale_large_n_dynamic_b1_f32) {
    this->test_compressed_int4_scale_large_n_gemv(false, true, 1, false, false);
}

// Primitive-level test for UINT2 (2-bit) compressed weights on the oneDNN path.
TEST(fully_connected_gpu, compressed_u2_weights_immad) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    // Input: 1x4 (batch=1, ifm=4), distinct per-lane values so a wrong 2-bit lane order is detected
    // Weights: 2x4 packed as u2 (4 values/byte, low-to-high bit order):
    //   ofm0 = [1, 2, 3, 0] -> byte 0x39, ofm1 = [3, 1, 0, 2] -> byte 0x87
    // Scale: 1.0 per output channel, so decompressed weight == raw u2 value
    // Expected output: ofm0 = 1*1+2*2+3*4+0*8 = 17, ofm1 = 3*1+1*2+0*4+2*8 = 21
    const int32_t batch_num = 1, ifm_num = 4, ofm_num = 2;

    auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
    auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u2, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {ofm_num, 1}, data_types::f16, format::bfyx });

    set_values(input_mem, std::vector<ov::float16>{1, 2, 4, 8});
    set_values(weights_mem, std::vector<uint8_t>{0x39, 0x87});
    set_values(scale_mem, std::vector<ov::float16>{1, 1});

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 2);

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("scale", scale_mem));
    topology.add(fc_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

    ASSERT_EQ(output_ptr.size(), size_t(ofm_num));
    EXPECT_NEAR(static_cast<float>(output_ptr[0]), 17.0f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(output_ptr[1]), 21.0f, 1e-2f);
}

// Primitive-level test for UINT2 (2-bit) decompression zero-point on the oneDNN path.
// Zero point dtype is u8 (not u2): the oneDNN u2 kernel cannot unpack a packed-u2 zero-point tensor
// (OpenCL build failure assigning __global u2 to __private int), so u2 is excluded from
// is_zp_compressed in fully_connected_onednn.hpp and only wider zp dtypes are supported.
TEST(fully_connected_gpu, compressed_u2_weights_zp_immad) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    // Weights unchanged from compressed_u2_weights_immad: ofm0 = [1, 2, 3, 0], ofm1 = [3, 1, 0, 2]
    // Zero point: zp0 = 0, zp1 = 1; input = [1, 2, 4, 8]
    // Decompressed weight = (raw - zp) * scale (scale == 1):
    //   ofm0 = [1, 2, 3, 0] - 0 = [1, 2, 3, 0]  -> 1+4+12+0 = 17
    //   ofm1 = [3, 1, 0, 2] - 1 = [2, 0, -1, 1] -> 2+0-4+8 = 6
    const int32_t batch_num = 1, ifm_num = 4, ofm_num = 2;

    auto input_ps = ov::PartialShape{ batch_num, 1, ifm_num };
    auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u2, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {ofm_num, 1}, data_types::f16, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {ofm_num, 1}, data_types::u8, format::bfyx });

    set_values(input_mem, std::vector<ov::float16>{1, 2, 4, 8});
    set_values(weights_mem, std::vector<uint8_t>{0x39, 0x87});
    set_values(scale_mem, std::vector<ov::float16>{1, 1});
    set_values(zp_mem, std::vector<uint8_t>{0, 1});

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "zp", data_types::f16, 3, 2);

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("scale", scale_mem));
    topology.add(data("zp", zp_mem));
    topology.add(fc_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

    ASSERT_EQ(output_ptr.size(), size_t(ofm_num));
    EXPECT_NEAR(static_cast<float>(output_ptr[0]), 17.0f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(output_ptr[1]), 6.0f, 1e-2f);
}

// u2 with a scale group_size oneDNN can't handle (smoke_MatMulCompressedWeights_corner_cases_basic:
// weights [16,32], group_size=4) must be routed to a non-oneDNN impl at selection and stay correct.
TEST(fully_connected_gpu, compressed_u2_weights_unaligned_group_size) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    const int32_t batch_num = 4, ifm_num = 16, ofm_num = 32, group_size = 4;
    const int32_t num_groups = ifm_num / group_size;

    auto input_ps = ov::PartialShape{ batch_num, ifm_num };
    auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u2, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {ofm_num, num_groups}, data_types::f16, format::bfyx });

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
    set_values(input_mem, input_data);
    // k=1 so every byte value 0..255 can occur, exercising all four 2-bit lanes
    auto packed_weights = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 4, 0, 255, 1);
    set_values(weights_mem, packed_weights);
    auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * num_groups, 0.5f, 2.0f);
    set_values(scale_mem, scale_data);

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 2, 2);

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("scale", scale_mem));
    topology.add(fc_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto fc_impl = network.get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_FALSE(fc_impl->is_onednn());

    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_ptr.size(), size_t(batch_num * ofm_num));

    for (int n = 0; n < batch_num; ++n) {
        for (int ofm = 0; ofm < ofm_num; ++ofm) {
            float ref = 0.0f;
            for (int ifm = 0; ifm < ifm_num; ++ifm) {
                const int w_idx = ofm * ifm_num + ifm;
                const uint8_t w = (packed_weights[w_idx / 4] >> ((w_idx % 4) * 2)) & 0x3;
                const float scale = static_cast<float>(scale_data[ofm * num_groups + ifm / group_size]);
                ref += static_cast<float>(input_data[n * ifm_num + ifm]) * static_cast<float>(w) * scale;
            }
            const float out = static_cast<float>(output_ptr[n * ofm_num + ofm]);
            ASSERT_NEAR(out, ref, 0.1f + 0.01f * std::fabs(ref)) << "Mismatch at b=" << n << " ofm=" << ofm;
        }
    }
}

// u2 with several scale (and optional zero-point) groups on the oneDNN path, checked against a CPU reference.
static void test_compressed_u2_grouped_onednn(int32_t group_size, bool with_zp, bool is_caching_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    const int32_t batch_num = 4, ifm_num = 64, ofm_num = 32;
    const int32_t num_groups = ifm_num / group_size;

    auto input_ps = ov::PartialShape{ batch_num, ifm_num };
    auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u2, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {ofm_num, num_groups}, data_types::f16, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {ofm_num, num_groups}, data_types::u8, format::bfyx });

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
    set_values(input_mem, input_data);
    auto packed_weights = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 4, 0, 255, 1);
    set_values(weights_mem, packed_weights);
    auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * num_groups, 0.5f, 2.0f);
    set_values(scale_mem, scale_data);
    auto zp_data = rg.generate_random_1d<uint8_t>(ofm_num * num_groups, 0, 3, 1);
    if (!with_zp)
        std::fill(zp_data.begin(), zp_data.end(), 0);
    set_values(zp_mem, zp_data);

    auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", with_zp ? "zp" : "", data_types::f16, 2, 2);

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(data("weights", weights_mem));
    topology.add(data("scale", scale_mem));
    if (with_zp)
        topology.add(data("zp", zp_mem));
    topology.add(fc_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input_mem);

    auto fc_impl = network->get_primitive("fc_prim")->get_impl();
    ASSERT_TRUE(fc_impl != nullptr);
    ASSERT_TRUE(fc_impl->is_onednn());

    auto outputs = network->execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());
    ASSERT_EQ(output_ptr.size(), size_t(batch_num * ofm_num));

    for (int n = 0; n < batch_num; ++n) {
        for (int ofm = 0; ofm < ofm_num; ++ofm) {
            float ref = 0.0f;
            for (int ifm = 0; ifm < ifm_num; ++ifm) {
                const int w_idx = ofm * ifm_num + ifm;
                const int g_idx = ofm * num_groups + ifm / group_size;
                const int w = (packed_weights[w_idx / 4] >> ((w_idx % 4) * 2)) & 0x3;
                const float w_dq = static_cast<float>(w - zp_data[g_idx]) * static_cast<float>(scale_data[g_idx]);
                ref += static_cast<float>(input_data[n * ifm_num + ifm]) * w_dq;
            }
            const float out = static_cast<float>(output_ptr[n * ofm_num + ofm]);
            ASSERT_NEAR(out, ref, 0.1f + 0.01f * std::fabs(ref)) << "Mismatch at b=" << n << " ofm=" << ofm;
        }
    }
}

TEST(fully_connected_gpu, compressed_u2_weights_grouped_scale_gs16_immad) {
    test_compressed_u2_grouped_onednn(16, false, false);
}

TEST(fully_connected_gpu, compressed_u2_weights_grouped_scale_gs32_immad) {
    test_compressed_u2_grouped_onednn(32, false, false);
}

TEST(fully_connected_gpu, compressed_u2_weights_grouped_scale_zp_gs16_immad) {
    test_compressed_u2_grouped_onednn(16, true, false);
}

TEST(fully_connected_gpu, compressed_u2_weights_grouped_scale_zp_gs32_immad) {
    test_compressed_u2_grouped_onednn(32, true, false);
}

TEST(fully_connected_gpu, compressed_u2_weights_grouped_scale_zp_gs32_immad_cached) {
    test_compressed_u2_grouped_onednn(32, true, true);
}

// Test weight zp for INT8 ASYM
TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_large_input_1025) {
    this->test_comp_weight_scale_zp(true, 1025, 1792, 4608, 128, 128, 1, WzpMode::AsymmetricScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_large) {
    this->test_comp_weight_scale_zp(true, 160, 2048, 2048, 128, 128, 2048, WzpMode::AsymmetricNonScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_32_ifm_1024) {
    this->test_comp_weight_scale_zp(true, 160, 1024, 1024, 32, 32, 1024, WzpMode::AsymmetricNonScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_32_ifm_2048) {
    this->test_comp_weight_scale_zp(true, 160, 2048, 2048, 32, 32, 1, WzpMode::AsymmetricScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_32_ifm_4096) {
    this->test_comp_weight_scale_zp(true, 320, 2048, 2048, 32, 32, 1, WzpMode::AsymmetricScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_small) {
    this->test_comp_weight_scale_zp(true, 16, 1024, 1024, 128, 128, 1024, WzpMode::AsymmetricNonScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_wzp_128_single) {
    this->test_comp_weight_scale_zp(true, 1, 1024, 1024, 128, 128, 1, WzpMode::AsymmetricScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

// Test per-token dyn-quan
TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_test_fake_per_token) {
    this->test_comp_weight_scale_zp(true, 600, 1024, 2048, -1, 32, 1024, WzpMode::AsymmetricNonScalar, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int4_scale_dynamic_quantize_test_per_token) {
    this->test_comp_weight_scale_zp(true, 600, 1024, 2048, -1, 1024, 1024, WzpMode::AsymmetricNonScalar, WeightMode::Bit4, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_test_per_token_small_scale) {
    this->test_comp_weight_scale_zp(true, 600, 1024, 2048, -1, 32, 1, WzpMode::AsymmetricScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_scale_dynamic_quantize_test_per_token_full_scale) {
    this->test_comp_weight_scale_zp(true, 600, 1024, 2048, -1, 1024, 1, WzpMode::AsymmetricScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
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

TEST_F(fully_connected_gpu_tests, compressed_int8_transpose_scale_zp) {
    this->test_compressed_int8_transpose_scale_zp(true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_parameter_scale_zp_per_channel) {
    this->test_compressed_int8_parameter_scale_zp(false, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_parameter_scale_zp_per_channel_transposed) {
    this->test_compressed_int8_parameter_scale_zp(false, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_parameter_scale_zp_per_group) {
    this->test_compressed_int8_parameter_scale_zp(true, false);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_parameter_scale_zp_per_group_transposed) {
    this->test_compressed_int8_parameter_scale_zp(true, true);
}

// Full-pipeline regression for runtime scale/zp: an f32 scale Parameter and a u8 zp Parameter feeding a
// compressed MatMul reach GPU:PostLPT through ConvertPrecision, so the scale arrives as a bare
// Convert(Parameter) and the zp as Convert(Parameter) over the Subtract operand. The pipeline must
// create FullyConnectedCompressed with an f16 scale (Convert preserved) and a u8 integer zero point,
// and the device must compute the dequantized values.
TEST_F(fully_connected_gpu_tests, compressed_int8_parameter_scale_zp_full_pipeline) {
    auto& engine = get_test_engine();
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    // weights u8 [N=4, K=16] = 5, data all ones -> out[n] = 16 * (5 - zp[n]) * scale[n]
    auto make_model = [](ov::PartialShape data_ps) {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, data_ps);
        auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 16}, {5});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
        auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 1});
        auto zp = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{4, 1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f32);
        auto sub = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
        auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scale);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data, mul, false, true);
        return std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{data, scale, zp});
    };

    auto model = make_model({2, 16});
    auto config = get_test_default_config(engine);
    config.finalize(context.get(), model.get());
    ov::intel_gpu::TransformationsPipeline pipeline(config, context);
    pipeline.apply(model);

    std::shared_ptr<ov::Node> fc_compressed;
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_type_name() == std::string("FullyConnectedCompressed")) {
            fc_compressed = op;
        }
    }
    ASSERT_TRUE(fc_compressed != nullptr) << "runtime scale/zp graph was not compressed";
    EXPECT_EQ(fc_compressed->get_input_element_type(3), ov::element::f16) << "scale input must keep the graph precision (its Convert must not be stripped)";
    EXPECT_EQ(fc_compressed->get_input_element_type(4), ov::element::u8) << "zero point input must stay an integer value";

    // Execute through the real plugin entry point, on the device selected by --device_suffix.
    ov::Core core;
    const auto device_name = get_engine_gpu_device_name();
    auto compiled = core.compile_model(make_model({2, 16}), device_name, ov::hint::inference_precision(ov::element::f16));
    const auto execution_devices = compiled.get_property(ov::execution_devices);
    ASSERT_FALSE(execution_devices.empty());
    // A single GPU is listed as "GPU" but reported as "GPU.0", so compare by uuid rather than by name.
    EXPECT_EQ(core.get_property(execution_devices.front(), ov::device::uuid).uuid, engine.get_device_info().uuid.uuid)
        << "test must run on the get_test_engine() device";
    auto request = compiled.create_infer_request();
    std::vector<float> data_data(2 * 16, 1.0f);
    std::vector<float> scale_data = {2.f, 1.f, 1.f, 2.f};
    std::vector<uint8_t> zp_data = {4, 4, 5, 4};
    request.set_input_tensor(0, ov::Tensor(ov::element::f32, {2, 16}, data_data.data()));
    request.set_input_tensor(1, ov::Tensor(ov::element::f32, {4, 1}, scale_data.data()));
    request.set_input_tensor(2, ov::Tensor(ov::element::u8, {4, 1}, zp_data.data()));
    request.infer();
    const float* output = request.get_output_tensor().data<const float>();
    const float expected[4] = {32.f, 16.f, 0.f, 32.f};  // 16 * (5 - zp) * scale
    for (int b = 0; b < 2; ++b) {
        for (int n = 0; n < 4; ++n) {
            EXPECT_NEAR(expected[n], output[b * 4 + n], 1e-1f) << "batch = " << b << ", output = " << n;
        }
    }
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

TEST_F(fully_connected_gpu_tests, compressed_int8_common_scale_b13) {
    this->test_compressed_int8_scale(false, true, 13, false, false, false, true);
}

TEST_F(fully_connected_gpu_tests, compressed_int8_common_scale_b13_cached) {
    this->test_compressed_int8_scale(true, true, 13, false, false, false, true);
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

TEST_F(fully_connected_gpu_tests, compressed_int8_dynamic_quantize_grouped_wzp_test) {
    this->test_comp_weight_scale_zp(false, 160, 1024, 1024, 32, 32, 32, WzpMode::AsymmetricNonScalar, WeightMode::Bit8, TargetDevice::SkipDgpu);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int8_group_scale_zp) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 32, 32, WzpMode::AsymmetricNonScalar, WeightMode::Bit8);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int8_per_oc_scale_zp) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 256, 256, WzpMode::AsymmetricNonScalar, WeightMode::Bit8);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int8_group_scale_scalar_zp) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 32, 32, WzpMode::Symmetric, WeightMode::Bit8);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int4_group_scale_zp) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 32, 32, WzpMode::AsymmetricNonScalar, WeightMode::Bit4);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int4_per_oc_scale_zp) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 256, 256, WzpMode::AsymmetricNonScalar, WeightMode::Bit4);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int4_group_scale_scalar_zp) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 32, 32, WzpMode::Symmetric, WeightMode::Bit4);
}

TEST_F(fully_connected_gpu_tests, onednn_acc_test_compressed_weight_int8_group_scale_scalar_zp_gs16) {
    this->test_comp_weight_scale_zp(false, 64, 256, 2048, 0, 16, 16, WzpMode::Symmetric, WeightMode::Bit8);
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
        const auto& [batch_sizes, input_f, output_f, fc_3d] = GetParam();

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

            mem_lock<OutputT, mem_lock_type::read> output_ptr(output_prim_mem, get_test_stream());

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
        cldnn::mem_lock<WeightsT, mem_lock_type::read> out_ptr(out_mem, get_test_stream());

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

        const auto& [b, in_f, in_x, in_y, out_f, in_fmt] = GetParam();

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

struct fc_bf_tiled_dyn_b_accuracy_test : public ::testing::TestWithParam<std::tuple<long int, long int, long int, bool>> {
    tests::random_generator rg;
    void SetUp() override {
        rg = tests::random_generator(GET_SUITE_NAME);
    }

    void test_accuracy(long int batch_num, long int ifm_num, long int ofm_num, bool with_post_op, long int scales_group_size = 128) {
        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({ {batch_num, ifm_num}, data_types::f16, format::bfyx });
        auto weights_mem = engine.allocate_memory({ {ofm_num, ifm_num}, data_types::u4, format::bfyx });
        auto scale_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });
        auto dcomp_zp_mem = engine.allocate_memory({ {ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx });

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -1.0f, 1.0f);
        set_values(input_mem, input_data);
        auto weights_data = rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 10);
        set_values(weights_mem, weights_data);
        auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -1.0f, 1.0f);
        set_values(scale_mem, scale_data);
        auto zp_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, -0.5f, 0.5f);
        set_values(dcomp_zp_mem, zp_data);

        auto dyn_layout = layout{ {-1, ifm_num}, data_types::f16, format::bfyx };

        cldnn::memory::ptr eltw_mem = nullptr;
        if (with_post_op) {
            eltw_mem = engine.allocate_memory({ {batch_num, ofm_num}, data_types::f16, format::bfyx });
            auto eltw_data = rg.generate_random_1d<ov::float16>(batch_num * ofm_num, -1.0f, 1.0f);
            set_values(eltw_mem, eltw_data);
        }

        auto build_topology = [&]() {
            auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "dcomp_zp", data_types::f16, 2, 2);
            topology topology(
                input_layout("input", dyn_layout),
                data("weights", weights_mem),
                data("scale", scale_mem),
                data("dcomp_zp", dcomp_zp_mem),
                fc_prim
            );
            if (with_post_op) {
                topology.add(data("eltw_data", eltw_mem));
                topology.add(eltwise("eltw", { input_info("fc_prim"), input_info("eltw_data") }, eltwise_mode::sum));
            }
            return topology;
        };

        auto out_name = with_post_op ? "eltw" : "fc_prim";

        // Reference: bfyx_ref kernel
        auto get_ref_results = [&]() {
            auto topo = build_topology();
            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
            network network(engine, topo, config);
            network.set_input_data("input", input_mem);
            auto outputs = network.execute();
            auto output_mem = outputs.at(out_name).get_memory();
            return engine.reinterpret_buffer(*output_mem, outputs.at(out_name).get_layout());
        };

        // Test: dyn_b kernel
        auto get_dyn_b_results = [&]() {
            auto topo = build_topology();
            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_property(ov::intel_gpu::optimize_data(true));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
            ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bf_tiled_dyn_b", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl_desc} }));
            network network(engine, topo, config);
            network.set_input_data("input", input_mem);
            auto outputs = network.execute();
            auto output_mem = outputs.at(out_name).get_memory();
            return engine.reinterpret_buffer(*output_mem, outputs.at(out_name).get_layout());
        };

        auto ref_mem = get_ref_results();
        auto test_mem = get_dyn_b_results();

        cldnn::mem_lock<ov::float16> ref_ptr(ref_mem, get_test_stream());
        cldnn::mem_lock<ov::float16> test_ptr(test_mem, get_test_stream());

        ASSERT_EQ(ref_ptr.size(), test_ptr.size());
        for (size_t i = 0; i < ref_ptr.size(); i++) {
            ASSERT_NEAR(float(ref_ptr[i]), float(test_ptr[i]), 9.0f)
                << "Mismatch at i=" << i
                << " ref=" << float(ref_ptr[i])
                << " dyn_b=" << float(test_ptr[i]);
        }
    }
};

TEST_P(fc_bf_tiled_dyn_b_accuracy_test, compare_with_ref) {
    auto [batch_num, ifm_num, ofm_num, with_post_op] = GetParam();
    test_accuracy(batch_num, ifm_num, ofm_num, with_post_op);
}

INSTANTIATE_TEST_SUITE_P(fully_connected_bf_tiled_dyn_b_accuracy,
                         fc_bf_tiled_dyn_b_accuracy_test,
                         ::testing::Values(
                             // Small batch, IFM < OFM (up/gate projection pattern)
                             std::make_tuple(3L, 256L, 1024L, false),
                             // Prime batch with tail, IFM > OFM (down projection pattern)
                             std::make_tuple(11L, 1024L, 256L, false),
                             // Aligned batch, IFM < OFM
                             std::make_tuple(16L, 256L, 1024L, false),
                             // Large prime batch, IFM < OFM (multi TILE_B + tail)
                             std::make_tuple(29L, 256L, 1024L, false),
                             // With post-op eltwise
                             std::make_tuple(16L, 256L, 1024L, true)));

// Common parameterized fixture for sub-byte (UINT2, UINT3) fully connected kernel tests across various matrix dimensions
class fully_connected_gpu_sub_byte_validation : public ::testing::TestWithParam<std::tuple<int, int, int, int, int, bool>> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<std::tuple<int, int, int, int, int, bool>>& obj) {
        int batch = std::get<0>(obj.param);
        int ifm = std::get<1>(obj.param);
        int ofm = std::get<2>(obj.param);
        int group_size = std::get<3>(obj.param);
        int seq_len = std::get<4>(obj.param);
        bool use_zp = std::get<5>(obj.param);
        auto name = "b" + std::to_string(batch) + "_ifm" + std::to_string(ifm) + "_ofm" + std::to_string(ofm) + "_g" + std::to_string(group_size);
        if (seq_len > 1)
            name += "_seq" + std::to_string(seq_len);
        if (use_zp)
            name += "_zp";
        return name;
    }
};

class fully_connected_gpu_u2_validation : public fully_connected_gpu_sub_byte_validation {};

// Validates UINT2 decompression correctness across multiple sizes using random data vs inline CPU-computed reference
TEST_P(fully_connected_gpu_u2_validation, various_sizes) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().dev_type == device_type::discrete_gpu)
        GTEST_SKIP();

    const int batch_num = std::get<0>(GetParam());
    const int ifm_num = std::get<1>(GetParam());
    const int ofm_num = std::get<2>(GetParam());
    const int scales_group_size = std::get<3>(GetParam());
    const int seq_len = std::get<4>(GetParam());  // 1 = 2D path, >1 = 3D (OUTPUT_3D) path
    const bool use_zp = std::get<5>(GetParam());
    const bool is_3d = seq_len > 1;

    ASSERT_EQ(ifm_num % 4, 0) << "ifm_num must be multiple of 4 for U2";
    ASSERT_EQ(ifm_num % scales_group_size, 0) << "ifm_num must be multiple of scales_group_size";

    auto input_shape = is_3d ? ov::PartialShape{batch_num, seq_len, ifm_num} : ov::PartialShape{batch_num, ifm_num};
    auto input_mem = engine.allocate_memory({input_shape, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u2, format::bfyx});
    auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
    auto zp_mem = use_zp ? engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx}) : nullptr;

    tests::random_generator rg(GET_SUITE_NAME);
    const int input_elems = is_3d ? batch_num * seq_len * ifm_num : batch_num * ifm_num;
    auto input_data = rg.generate_random_1d<ov::float16>(input_elems, -1.0f, 1.0f);
    set_values(input_mem, input_data);

    std::vector<uint8_t> packed_weights;
    std::vector<uint8_t> unpacked_weights;
    for (int ofm = 0; ofm < ofm_num; ++ofm) {
        for (int ifm_byte = 0; ifm_byte < ifm_num / 4; ++ifm_byte) {
            uint8_t packed = 0;
            for (int i = 0; i < 4; ++i) {
                uint8_t val = (ifm_byte * 4 + i) % 4;
                unpacked_weights.push_back(val);
                packed |= (val << (i * 2));
            }
            packed_weights.push_back(packed);
        }
    }
    set_values(weights_mem, packed_weights);

    auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 0.5f, 2.0f);
    set_values(scale_mem, scale_data);

    std::vector<ov::float16> zp_data;
    if (use_zp) {
        zp_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 0.5f, 1.5f);
        set_values(zp_mem, zp_data);
    }

    auto in_layout = layout{input_shape, data_types::f16, format::bfyx};
    const int input_size = is_3d ? 3 : 2;

    topology topology(input_layout("input", in_layout), data("weights", weights_mem), data("scale", scale_mem));
    if (use_zp)
        topology.add(data("zp", zp_mem));
    topology.add(fully_connected("fc_prim", input_info("input"), "weights", "", "scale", use_zp ? "zp" : "", data_types::f16, input_size, 2));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
    config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->set_input_data("input", input_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto inst = network->get_primitive("fc_prim");
    auto impl = inst->get_impl();
    ASSERT_NE(impl, nullptr);

    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

    const int outer_count = is_3d ? batch_num * seq_len : batch_num;
    std::vector<float> expected_output(outer_count * ofm_num);
    for (int n = 0; n < outer_count; ++n) {
        for (int ofm = 0; ofm < ofm_num; ++ofm) {
            float acc = 0.0f;
            for (int ifm = 0; ifm < ifm_num; ++ifm) {
                int group_id = ifm / scales_group_size;
                uint8_t weight_u2 = unpacked_weights[ofm * ifm_num + ifm];
                float scale = static_cast<float>(scale_data[ofm * (ifm_num / scales_group_size) + group_id]);
                float zp = use_zp ? static_cast<float>(zp_data[ofm * (ifm_num / scales_group_size) + group_id]) : 0.0f;
                float weight_f16 = (static_cast<float>(weight_u2) - zp) * scale;
                float inp = static_cast<float>(input_data[n * ifm_num + ifm]);
                acc += inp * weight_f16;
            }
            expected_output[n * ofm_num + ofm] = acc;
        }
    }

    ASSERT_EQ(output_ptr.size(), expected_output.size());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        ASSERT_NEAR(expected_output[i], static_cast<float>(output_ptr[i]), 2.0f) << "Mismatch at output index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_gpu_u2_validation,
                         ::testing::Values(std::make_tuple(1, 4, 4, 4, 1, false),        // minimum size (single group)
                                           std::make_tuple(2, 16, 8, 4, 1, false),       // multiple batches, small group size
                                           std::make_tuple(4, 64, 32, 32, 1, false),     // group_size == ifm (single group per row)
                                           std::make_tuple(1, 128, 64, 32, 1, false),    // larger dimensions, multiple groups
                                           std::make_tuple(8, 256, 128, 64, 1, false),   // large batch + large matrix
                                           std::make_tuple(1, 512, 256, 128, 1, false),  // stress test
                                           std::make_tuple(2, 64, 63, 16, 1, false),     // odd ofm_num
                                           std::make_tuple(2, 60, 64, 4, 1, false),      // odd number of scale groups
                                           std::make_tuple(2, 8, 4, 4, 3, false),        // 3D (OUTPUT_3D) path
                                           std::make_tuple(1, 16, 8, 4, 5, false),       // 3D with longer sequence
                                           std::make_tuple(2, 16, 8, 4, 1, true),        // with zero-point
                                           std::make_tuple(1, 64, 32, 16, 1, true),      // with zero-point, larger
                                           std::make_tuple(2, 8, 4, 4, 3, true)          // 3D with zero-point
                                           ),
                         fully_connected_gpu_u2_validation::GetTestCaseName);

// TernOCL ternary FullyConnected (fully_connected_ternocl_int2): u2 codes {0, 1, 2} with the scalar
// zero point 1, i.e. weights {-1, 0, +1}, and one f16 scale per 128-group along K.
namespace {

enum class ternary_epilogue { none, bias, sigmoid, residual };

struct ternary_fc_params {
    int64_t m, n, k;
    ternary_epilogue epilogue;
    bool out_f32;
    bool int8_prefill;
    bool hadamard;
};

std::ostream& operator<<(std::ostream& os, const ternary_fc_params& p) {
    static const char* epi[] = {"none", "bias", "sigmoid", "residual"};
    return os << "m" << p.m << "_n" << p.n << "_k" << p.k << "_" << epi[static_cast<int>(p.epilogue)] << (p.out_f32 ? "_f32" : "")
              << (p.int8_prefill ? "_int8" : "") << (p.hadamard ? "_had" : "");
}

bool ternocl_int2_supported(engine& engine) {
    const auto& info = engine.get_device_info();
    const auto runtime = engine.runtime_type();
    return (runtime == runtime_types::ocl || runtime == runtime_types::ze) && info.arch >= gpu_arch::xe2 && info.supports_immad;
}

// Normalised Sylvester H_1024 applied to s * x, per 1024-wide block of every row.
void hadamard_1024(std::vector<float>& x, int64_t m, int64_t k, const std::vector<int8_t>& signs) {
    std::vector<float> blk(1024);
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t b0 = 0; b0 < k; b0 += 1024) {
            for (int64_t i = 0; i < 1024; ++i)
                blk[i] = x[r * k + b0 + i] * signs[b0 + i];
            for (int64_t h = 1; h < 1024; h <<= 1) {
                for (int64_t i = 0; i < 1024; i += 2 * h) {
                    for (int64_t j = i; j < i + h; ++j) {
                        const float a = blk[j], c = blk[j + h];
                        blk[j] = a + c;
                        blk[j + h] = a - c;
                    }
                }
            }
            for (int64_t i = 0; i < 1024; ++i)
                x[r * k + b0 + i] = blk[i] / 32.0f;
        }
    }
}

class fully_connected_gpu_ternocl_int2 : public ::testing::TestWithParam<ternary_fc_params> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<ternary_fc_params>& obj) {
        std::ostringstream ss;
        ss << obj.param;
        return ss.str();
    }

    // Runs the FC on M rows of a [-1, K] (dynamic) or [M, K] input and checks it against a host reference.
    // With is_caching_test the network is exported and imported before it runs.
    static void run(const ternary_fc_params& p, bool dynamic, const std::vector<int64_t>& ms, bool is_caching_test = false) {
        auto& engine = get_test_engine();
        const int64_t groups = p.k / 128;
        tests::random_generator rg(GET_SUITE_NAME);

        // About 40% zeros, as in the ternary checkpoints.
        auto u = rg.generate_random_1d<float>(p.n * p.k, 0.0f, 1.0f, 100);
        std::vector<int8_t> w(p.n * p.k);
        std::vector<uint8_t> packed(p.n * p.k / 4, 0);
        for (int64_t i = 0; i < p.n * p.k; ++i) {
            w[i] = u[i] < 0.4f ? 0 : (u[i] < 0.7f ? 1 : -1);
            packed[i / 4] |= static_cast<uint8_t>((w[i] + 1) << (2 * (i % 4)));
        }
        auto scales = rg.generate_random_1d<ov::float16>(p.n * groups, 0.01f, 0.1f, 1000);
        auto bias = rg.generate_random_1d<ov::float16>(p.n, -1.0f, 1.0f);
        std::vector<int8_t> signs;
        if (p.hadamard) {
            auto s = rg.generate_random_1d<float>(p.k, -1.0f, 1.0f);
            for (auto v : s)
                signs.push_back(v < 0 ? -1 : 1);
        }

        auto weights_mem = engine.allocate_memory({{p.n, p.k}, data_types::u2, format::bfyx});
        auto scale_mem = engine.allocate_memory({{p.n, groups}, data_types::f16, format::bfyx});
        set_values(weights_mem, packed);
        set_values(scale_mem, scales);
        const auto out_dt = p.out_f32 ? data_types::f32 : data_types::f16;

        const int64_t max_m = *std::max_element(ms.begin(), ms.end());
        auto in_layout = dynamic ? layout{{-1, p.k}, data_types::f16, format::bfyx} : layout{{max_m, p.k}, data_types::f16, format::bfyx};
        topology topology(input_layout("input", in_layout), data("weights", weights_mem), data("scale", scale_mem));
        const bool has_bias = p.epilogue == ternary_epilogue::bias;
        if (has_bias) {
            auto bias_mem = engine.allocate_memory({{1, p.n}, out_dt, format::bfyx});
            if (p.out_f32) {
                std::vector<float> b(bias.begin(), bias.end());
                set_values(bias_mem, b);
            } else {
                set_values(bias_mem, bias);
            }
            topology.add(data("bias", bias_mem));
        }
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", has_bias ? "bias" : "", "scale", "", out_dt, 2, 2);
        fc.decompression_zero_point_scalar = 1;
        if (p.hadamard) {
            fc.hadamard_block = 1024;
            fc.hadamard_signs = signs;
        }
        topology.add(fc);
        std::string out_id = "fc_prim";
        if (p.epilogue == ternary_epilogue::sigmoid) {
            topology.add(activation("act", input_info("fc_prim"), activation_func::logistic));
            out_id = "act";
        } else if (p.epilogue == ternary_epilogue::residual) {
            auto res_layout = dynamic ? layout{{-1, p.n}, data_types::f16, format::bfyx} : layout{{max_m, p.n}, data_types::f16, format::bfyx};
            topology.add(input_layout("residual", res_layout));
            topology.add(eltwise("sum", {input_info("fc_prim"), input_info("residual")}, eltwise_mode::sum, out_dt));
            out_id = "sum";
        }
        const bool fused_epilogue = p.epilogue == ternary_epilogue::sigmoid || p.epilogue == ternary_epilogue::residual;
        if (fused_epilogue) {
            // An eltwise that is a network output is never fused, so give the epilogue a consumer.
            topology.add(reorder("out", input_info(out_id), format::bfyx, data_types::f32));
            out_id = "out";
        }
        const bool read_f32 = p.out_f32 || fused_epilogue;

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::ternocl_int2_int8_prefill(p.int8_prefill));
        config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // The weights are packed into a constant of their own; the u2 source is left untouched.
        {
            cldnn::mem_lock<uint8_t, mem_lock_type::read> src(weights_mem, get_test_stream());
            ASSERT_TRUE(std::equal(packed.begin(), packed.end(), src.begin()));
        }

        for (const int64_t m : ms) {
            auto x = rg.generate_random_1d<ov::float16>(m * p.k, -1.0f, 1.0f, 100);
            auto input_mem = engine.allocate_memory({{m, p.k}, data_types::f16, format::bfyx});
            set_values(input_mem, x);
            net->set_input_data("input", input_mem);
            std::vector<ov::float16> res;
            if (p.epilogue == ternary_epilogue::residual) {
                res = rg.generate_random_1d<ov::float16>(m * p.n, -1.0f, 1.0f);
                auto res_mem = engine.allocate_memory({{m, p.n}, data_types::f16, format::bfyx});
                set_values(res_mem, res);
                net->set_input_data("residual", res_mem);
            }
            auto outputs = net->execute();

            ASSERT_EQ(net->get_primitive("fc_prim")->get_impl()->get_kernel_name(), "ternocl_int2");
            if (fused_epilogue) {
                ASSERT_TRUE(net->get_primitive("fc_prim")->get_node().has_fused_primitives());
            }

            std::vector<float> xf(x.begin(), x.end());
            if (p.hadamard)
                hadamard_1024(xf, m, p.k, signs);
            std::vector<float> ref(m * p.n);
            float max_abs = 0.0f;
            for (int64_t r = 0; r < m; ++r) {
                for (int64_t n = 0; n < p.n; ++n) {
                    float acc = 0.0f;
                    for (int64_t g = 0; g < groups; ++g) {
                        float part = 0.0f;
                        for (int64_t kk = g * 128; kk < (g + 1) * 128; ++kk)
                            part += xf[r * p.k + kk] * w[n * p.k + kk];
                        acc += part * static_cast<float>(scales[n * groups + g]);
                    }
                    if (has_bias)
                        acc += static_cast<float>(bias[n]);
                    if (p.epilogue == ternary_epilogue::sigmoid)
                        acc = 1.0f / (1.0f + std::exp(-acc));
                    if (p.epilogue == ternary_epilogue::residual)
                        acc += static_cast<float>(res[r * p.n + n]);
                    ref[r * p.n + n] = acc;
                    max_abs = std::max(max_abs, std::fabs(acc));
                }
            }

            // int8 prefill quantizes the activations per 128-group (~1/127 relative per element).
            const float tol = (p.int8_prefill && m > 8 ? 0.02f : 0.005f) * std::max(max_abs, 1.0f);
            // A dynamic network may hand back a buffer larger than the current shape; only m * n values are valid.
            ASSERT_EQ(outputs.at(out_id).get_layout().count(), ref.size());
            auto out_mem = outputs.at(out_id).get_memory();
            std::vector<float> out;
            if (read_f32) {
                cldnn::mem_lock<float> l(out_mem, get_test_stream());
                out.assign(l.begin(), l.begin() + ref.size());
            } else {
                cldnn::mem_lock<ov::float16> l(out_mem, get_test_stream());
                out.assign(l.begin(), l.begin() + ref.size());
            }
            for (size_t i = 0; i < ref.size(); ++i)
                ASSERT_NEAR(ref[i], out[i], tol) << "m=" << m << " index " << i;
        }
    }
};

}  // namespace

TEST_P(fully_connected_gpu_ternocl_int2, static_shape) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    // Post-op fusion into an f16 OCL FC is only done for dynamic shapes (as in the LLMs), so the epilogue cases use one.
    const auto& p = GetParam();
    const bool dynamic = p.epilogue == ternary_epilogue::sigmoid || p.epilogue == ternary_epilogue::residual;
    run(p, dynamic, {p.m});
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_gpu_ternocl_int2,
                         ::testing::Values(
                             // GEMV classes (M = 1, 2, <= 4, <= 8), then the M-tiled GEMM bands
                             ternary_fc_params{1, 256, 1024, ternary_epilogue::none, false, false, false},
                             ternary_fc_params{2, 512, 512, ternary_epilogue::none, false, false, false},
                             ternary_fc_params{3, 1024, 256, ternary_epilogue::none, false, false, false},
                             ternary_fc_params{7, 256, 2048, ternary_epilogue::none, false, false, false},
                             ternary_fc_params{12, 256, 1024, ternary_epilogue::none, false, false, false},
                             ternary_fc_params{33, 512, 512, ternary_epilogue::none, false, false, false},
                             ternary_fc_params{70, 256, 1024, ternary_epilogue::none, false, false, false},
                             // as many groups as output columns: [N, groups] and [groups, N] scales have equal shapes
                             ternary_fc_params{1, 16, 2048, ternary_epilogue::none, false, false, false},
                             // epilogues and output type
                             ternary_fc_params{1, 256, 1024, ternary_epilogue::bias, false, false, false},
                             ternary_fc_params{20, 256, 1024, ternary_epilogue::bias, false, false, false},
                             ternary_fc_params{1, 256, 1024, ternary_epilogue::sigmoid, false, false, false},
                             ternary_fc_params{1, 256, 1024, ternary_epilogue::residual, false, false, false},
                             ternary_fc_params{17, 256, 1024, ternary_epilogue::residual, false, false, false},
                             ternary_fc_params{5, 256, 1024, ternary_epilogue::none, true, false, false},
                             ternary_fc_params{40, 256, 1024, ternary_epilogue::none, true, false, false},
                             // int2 x int8 DPAS prefill
                             ternary_fc_params{12, 256, 1024, ternary_epilogue::none, false, true, false},
                             ternary_fc_params{70, 512, 2048, ternary_epilogue::none, false, true, false},
                             // Hadamard input rotation (rotated-basis checkpoints)
                             ternary_fc_params{1, 256, 2048, ternary_epilogue::none, false, false, true},
                             ternary_fc_params{20, 256, 1024, ternary_epilogue::none, false, false, true},
                             ternary_fc_params{20, 256, 1024, ternary_epilogue::none, false, true, true}),
                         fully_connected_gpu_ternocl_int2::GetTestCaseName);

TEST(fully_connected_gpu_ternocl_int2_dynamic, gemv_and_gemm_on_one_network) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 1024, ternary_epilogue::none, false, false, false}, true, {1, 33, 4, 70, 1});
}

TEST(fully_connected_gpu_ternocl_int2_dynamic, residual_epilogue) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 1024, ternary_epilogue::residual, false, false, false}, true, {1, 17, 2});
}

TEST(fully_connected_gpu_ternocl_int2_cached, static_gemv) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 1024, ternary_epilogue::none, false, false, false}, false, {1}, true);
}

TEST(fully_connected_gpu_ternocl_int2_cached, dynamic_residual) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 1024, ternary_epilogue::residual, false, false, false}, true, {1, 17}, true);
}

TEST(fully_connected_gpu_ternocl_int2_cached, dynamic_hadamard) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 2048, ternary_epilogue::none, false, false, true}, true, {1, 20}, true);
}

TEST(fully_connected_gpu_ternocl_int2_cached, static_int8_prefill_bias) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{20, 256, 1024, ternary_epilogue::bias, false, true, false}, false, {20}, true);
}

// fwht, quant and int8_gemm: the active stages are not in declaration order.
TEST(fully_connected_gpu_ternocl_int2_cached, static_hadamard_int8_prefill) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{40, 256, 1024, ternary_epilogue::none, false, true, true}, false, {40}, true);
}

// A dynamic impl with int8 prefill switches between the GEMV stage (M <= 8) and quant + int8 GEMM
// (M > 8, one GEMM per M band), which also changes its scratch buffers.
TEST(fully_connected_gpu_ternocl_int2_cached, dynamic_int8_prefill) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 1024, ternary_epilogue::none, false, true, false}, true, {1, 8, 9, 20, 40, 70, 4, 1}, true);
}

TEST(fully_connected_gpu_ternocl_int2_cached, dynamic_hadamard_int8_prefill) {
    if (!ternocl_int2_supported(get_test_engine()))
        GTEST_SKIP();
    fully_connected_gpu_ternocl_int2::run(ternary_fc_params{1, 256, 1024, ternary_epilogue::none, false, true, true}, true, {1, 8, 9, 20, 40, 70, 4, 1}, true);
}

// Two FCs on one weight constant: both run the TernOCL impl on the same packed weights.
TEST(fully_connected_gpu_ternocl_int2_dynamic, shared_weights) {
    auto& engine = get_test_engine();
    if (!ternocl_int2_supported(engine))
        GTEST_SKIP();
    const int64_t m = 3, n = 256, k = 1024, groups = k / 128;
    tests::random_generator rg(GET_SUITE_NAME);
    auto u = rg.generate_random_1d<float>(n * k, 0.0f, 1.0f, 100);
    std::vector<int8_t> w(n * k);
    std::vector<uint8_t> packed(n * k / 4, 0);
    for (int64_t i = 0; i < n * k; ++i) {
        w[i] = u[i] < 0.4f ? 0 : (u[i] < 0.7f ? 1 : -1);
        packed[i / 4] |= static_cast<uint8_t>((w[i] + 1) << (2 * (i % 4)));
    }
    auto scales = rg.generate_random_1d<ov::float16>(n * groups, 0.01f, 0.1f, 1000);
    auto weights_mem = engine.allocate_memory({{n, k}, data_types::u2, format::bfyx});
    auto scale_mem = engine.allocate_memory({{n, groups}, data_types::f16, format::bfyx});
    set_values(weights_mem, packed);
    set_values(scale_mem, scales);

    const layout in_layout{{-1, k}, data_types::f16, format::bfyx};
    topology topology(input_layout("input0", in_layout), input_layout("input1", in_layout), data("weights", weights_mem), data("scale", scale_mem));
    for (const auto* id : {"0", "1"}) {
        auto fc = fully_connected(std::string("fc") + id, input_info(std::string("input") + id), "weights", "", "scale", "", data_types::f16, 2, 2);
        fc.decompression_zero_point_scalar = 1;
        topology.add(fc);
    }
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
    network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), false);

    std::vector<std::vector<ov::float16>> xs;
    for (const auto* id : {"0", "1"}) {
        xs.push_back(rg.generate_random_1d<ov::float16>(m * k, -1.0f, 1.0f, 100));
        auto input_mem = engine.allocate_memory({{m, k}, data_types::f16, format::bfyx});
        set_values(input_mem, xs.back());
        net->set_input_data(std::string("input") + id, input_mem);
    }
    auto outputs = net->execute();
    for (size_t f = 0; f < 2; ++f) {
        const auto id = "fc" + std::to_string(f);
        ASSERT_EQ(net->get_primitive(id)->get_impl()->get_kernel_name(), "ternocl_int2");
        cldnn::mem_lock<ov::float16> out(outputs.at(id).get_memory(), get_test_stream());
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t c = 0; c < n; ++c) {
                float acc = 0.0f;
                for (int64_t g = 0; g < groups; ++g) {
                    float part = 0.0f;
                    for (int64_t kk = g * 128; kk < (g + 1) * 128; ++kk)
                        part += static_cast<float>(xs[f][r * k + kk]) * w[c * k + kk];
                    acc += part * static_cast<float>(scales[c * groups + g]);
                }
                ASSERT_NEAR(acc, static_cast<float>(out[r * n + c]), 0.005f * std::max(std::fabs(acc), 1.0f)) << id << " row " << r << " col " << c;
            }
        }
    }
    cldnn::mem_lock<uint8_t, mem_lock_type::read> src(weights_mem, get_test_stream());
    ASSERT_TRUE(std::equal(packed.begin(), packed.end(), src.begin()));
}

TEST(fully_connected_gpu_ternocl_int2_dynamic, non_ternary_zero_point_falls_back) {
    auto& engine = get_test_engine();
    if (!ternocl_int2_supported(engine))
        GTEST_SKIP();
    auto weights_mem = engine.allocate_memory({{256, 1024}, data_types::u2, format::bfyx});
    auto scale_mem = engine.allocate_memory({{256, 8}, data_types::f16, format::bfyx});
    set_values(weights_mem, std::vector<uint8_t>(256 * 1024 / 4, 0x55));
    set_values(scale_mem, std::vector<ov::float16>(256 * 8, ov::float16(0.5f)));
    auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 2, 2);
    fc.decompression_zero_point_scalar = 2;
    topology topology(input_layout("input", layout{{1, 1024}, data_types::f16, format::bfyx}), data("weights", weights_mem), data("scale", scale_mem), fc);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), false);
    auto input_mem = engine.allocate_memory({{1, 1024}, data_types::f16, format::bfyx});
    set_values(input_mem, std::vector<ov::float16>(1024, ov::float16(1.0f)));
    net->set_input_data("input", input_mem);
    net->execute();
    ASSERT_NE(net->get_primitive("fc_prim")->get_impl()->get_kernel_name(), "ternocl_int2");
}

namespace {
// Counts the async compilation tasks queued by the implementations factory; the tasks still run on a real context.
class counting_compilation_context : public ICompilationContext {
public:
    explicit counting_compilation_context(std::shared_ptr<ICompilationContext> context) : _context(std::move(context)) {}
    void push_task(kernel_impl_params key, Task&& task) override {
        pushed_tasks++;
        _context->push_task(std::move(key), std::move(task));
    }
    void remove_keys(std::vector<kernel_impl_params>&& keys) override { _context->remove_keys(std::move(keys)); }
    bool is_stopped() override { return _context->is_stopped(); }
    void cancel() override { _context->cancel(); }
    void wait_all() override { _context->wait_all(); }

    size_t pushed_tasks = 0;

private:
    std::shared_ptr<ICompilationContext> _context;
};

// Runs a dynamic u8 compressed FC (scale + zero point) on several new shapes and returns the number of queued async compilation tasks.
size_t count_async_compilation_tasks(impl_types fc_impl_type) {
    auto& engine = get_test_engine();
    const int64_t ifm_num = 64, ofm_num = 32;
    auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u8, format::bfyx});
    auto scale_mem = engine.allocate_memory({{ofm_num, 1}, data_types::f16, format::bfyx});
    auto zp_mem = engine.allocate_memory({{ofm_num, 1}, data_types::u8, format::bfyx});
    set_values(weights_mem, std::vector<uint8_t>(ofm_num * ifm_num, 3));
    set_values(scale_mem, std::vector<ov::float16>(ofm_num, ov::float16(0.5f)));
    set_values(zp_mem, std::vector<uint8_t>(ofm_num, 1));

    topology topology(input_layout("input", layout{{-1, ifm_num}, data_types::f16, format::bfyx}),
                      data("weights", weights_mem),
                      data("scale", scale_mem),
                      data("zp", zp_mem),
                      fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "zp", data_types::f16, 2, 2));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", {format::bfyx, "", fc_impl_type}}}));

    auto context = std::make_shared<counting_compilation_context>(program::make_compilation_context(config));
    auto prog = program::build_program(engine, topology, config, program::make_task_executor(config), context);
    auto net = network::allocate_network(engine, prog);
    for (int64_t batch : {1, 3, 17, 40}) {
        auto input_mem = engine.allocate_memory({{batch, ifm_num}, data_types::f16, format::bfyx});
        set_values(input_mem, std::vector<ov::float16>(batch * ifm_num, ov::float16(1.0f)));
        net->set_input_data("input", input_mem);
        net->execute();
        EXPECT_EQ(net->get_primitive("fc_prim")->get_impl()->is_onednn(), fc_impl_type == impl_types::onednn);
    }
    context->wait_all();
    return context->pushed_tasks;
}
}  // namespace

#ifdef ENABLE_ONEDNN_FOR_GPU
// The oneDNN FC has no shape-agnostic impl, so each new shape is compiled synchronously. An async compilation task for the same
// shape would only duplicate that work and run create() concurrently on the same onednn attributes, which create() modifies.
TEST(fully_connected_onednn_gpu, dynamic_compressed_no_async_compilation_without_shape_agnostic_impl) {
    if (!get_test_engine().get_device_info().supports_immad)
        GTEST_SKIP();
    ASSERT_EQ(count_async_compilation_tasks(impl_types::onednn), size_t(0));
}
#endif

// Async compilation is kept when a shape-agnostic impl serves the new shapes meanwhile.
TEST(fully_connected_gpu, dynamic_compressed_async_compilation_with_shape_agnostic_impl) {
    ASSERT_GT(count_async_compilation_tasks(impl_types::ocl), size_t(0));
}

class fully_connected_gpu_u3_validation : public fully_connected_gpu_sub_byte_validation {};

// Validates UINT3 decompression correctness in the reference kernel. u3 is a tightly packed LSB-first bit stream,
// so values straddle byte boundaries and rows are not byte-aligned when ifm_num * 3 is not a multiple of 8
TEST_P(fully_connected_gpu_u3_validation, various_sizes) {
    auto& engine = get_test_engine();

    const int batch_num = std::get<0>(GetParam());
    const int ifm_num = std::get<1>(GetParam());
    const int ofm_num = std::get<2>(GetParam());
    const int scales_group_size = std::get<3>(GetParam());
    const int seq_len = std::get<4>(GetParam());  // 1 = 2D path, >1 = 3D (OUTPUT_3D) path
    const bool use_zp = std::get<5>(GetParam());
    const bool is_3d = seq_len > 1;

    ASSERT_EQ(ifm_num % scales_group_size, 0) << "ifm_num must be multiple of scales_group_size";

    auto input_shape = is_3d ? ov::PartialShape{batch_num, seq_len, ifm_num} : ov::PartialShape{batch_num, ifm_num};
    auto input_mem = engine.allocate_memory({input_shape, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::u3, format::bfyx});
    auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx});
    auto zp_mem = use_zp ? engine.allocate_memory({{ofm_num, ifm_num / scales_group_size}, data_types::f16, format::bfyx}) : nullptr;

    tests::random_generator rg(GET_SUITE_NAME);
    const int input_elems = is_3d ? batch_num * seq_len * ifm_num : batch_num * ifm_num;
    auto input_data = rg.generate_random_1d<ov::float16>(input_elems, -1.0f, 1.0f);
    set_values(input_mem, input_data);

    const size_t weights_count = static_cast<size_t>(ofm_num) * ifm_num;
    auto unpacked_weights = rg.generate_random_1d<uint8_t>(weights_count, 0, 7, 1);
    std::vector<uint8_t> packed_weights((weights_count * 3 + 7) / 8, 0);
    for (size_t i = 0; i < weights_count; ++i) {
        const uint32_t bits = static_cast<uint32_t>(unpacked_weights[i] & 0x7) << ((i * 3) % 8);
        const size_t byte_idx = i * 3 / 8;
        packed_weights[byte_idx] |= static_cast<uint8_t>(bits);
        if (bits > 0xFF)
            packed_weights[byte_idx + 1] |= static_cast<uint8_t>(bits >> 8);
    }
    ASSERT_EQ(weights_mem->size(), packed_weights.size());
    set_values(weights_mem, packed_weights);

    auto scale_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 0.5f, 2.0f);
    set_values(scale_mem, scale_data);

    std::vector<ov::float16> zp_data;
    if (use_zp) {
        zp_data = rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / scales_group_size, 2.5f, 4.5f);
        set_values(zp_mem, zp_data);
    }

    auto in_layout = layout{input_shape, data_types::f16, format::bfyx};
    const int input_size = is_3d ? 3 : 2;

    topology topology(input_layout("input", in_layout), data("weights", weights_mem), data("scale", scale_mem));
    if (use_zp)
        topology.add(data("zp", zp_mem));
    topology.add(fully_connected("fc_prim", input_info("input"), "weights", "", "scale", use_zp ? "zp" : "", data_types::f16, input_size, 2));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc fc_impl_desc = {format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl_desc}}));
    config.set_user_property(ov::hint::dynamic_quantization_group_size(0));

    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    network->set_input_data("input", input_mem);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output_mem, get_test_stream());

    const int outer_count = is_3d ? batch_num * seq_len : batch_num;
    std::vector<float> expected_output(outer_count * ofm_num);
    for (int n = 0; n < outer_count; ++n) {
        for (int ofm = 0; ofm < ofm_num; ++ofm) {
            float acc = 0.0f;
            for (int ifm = 0; ifm < ifm_num; ++ifm) {
                int group_id = ifm / scales_group_size;
                uint8_t weight_u3 = unpacked_weights[ofm * ifm_num + ifm];
                float scale = static_cast<float>(scale_data[ofm * (ifm_num / scales_group_size) + group_id]);
                float zp = use_zp ? static_cast<float>(zp_data[ofm * (ifm_num / scales_group_size) + group_id]) : 0.0f;
                float weight_f16 = (static_cast<float>(weight_u3) - zp) * scale;
                float inp = static_cast<float>(input_data[n * ifm_num + ifm]);
                acc += inp * weight_f16;
            }
            expected_output[n * ofm_num + ofm] = acc;
        }
    }

    ASSERT_EQ(output_ptr.size(), expected_output.size());
    for (size_t i = 0; i < expected_output.size(); ++i) {
        ASSERT_NEAR(expected_output[i], static_cast<float>(output_ptr[i]), 2.0f) << "Mismatch at output index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_gpu_u3_validation,
                         ::testing::Values(std::make_tuple(1, 8, 4, 8, 1, false),         // minimum byte-aligned row
                                           std::make_tuple(2, 16, 8, 4, 1, false),        // multiple batches, small group size
                                           std::make_tuple(4, 64, 32, 32, 1, false),      // multiple groups per row
                                           std::make_tuple(1, 512, 256, 128, 1, false),   // larger dimensions
                                           std::make_tuple(2, 12, 7, 4, 1, false),        // rows not byte-aligned, odd ofm_num
                                           std::make_tuple(2, 60, 63, 4, 1, false),       // rows not byte-aligned, many groups
                                           std::make_tuple(2, 12, 4, 4, 3, false),        // 3D (OUTPUT_3D) path
                                           std::make_tuple(2, 16, 8, 4, 1, true),         // with zero-point
                                           std::make_tuple(1, 60, 33, 12, 1, true),       // zero-point, rows not byte-aligned
                                           std::make_tuple(2, 12, 5, 4, 3, true)          // 3D with zero-point
                                           ),
                         fully_connected_gpu_u3_validation::GetTestCaseName);

#ifdef ENABLE_ONEDNN_FOR_GPU
namespace {
struct onednn_int4_fc_params {
    size_t k;
    bool is_3d;
    bool is_u4;
    bool expect_onednn;
    bool is_f32 = false;
};

class fully_connected_onednn_int4_alignment : public ::testing::TestWithParam<onednn_int4_fc_params> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<onednn_int4_fc_params>& obj) {
        const auto& p = obj.param;
        return std::string(p.is_u4 ? "u4" : "i4") + "_k" + std::to_string(p.k) + (p.is_3d ? "_3d" : "_2d") + (p.is_f32 ? "_f32" : "_f16");
    }
};

template <typename T>
void run_onednn_int4_fc_alignment(const onednn_int4_fc_params& p) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN INT4 FC requires XMX";

    tests::random_generator rg(GET_SUITE_NAME);
    constexpr size_t rows = 32;
    constexpr size_t ofm = 32;
    const size_t k = p.k;
    const int zero_point = p.is_u4 ? 8 : 0;
    const float scale_value = 1.0f / 16;
    const auto activation_type = p.is_f32 ? data_types::f32 : data_types::f16;

    const auto in_shape = p.is_3d ? ov::PartialShape{2, static_cast<int64_t>(rows / 2), static_cast<int64_t>(k)}
                                  : ov::PartialShape{static_cast<int64_t>(rows), static_cast<int64_t>(k)};
    auto input_mem = engine.allocate_memory({in_shape, activation_type, format::bfyx});
    auto weights_mem = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(ofm), static_cast<int64_t>(k)},
                                               p.is_u4 ? data_types::u4 : data_types::i4, format::bfyx});
    auto scale_mem = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(ofm), 1}, data_types::f16, format::bfyx});

    auto input = rg.generate_random_1d<T>(rows * k, -1.0f, 1.0f);
    std::vector<uint8_t> weights((ofm * k + 1) / 2);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<uint8_t>((i * 5 + 3) % 16 | ((i * 7 + 1) % 16) << 4);
    set_values(input_mem, input);
    set_values(weights_mem, weights);
    set_values(scale_mem, std::vector<ov::float16>(ofm, ov::float16(scale_value)));

    topology topo;
    topo.add(input_layout("input", input_mem->get_layout()));
    topo.add(data("weights", weights_mem));
    topo.add(data("scale", scale_mem));
    if (p.is_u4) {
        auto zp_mem = engine.allocate_memory({ov::PartialShape{1, 1}, data_types::u8, format::bfyx});
        set_values<uint8_t>(zp_mem, {static_cast<uint8_t>(zero_point)});
        topo.add(data("zp", zp_mem));
    }
    auto fc = fully_connected("fc", input_info("input"), "weights", "", "scale", p.is_u4 ? "zp" : "", activation_type, in_shape.size(), 2);
    if (p.is_u4)
        fc.decompression_zero_point_scalar = zero_point;
    topo.add(fc);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"fc", {format::bfyx, "", impl_types::onednn}}}));
    network net(engine, topo, config);
    net.set_input_data("input", input_mem);
    auto outputs = net.execute();

    const auto impl = net.get_primitive("fc")->get_impl();
    if (!p.is_3d) {
        ASSERT_TRUE(impl->is_onednn());
    }
    if (p.expect_onednn) {
        EXPECT_EQ(impl->get_kernel_name().find("ref"), std::string::npos) << "oneDNN kernel: " << impl->get_kernel_name();
    }

    auto output_mem = outputs.at("fc").get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output(output_mem, get_test_stream());
    ASSERT_EQ(output.size(), rows * ofm);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t n = 0; n < ofm; ++n) {
            float expected = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                const auto index = n * k + c;
                int w = (weights[index / 2] >> (4 * (index % 2))) & 0xf;
                if (!p.is_u4 && w >= 8)
                    w -= 16;
                expected += static_cast<float>(input[r * k + c]) * static_cast<float>(w - zero_point) * scale_value;
            }
            ASSERT_NEAR(static_cast<float>(output[r * ofm + n]), expected, 0.1f) << "row=" << r << ", ofm=" << n;
        }
    }
}

TEST_P(fully_connected_onednn_int4_alignment, compare_with_host_reference) {
    const auto p = GetParam();
    if (p.is_f32)
        run_onednn_int4_fc_alignment<float>(p);
    else
        run_onednn_int4_fc_alignment<ov::float16>(p);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_onednn_int4_alignment,
                         ::testing::Values(onednn_int4_fc_params{9, false, true, true},
                                           onednn_int4_fc_params{9, false, false, true},
                                           onednn_int4_fc_params{33, false, true, true},
                                           onednn_int4_fc_params{33, false, false, true},
                                           onednn_int4_fc_params{11, false, true, true},
                                           onednn_int4_fc_params{11, false, false, true},
                                           onednn_int4_fc_params{13, false, true, true},
                                           onednn_int4_fc_params{13, false, false, true},
                                           onednn_int4_fc_params{15, false, true, true},
                                           onednn_int4_fc_params{15, false, false, true},
                                           onednn_int4_fc_params{10, false, true, false},
                                           onednn_int4_fc_params{10, false, false, false},
                                           onednn_int4_fc_params{9, false, true, false, true},
                                           onednn_int4_fc_params{9, false, false, false, true},
                                           onednn_int4_fc_params{9, true, true, false},
                                           onednn_int4_fc_params{33, true, false, false}),
                         fully_connected_onednn_int4_alignment::GetTestCaseName);
}  // namespace
#endif

// fully_connected_gpu_gemv has no tail handling for a partial OFM tile (GWS[0] = OFM / ofm_per_work_item, LWS[0] = 16).
// It must be rejected at Validate for an unaligned OFM, also for the shape-agnostic impl built with a dynamic batch.
TEST_F(fully_connected_gpu_tests, gemv_rejects_unaligned_ofm_with_dynamic_batch) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto supports_immad = engine.get_device_info().supports_immad;
    const long int ifm_num = 1024;
    const long int group_size = 128;

    for (long int ofm_num : {1024L, 1000L}) {
        SCOPED_TRACE("OFM=" + std::to_string(ofm_num));
        auto input_mem = engine.allocate_memory({{1, ifm_num}, data_types::f16, format::bfyx});
        auto weights_mem = engine.allocate_memory({{ofm_num, ifm_num}, data_types::i4, format::bfyx});
        auto scale_mem = engine.allocate_memory({{ofm_num, ifm_num / group_size}, data_types::f16, format::bfyx});
        auto dcomp_zp_mem = engine.allocate_memory({{1, 1, 1, 1}, data_types::u8, format::bfyx});
        set_values<int8_t>(dcomp_zp_mem, {8});
        set_values(input_mem, rg.generate_random_1d<ov::float16>(ifm_num, -1.0f, 1.0f));
        set_values(weights_mem, rg.generate_random_1d<uint8_t>(ofm_num * ifm_num / 2, 0, 5));
        set_values(scale_mem, rg.generate_random_1d<ov::float16>(ofm_num * ifm_num / group_size, -2.0f, 2.0f));

        // Fresh topology per network: program build may modify the primitive descriptors.
        auto make_topology = [&]() {
            auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", supports_immad ? "dcomp_zp" : "", data_types::f16, 2, 2);
            fc_prim.decompression_zero_point_scalar = 8;
            return topology(input_layout("input", layout{{-1, ifm_num}, data_types::f16, format::bfyx}),
                            data("weights", weights_mem),
                            data("scale", scale_mem),
                            data("dcomp_zp", dcomp_zp_mem),
                            fc_prim);
        };

        auto make_config = [&](const std::string& forced_kernel) {
            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_property(ov::intel_gpu::optimize_data(true));
            config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
            if (!forced_kernel.empty()) {
                config.set_property(
                    ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", {format::bfyx, forced_kernel, impl_types::ocl}}}));
            }
            return config;
        };
        auto execute = [&](network& net) {
            net.set_input_data("input", input_mem);
            return net.execute().at("fc_prim").get_memory();
        };

        network ref_net(engine, make_topology(), make_config("fully_connected_gpu_bfyx_ref"));
        auto ref_mem = execute(ref_net);

        auto kernel_name_of = [](network& net) {
            auto impl = net.get_primitive("fc_prim")->get_impl();
            return impl ? impl->get_kernel_name() : std::string("<none>");
        };
        auto is_gemv = [](const std::string& kernel_name) {
            return kernel_name.find("fully_connected_gpu_gemv") != std::string::npos;
        };

        // With a dynamic batch the shape-agnostic impl is selected at build time, before any enqueue, so this checks
        // the Validate decision.
        const bool ofm_aligned = ofm_num % 16 == 0;
        auto gemv_net = std::make_shared<network>(engine, make_topology(), make_config("fully_connected_gpu_gemv"));
        const auto shape_agnostic_kernel = kernel_name_of(*gemv_net);
        EXPECT_EQ(is_gemv(shape_agnostic_kernel), ofm_aligned) << "shape-agnostic kernel " << shape_agnostic_kernel;

        auto net = ofm_aligned ? gemv_net : std::make_shared<network>(engine, make_topology(), make_config(""));
        auto out_mem = execute(*net);
        const auto kernel_name = kernel_name_of(*net);
        EXPECT_EQ(is_gemv(kernel_name), ofm_aligned) << "kernel " << kernel_name;

        compare_outputs<ov::float16>(out_mem, ref_mem);
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
// A residual add fused into an f32 oneDNN fully_connected must reach oneDNN as a binary post-op. As a sum post-op it
// becomes GEMM beta = 1, and oneDNN may then split K between work-groups that add their partial sums to dst with
// float atomics, which makes the f32 result change from run to run.
TEST(fully_connected_onednn_gpu, f32_residual_add_fused_as_binary_post_op) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "oneDNN fully_connected requires a device with immad";

    const int64_t m = 32, k = 64, n = 16;
    tests::random_generator rg(GET_SUITE_NAME);
    auto x = rg.generate_random_1d<float>(m * k, -1.0f, 1.0f);
    auto w = rg.generate_random_1d<float>(n * k, -1.0f, 1.0f);
    auto h = rg.generate_random_1d<float>(m * n, -1.0f, 1.0f);

    auto input_mem = engine.allocate_memory({{m, k}, data_types::f32, format::bfyx});
    auto hidden_mem = engine.allocate_memory({{m, n}, data_types::f32, format::bfyx});
    auto weights_mem = engine.allocate_memory({{n, k}, data_types::f32, format::bfyx});
    set_values(input_mem, x);
    set_values(hidden_mem, h);
    set_values(weights_mem, w);

    topology topology(input_layout("input", input_mem->get_layout()),
                      input_layout("hidden", hidden_mem->get_layout()),
                      data("weights", weights_mem),
                      // The addend is computed by the network: a network input never becomes a sum post-op.
                      activation("residual", input_info("hidden"), activation_func::abs),
                      fully_connected("fc", input_info("input"), "weights"),
                      eltwise("sum", {input_info("fc"), input_info("residual")}, eltwise_mode::sum),
                      reorder("out", input_info("sum"), format::bfyx, data_types::f32));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc", {format::bfyx, "", impl_types::onednn}}}));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    network.set_input_data("hidden", hidden_mem);
    auto outputs = network.execute();

    // Look up by type, not id: the node is renamed to "out" when the redundant reorder is dropped.
    const auto& infos = network.get_primitives_info();
    auto fc_info = std::find_if(infos.begin(), infos.end(), [](const primitive_info& info) {
        return info.type_id == "fully_connected";
    });
    ASSERT_NE(fc_info, infos.end());
    ASSERT_NE(std::find(fc_info->c_fused_ids.begin(), fc_info->c_fused_ids.end(), "sum"), fc_info->c_fused_ids.end())
        << "the residual add did not fuse into the fully_connected";
    auto fc_inst = network.get_primitive(fc_info->original_id);
    ASSERT_TRUE(fc_inst->get_impl()->is_onednn());
    const auto& post_ops = fc_inst->get_fused_primitives_onednn();
    auto has_post_op = [&](onednn_post_op_type type) {
        return std::any_of(post_ops.begin(), post_ops.end(), [&](const fused_primitive_desc_onednn& p) {
            return p.op_type == type;
        });
    };
    ASSERT_TRUE(has_post_op(onednn_post_op_type::binary_add));
    ASSERT_FALSE(has_post_op(onednn_post_op_type::sum));

    cldnn::mem_lock<float> out(outputs.at("out").get_memory(), get_test_stream());
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float ref = std::abs(h[i * n + j]);
            for (int64_t kk = 0; kk < k; ++kk)
                ref += x[i * k + kk] * w[j * k + kk];
            ASSERT_NEAR(out[i * n + j], ref, 1e-4f) << "at (" << i << ", " << j << ")";
        }
    }
}
#endif  // ENABLE_ONEDNN_FOR_GPU
