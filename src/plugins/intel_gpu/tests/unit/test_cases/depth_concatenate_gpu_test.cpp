// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/reshape.hpp>

using namespace cldnn;
using namespace tests;

TEST(depth_concatenate_f32_gpu, test01) {
    //  Input count : 2
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //
    //  Input1:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //
    //  Input2:
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //  1    0.1  :f2
    //  0.3 -0.5  :f3
    //  0   -0.2  :f4
    //

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::yxfb, {2, 2, 1, 1}});
    auto input2 = engine.allocate_memory({data_types::f32, format::yxfb, {2, 3, 1, 1}});

    set_values(input1, {0.5f, 0.7f, 0.2f, 0.4f});
    set_values(input2, {1.0f, 0.1f, 0.3f, -0.5f, 0.0f, -0.2f});

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(concatenation("depth1", { input_info("input1"), input_info("input2") }, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_FLOAT_EQ(0.5f, output_ptr[0]);
    ASSERT_FLOAT_EQ(0.7f, output_ptr[1]);
    ASSERT_FLOAT_EQ(0.2f, output_ptr[2]);
    ASSERT_FLOAT_EQ(0.4f, output_ptr[3]);
    ASSERT_FLOAT_EQ(1.0f, output_ptr[4]);
    ASSERT_FLOAT_EQ(0.1f, output_ptr[5]);
    ASSERT_FLOAT_EQ(0.3f, output_ptr[6]);
    ASSERT_FLOAT_EQ(-0.5f, output_ptr[7]);
    ASSERT_FLOAT_EQ(0.0f, output_ptr[8]);
    ASSERT_FLOAT_EQ(-0.2f, output_ptr[9]);
}

template <data_types DType>
void concat_basic_with_reorder() {
    //  Input count : 2
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //
    //  Input1:
    //  2.5  3.7  :f0
    //  0.2  1.4  :f1
    //
    //  Input2:
    //  1    4.1  :f0
    // -4.3 -7.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  2    3  :f0
    //  0    1  :f1
    //  1    4  :f2
    // -4   -7  :f3
    //  0    0  :f4
    //

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::yxfb, {2, 2, 1, 1}});
    auto input2 = engine.allocate_memory({data_types::f32, format::yxfb, {2, 3, 1, 1}});
    auto outs = {2.0f, 3.0f, 0.0f, 1.0f, 1.0f, 4.0f, -4.0f, -7.0f, 0.0f, 0.0f};
    set_values(input1, {2.5f, 3.7f, 0.2f, 1.4f});
    set_values(input2, {1.0f, 4.1f, -4.3f, -7.5f, 0.0f, -0.2f});

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reorder("to_int1", input_info("input1"), {DType, format::yxfb, {2, 2, 1, 1}}));
    topology.add(reorder("to_int2", input_info("input2"), {DType, format::yxfb, {2, 3, 1, 1}}));
    topology.add(concatenation("depth1", { input_info("to_int1"), input_info("to_int2") }, 1));
    topology.add(reorder("to_float", input_info("depth1"), {data_types::f32, format::yxfb, {2, 5, 1, 1}}));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "to_float");

    auto output = outputs.at("to_float").get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    int ptr_cntr = 0;
    for (const auto& ref : outs) {
        ASSERT_FLOAT_EQ(ref, output_ptr[ptr_cntr++]);
    }
}

TEST(depth_concatenate_int8_gpu, concat_basic) {
    concat_basic_with_reorder<data_types::i8>();
}

TEST(depth_concatenate_int32_gpu, concat_basic) {
    concat_basic_with_reorder<data_types::i32>();
}

TEST(depth_concatenate_int64_gpu, concat_basic) {
    concat_basic_with_reorder<data_types::i64>();
}

TEST(depth_concatenate_f32_gpu, test02) {
    //  Input count : 3 (yxfb, yxfb, bfyx)
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //  Input3 : 2x 1x1 x 3
    //
    //  Input1 (yxfb):
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //
    //  Input2 (yxfb):
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Input3 (bfyx):
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //  1    0.1  :f2
    //  0.3 -0.5  :f3
    //  0   -0.2  :f4
    //  1    0.1  :f5
    //  0.3 -0.5  :f6
    //  0   -0.2  :f7
    //

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::yxfb, {2, 2, 1, 1}});
    auto input2 = engine.allocate_memory({data_types::f32, format::yxfb, {2, 3, 1, 1}});
    auto input3 = engine.allocate_memory({data_types::f32, format::bfyx, {2, 3, 1, 1}});

    set_values(input1, {0.5f, 0.7f, 0.2f, 0.4f});
    set_values(input2, {1.0f, 0.1f, 0.3f, -0.5f, 0.0f, -0.2f});
    set_values(input3, {1.0f, 0.3f, 0.0f, 0.1f, -0.5f, -0.2f});

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(input_layout("input3", input3->get_layout()));
    topology.add(concatenation("depth1", { input_info("input1"), input_info("input2"), input_info("input3") }, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_FLOAT_EQ(0.5f, output_ptr[0]);
    ASSERT_FLOAT_EQ(0.7f, output_ptr[1]);
    ASSERT_FLOAT_EQ(0.2f, output_ptr[2]);
    ASSERT_FLOAT_EQ(0.4f, output_ptr[3]);
    ASSERT_FLOAT_EQ(1.0f, output_ptr[4]);
    ASSERT_FLOAT_EQ(0.1f, output_ptr[5]);
    ASSERT_FLOAT_EQ(0.3f, output_ptr[6]);
    ASSERT_FLOAT_EQ(-0.5f, output_ptr[7]);
    ASSERT_FLOAT_EQ(0.0f, output_ptr[8]);
    ASSERT_FLOAT_EQ(-0.2f, output_ptr[9]);
    ASSERT_FLOAT_EQ(1.0f, output_ptr[10]);
    ASSERT_FLOAT_EQ(0.1f, output_ptr[11]);
    ASSERT_FLOAT_EQ(0.3f, output_ptr[12]);
    ASSERT_FLOAT_EQ(-0.5f, output_ptr[13]);
    ASSERT_FLOAT_EQ(0.0f, output_ptr[14]);
    ASSERT_FLOAT_EQ(-0.2f, output_ptr[15]);
}

TEST(concatenate_f32_gpu, test_concatenation_of_pool_and_unpool) {
    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 2, 2}});
    auto weights = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 2, 1}});

    set_values(input1, {16.0f, 32.0f, 128.0f, 256.0f});
    set_values(weights, {.1f, .2f});
    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(pooling("pool1", input_info("input1"),
                         cldnn::pooling_mode::max,
                         {1, 2}, /*kernel*/
                         {1, 1}  /*stride*/
                         ));
    topology.add(resample("unpool1", input_info("input1"), tensor(1, 1, 2, 2), 0, resample::InterpolateOp::InterpolateMode::NEAREST));
    topology.add(concatenation("concat1", { input_info("pool1"), input_info("unpool1") }, 3));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", input_info("concat1"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);

    auto outputs = network.execute({});
    auto output = outputs.at("conv").get_memory();
    std::vector<float> out_ref = {6.4f, 8.f, 51.2f, 64.f};
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(output_ptr[i], out_ref[i], 1e-3);
    }
}

TEST(depth_concatenate_f32_gpu, test03_cascade_concat_opt) {
    //  Test for cascade concatenation optimization.
    //  Despite having concatenations one after another and connected to different non padded activation primitives,
    //  graph should remove all concatenations from execution.

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 2, 2, 1}});

    set_values(input1, {16.0f, 32.0f, 128.0f, 256.0f});

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(activation("relu1", input_info("input1"), activation_func::relu));
    topology.add(activation("relu2", input_info("relu1"), activation_func::sqrt));
    topology.add(concatenation("depth1", { input_info("relu2"), input_info("relu1") }, 1));
    topology.add(activation("relu3", input_info("depth1"), activation_func::sqrt));
    topology.add(concatenation("depth2", { input_info("relu3"), input_info("depth1") }, 1));
    topology.add(activation("relu4", input_info("depth2"), activation_func::sqrt));
    topology.add(concatenation("depth3", { input_info("relu4"), input_info("depth2") }, 1));
    topology.add(activation("relu5", input_info("depth3"), activation_func::relu));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);

    auto outputs = network.execute({});
    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
    auto executed_primitives = network.get_executed_primitives();

    ASSERT_TRUE(executed_primitives.count("depth1") == 0);
    ASSERT_TRUE(executed_primitives.count("depth2") == 0);
    ASSERT_TRUE(executed_primitives.count("depth3") == 0);

    ASSERT_NEAR(1.4142f, output_ptr[0], 1e-3);
    ASSERT_NEAR(1.5422f, output_ptr[1], 1e-3);
    ASSERT_NEAR(1.8340f, output_ptr[2], 1e-3);
    ASSERT_NEAR(2.0f, output_ptr[3], 1e-3);
    ASSERT_NEAR(2.0f, output_ptr[4], 1e-3);
    ASSERT_NEAR(2.3784f, output_ptr[5], 1e-3);
    ASSERT_NEAR(3.3635f, output_ptr[6], 1e-3);
    ASSERT_NEAR(4.0f, output_ptr[7], 1e-3);
    ASSERT_NEAR(2.0f, output_ptr[8], 1e-3);
    ASSERT_NEAR(2.3784f, output_ptr[9], 1e-3);
    ASSERT_NEAR(3.3635f, output_ptr[10], 1e-3);
    ASSERT_NEAR(4.0f, output_ptr[11], 1e-3);
    ASSERT_NEAR(4.0f, output_ptr[12], 1e-3);
    ASSERT_NEAR(5.6568f, output_ptr[13], 1e-3);
    ASSERT_NEAR(11.3137f, output_ptr[14], 1e-3);
    ASSERT_NEAR(16.0f, output_ptr[15], 1e-3);
}

TEST(depth_concatenate_f32_gpu, test04_fused_relu) {
    // 2 inputs of size 3x10x10 concatenated on f axis with fused relu

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 3, 10, 10}});
    auto input2 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 3, 10, 10}});

    std::vector<float> input1_vec = rg.generate_random_1d<float>(input1->count(), -10, 10);
    set_values(input1, input1_vec);
    std::vector<float> input2_vec = rg.generate_random_1d<float>(input2->count(), -10, 10);
    set_values(input2, input2_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(concatenation("depth1", { input_info("input1"), input_info("input2") }, 1));
    topology.add(activation("relu1", input_info("depth1"), activation_func::relu));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "relu1");

    auto output = outputs.at("relu1").get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    unsigned int input_element_count = 300;
    for (unsigned int i = 0; i < 600; i++) {
        if (i < input_element_count)
            ASSERT_FLOAT_EQ(input1_vec[i] < 0.0f ? 0.0f : input1_vec[i], output_ptr[i]);
        else
            ASSERT_FLOAT_EQ(input2_vec[i - input_element_count] < 0.0f ? 0.0f : input2_vec[i - input_element_count], output_ptr[i]);
    }
}

TEST(depth_concatenate_f32_gpu, test05_different_formats) {
    // 2 inputs of size 3x2x2 concatenated on f axis

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({data_types::f32, format::bfyx, {1, 3, 2, 2}});
    auto input2 = engine.allocate_memory({data_types::f32, format::yxfb, {1, 3, 2, 2}});

    set_values(input1, {1.0f, 1.0f, 1.0f, 1.0f,
                        2.0f, 2.0f, 2.0f, 2.0f,
                        3.0f, 3.0f, 3.0f, 3.0f});
    set_values(input2, {-1.0f, -2.0f, -3.0f,
                        -1.0f, -2.0f, -3.0f,
                        -1.0f, -2.0f, -3.0f,
                        -1.0f, -2.0f, -3.0f});

    std::vector<float> out_ref = {
        1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f,
        -1.0f, -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f, -2.0f,
        -3.0f, -3.0f, -3.0f, -3.0f};

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(reshape("reshape1", input_info("input1"), {1, 3, 2, 2}));
    topology.add(reshape("reshape2", input_info("input2"), {1, 3, 2, 2}));
    topology.add(concatenation("depth1", { input_info("reshape1"), input_info("reshape2") }, 1));
    topology.add(reorder("output", input_info("depth1"), format::bfyx, data_types::f32));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    int cntr = 0;
    for (float val : output_ptr) {
        ASSERT_EQ(val, out_ref[cntr++]);
    }
}

TEST(depth_concatenate_f32_gpu, test06_padded_input) {
    // input1 - activation - concatenation - concatenation - reorder
    //                     /                /
    // input2 - activation -  convolution* /
    //
    // *Convolution has input offset so it should be propagated, both back to reorders and to second concatenation.
    // As a result both concatenations should be optimized out and convolution should use optimized implementation.

    const int32_t input_f = 32;
    const int32_t output_f = 3 * input_f;

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad) {
        return;
    }
    auto input1 = engine.allocate_memory({ data_types::f16, format::fs_b_yx_fsv32, {1, input_f, 1, 1} });
    auto input2 = engine.allocate_memory({ data_types::f16, format::fs_b_yx_fsv32, {1, input_f, 1, 1} });

    auto input1_data = rg.generate_random_4d<ov::float16>(1, input_f, 1, 1, -1, 1);
    auto input2_data = rg.generate_random_4d<ov::float16>(1, input_f, 1, 1, -1, 1);
    set_values(input1, flatten_4d(format::bfyx, input1_data));
    set_values(input2, flatten_4d(format::bfyx, input2_data));

    auto weights = engine.allocate_memory({ data_types::f16, format::oiyx, {input_f, input_f, 3, 3} });
    // Construct weights for convolution that just double input values.
    VVVVF<ov::float16> weights_data;
    weights_data.resize(input_f);
    for (size_t oi = 0; oi < input_f; ++oi) {
        weights_data[oi].resize(input_f, VVF<ov::float16>(3, VF<ov::float16>(3, ov::float16(0.f))));
        weights_data[oi][oi][1][1] = 2.f;
    }
    set_values(weights, flatten_4d(format::bfyx, weights_data));

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(activation("actv1", input_info("input1"), activation_func::linear, { 0.75f, 0.0f }));
    topology.add(activation("actv2", input_info("input2"), activation_func::linear, { 0.5f, 0.0f }));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", input_info("actv2"), { "weights" }, {}, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    topology.add(concatenation("depth1", { input_info("actv1"), input_info("actv2") }, 1));
    topology.add(concatenation("depth2", { input_info("depth1"), input_info("conv") }, 1));
    topology.add(reorder("output", input_info("depth2"), format::bfyx, data_types::f32));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", ov::intel_gpu::ImplementationDesc{format::fs_b_yx_fsv32, ""} } }));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");
    // Check that all concatenations have been optimized out.
    auto executed_primitives = network.get_executed_primitives();
    ASSERT_TRUE(executed_primitives.count("depth1") == 0);
    ASSERT_TRUE(executed_primitives.count("depth2") == 0);
    // Check that convolution was able to use optimzed kernel.
    for (auto& info : network.get_primitives_info()) {
        if (info.original_id == "conv") {
            ASSERT_TRUE(info.kernel_id.find("ref") == std::string::npos) << " selected kernel: " << info.kernel_id;
        }
    }

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output->count(), output_f);
    for (size_t i = 0; i < output_f; ++i) {
        auto& val = output_ptr[i];
        float ref;
        if (i < input_f)
            ref = 0.75f * static_cast<float>(input1_data[0][i % input_f][0][0]);
        else if (i < 2 * input_f)
            ref = 0.5f * static_cast<float>(input2_data[0][i % input_f][0][0]);
        else
            ref = static_cast<float>(input2_data[0][i % input_f][0][0]);

        ASSERT_EQ(val, ref) << " at i=" << i;
    }
}

TEST(depth_concatenate_f32_gpu, test07_padded_output) {
    // input1 - activation - concatenation - convolution - reorder
    // input2 - activation /
    //
    // *Convolution has input offset so it should be propagated back to activations.
    const int32_t input_f = 32;
    const int32_t output_f = 2 * input_f;

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad) {
        // Currently, oneDNN does NOT support input/output padding
        return;
    }

    auto input1 = engine.allocate_memory({ data_types::f16, format::fs_b_yx_fsv32, {1, input_f, 1, 1} });
    auto input2 = engine.allocate_memory({ data_types::f16, format::fs_b_yx_fsv32, {1, input_f, 1, 1} });

    auto input1_data = rg.generate_random_4d<ov::float16>(1, input_f, 1, 1, -1, 1);
    auto input2_data = rg.generate_random_4d<ov::float16>(1, input_f, 1, 1, -1, 1);
    set_values(input1, flatten_4d(format::bfyx, input1_data));
    set_values(input2, flatten_4d(format::bfyx, input2_data));

    auto weights = engine.allocate_memory({ data_types::f16, format::oiyx, {output_f, output_f, 3, 3} });
    // Construct weights for convolution that just double input values.
    VVVVF<ov::float16> weights_data;
    weights_data.resize(output_f);
    for (size_t oi = 0; oi < output_f; ++oi) {
        weights_data[oi].resize(output_f, VVF<ov::float16>(3, VF<ov::float16>(3, ov::float16(0.f))));
        weights_data[oi][oi][1][1] = 2.f;
    }
    set_values(weights, flatten_4d(format::bfyx, weights_data));

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(activation("actv1", input_info("input1"), activation_func::linear, { 0.75f, 0.0f }));
    topology.add(activation("actv2", input_info("input2"), activation_func::linear, { 0.5f, 0.0f }));
    topology.add(concatenation("depth1", { input_info("actv1"), input_info("actv2") }, 1));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", input_info("depth1"), { "weights" }, {}, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    topology.add(reorder("output", input_info("conv"), format::bfyx, data_types::f32));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", ov::intel_gpu::ImplementationDesc{format::fs_b_yx_fsv32, ""} } }));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");
    // Check that all concatenations have been optimized out.
    auto executed_primitives = network.get_executed_primitives();
    ASSERT_TRUE(executed_primitives.count("depth1") == 0);
    // Check that convolution was able to use optimzed kernel.
    for (auto& info : network.get_primitives_info()) {
        if (info.original_id == "conv") {
            ASSERT_TRUE(info.kernel_id.find("ref") == std::string::npos) << " selected kernel: " << info.kernel_id;
        }
    }

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output->count(), output_f);
    for (size_t i = 0; i < output_f; ++i) {
        auto& val = output_ptr[i];
        float ref;
        if (i < input_f)
            ref = 1.5f * static_cast<float>(input1_data[0][i % input_f][0][0]);
        else
            ref = static_cast<float>(input2_data[0][i % input_f][0][0]);

        ASSERT_EQ(val, ref) << " at i=" << i;
    }
}

TEST(depth_concatenate_f32_gpu, test07_concat_is_output) {
    // input1 - activation - concatenation
    // input2 - activation /
    //
    // As concatenation is output it should not be optimizex out.
    const int32_t input_f = 16;
    const int32_t output_f = 2 * input_f;

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, {1, input_f, 1, 1} });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, {1, input_f, 1, 1} });

    auto input1_data = rg.generate_random_4d<float>(1, input_f, 1, 1, -1, 1);
    auto input2_data = rg.generate_random_4d<float>(1, input_f, 1, 1, -1, 1);
    set_values(input1, flatten_4d(format::bfyx, input1_data));
    set_values(input2, flatten_4d(format::bfyx, input2_data));

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(activation("actv1", input_info("input1"), activation_func::linear, { 0.75f, 0.0f }));
    topology.add(activation("actv2", input_info("input2"), activation_func::linear, { 0.5f, 0.0f }));
    topology.add(concatenation("depth1", { input_info("actv1"), input_info("actv2") }, 1));

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "depth1");
    // Check that concatenation haven't been optimized out.
    auto executed_primitives = network.get_executed_primitives();
    ASSERT_TRUE(executed_primitives.count("depth1") == 1);

    auto output = outputs.at("depth1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output->count(), output_f);
    for (size_t i = 0; i < output_f; ++i) {
        auto& val = output_ptr[i];
        float ref;
        if (i < input_f)
            ref = 0.75f * input1_data[0][i % input_f][0][0];
        else
            ref = 0.5f * input2_data[0][i % input_f][0][0];

        ASSERT_EQ(val, ref) << " at i=" << i;
    }
}

TEST(depth_concatenate_f32_gpu, concat_with_different_format_inputs) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    const int in1_f = 2, in2_f = 1;
    const int b = 2, x = 2, y = 4;
    auto input1 = engine.allocate_memory({ data_types::f32, format::yxfb,{ b, in1_f, y, x } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ b, in2_f, y, x } });
    unsigned input2_start_value = (unsigned)input1->count() + 1;

    std::vector<float> in1(input1->count());
    std::vector<float> in2(input2->count());

    for (unsigned i = 0; i < input1->count(); i++)
    {
        in1[i] = (float)(i + 1);
    }

    for (unsigned i = 0; i < input2->count(); i++)
    {
        in2[i] = (float)(i + input2_start_value);
    }

    set_values(input1, in1);
    set_values(input2, in2);

    // Special constrution of topology to run buffer fusing optimization
    // for concatenation with different format inputs
    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(concatenation("depth1", { input_info("input1") }, 1));
    topology.add(concatenation("depth2", { input_info("input2") }, 1));
    // In the step below there will be run of buffer fusing optimization for concatenation with
    // Input1 YXFB, Input2 BFYX and Output YXFB
    topology.add(concatenation("depth3", { input_info("depth1"), input_info("depth2") }, 1));
    topology.add(concatenation("depth4", { input_info("depth3") }, 1));

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "depth4");

    auto output = outputs.at("depth4").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    int input1_values_count = in1_f * x;
    int input2_values_count = in2_f * x;
    int all_values_count = input1_values_count + input2_values_count;
    int input2_batch_offset = x * y;
    int out_offset = 0;

    for (unsigned i = 0; i < input1->count(); i++)
    {
        int value = i + 1;
        ASSERT_FLOAT_EQ(float(value), output_ptr[out_offset++]);

        if ((value % input1_values_count) == 0)
        {
            out_offset += input2_values_count;
        }
    }

    out_offset = input1_values_count;
    for (unsigned i = 0; i < input2->count() / b; i++)
    {
        for (unsigned j = 0; j < b; j++)
        {
            int value = i + input2_start_value + j * input2_batch_offset;
            ASSERT_FLOAT_EQ(float(value), output_ptr[out_offset++]);

            if ((out_offset % all_values_count) == 0)
            {
                out_offset += input1_values_count;
            }
        }
    }
}

TEST(depth_concatenate_f32_gpu, concat_with_reshape_input) {

    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2,4,1,2 } });

    std::vector<float> values = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f ,
        0.11f, 0.22f, 0.33f, 0.44f ,
        0.55f, 0.66f, 0.77f, 0.88f };

    set_values(input1, values);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(reshape("reshape", input_info("input1"), tensor(2, 1, 4, 2)));
    topology.add(concatenation("depth1", { input_info("reshape") }, 1));
    topology.add(concatenation("depth2", { input_info("depth1") }, 1));

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);

    auto outputs = network.execute({});
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "depth2");

    auto output = outputs.at("depth2").get_memory();

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++)
    {
        ASSERT_FLOAT_EQ(values[i], output_ptr[i]);
    }
}

TEST(depth_concatenate_i32_gpu, optimize_data01) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    auto input = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 1, 1}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout()));
    topology.add(cldnn::concatenation("int1", { input_info("input") }, 1));
    topology.add(cldnn::concatenation("result1", { input_info("int1") }, 1));
    topology.add(cldnn::concatenation("result2", { input_info("int1") }, 1));

    std::vector<int> input_data = {4};
    std::vector<int> out_data = {4};
    set_values(input, input_data);

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    for (auto& it : outputs) {
        cldnn::mem_lock<int> output_ptr(it.second.get_memory(), get_test_stream());
        ASSERT_EQ(output_ptr[0], out_data[0]);
    }
}

TEST(depth_concatenate_i32_gpu, optimize_data02) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    auto input1 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});
    auto input3 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});
    auto input4 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});

    topology topology;
    topology.add(
        input_layout("input1", input1->get_layout()));
    topology.add(
        input_layout("input2", input2->get_layout()));
    topology.add(
        input_layout("input3", input3->get_layout()));
    topology.add(
        input_layout("input4", input4->get_layout()));

    topology.add(cldnn::concatenation("concat1", { input_info("input1"), input_info("input2") }, 3));
    topology.add(cldnn::concatenation("concat2", { input_info("input3"), input_info("input4") }, 3));
    topology.add(cldnn::concatenation("concat3", { input_info("input2"), input_info("input4") }, 3));

    topology.add(cldnn::concatenation("concat4", { input_info("concat1"), input_info("concat2") }, 3));
    topology.add(cldnn::concatenation("concat5", { input_info("concat2"), input_info("concat3") }, 3));

    topology.add(cldnn::concatenation("concat6", { input_info("concat4"), input_info("concat5") }, 3));

    std::vector<int> input_data1 =
        {1, 2,
         3, 4};

    std::vector<int> input_data2 =
        {5, 6,
         7, 8};

    std::vector<int> input_data3 =
        {9, 10,
         11, 12};

    std::vector<int> input_data4 =
        {12, 14,
         15, 16};

    std::vector<int> c6_data =
        {1, 2, 5, 6, 9, 10, 12, 14, 9, 10, 12, 14, 5, 6, 12, 14,
         3, 4, 7, 8, 11, 12, 15, 16, 11, 12, 15, 16, 7, 8, 15, 16};

    set_values(input1, input_data1);
    set_values(input2, input_data2);
    set_values(input3, input_data3);
    set_values(input4, input_data4);

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);
    network.set_input_data("input4", input4);
    auto outputs = network.execute();

    cldnn::mem_lock<int> output_concat6(outputs.at("concat6").get_memory(), get_test_stream());

    for (size_t i = 0; i < output_concat6.size(); i++) {
        ASSERT_EQ(output_concat6[i], c6_data[i]);
    }
}

TEST(depth_concatenate_i32_gpu, optimize_data03) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    auto input1 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});

    topology topology;
    topology.add(
        input_layout("input1", input1->get_layout()));

    topology.add(cldnn::concatenation("concat1", { input_info("input1") }, 3));

    topology.add(cldnn::concatenation("concat2", { input_info("concat1") }, 3));
    topology.add(cldnn::concatenation("concat3", { input_info("concat1") }, 3));

    topology.add(cldnn::concatenation("concat4", { input_info("concat3") }, 3));

    std::vector<int> input_data1 =
        {1, 2,
         3, 4};

    std::vector<int> output_data =
        {1, 2,
         3, 4};

    set_values(input1, input_data1);

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);

    auto outputs = network.execute();

    for (auto& it : outputs) {
        cldnn::mem_lock<int> output_ptr(it.second.get_memory(), get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_EQ(output_ptr[i], output_data[i]);
        }
    }
}

TEST(depth_concatenate_i32_gpu, optimize_data04) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    auto input1 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});

    topology topology;
    topology.add(
        input_layout("input1", input1->get_layout()));

    topology.add(cldnn::concatenation("concat1", { input_info("input1") }, 3));

    topology.add(cldnn::concatenation("concat2", { input_info("concat1") }, 3));
    topology.add(cldnn::concatenation("concat3", { input_info("concat1") }, 3));

    topology.add(cldnn::concatenation("concat4", { input_info("concat2"), input_info("concat3") }, 3));

    std::vector<int> input_data1 =
        {1, 2,
         3, 4};

    std::vector<int> output_data =
        {1, 2, 1, 2,
         3, 4, 3, 4};

    set_values(input1, input_data1);

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);

    auto outputs = network.execute();

    for (auto& it : outputs) {
        cldnn::mem_lock<int> output_ptr(it.second.get_memory(), get_test_stream());
        for (size_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_EQ(output_ptr[i], output_data[i]);
        }
    }
}

TEST(depth_concatenate_i32_gpu, optimize_data05) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    auto input1 = engine.allocate_memory({data_types::i32, format::bfyx, {1, 1, 2, 2}});

    topology topology;
    topology.add(
        input_layout("input1", input1->get_layout()));

    topology.add(cldnn::concatenation("concat1", { input_info("input1") }, 3));

    topology.add(cldnn::concatenation("concat2", { input_info("concat1") }, 3));
    topology.add(cldnn::concatenation("concat3", { input_info("concat1") }, 3));

    topology.add(cldnn::concatenation("concat4", { input_info("concat2"), input_info("concat3") }, 3));
    topology.add(cldnn::concatenation("concat5", { input_info("concat1"), input_info("concat4") }, 3));

    std::vector<int> input_data1 =
        {1, 2,
         3, 4};

    std::vector<int> c5_data =
        {1, 2, 1, 2, 1, 2,
         3, 4, 3, 4, 3, 4};

    set_values(input1, input_data1);

    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);

    auto outputs = network.execute();

    cldnn::mem_lock<int> output_concat5(outputs.at("concat5").get_memory(), get_test_stream());

    for (size_t i = 0; i < output_concat5.size(); i++) {
        ASSERT_EQ(output_concat5[i], c5_data[i]);
    }
}

template <typename T>
void test_depth_concatenate_f32_gpu_basic_bfwzyx_along_w(bool is_caching_test) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    const int b = 2;
    const int f = 3;
    const int x = 2;
    const int y = 5;
    const int z = 7;
    const int w = 9;

    auto input1_layout = layout(data_types::f32, format::bfwzyx, tensor{batch(b), feature(f), spatial(x, y, z, w)});
    auto input1 = engine.allocate_memory(input1_layout);
    auto output_layout = layout(data_types::f32, format::bfwzyx, tensor{batch(b), feature(f), spatial(x, y, z, w * 2)});

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(concatenation("concat", { input_info("input1"), input_info("input1") }, 2));

    auto input_data = rg.generate_random_1d<T>(input1->count(), -1, 1);

    auto expected_output = std::vector<T>(input1->count() * 2);

    for (int bi = 0; bi < b; bi++)
        for (int fi = 0; fi < f; fi++)
            for (int wi = 0; wi < w * 2; wi++)
                for (int zi = 0; zi < z; zi++)
                    for (int yi = 0; yi < y; yi++)
                        for (int xi = 0; xi < x; xi++) {
                            auto out_offset = output_layout.get_linear_offset(tensor{batch(bi), feature(fi), spatial(xi, yi, zi, wi)});
                            auto in_offset = input1_layout.get_linear_offset(tensor{batch(bi), feature(fi), spatial(xi, yi, zi, wi % w)});

                            expected_output[out_offset] = input_data[in_offset];
                        }

    set_values(input1, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input1", input1);

    auto outputs = network->execute();

    cldnn::mem_lock<T> output_concat(outputs.at("concat").get_memory(), get_test_stream());

    ASSERT_EQ(output_concat.size(), expected_output.size());
    for (size_t i = 0; i < output_concat.size(); i++) {
        ASSERT_EQ(output_concat[i], expected_output[i]);
    }
}

TEST(depth_concatenate_f32_gpu, basic_bfwzyx_along_w) {
    test_depth_concatenate_f32_gpu_basic_bfwzyx_along_w<float>(false);
}

TEST(export_import_depth_concatenate_f32_gpu, basic_bfwzyx_along_w) {
    test_depth_concatenate_f32_gpu_basic_bfwzyx_along_w<float>(true);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO: this should be done using TEST_P or some equivallent construct
static network::ptr setup_depth_concatatenate_network(const std::vector<data_types> dts, const std::vector<tensor> ts, const std::vector<cldnn::format> fmt) {
    assert(dts.size() == ts.size());
    const size_t sz = ts.size();

    auto& engine = get_test_engine();
    topology topology;

    std::vector<input_info> input_names;
    input_names.resize(sz);

    for (size_t i = 0; i < sz; ++i) {
        auto input = engine.allocate_memory({dts[i], fmt[i], ts[i]});

        input_names[i] = input_info("input");
        input_names[i].pid += std::to_string(i);

        topology.add(input_layout(input_names[i].pid, input->get_layout()));
    }
    //TODO: ask Uzi if something tests cases where there's missing input_names (nodes not present in the topology, etc.)
    topology.add(concatenation("depth_concat_node", input_names, 1));

    return network::build_network(engine, topology, get_test_default_config(engine));
}

TEST(NegativeDepthConcatenateTest, DISABLED_TestAll) {
    auto d = data_types::f32;
    auto od = data_types::f16;

    auto f = format::bfyx;

    std::vector<int> t{1, 2, 3, 4};
    std::vector<int> t0{7, 2, 3, 4};
    std::vector<int> t1{1, 2, 7, 4};
    std::vector<int> t2{1, 2, 3, 7};

    //TODO: should be ASSERT_THROW(statement, exception_type) - but what exception type?
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({}, {}, {}));

    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, od}, {tensor(t), tensor(t)}, {f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d}, {tensor(t), tensor(t0)}, {f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d}, {tensor(t), tensor(t1)}, {f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d}, {tensor(t), tensor(t2)}, {f, f}));

    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, od, d}, {tensor(t), tensor(t), tensor(t)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, od}, {tensor(t), tensor(t), tensor(t)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, d}, {tensor(t), tensor(t0), tensor(t)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, d}, {tensor(t), tensor(t1), tensor(t)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, d}, {tensor(t), tensor(t2), tensor(t)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, d}, {tensor(t), tensor(t), tensor(t0)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, d}, {tensor(t), tensor(t), tensor(t1)}, {f, f, f}));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({d, d, d}, {tensor(t), tensor(t), tensor(t2)}, {f, f, f}));
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class depth_concatenate_test : public tests::generic_test {
public:
    static void TearDownTestCase() {
        all_generic_params.clear();
        all_layer_params.clear();
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params(int i) {
        std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;

        switch (i) {
            case 1:
                all_layer_params.emplace_back(new concatenation("depth_concatenate", { input_info("input0") }, 1));
                break;
            case 2:
                all_layer_params.emplace_back(new concatenation("depth_concatenate", { input_info("input0"), input_info("input1") }, 1));
                break;
            case 3:
                all_layer_params.emplace_back(new concatenation("depth_concatenate", { input_info("input0"), input_info("input1"), input_info("input2") }, 1));
                break;
            default:
                assert(0);
        }

        return all_layer_params;
    }

    static std::vector<std::shared_ptr<tests::test_params>> generate_generic_test_params(int input_count) {
        std::vector<std::shared_ptr<tests::test_params>> all_generic_params;

        auto data_types = test_data_types();

        for (cldnn::data_types dt : data_types)
            for (int32_t b : test_batch_sizes)
                for (tensor& t : test_input_sizes) {
                    const int w = t.spatial[0];
                    const int h = t.spatial[1];

                    switch (input_count) {
                        case 1:
                            for (auto f0 : test_feature_sizes) {
                                std::shared_ptr<tests::test_params> tp = std::make_shared<test_params>();
                                tp->data_type = dt;

                                tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(b, f0, w, h)));

                                all_generic_params.emplace_back(tp);
                            }
                            break;
                        case 2:
                            for (auto f0 : test_feature_sizes)
                                for (auto f1 : test_feature_sizes) {
                                    std::shared_ptr<tests::test_params> tp = std::make_shared<test_params>();
                                    tp->data_type = dt;

                                    tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(b, f0, w, h)));
                                    tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(b, f1, w, h)));

                                    all_generic_params.emplace_back(tp);
                                }
                            break;
                        case 3:
                            for (auto f0 : test_feature_sizes)
                                for (auto f1 : test_feature_sizes)
                                    for (auto f2 : test_feature_sizes) {
                                        std::shared_ptr<tests::test_params> tp = std::make_shared<test_params>();
                                        tp->data_type = dt;

                                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(b, f0, w, h)));
                                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(b, f1, w, h)));
                                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(b, f2, w, h)));

                                        all_generic_params.emplace_back(tp);
                                    }
                            break;
                        default:
                            assert(0);
                    }
                }

        return all_generic_params;
    }

    static std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> generate_all_test_params() {
        std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> res;

        for (int i = 1; i <= 3; ++i) {
            auto tpv = generate_generic_test_params(i);
            auto pv = generate_specific_test_params(i);

            all_generic_params.insert(all_generic_params.end(), tpv.begin(), tpv.end());
            all_layer_params.insert(all_layer_params.end(), pv.begin(), pv.end());

            for (auto& tp : tpv)
                for (auto& p : pv)
                    res.emplace_back(tp, p);
        }

        return res;
    }

    bool is_format_supported(cldnn::format format) override {
        return format == cldnn::format::bfyx;
    }

    cldnn::tensor get_expected_output_tensor() override {
        cldnn::tensor::value_type features = 0;
        for (const auto& t : generic_params->input_layouts) {
            features += t.feature();
        }

        const auto& t = generic_params->input_layouts[0].get_tensor();
        return {t.batch[0], features, t.spatial[0], t.spatial[1]};
    }

    template <typename Type>
    memory::ptr generate_reference_typed(const std::vector<memory::ptr>& inputs) {
        assert(!inputs.empty());

        const int in_b = inputs[0]->get_layout().batch();
        const int in_h = inputs[0]->get_layout().spatial(1);
        const int in_w = inputs[0]->get_layout().spatial(0);

        int out_f = 0;

        for (const memory::ptr& input : inputs) {
            assert(input->get_layout().batch() == in_b);
            assert(input->get_layout().spatial(1) == in_h);
            assert(input->get_layout().spatial(0) == in_w);

            out_f += input->get_layout().feature();

            assert(input->get_layout().data_type == inputs[0]->get_layout().data_type);
            assert(input->get_layout().format.value == inputs[0]->get_layout().format.value);
        }

        //Output is bfyx
        auto output = engine.allocate_memory(cldnn::layout(inputs[0]->get_layout().data_type, cldnn::format::bfyx, tensor(in_b, out_f, in_w, in_h)));
        cldnn::mem_lock<Type> out_mem(output, get_test_stream());

        int out_f_off = 0;
        for (const memory::ptr& input : inputs) {
            const auto input_desc = get_linear_memory_desc(input->get_layout());
            const auto output_desc = get_linear_memory_desc(output->get_layout());

            const int in_f = input->get_layout().feature();
            cldnn::mem_lock<Type> in_mem(input, get_test_stream());

            for (int n = 0; n < in_b; ++n)
                for (int f = 0; f < in_f; ++f)
                    for (int y = 0; y < in_h; ++y)
                        for (int x = 0; x < in_w; ++x) {
                            const size_t in_idx = get_linear_index(input->get_layout(), n, f, y, x, input_desc);
                            const size_t out_idx = get_linear_index(output->get_layout(), n, out_f_off + f, y, x, output_desc);

                            out_mem[out_idx] = in_mem[in_idx];
                        }

            out_f_off += in_f;
        }

        return output;
    }

    virtual memory::ptr generate_reference(const std::vector<memory::ptr>& inputs) override {
        if (generic_params->data_type == data_types::f32) {
            return generate_reference_typed<float>(inputs);
        } else {
            return generate_reference_typed<ov::float16>(inputs);
        }
    }

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>>& info) {
        std::stringstream res;

        const auto& p = std::get<0>(info.param);

        assert(p->data_type == data_types::f32 ||
               p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i) {
            const auto chans = p->fmt.order();

            res << "_"
                << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].get_tensor().sizes(p->fmt).size(); ++j) {
                res << chans[j] << p->input_layouts[i].get_tensor().sizes(p->fmt)[j];
            }
        }

        return res.str();
    }

private:
    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;
};

std::vector<std::shared_ptr<cldnn::primitive>> depth_concatenate_test::all_layer_params = {};
std::vector<std::shared_ptr<tests::test_params>> depth_concatenate_test::all_generic_params = {};

TEST_P(depth_concatenate_test, DEPTHCONCATENATE) {
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_DEPTHCONCATENATE,
                        depth_concatenate_test,
                        ::testing::ValuesIn(depth_concatenate_test::generate_all_test_params()),
                        depth_concatenate_test::custom_param_name);
