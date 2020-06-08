/*
// Copyright (c) 2016-2020 Intel Corporation
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
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/convolution.hpp"
#include "api/eltwise.hpp"
#include "api/reorder.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/data.hpp>
#include <api/depth_to_space.hpp>

#include <api_extension/fused_conv_eltwise.hpp>

#include <cassert>
#include <cmath>
#include <gmock/gmock.h>
#include <limits>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(fused_conv_eltwise, basic_0)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out", "eltwise", format::bfyx, data_types::f32));

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, basic_image2d)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 4, 128, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 3, 256, 4 } });
    auto weights = memory::allocate(engine, { data_types::f16, format::bfyx, { 12, 4, 1, 1 } });

    auto input_data1 = generate_random_4d<FLOAT16>(1, 4, 2, 128, -1, 1);
    auto input_data1_bfyx = flatten_4d(format::bfyx, input_data1);
    set_values(input, input_data1_bfyx);

    auto input_data2 = generate_random_4d<FLOAT16>(1, 3, 4, 256, -1, 1);
    auto input_data2_bfyx = flatten_4d(format::bfyx, input_data2);
    set_values(input2, input_data2_bfyx);

    auto weights_data= generate_random_4d<FLOAT16>(12, 4, 1, 1, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    set_values(weights, weights_data_bfyx);

    topology topology_act(
        input_layout("input", input.get_layout()),
        input_layout("input2", input2.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        depth_to_space("depth_to_space", "conv", 2, depth_to_space_mode::blocks_first),
        eltwise("eltwise", "input2", "depth_to_space", eltwise_mode::sum)
    );

    build_options opt_act;
    opt_act.set_option(build_option::optimize_data(true));
    network network_act(engine, topology_act, opt_act);
    network_act.set_input_data("input", input);
    network_act.set_input_data("input2", input2);

    auto outputs_act = network_act.execute();
    EXPECT_EQ(outputs_act.size(), size_t(1));
    EXPECT_EQ(outputs_act.begin()->first, "eltwise");

    auto output_act = outputs_act.begin()->second.get_memory();
    auto&& out_act_layout = output_act.get_layout();
    auto out_act_ptr = output_act.pointer<uint8_t>();

    topology topology_ref(
        input_layout("input", input.get_layout()),
        input_layout("input2", input2.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        depth_to_space("depth_to_space", "conv", 2, depth_to_space_mode::blocks_first),
        eltwise("eltwise", "input2", "depth_to_space", eltwise_mode::sum),
        reorder("out", "eltwise", format::image_2d_rgba, data_types::u8));

    build_options opt_ref;
    opt_ref.set_option(build_option::optimize_data(false));
    network network_ref(engine, topology_ref, opt_ref);
    network_ref.set_input_data("input", input);
    network_ref.set_input_data("input2", input2);

    auto outputs_ref = network_ref.execute();
    EXPECT_EQ(outputs_ref.size(), size_t(1));
    EXPECT_EQ(outputs_ref.begin()->first, "out");

    auto output_ref = outputs_ref.begin()->second.get_memory();
    auto&& out_ref_layout = output_ref.get_layout();
    auto out_ref_ptr = output_ref.pointer<uint8_t>();

    for (int i = 0;i < 3 * 256 * 4;i++) {
        EXPECT_EQ(out_act_ptr[i], out_ref_ptr[i]);
    }
}

TEST(fused_conv_eltwise, dont_fuse_if_conv_elt_are_outputs)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
        });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("out", "input", "conv", eltwise_mode::sum));

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

template<typename InputTy,
         typename OutputTy>
class FusedConvTest : public testing::Test
{
protected:
    static constexpr bool is_pure_float = std::is_same<InputTy, float>::value;
    using OutputPreActivationTy = typename std::conditional<is_pure_float, float, int32_t>::type;
    using WeightsTy = typename std::conditional<is_pure_float, float, int8_t>::type;
    using BiasesTy = typename std::conditional<is_pure_float, float, int32_t>::type;

    topology the_topology;

    std::vector<InputTy> input_values;
    std::vector<WeightsTy> weights_values;
    std::vector<BiasesTy> biases_values;
    // Note, not all of the quantization/calibration factors are used in all the
    // tests. However, I didn't come up with a way to correctly reflect that
    // while unifying the boileplate testing code.
    static constexpr float ignore = std::numeric_limits<float>::quiet_NaN();

    // Eltw part.
    std::vector<InputTy> non_conv_input_values;
    std::vector<OutputPreActivationTy> output_pre_relu;

    void add_feature(std::vector<InputTy> input,
                     std::vector<WeightsTy> weights,
                     BiasesTy bias,
                     std::vector<InputTy> non_conv_input,
                     std::vector<OutputPreActivationTy> output)
    {
        assert(non_conv_input.size() == output.size());
        input_values.insert(input_values.end(), input.begin(), input.end());
        weights_values.insert(
            weights_values.end(), weights.begin(), weights.end());
        biases_values.push_back(bias);
        non_conv_input_values.insert(non_conv_input_values.end(),
                                     non_conv_input.begin(),
                                     non_conv_input.end());
        output_pre_relu.insert(
            output_pre_relu.end(), output.begin(), output.end());
    }

    void do_test(const fused_conv_eltwise& fused_prim)
    {
        const auto& engine = get_test_engine();

        int n_features = static_cast<int>(biases_values.size());

        auto input_shape = tensor(1, n_features, 4, 1);
        auto weights_shape = tensor(n_features, n_features, 3, 1);
        auto biases_shape = tensor(1, n_features, 1, 1);
        auto sum_input_shape = tensor(1, n_features, 2, 1);

        auto input = memory::allocate(
            engine,
            {type_to_data_type<InputTy>::value, format::bfyx, input_shape});
        auto weights = memory::allocate(
            engine,
            {type_to_data_type<WeightsTy>::value, format::bfyx, weights_shape});

        auto biases = memory::allocate(
            engine,
            {type_to_data_type<BiasesTy>::value, format::bfyx, biases_shape});
        auto sum_input = memory::allocate(
            engine,
            {type_to_data_type<InputTy>::value, format::bfyx, sum_input_shape});

        set_values(input, input_values);
        std::vector<WeightsTy> post_processed_weights_values(n_features
                                                             * n_features * 3);
        for (int output_feature = 0; output_feature < n_features; ++output_feature)
            for (int input_feature = 0; input_feature < n_features;
                 ++input_feature)
                for (int x = 0; x < 3; ++x)
                {
                    int idx =
                        output_feature * n_features * 3 + input_feature * 3 + x;
                    if (input_feature == output_feature)
                        post_processed_weights_values[idx] =
                            weights_values[input_feature * 3 + x];
                    else
                        post_processed_weights_values[idx] = 0;
                }
        set_values(weights, post_processed_weights_values);
        set_values(biases, biases_values);
        set_values(sum_input, non_conv_input_values);

        the_topology.add(input_layout("input", input.get_layout()));
        the_topology.add(data("weights", weights));
        the_topology.add(data("biases", biases));
        the_topology.add(data("sum_input", sum_input));
        the_topology.add(fused_prim);

        build_options opts;
        opts.set_option(build_option::optimize_data(false));

        network network(engine, the_topology, opts);
        network.set_input_data("input", input);

        auto outputs = network.execute();

        auto output_memory = outputs.at("fused_conv").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<OutputTy>();
        int y_size = output_layout.size.spatial[1];
        int x_size = output_layout.size.spatial[0];
        int f_size = output_layout.size.feature[0];
        int b_size = output_layout.size.batch[0];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(y_size, 1);
        EXPECT_EQ(x_size, 2);
        EXPECT_EQ(f_size, n_features);
        EXPECT_EQ(b_size, 1);

        for (int f = 0; f < f_size; f++)
            for (int x = 0; x < x_size; ++x)
            {
                // printf("f: %d, x: %d\n", f, x);
                OutputPreActivationTy expected =
                    pre_relu_to_output(output_pre_relu[f * x_size + x]);
                auto actual = static_cast<OutputPreActivationTy>(
                    output_ptr[f * x_size + x]);
                expect_eq(expected, actual);
            }
    }

private:
    template<typename T = OutputPreActivationTy>
    static typename std::enable_if<std::is_floating_point<T>::value>::type
    expect_eq(const OutputPreActivationTy& lhs, const OutputPreActivationTy& rhs)
    {
        EXPECT_NEAR(lhs, rhs, 0.001f);
    }

    template<typename T = OutputPreActivationTy>
    static typename std::enable_if<std::is_integral<T>::value>::type
    expect_eq(const OutputPreActivationTy& lhs, const OutputPreActivationTy& rhs)
    {
        EXPECT_EQ(lhs, rhs);
    }

    template <typename T>
    static T pre_relu_to_output(T pre_relu) {
      // No std::clamp before C++17 :(
      return std::min(
          static_cast<T>(std::numeric_limits<OutputTy>::max()),
          std::max(static_cast<T>(std::numeric_limits<OutputTy>::lowest()),
                   std::max(static_cast<T>(0), pre_relu)));
    }
};

class FusedConvTest_all_float : public FusedConvTest<float, float>
{};

TEST_F(FusedConvTest_all_float, DISABLED_basic) {
    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                1.0f,                         // bias
                {-10.0f, -10.0f},             // non_conv_input
                {241.0f, 242.0f});            // output_pre_relu

    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                0.0f,                         // bias
                {-10.0f, -11.0f},             // non_conv_input
                {480.0f, 480.0f});            // output_pre_relu

    do_test(fused_conv_eltwise("fused_conv",
                               "input",
                               "sum_input",
                               eltwise_mode::sum,
                               {"weights"},
                               {"biases"},
                               {},
                               {},
                               1.0f, // conv_i_quantization_factor
                               1.0f, // non_conv_scale
                               "",
                               {{1, 1, 1, 1}}, // eltw_stride
                               {1, 1, 1, 1},   // stride
                               {0, 0, 0, 0},   // input_offset
                               {1, 1, 1, 1},   // dilation
                               false,          // conv_with_activation
                               0.0f,           // con_activation_slp
                               true,           // eltw_activation
                               0.0f));         // eltw_activation_slp
}

class FusedConvTest_no_conv_calibration : public FusedConvTest<float, float>
{};

TEST_F(FusedConvTest_no_conv_calibration, DISABLED_basic) {
    // That might happen if both conv output and non-conv input happen to be
    // normalized to the same dynamic range of if tensor-wise (instead of
    // per-channel) calibration is used. Also, a similar thing might happen for
    // a convolution with calibration without quantization (which is the real
    // target of this test, needed for the Inference Engine).

    // add_feature contains data for conv quantization/calibration, but the
    // primitive won't use it. It's just much easier to unify different tests
    // this way.
    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                1.0f,                         // bias
                {-10.0f, -10.0f},             // non_conv_input
                {241.0f, 242.0f});            // output_pre_relu

    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                0.0f,                         // bias
                {-10.0f, -11.0f},             // non_conv_input
                {480.0f, 480.0f});            // output_pre_relu

    do_test(fused_conv_eltwise("fused_conv",
                               "input",
                               "sum_input",
                               eltwise_mode::sum,
                               {"weights"},
                               {"biases"},
                               {},
                               {},   // conv_output_calibration
                               1.0f, // conv_i_quantization_factor
                               1.0f, // non_conv_scale
                               "",
                               {{1, 1, 1, 1}}, // eltw_stride
                               {1, 1, 1, 1},   // stride
                               {0, 0, 0, 0},   // input_offset
                               {1, 1, 1, 1},   // dilation
                               false,          // conv_with_activation
                               0.0f,           // con_activation_slp
                               true,           // eltw_activation
                               0.0f));         // eltw_activation_slp
}
