// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dnnl.hpp>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

static int tensor_volume(const dnnl::memory::dims& t) {
    int x = 1;
    for (const auto i : t)
        x *= i;
    return x;
}

void test() {
    using namespace dnnl;

    auto cpu_engine = engine(engine::cpu, 0);

    const int mb = 2;
    const int groups = 2;
    memory::dims input_tz = {mb, 256, 13, 13};
    memory::dims weights_tz = {groups, 384 / groups, 256 / groups, 3, 3};
    memory::dims bias_tz = {384};
    memory::dims strides = {1, 1};
    memory::dims padding = {0, 0};
    memory::dims output_tz = {
        mb,
        384,
        (input_tz[2] + 2 * padding[0] - weights_tz[3]) / strides[0] + 1,
        (input_tz[3] + 2 * padding[1] - weights_tz[4]) / strides[1] + 1,
    };

    std::vector<float> input(tensor_volume(input_tz), .0f);
    std::vector<float> weights(tensor_volume(weights_tz), .0f);
    std::vector<float> bias(tensor_volume(bias_tz), .0f);
    std::vector<float> output(tensor_volume(output_tz), .0f);

    auto c3_src_desc = memory::desc({input_tz}, memory::data_type::f32, memory::format::nchw);
    auto c3_weights_desc = memory::desc({weights_tz}, memory::data_type::f32, memory::format::goihw);
    auto c3_bias_desc = memory::desc({bias_tz}, memory::data_type::f32, memory::format::x);
    auto c3_dst_desc = memory::desc({output_tz}, memory::data_type::f32, memory::format::nchw);

    auto c3_src = memory({c3_src_desc, cpu_engine}, input.data());
    auto c3_weights = memory({c3_weights_desc, cpu_engine}, weights.data());
    auto c3_bias = memory({c3_bias_desc, cpu_engine}, bias.data());
    auto c3_dst = memory({c3_dst_desc, cpu_engine}, output.data());

    auto c3 =
        convolution_forward(convolution_forward::primitive_desc(convolution_forward::desc(prop_kind::forward,
                                                                                          algorithm::convolution_direct,
                                                                                          c3_src_desc,
                                                                                          c3_weights_desc,
                                                                                          c3_bias_desc,
                                                                                          c3_dst_desc,
                                                                                          strides,
                                                                                          padding,
                                                                                          padding,
                                                                                          padding_kind::zero),
                                                                cpu_engine),
                            c3_src,
                            c3_weights,
                            c3_bias,
                            c3_dst);

    stream(stream::kind::eager).submit({c3}).wait();
}

TEST(dnnl, engine) {
    EXPECT_NO_THROW(test());
}
