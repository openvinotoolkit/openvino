/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <numeric>
#include <string>
#include "mkldnn.hpp"

using namespace mkldnn;

void simple_net_int8() {
    auto cpu_engine = engine(engine::cpu, 0);

    /* Create a vector to store the topology primitives */
    std::vector<primitive> net;

    const int batch = 8;

    /* AlexNet: conv3
     * {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
     * strides: {1, 1}
     */
    memory::dims conv_src_tz = { batch, 256, 13, 13 };
    memory::dims conv_weights_tz = { 384, 256, 3, 3 };
    memory::dims conv_bias_tz = { 384 };
    memory::dims conv_dst_tz = { batch, 384, 13, 13 };
    memory::dims conv_strides = { 1, 1 };
    auto conv_padding = { 1, 1 };

    /* Set Scaling mode for int8 quantizing */
    const std::vector<float> src_scales = { 1.8f };
    const std::vector<float> weight_scales = { 2.0f };
    const std::vector<float> bias_scales = { 1.0f };
    const std::vector<float> dst_scales = { 0.55f };
    /* assign halves of vector with arbitrary values */
    std::vector<float> conv_scales(384);
    const int scales_half = 384 / 2;
    std::fill(conv_scales.begin(), conv_scales.begin() + scales_half, 0.3f);
    std::fill(conv_scales.begin() + scales_half + 1, conv_scales.end(), 0.8f);

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;
    const int conv_mask = 2; // 1 << output_channel_dim

    /* Allocate input and output buffers for user data */
    std::vector<float> user_src(batch * 256 * 13 * 13);
    std::vector<float> user_dst(batch * 384 * 13 * 13);

    /* Allocate and fill buffers for weights and bias */
    std::vector<float> conv_weights(std::accumulate(conv_weights_tz.begin(),
            conv_weights_tz.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
            conv_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto user_src_memory = memory(
            { { { conv_src_tz }, memory::data_type::f32, memory::format::nchw },
                    cpu_engine },
            user_src.data());
    auto user_weights_memory
            = memory({ { { conv_weights_tz }, memory::data_type::f32,
                               memory::format::oihw },
                             cpu_engine },
                    conv_weights.data());
    auto user_bias_memory = memory(
            { { { conv_bias_tz }, memory::data_type::f32, memory::format::x },
                    cpu_engine },
            conv_bias.data());

    /* create memory descriptors for convolution data w/ no specified format */
    auto conv_src_md = memory::desc(
            { conv_src_tz }, memory::data_type::u8, memory::format::any);
    auto conv_bias_md = memory::desc(
            { conv_bias_tz }, memory::data_type::s8, memory::format::any);
    auto conv_weights_md = memory::desc(
            { conv_weights_tz }, memory::data_type::s8, memory::format::any);
    auto conv_dst_md = memory::desc(
            { conv_dst_tz }, memory::data_type::u8, memory::format::any);

    /* create a convolution */
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
            conv_dst_md, conv_strides, conv_padding, conv_padding,
            padding_kind::zero);

    /* define the convolution attributes */
    primitive_attr conv_attr;
    conv_attr.set_int_output_round_mode(round_mode::round_nearest);
    conv_attr.set_output_scales(conv_mask, conv_scales);

    /* AlexNet: execute ReLU as PostOps */
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    /* check if int8 convolution is supported */
    try {
        auto conv_prim_desc = convolution_forward::primitive_desc(
                conv_desc, conv_attr, cpu_engine);
    } catch (error &e) {
        if (e.status == mkldnn_unimplemented) {
            std::cerr << "AVX512-BW support or Intel(R) MKL dependency is "
            "required for int8 convolution" << std::endl;
        }
        throw;
    }

    auto conv_prim_desc = convolution_forward::primitive_desc(
            conv_desc, conv_attr, cpu_engine);

    /* Next: create memory primitives for the convolution's input data
     * and use reorder to quantize the values into int8 */
    auto conv_src_memory = memory(conv_prim_desc.src_primitive_desc());
    primitive_attr src_attr;
    src_attr.set_int_output_round_mode(round_mode::round_nearest);
    src_attr.set_output_scales(src_mask, src_scales);
    auto src_reorder_pd
            = reorder::primitive_desc(user_src_memory.get_primitive_desc(),
                    conv_src_memory.get_primitive_desc(), src_attr);
    net.push_back(reorder(src_reorder_pd, user_src_memory, conv_src_memory));

    auto conv_weights_memory = memory(conv_prim_desc.weights_primitive_desc());
    primitive_attr weight_attr;
    weight_attr.set_int_output_round_mode(round_mode::round_nearest);
    weight_attr.set_output_scales(weight_mask, weight_scales);
    auto weight_reorder_pd
            = reorder::primitive_desc(user_weights_memory.get_primitive_desc(),
                    conv_weights_memory.get_primitive_desc(), weight_attr);
    net.push_back(reorder(
            weight_reorder_pd, user_weights_memory, conv_weights_memory));

    auto conv_bias_memory = memory(conv_prim_desc.bias_primitive_desc());
    primitive_attr bias_attr;
    bias_attr.set_int_output_round_mode(round_mode::round_nearest);
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd
            = reorder::primitive_desc(user_bias_memory.get_primitive_desc(),
                    conv_bias_memory.get_primitive_desc(), bias_attr);
    net.push_back(reorder(bias_reorder_pd, user_bias_memory, conv_bias_memory));

    auto conv_dst_memory = memory(conv_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, conv_src_memory,
            conv_weights_memory, conv_bias_memory, conv_dst_memory));

    /* Convert data back into fp32 and compare values with u8.
     * Note: data is unsigned since there are no negative values
     * after ReLU */

    /* Create a memory primitive for user data output */
    auto user_dst_memory = memory(
            { { { conv_dst_tz }, memory::data_type::f32, memory::format::nchw },
                    cpu_engine },
            user_dst.data());

    primitive_attr dst_attr;
    dst_attr.set_int_output_round_mode(round_mode::round_nearest);
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = reorder::primitive_desc(conv_dst_memory.get_primitive_desc(),
                    user_dst_memory.get_primitive_desc(), dst_attr);

    /* Convert the destination memory from convolution into user
     * data format if necessary */
    if (conv_dst_memory != user_dst_memory) {
        net.push_back(
                reorder(dst_reorder_pd, conv_dst_memory, user_dst_memory));
    }

    stream(stream::kind::eager).submit(net).wait();

}

int main(int argc, char **argv) {
    try {
        /* Notes:
         * On convolution creating: check for Intel(R) MKL dependency execution.
         * output: warning if not found. */
        simple_net_int8();
        std::cout << "Simple-net-int8 example passed!" << std::endl;
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
