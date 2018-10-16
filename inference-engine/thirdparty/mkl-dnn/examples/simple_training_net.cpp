/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include <math.h>
#include <string>
#include "mkldnn.hpp"

using namespace mkldnn;

void simple_net()
{
    auto cpu_engine = engine(engine::cpu, 0);

    const int batch = 32;

    std::vector<float> net_src(batch * 3 * 227 * 227);
    std::vector<float> net_dst(batch * 96 * 27 * 27);

    /* initializing non-zero values for src */
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = sinf((float)i);

    /* AlexNet: conv
     * {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
     * strides: {4, 4}
     */
    memory::dims conv_src_tz = { batch, 3, 227, 227 };
    memory::dims conv_weights_tz = { 96, 3, 11, 11 };
    memory::dims conv_bias_tz = { 96 };
    memory::dims conv_dst_tz = { batch, 96, 55, 55 };
    memory::dims conv_strides = { 4, 4 };
    auto conv_padding = { 0, 0 };

    std::vector<float> conv_weights(
            std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1,
                            std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
                                                 conv_bias_tz.end(), 1,
                                                 std::multiplies<uint32_t>()));

    /* initializing non-zero values for weights and bias */
    for (size_t i = 0; i < conv_weights.size(); ++i)
        conv_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv_bias.size(); ++i)
        conv_bias[i] = sinf((float)i);

    /* create memory for user data */
    auto conv_user_src_memory = memory(
            { { { conv_src_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            net_src.data());
    auto conv_user_weights_memory
            = memory({ { { conv_weights_tz }, memory::data_type::f32,
                         memory::format::oihw },
                       cpu_engine },
                     conv_weights.data());
    auto conv_user_bias_memory = memory(
            { { { conv_bias_tz }, memory::data_type::f32, memory::format::x },
              cpu_engine },
            conv_bias.data());

    /* create mmemory descriptors for convolution data w/ no specified
     * format(`any`)
     * format `any` lets a primitive(convolution in this case)
     * chose the memory format preferred for best performance. */
    auto conv_src_md = memory::desc({ conv_src_tz }, memory::data_type::f32,
                                    memory::format::any);
    auto conv_bias_md = memory::desc({ conv_bias_tz }, memory::data_type::f32,
                                     memory::format::any);
    auto conv_weights_md = memory::desc(
            { conv_weights_tz }, memory::data_type::f32, memory::format::any);
    auto conv_dst_md = memory::desc({ conv_dst_tz }, memory::data_type::f32,
                                    memory::format::any);

    /* create a convolution primitive descriptor */
    auto conv_desc = convolution_forward::desc(
            prop_kind::forward, convolution_direct, conv_src_md,
            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
            conv_padding, conv_padding, padding_kind::zero);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);

    /* create reorder primitives between user input and conv src if needed */
    auto conv_src_memory = conv_user_src_memory;
    bool reorder_conv_src = false;
    primitive conv_reorder_src;
    if (memory::primitive_desc(conv_pd.src_primitive_desc())
        != conv_user_src_memory.get_primitive_desc()) {
        conv_src_memory = memory(conv_pd.src_primitive_desc());
        conv_reorder_src = reorder(conv_user_src_memory, conv_src_memory);
        reorder_conv_src = true;
    }

    auto conv_weights_memory = conv_user_weights_memory;
    bool reorder_conv_weights = false;
    primitive conv_reorder_weights;
    if (memory::primitive_desc(conv_pd.weights_primitive_desc())
        != conv_user_weights_memory.get_primitive_desc()) {
        conv_weights_memory = memory(conv_pd.weights_primitive_desc());
        conv_reorder_weights
                = reorder(conv_user_weights_memory, conv_weights_memory);
        reorder_conv_weights = true;
    }

    /* create memory primitive for conv dst */
    auto conv_dst_memory = memory(conv_pd.dst_primitive_desc());

    /* finally create a convolution primitive */
    auto conv
            = convolution_forward(conv_pd, conv_src_memory, conv_weights_memory,
                                  conv_user_bias_memory, conv_dst_memory);

    /* AlexNet: relu
     * {batch, 96, 55, 55} -> {batch, 96, 55, 55}
     */
    const float negative_slope = 1.0;

    /* create relu primitive desc */
    /* keep memory format of source same as the format of convolution
     * output in order to avoid reorder */
    auto relu_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv_pd.dst_primitive_desc().desc(),
            negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu_engine);

    /* create relu dst memory primitive */
    auto relu_dst_memory = memory(relu_pd.dst_primitive_desc());

    /* finally create a relu primitive */
    auto relu = eltwise_forward(relu_pd, conv_dst_memory, relu_dst_memory);

    /* AlexNet: lrn
     * {batch, 96, 55, 55} -> {batch, 96, 55, 55}
     * local size: 5
     * alpha: 0.0001
     * beta: 0.75
     * k: 1.0
     */
    const uint32_t local_size = 5;
    const float alpha = 0.0001;
    const float beta = 0.75;
    const float k = 1.0;

    /* create a lrn primitive descriptor */
    auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels,
                                      relu_pd.dst_primitive_desc().desc(),
                                      local_size, alpha, beta, k);
    auto lrn_pd = lrn_forward::primitive_desc(lrn_desc, cpu_engine);

    /* create lrn dst memory */
    auto lrn_dst_memory = memory(lrn_pd.dst_primitive_desc());

    /* create workspace only in training and only for forward primitive*/
    /* query lrn_pd for workspace, this memory will be shared with forward lrn*/
    auto lrn_workspace_memory = memory(lrn_pd.workspace_primitive_desc());

    /* finally create a lrn primitive */
    auto lrn = lrn_forward(lrn_pd, relu_dst_memory, lrn_workspace_memory,
                           lrn_dst_memory);

    /* AlexNet: pool
     * {batch, 96, 55, 55} -> {batch, 96, 27, 27}
     * kernel: {3, 3}
     * strides: {2, 2}
     */
    memory::dims pool_dst_tz = { batch, 96, 27, 27 };
    memory::dims pool_kernel = { 3, 3 };
    memory::dims pool_strides = { 2, 2 };
    auto pool_padding = { 0, 0 };

    /* create memory for pool dst data in user format */
    auto pool_user_dst_memory = memory(
            { { { pool_dst_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            net_dst.data());

    /* create pool dst memory descriptor in format any */
    auto pool_dst_md = memory::desc({ pool_dst_tz }, memory::data_type::f32,
                                    memory::format::any);

    /* create a pooling primitive descriptor */
    auto pool_desc = pooling_forward::desc(
            prop_kind::forward, pooling_max,
            lrn_dst_memory.get_primitive_desc().desc(), pool_dst_md,
            pool_strides, pool_kernel, pool_padding, pool_padding,
            padding_kind::zero);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, cpu_engine);

    /* create reorder primitive between pool dst and user dst format
     * if needed */
    auto pool_dst_memory = pool_user_dst_memory;
    bool reorder_pool_dst = false;
    primitive pool_reorder_dst;
    if (memory::primitive_desc(pool_pd.dst_primitive_desc())
        != pool_user_dst_memory.get_primitive_desc()) {
        pool_dst_memory = memory(pool_pd.dst_primitive_desc());
        pool_reorder_dst = reorder(pool_dst_memory, pool_user_dst_memory);
        reorder_pool_dst = true;
    }

    /* create pooling workspace memory if training */
    auto pool_workspace_memory = memory(pool_pd.workspace_primitive_desc());

    /* finally create a pooling primitive */
    auto pool = pooling_forward(pool_pd, lrn_dst_memory, pool_dst_memory,
                                pool_workspace_memory);

    /* build forward net */
    std::vector<primitive> net_fwd;
    if (reorder_conv_src)
        net_fwd.push_back(conv_reorder_src);
    if (reorder_conv_weights)
        net_fwd.push_back(conv_reorder_weights);
    net_fwd.push_back(conv);
    net_fwd.push_back(relu);
    net_fwd.push_back(lrn);
    net_fwd.push_back(pool);
    if (reorder_pool_dst)
        net_fwd.push_back(pool_reorder_dst);

    /*----------------------------------------------------------------------*/
    /*----------------- Backward Stream -------------------------------------*/
    /* ... user diff_data ...*/
    std::vector<float> net_diff_dst(batch * 96 * 27 * 27);
    for (size_t i = 0; i < net_diff_dst.size(); ++i)
        net_diff_dst[i] = sinf((float)i);

    /* create memory for user diff dst data */
    auto pool_user_diff_dst_memory = memory(
            { { { pool_dst_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            net_diff_dst.data());

    /* Backward pooling */
    /* create memory descriptorsfor pooling */
    auto pool_diff_src_md = lrn_dst_memory.get_primitive_desc().desc();
    auto pool_diff_dst_md = pool_dst_memory.get_primitive_desc().desc();

    /* create backward pooling descriptor*/
    auto pool_bwd_desc = pooling_backward::desc(
            pooling_max, pool_diff_src_md, pool_diff_dst_md, pool_strides,
            pool_kernel, pool_padding, pool_padding, padding_kind::zero);
    /* backward primitive descriptor needs to hint forward descriptor */
    auto pool_bwd_pd = pooling_backward::primitive_desc(pool_bwd_desc,
                                                        cpu_engine, pool_pd);

    /* create reorder primitive between user diff dst and pool diff dst
     * if required */
    auto pool_diff_dst_memory = pool_user_diff_dst_memory;
    primitive pool_reorder_diff_dst;
    bool reorder_pool_diff_dst = false;
    if (memory::primitive_desc(pool_dst_memory.get_primitive_desc())
        != pool_user_diff_dst_memory.get_primitive_desc()) {
        pool_diff_dst_memory = memory(pool_dst_memory.get_primitive_desc());
        pool_reorder_diff_dst
                = reorder(pool_user_diff_dst_memory, pool_diff_dst_memory);
        reorder_pool_diff_dst = true;
    }

    /* create memory primitive for pool diff src */
    auto pool_diff_src_memory = memory(pool_bwd_pd.diff_src_primitive_desc());

    /* finally create backward pooling primitive */
    auto pool_bwd
            = pooling_backward(pool_bwd_pd, pool_diff_dst_memory,
                               pool_workspace_memory, pool_diff_src_memory);

    /* Backward lrn */
    auto lrn_diff_dst_md = lrn_dst_memory.get_primitive_desc().desc();

    /* create backward lrn primitive descriptor */
    auto lrn_bwd_desc = lrn_backward::desc(
            lrn_across_channels, lrn_pd.src_primitive_desc().desc(),
            lrn_diff_dst_md, local_size, alpha, beta, k);
    auto lrn_bwd_pd
            = lrn_backward::primitive_desc(lrn_bwd_desc, cpu_engine, lrn_pd);

    /* create memory for lrn diff src */
    auto lrn_diff_src_memory = memory(lrn_bwd_pd.diff_src_primitive_desc());

    /* finally create a lrn backward primitive */
    // backward lrn needs src: relu dst in this topology
    auto lrn_bwd
            = lrn_backward(lrn_bwd_pd, relu_dst_memory, pool_diff_src_memory,
                           lrn_workspace_memory, lrn_diff_src_memory);

    /* Backward relu */
    auto relu_diff_dst_md = lrn_diff_src_memory.get_primitive_desc().desc();
    auto relu_src_md = conv_pd.dst_primitive_desc().desc();

    /* create backward relu primitive_descriptor */
    auto relu_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
            relu_diff_dst_md, relu_src_md, negative_slope);
    auto relu_bwd_pd
            = eltwise_backward::primitive_desc(relu_bwd_desc, cpu_engine, relu_pd);

    /* create memory for relu diff src */
    auto relu_diff_src_memory = memory(relu_bwd_pd.diff_src_primitive_desc());

    /* finally create a backward relu primitive */
    auto relu_bwd = eltwise_backward(relu_bwd_pd, conv_dst_memory,
                                  lrn_diff_src_memory, relu_diff_src_memory);

    /* Backward convolution with respect to weights */
    /* create user format diff weights and diff bias memory */
    std::vector<float> conv_user_diff_weights_buffer(
            std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1,
                            std::multiplies<uint32_t>()));
    std::vector<float> conv_diff_bias_buffer(
            std::accumulate(conv_bias_tz.begin(), conv_bias_tz.end(), 1,
                            std::multiplies<uint32_t>()));

    auto conv_user_diff_weights_memory
            = memory({ { { conv_weights_tz }, memory::data_type::f32,
                         memory::format::nchw },
                       cpu_engine },
                     conv_user_diff_weights_buffer.data());
    auto conv_diff_bias_memory = memory(
            { { { conv_bias_tz }, memory::data_type::f32, memory::format::x },
              cpu_engine },
            conv_diff_bias_buffer.data());

    /* create memory primitives descriptors */

    auto conv_bwd_src_md = memory::desc({ conv_src_tz }, memory::data_type::f32,
                                        memory::format::any);
    auto conv_diff_bias_md = memory::desc(
            { conv_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv_diff_weights_md = memory::desc(
            { conv_weights_tz }, memory::data_type::f32, memory::format::any);
    auto conv_diff_dst_md = memory::desc(
            { conv_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create backward convolution primitive descriptor */
    auto conv_bwd_weights_desc = convolution_backward_weights::desc(
            convolution_direct, conv_bwd_src_md, conv_diff_weights_md,
            conv_diff_bias_md, conv_diff_dst_md, conv_strides, conv_padding,
            conv_padding, padding_kind::zero);
    auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, cpu_engine, conv_pd);

    /* for best performance convolution backward might chose
     * different memory format for src and diff_dst
     * than the memory formats preferred by forward convolution
     * for src and dst respectively */
    /* create reorder primitives for src from forward convolution to the
     * format chosen by backward convolution */
    auto conv_bwd_src_memory = conv_src_memory;
    primitive conv_bwd_reorder_src;
    auto reorder_conv_bwd_src = false;
    if (memory::primitive_desc(conv_bwd_weights_pd.src_primitive_desc())
        != conv_src_memory.get_primitive_desc())
    {
        conv_bwd_src_memory = memory(conv_bwd_weights_pd.src_primitive_desc());
        conv_bwd_reorder_src = reorder(conv_src_memory, conv_bwd_src_memory);
        reorder_conv_bwd_src = true;
    }

    /* create reorder primitives for diff_dst between diff_src from relu_bwd
     * and format preferred by conv_diff_weights */
    auto conv_diff_dst_memory = relu_diff_src_memory;
    primitive conv_reorder_diff_dst;
    auto reorder_conv_diff_dst = false;
    if (memory::primitive_desc(conv_bwd_weights_pd.diff_dst_primitive_desc())
        != relu_diff_src_memory.get_primitive_desc())
    {
        conv_diff_dst_memory
                = memory(conv_bwd_weights_pd.diff_dst_primitive_desc());
        conv_reorder_diff_dst
                = reorder(relu_diff_src_memory, conv_diff_dst_memory);
        reorder_conv_diff_dst = true;
    }

    /* create reorder primitives between conv diff weights and user diff weights
     * if needed */
    auto conv_diff_weights_memory = conv_user_diff_weights_memory;
    primitive conv_reorder_diff_weights;
    bool reorder_conv_diff_weights = false;
    if (memory::primitive_desc(
                conv_bwd_weights_pd.diff_weights_primitive_desc())
        != conv_user_diff_weights_memory.get_primitive_desc()) {
        conv_diff_weights_memory
                = memory(conv_bwd_weights_pd.diff_weights_primitive_desc());
        conv_reorder_diff_weights = reorder(conv_diff_weights_memory,
                                            conv_user_diff_weights_memory);
        reorder_conv_diff_weights = true;
    }

    /* finally create backward convolution primitive */
    auto conv_bwd_weights = convolution_backward_weights(
            conv_bwd_weights_pd, conv_bwd_src_memory, conv_diff_dst_memory,
            conv_diff_weights_memory, conv_diff_bias_memory);

    /* build backward propagation net */
    std::vector<primitive> net_bwd;
    if (reorder_pool_diff_dst)
        net_bwd.push_back(pool_reorder_diff_dst);
    net_bwd.push_back(pool_bwd);
    net_bwd.push_back(lrn_bwd);
    net_bwd.push_back(relu_bwd);
    if (reorder_conv_bwd_src)
        net_bwd.push_back(conv_bwd_reorder_src);
    if (reorder_conv_diff_dst)
        net_bwd.push_back(conv_reorder_diff_dst);
    net_bwd.push_back(conv_bwd_weights);
    if (reorder_conv_diff_weights)
        net_bwd.push_back(conv_reorder_diff_weights);

    int n_iter = 1; //number of iterations for training
    /* execute */
    while (n_iter) {
        /* forward */
        stream(stream::kind::eager).submit(net_fwd).wait();

        /* update net_diff_dst */
        // auto net_output = pool_user_dst_memory.get_data_handle();
        /*..user updates net_diff_dst using net_output...*/
        // some user defined func update_diff_dst(net_diff_dst.data(),
        // net_output)

        stream(stream::kind::eager).submit(net_bwd).wait();
        /* update weights and bias using diff weights and bias*/
        // auto net_diff_weights
        //     = conv_user_diff_weights_memory.get_data_handle();
        // auto net_diff_bias = conv_diff_bias_memory.get_data_handle();
        /* ...user updates weights and bias using diff weights and bias...*/
        // some user defined func update_weights(conv_weights.data(),
        // conv_bias.data(), net_diff_weights, net_diff_bias);

        --n_iter;
    }
}

int main(int argc, char **argv)
{
    try
    {
        simple_net();
        std::cout << "passed" << std::endl;
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
