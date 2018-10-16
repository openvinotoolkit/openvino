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

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>

#include "mkldnn.hpp"

using namespace mkldnn;

using namespace std;

void simple_net(int times = 100) {

    auto cpu_engine = engine(engine::cpu, 0);

    /* Create a vector primitive to hold the network. For efficienty purpose,
     * weights are stored in a separate net to perform reordering only once. */
    std::vector<primitive> net;
    std::vector<primitive> net_weights;

    const int batch = 1;

    /* AlexNet: conv1
     * {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
     * strides: {4, 4}
     */
    memory::dims conv1_src_tz = { batch, 3, 227, 227 };
    memory::dims conv1_weights_tz = { 96, 3, 11, 11 };
    memory::dims conv1_bias_tz = { 96 };
    memory::dims conv1_dst_tz = { batch, 96, 55, 55 };
    memory::dims conv1_strides = { 4, 4 };
    auto conv1_padding = { 0, 0 };

    /* Allocate input and output buffers for user data */
    std::vector<float> user_src(batch * 3 * 227 * 227);
    std::vector<float> user_dst(batch * 1000);

    /* Allocate and fill buffers for weights and bias */
    std::vector<float> conv1_weights(std::accumulate(
            conv1_weights_tz.begin(), conv1_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    std::vector<float> conv1_bias(std::accumulate(conv1_bias_tz.begin(),
            conv1_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto user_src_memory
            = memory({ { { conv1_src_tz }, memory::data_type::f32,
                               memory::format::nchw },
                             cpu_engine },
                    user_src.data());
    auto user_weights_memory
            = memory({ { { conv1_weights_tz }, memory::data_type::f32,
                               memory::format::oihw },
                             cpu_engine },
                    conv1_weights.data());
    auto user_bias_memory = memory(
            { { { conv1_bias_tz }, memory::data_type::f32, memory::format::x },
                    cpu_engine },
            conv1_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv1_src_md = memory::desc(
            { conv1_src_tz }, memory::data_type::f32, memory::format::any);
    auto conv1_bias_md = memory::desc(
            { conv1_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv1_weights_md = memory::desc(
            { conv1_weights_tz }, memory::data_type::f32, memory::format::any);
    auto conv1_dst_md = memory::desc(
            { conv1_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a convolution */
    auto conv1_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv1_src_md,
            conv1_weights_md, conv1_bias_md, conv1_dst_md, conv1_strides,
            conv1_padding, conv1_padding, padding_kind::zero);
    auto conv1_prim_desc
            = convolution_forward::primitive_desc(conv1_desc, cpu_engine);

    /* create reorders for data and weights if layout requested by
     * convolution is different from NCHW/OIHW */
    auto conv1_src_memory = user_src_memory;
    if (memory::primitive_desc(conv1_prim_desc.src_primitive_desc())
            != user_src_memory.get_primitive_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_primitive_desc());
        net.push_back(reorder(user_src_memory, conv1_src_memory));
    }

    auto conv1_weights_memory = user_weights_memory;
    if (memory::primitive_desc(conv1_prim_desc.weights_primitive_desc())
            != user_weights_memory.get_primitive_desc()) {
        conv1_weights_memory
                = memory(conv1_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(user_weights_memory, conv1_weights_memory));
    }

    auto conv1_dst_memory = memory(conv1_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv1_prim_desc, conv1_src_memory,
            conv1_weights_memory, user_bias_memory,
            conv1_dst_memory));

    /* AlexNet: relu1
     * {batch, 96, 55, 55} -> {batch, 96, 55, 55}
     */
    const float negative1_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu,
            conv1_dst_memory.get_primitive_desc().desc(), negative1_slope);
    auto relu1_prim_desc
            = eltwise_forward::primitive_desc(relu1_desc, cpu_engine);

    net.push_back(eltwise_forward(
            relu1_prim_desc, conv1_dst_memory, conv1_dst_memory));

    /* AlexNet: lrn1
     * {batch, 96, 55, 55} -> {batch, 96, 55, 55}
     * local size: 5
     * alpha1: 0.0001
     * beta1: 0.75
     */
    const uint32_t local1_size = 5;
    const float alpha1 = 0.0001f;
    const float beta1 = 0.75f;
    const float k1 = 1.0f;

    /* create lrn primitive and add it to net */
    auto lrn1_desc = lrn_forward::desc(prop_kind::forward_inference,
            lrn_across_channels,
            conv1_dst_memory.get_primitive_desc().desc(), local1_size,
            alpha1, beta1, k1);
    auto lrn1_prim_desc
            = lrn_forward::primitive_desc(lrn1_desc, cpu_engine);
    auto lrn1_dst_memory = memory(lrn1_prim_desc.dst_primitive_desc());

    net.push_back(
            lrn_forward(lrn1_prim_desc, conv1_dst_memory, lrn1_dst_memory));

    /* AlexNet: pool1
     * {batch, 96, 55, 55} -> {batch, 96, 27, 27}
     * kernel: {3, 3}
     * strides: {2, 2}
     */

    memory::dims pool1_dst_tz = { batch, 96, 27, 27 };
    memory::dims pool1_kernel = { 3, 3 };
    memory::dims pool1_strides = { 2, 2 };
    auto pool_padding = { 0, 0 };

    auto pool1_dst_md = memory::desc(
            { pool1_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a pooling */
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
            pooling_max, lrn1_dst_memory.get_primitive_desc().desc(),
            pool1_dst_md, pool1_strides, pool1_kernel, pool_padding,
            pool_padding, padding_kind::zero);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, cpu_engine);
    auto pool1_dst_memory = memory(pool1_pd.dst_primitive_desc());

    /* create pooling primitive an add it to net */
    net.push_back(
            pooling_forward(pool1_pd, lrn1_dst_memory, pool1_dst_memory));

    /* AlexNet: conv2
    * {batch, 96, 27, 27} (x) {2, 128, 48, 5, 5} -> {batch, 256, 27, 27}
    * strides: {1, 1}
    */
    memory::dims conv2_src_tz = { batch, 96, 27, 27 };
    memory::dims conv2_weights_tz = { 2, 128, 48, 5, 5 };
    memory::dims conv2_bias_tz = { 256 };
    memory::dims conv2_dst_tz = { batch, 256, 27, 27 };
    memory::dims conv2_strides = { 1, 1 };
    auto conv2_padding = { 2, 2 };

    std::vector<float> conv2_weights(std::accumulate(
            conv2_weights_tz.begin(), conv2_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    std::vector<float> conv2_bias(std::accumulate(conv2_bias_tz.begin(),
            conv2_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv2_user_weights_memory
            = memory({ { { conv2_weights_tz }, memory::data_type::f32,
                               memory::format::goihw },
                             cpu_engine },
                    conv2_weights.data());
    auto conv2_user_bias_memory
            = memory({ { { conv2_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    conv2_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv2_src_md = memory::desc(
            { conv2_src_tz }, memory::data_type::f32, memory::format::any);
    auto conv2_bias_md = memory::desc(
            { conv2_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv2_weights_md = memory::desc({ conv2_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto conv2_dst_md = memory::desc(
            { conv2_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a convolution */
    auto conv2_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv2_src_md,
            conv2_weights_md, conv2_bias_md, conv2_dst_md, conv2_strides,
            conv2_padding, conv2_padding, padding_kind::zero);
    auto conv2_prim_desc
            = convolution_forward::primitive_desc(conv2_desc, cpu_engine);

    auto conv2_src_memory = pool1_dst_memory;
    if (memory::primitive_desc(conv2_prim_desc.src_primitive_desc())
            != conv2_src_memory.get_primitive_desc()) {
        conv2_src_memory = memory(conv2_prim_desc.src_primitive_desc());
        net.push_back(reorder(pool1_dst_memory, conv2_src_memory));
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    if (memory::primitive_desc(conv2_prim_desc.weights_primitive_desc())
            != conv2_user_weights_memory.get_primitive_desc()) {
        conv2_weights_memory
                = memory(conv2_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(conv2_user_weights_memory, conv2_weights_memory));
    }

    auto conv2_dst_memory = memory(conv2_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv2_prim_desc, conv2_src_memory,
            conv2_weights_memory, conv2_user_bias_memory,
            conv2_dst_memory));

    /* AlexNet: relu2
    * {batch, 256, 27, 27} -> {batch, 256, 27, 27}
    */
    const float negative2_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu,
            conv2_dst_memory.get_primitive_desc().desc(), negative2_slope);
    auto relu2_prim_desc
            = eltwise_forward::primitive_desc(relu2_desc, cpu_engine);

    net.push_back(eltwise_forward(
            relu2_prim_desc, conv2_dst_memory, conv2_dst_memory));

    /* AlexNet: lrn2
     * {batch, 256, 27, 27} -> {batch, 256, 27, 27}
     * local size: 5
     * alpha2: 0.0001
     * beta2: 0.75
     */
    const uint32_t local2_size = 5;
    const float alpha2 = 0.0001f;
    const float beta2 = 0.75f;
    const float k2 = 1.0f;

    /* create lrn primitive and add it to net */
    auto lrn2_desc = lrn_forward::desc(prop_kind::forward_inference,
            lrn_across_channels,
            conv2_prim_desc.dst_primitive_desc().desc(), local2_size,
            alpha2, beta2, k2);
    auto lrn2_prim_desc
            = lrn_forward::primitive_desc(lrn2_desc, cpu_engine);
    auto lrn2_dst_memory = memory(lrn2_prim_desc.dst_primitive_desc());

    net.push_back(
            lrn_forward(lrn2_prim_desc, conv2_dst_memory, lrn2_dst_memory));

    /* AlexNet: pool2
    * {batch, 256, 27, 27} -> {batch, 256, 13, 13}
    * kernel: {3, 3}
    * strides: {2, 2}
    */

    memory::dims pool2_dst_tz = { batch, 256, 13, 13 };
    memory::dims pool2_kernel = { 3, 3 };
    memory::dims pool2_strides = { 2, 2 };
    auto pool2_padding = { 0, 0 };

    auto pool2_dst_md = memory::desc(
            { pool2_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a pooling */
    auto pool2_desc = pooling_forward::desc(prop_kind::forward_inference,
            pooling_max, lrn2_dst_memory.get_primitive_desc().desc(),
            pool2_dst_md, pool2_strides, pool2_kernel, pool2_padding,
            pool2_padding, padding_kind::zero);
    auto pool2_pd = pooling_forward::primitive_desc(pool2_desc, cpu_engine);

    auto pool2_dst_memory = memory(pool2_pd.dst_primitive_desc());

    /* create pooling primitive an add it to net */
    net.push_back(
            pooling_forward(pool2_pd, lrn2_dst_memory, pool2_dst_memory));

    // -------
    /* AlexNet: conv3
    * {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13};
    * strides: {1, 1}
    */
    memory::dims conv3_src_tz = { batch, 256, 13, 13 };
    memory::dims conv3_weights_tz = { 384, 256, 3, 3 };
    memory::dims conv3_bias_tz = { 384 };
    memory::dims conv3_dst_tz = { batch, 384, 13, 13 };
    memory::dims conv3_strides = { 1, 1 };
    auto conv3_padding = { 1, 1 };

    std::vector<float> conv3_weights(std::accumulate(
            conv3_weights_tz.begin(), conv3_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    std::vector<float> conv3_bias(std::accumulate(conv3_bias_tz.begin(),
            conv3_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv3_user_weights_memory
            = memory({ { { conv3_weights_tz }, memory::data_type::f32,
                               memory::format::oihw },
                             cpu_engine },
                    conv3_weights.data());
    auto conv3_user_bias_memory
            = memory({ { { conv3_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    conv3_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv3_src_md = memory::desc(
            { conv3_src_tz }, memory::data_type::f32, memory::format::any);
    auto conv3_bias_md = memory::desc(
            { conv3_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv3_weights_md = memory::desc({ conv3_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto conv3_dst_md = memory::desc(
            { conv3_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a convolution */
    auto conv3_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv3_src_md,
            conv3_weights_md, conv3_bias_md, conv3_dst_md, conv3_strides,
            conv3_padding, conv3_padding, padding_kind::zero);
    auto conv3_prim_desc
            = convolution_forward::primitive_desc(conv3_desc, cpu_engine);

    auto conv3_src_memory = pool2_dst_memory;
    if (memory::primitive_desc(conv3_prim_desc.src_primitive_desc())
            != conv3_src_memory.get_primitive_desc()) {
        conv3_src_memory = memory(conv3_prim_desc.src_primitive_desc());
        net.push_back(reorder(pool2_dst_memory, conv3_src_memory));
    }

    auto conv3_weights_memory = conv3_user_weights_memory;
    if (memory::primitive_desc(conv3_prim_desc.weights_primitive_desc())
            != conv3_user_weights_memory.get_primitive_desc()) {
        conv3_weights_memory
                = memory(conv3_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(conv3_user_weights_memory, conv3_weights_memory));
    }

    auto conv3_dst_memory = memory(conv3_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv3_prim_desc, conv3_src_memory,
            conv3_weights_memory, conv3_user_bias_memory,
            conv3_dst_memory));

    /* AlexNet: relu3
    * {batch, 384, 13, 13} -> {batch, 384, 13, 13}
    */
    const float negative3_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu3_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu,
            conv3_dst_memory.get_primitive_desc().desc(), negative3_slope);
    auto relu3_prim_desc
            = eltwise_forward::primitive_desc(relu3_desc, cpu_engine);

    net.push_back(eltwise_forward(
            relu3_prim_desc, conv3_dst_memory, conv3_dst_memory));

    /* AlexNet: conv4
    * {batch, 384, 13, 13} (x)  {2, 192, 192, 3, 3}; -> {batch, 384, 13,
    * 13};
    * strides: {1, 1}
    */
    memory::dims conv4_src_tz = { batch, 384, 13, 13 };
    memory::dims conv4_weights_tz = { 2, 192, 192, 3, 3 };
    memory::dims conv4_bias_tz = { 384 };
    memory::dims conv4_dst_tz = { batch, 384, 13, 13 };
    memory::dims conv4_strides = { 1, 1 };
    auto conv4_padding = { 1, 1 };

    std::vector<float> conv4_weights(std::accumulate(
            conv4_weights_tz.begin(), conv4_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    std::vector<float> conv4_bias(std::accumulate(conv4_bias_tz.begin(),
            conv4_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv4_user_weights_memory
            = memory({ { { conv4_weights_tz }, memory::data_type::f32,
                               memory::format::goihw },
                             cpu_engine },
                    conv4_weights.data());
    auto conv4_user_bias_memory
            = memory({ { { conv4_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    conv4_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv4_src_md = memory::desc(
            { conv4_src_tz }, memory::data_type::f32, memory::format::any);
    auto conv4_bias_md = memory::desc(
            { conv4_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv4_weights_md = memory::desc({ conv4_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto conv4_dst_md = memory::desc(
            { conv4_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a convolution */
    auto conv4_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv4_src_md,
            conv4_weights_md, conv4_bias_md, conv4_dst_md, conv4_strides,
            conv4_padding, conv4_padding, padding_kind::zero);
    auto conv4_prim_desc
            = convolution_forward::primitive_desc(conv4_desc, cpu_engine);

    auto conv4_src_memory = conv3_dst_memory;
    if (memory::primitive_desc(conv4_prim_desc.src_primitive_desc())
            != conv4_src_memory.get_primitive_desc()) {
        conv4_src_memory = memory(conv4_prim_desc.src_primitive_desc());
        net.push_back(reorder(conv3_dst_memory, conv4_src_memory));
    }

    auto conv4_weights_memory = conv4_user_weights_memory;
    if (memory::primitive_desc(conv4_prim_desc.weights_primitive_desc())
            != conv4_user_weights_memory.get_primitive_desc()) {
        conv4_weights_memory
                = memory(conv4_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(conv4_user_weights_memory, conv4_weights_memory));
    }

    auto conv4_dst_memory = memory(conv4_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv4_prim_desc, conv4_src_memory,
            conv4_weights_memory, conv4_user_bias_memory,
            conv4_dst_memory));

    /* AlexNet: relu4
    * {batch, 384, 13, 13} -> {batch, 384, 13, 13}
    */
    const float negative4_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu4_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu,
            conv4_dst_memory.get_primitive_desc().desc(), negative4_slope);
    auto relu4_prim_desc
            = eltwise_forward::primitive_desc(relu4_desc, cpu_engine);

    net.push_back(eltwise_forward(
            relu4_prim_desc, conv4_dst_memory, conv4_dst_memory));

    /* AlexNet: conv5
    * {batch, 384, 13, 13} (x)  {2, 128, 192, 3, 3}; -> {batch, 256, 13,
    * 13};
    * strides: {1, 1}
    */
    memory::dims conv5_weights_tz = { 2, 128, 192, 3, 3 };
    memory::dims conv5_bias_tz = { 256 };
    memory::dims conv5_dst_tz = { batch, 256, 13, 13 };
    memory::dims conv5_strides = { 1, 1 };
    auto conv5_padding = { 1, 1 };

    std::vector<float> conv5_weights(std::accumulate(
            conv5_weights_tz.begin(), conv5_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    std::vector<float> conv5_bias(std::accumulate(conv5_bias_tz.begin(),
            conv5_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv5_user_weights_memory
            = memory({ { { conv5_weights_tz }, memory::data_type::f32,
                               memory::format::goihw },
                             cpu_engine },
                    conv5_weights.data());
    auto conv5_user_bias_memory
            = memory({ { { conv5_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    conv5_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv5_bias_md = memory::desc(
            { conv5_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv5_weights_md = memory::desc({ conv5_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto conv5_dst_md = memory::desc(
            { conv5_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a convolution */
    auto conv5_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct,
            conv4_dst_memory.get_primitive_desc().desc(), conv5_weights_md,
            conv5_bias_md, conv5_dst_md, conv5_strides, conv5_padding,
            conv5_padding, padding_kind::zero);
    auto conv5_prim_desc
            = convolution_forward::primitive_desc(conv5_desc, cpu_engine);

    auto conv5_src_memory = conv4_dst_memory;
    if (memory::primitive_desc(conv5_prim_desc.src_primitive_desc())
            != conv5_src_memory.get_primitive_desc()) {
        conv5_src_memory = memory(conv5_prim_desc.src_primitive_desc());
        net.push_back(reorder(conv4_dst_memory, conv5_src_memory));
    }

    auto conv5_weights_memory = conv5_user_weights_memory;
    if (memory::primitive_desc(conv5_prim_desc.weights_primitive_desc())
            != conv5_user_weights_memory.get_primitive_desc()) {
        conv5_weights_memory
                = memory(conv5_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(conv5_user_weights_memory, conv5_weights_memory));
    }

    auto conv5_dst_memory = memory(conv5_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv5_prim_desc, conv5_src_memory,
            conv5_weights_memory, conv5_user_bias_memory,
            conv5_dst_memory));

    /* AlexNet: relu5
    * {batch, 256, 13, 13} -> {batch, 256, 13, 13}
    */
    const float negative5_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu5_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu,
            conv5_dst_memory.get_primitive_desc().desc(), negative5_slope);
    auto relu5_prim_desc
            = eltwise_forward::primitive_desc(relu5_desc, cpu_engine);

    net.push_back(eltwise_forward(
            relu5_prim_desc, conv5_dst_memory, conv5_dst_memory));

    /* AlexNet: pool5
    * {batch, 256, 13, 13} -> {batch, 256, 6, 6}
    * kernel: {3, 3}
    * strides: {2, 2}
    */

    memory::dims pool5_dst_tz = { batch, 256, 6, 6 };
    memory::dims pool5_kernel = { 3, 3 };
    memory::dims pool5_strides = { 2, 2 };
    auto pool5_padding = { 0, 0 };

    std::vector<float> pool5_dst(std::accumulate(pool5_dst_tz.begin(),
            pool5_dst_tz.end(), 1, std::multiplies<uint32_t>()));

    auto pool5_dst_md = memory::desc(
            { pool5_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a pooling */
    auto pool5_desc = pooling_forward::desc(prop_kind::forward_inference,
            pooling_max, conv5_dst_memory.get_primitive_desc().desc(),
            pool5_dst_md, pool5_strides, pool5_kernel, pool5_padding,
            pool5_padding, padding_kind::zero);
    auto pool5_pd = pooling_forward::primitive_desc(pool5_desc, cpu_engine);

    auto pool5_dst_memory = memory(pool5_pd.dst_primitive_desc());

    /* create pooling primitive an add it to net */
    net.push_back(
            pooling_forward(pool5_pd, conv5_dst_memory, pool5_dst_memory));

    /**
     * fc6 inner product {batch, 256, 6, 6} (x) {4096, 256, 6, 6}-> {batch,
     * 4096}
     */
    memory::dims fc6_src_tz = { batch, 256, 6, 6 };
    memory::dims fc6_weights_tz = { 4096, 256, 6, 6 };
    memory::dims fc6_bias_tz = { 4096 };
    memory::dims fc6_dst_tz = { batch, 4096 };

    std::vector<float> fc6_weights(std::accumulate(fc6_weights_tz.begin(),
            fc6_weights_tz.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> fc6_bias(std::accumulate(fc6_bias_tz.begin(),
            fc6_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto fc6_user_weights_memory
            = memory({ { { fc6_weights_tz }, memory::data_type::f32,
                               memory::format::oihw },
                             cpu_engine },
                    fc6_weights.data());

    auto fc6_user_bias_memory
            = memory({ { { fc6_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    fc6_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto fc6_src_md = memory::desc(
            { fc6_src_tz }, memory::data_type::f32, memory::format::any);
    auto fc6_bias_md = memory::desc(
            { fc6_bias_tz }, memory::data_type::f32, memory::format::any);
    auto fc6_weights_md = memory::desc({ fc6_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto fc6_dst_md = memory::desc(
            { fc6_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a inner_product */
    auto fc6_desc
            = inner_product_forward::desc(prop_kind::forward_inference,
                    fc6_src_md, fc6_weights_md, fc6_bias_md, fc6_dst_md);
    auto fc6_prim_desc
            = inner_product_forward::primitive_desc(fc6_desc, cpu_engine);

    auto fc6_src_memory = pool5_dst_memory;
    if (memory::primitive_desc(fc6_prim_desc.src_primitive_desc())
            != fc6_src_memory.get_primitive_desc()) {
        fc6_src_memory = memory(fc6_prim_desc.src_primitive_desc());
        net.push_back(reorder(pool5_dst_memory, fc6_src_memory));
    }

    auto fc6_weights_memory = fc6_user_weights_memory;
    if (memory::primitive_desc(fc6_prim_desc.weights_primitive_desc())
            != fc6_user_weights_memory.get_primitive_desc()) {
        fc6_weights_memory = memory(fc6_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(fc6_user_weights_memory, fc6_weights_memory));
    }

    auto fc6_dst_memory = memory(fc6_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(inner_product_forward(fc6_prim_desc, fc6_src_memory,
            fc6_weights_memory, fc6_user_bias_memory, fc6_dst_memory));

    /**
     * fc7 inner product {batch, 4096} (x) {4096, 4096}-> {batch, 4096}
     */
    memory::dims fc7_weights_tz = { 4096, 4096 };
    memory::dims fc7_bias_tz = { 4096 };
    memory::dims fc7_dst_tz = { batch, 4096 };

    std::vector<float> fc7_weights(std::accumulate(fc7_weights_tz.begin(),
            fc7_weights_tz.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> fc7_bias(std::accumulate(fc7_bias_tz.begin(),
            fc7_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto fc7_user_weights_memory
            = memory({ { { fc7_weights_tz }, memory::data_type::f32,
                               memory::format::nc },
                             cpu_engine },
                    fc7_weights.data());

    auto fc7_user_bias_memory
            = memory({ { { fc7_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    fc7_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto fc7_bias_md = memory::desc(
            { fc7_bias_tz }, memory::data_type::f32, memory::format::any);
    auto fc7_weights_md = memory::desc({ fc7_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto fc7_dst_md = memory::desc(
            { fc7_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a inner_product */
    auto fc7_desc
            = inner_product_forward::desc(prop_kind::forward_inference,
                    fc6_dst_memory.get_primitive_desc().desc(),
                    fc7_weights_md, fc7_bias_md, fc7_dst_md);
    auto fc7_prim_desc
            = inner_product_forward::primitive_desc(fc7_desc, cpu_engine);

    auto fc7_weights_memory = fc7_user_weights_memory;
    if (memory::primitive_desc(fc7_prim_desc.weights_primitive_desc())
            != fc7_user_weights_memory.get_primitive_desc()) {
        fc7_weights_memory = memory(fc7_prim_desc.weights_primitive_desc());
        net.push_back(reorder(fc7_user_weights_memory, fc7_weights_memory));
    }

    auto fc7_dst_memory = memory(fc7_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(inner_product_forward(fc7_prim_desc, fc6_dst_memory,
            fc7_weights_memory, fc7_user_bias_memory, fc7_dst_memory));

    /**
    * fc8 inner product {batch, 4096} (x) {1000, 4096}-> {batch, 1000}
    */
    memory::dims fc8_weights_tz = { 1000, 4096 };
    memory::dims fc8_bias_tz = { 1000 };
    memory::dims fc8_dst_tz = { batch, 1000 };

    std::vector<float> fc8_weights(std::accumulate(fc8_weights_tz.begin(),
            fc8_weights_tz.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> fc8_bias(std::accumulate(fc8_bias_tz.begin(),
            fc8_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto fc8_user_weights_memory
            = memory({ { { fc8_weights_tz }, memory::data_type::f32,
                               memory::format::nc },
                             cpu_engine },
                    fc8_weights.data());

    auto fc8_user_bias_memory
            = memory({ { { fc8_bias_tz }, memory::data_type::f32,
                               memory::format::x },
                             cpu_engine },
                    fc8_bias.data());

    auto user_dst_memory = memory({ { { fc8_dst_tz }, memory::data_type::f32,
                                           memory::format::nc },
                                         cpu_engine },
            user_dst.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto fc8_bias_md = memory::desc(
            { fc8_bias_tz }, memory::data_type::f32, memory::format::any);
    auto fc8_weights_md = memory::desc({ fc8_weights_tz },
            memory::data_type::f32, memory::format::any);
    auto fc8_dst_md = memory::desc(
            { fc8_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a inner_product */
    auto fc8_desc
            = inner_product_forward::desc(prop_kind::forward_inference,
                    fc7_dst_memory.get_primitive_desc().desc(),
                    fc8_weights_md, fc8_bias_md, fc8_dst_md);
    auto fc8_prim_desc
            = inner_product_forward::primitive_desc(fc8_desc, cpu_engine);

    auto fc8_weights_memory = fc8_user_weights_memory;
    if (memory::primitive_desc(fc8_prim_desc.weights_primitive_desc())
            != fc8_user_weights_memory.get_primitive_desc()) {
        fc8_weights_memory = memory(fc8_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(fc8_user_weights_memory, fc8_weights_memory));
    }

    auto fc8_dst_memory = memory(fc8_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(inner_product_forward(fc8_prim_desc, fc7_dst_memory,
            fc8_weights_memory, fc8_user_bias_memory, fc8_dst_memory));

    /* create reorder between internal and user data if it is needed and
     *  add it to net after pooling */
    if (fc8_dst_memory != user_dst_memory) {
        net.push_back(reorder(fc8_dst_memory, user_dst_memory));
    }

    stream(stream::kind::eager).submit(net_weights).wait();
    for (int j = 0; j < times; ++j) {
        stream(stream::kind::eager).submit(net).wait();
    }
}

int main(int argc, char **argv) {
    try {
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                             chrono::steady_clock::now().time_since_epoch())
                             .count();
        int times = 1000;
        simple_net(times);
        auto end = chrono::duration_cast<chrono::milliseconds>(
                           chrono::steady_clock::now().time_since_epoch())
                           .count();
        cout << "Use time " << (end - begin) / (times + 0.0) << "\n";
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
