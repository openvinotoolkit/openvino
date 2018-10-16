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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "mkldnn.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

int tensor_volume(const mkldnn::tensor::dims &t)
{
    int x = 1;
    for (size_t i = 0; i < t.size(); ++i) x *= t[i];
    return x;
}

int doit(bool lazy) {
    using namespace mkldnn;

    /* AlexNet: c3
     * {2, 256, 13, 13} (x) {384, 256, 3, 3} -> {2, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    // TODO: make tensor desc optional and default to N C X1 .. XN

    // XXX: descs for memory should be not necessary!

    const int mb = 2;
    const int groups = 2;
    tensor::dims input_tz = {mb, 256, 13, 13};
    tensor::dims weights_tz = {groups, 384/groups, 256/groups, 3, 3};
    tensor::dims bias_tz = {384};
    tensor::dims strides = {1, 1};
    tensor::dims padding = {0, 0};
    tensor::dims output_tz = {mb, 384,
        (input_tz[2] + 2*padding[0] - weights_tz[3])/strides[0] + 1,
        (input_tz[3] + 2*padding[1] - weights_tz[4])/strides[1] + 1,
    };

    /* prepare actual data */
    std::vector<float> input(tensor_volume(input_tz), .0f);
    std::vector<float> weights(tensor_volume(weights_tz), .0f);
    std::vector<float> bias(tensor_volume(bias_tz), .0f);
    std::vector<float> output(tensor_volume(output_tz), .0f);

    /* mkl-dnn starts here */
    auto c3_src_desc = memory::desc({input_tz}, memory::precision::f32, memory::format::nchw);
    auto c3_weights_desc = memory::desc({weights_tz}, memory::precision::f32, memory::format::goihw);
    auto c3_bias_desc = memory::desc({bias_tz}, memory::precision::f32, memory::format::x);
    auto c3_dst_desc = memory::desc({output_tz}, memory::precision::f32, memory::format::nchw);

    auto c3_src = memory({c3_src_desc, cpu_engine}, input.data());
    auto c3_weights = memory({c3_weights_desc, cpu_engine}, weights.data());
    auto c3_bias = memory({c3_bias_desc, cpu_engine}, bias.data());
    auto c3_dst = memory({c3_dst_desc, cpu_engine}, output.data());

#if 0
    auto c3_desc = convolution::desc(prop_kind::forward, convolution::direct,
            c3_src_desc, c3_weights_desc, c3_bias_desc, c3_dst_desc,
            {0, 0, 1, 1}, {0, 0, 1, 1}, padding_kind::zero);
    auto c3_primitive_desc = convolution::primitive_desc(c3_desc, cpu_engine);
    auto c3 = convolution(c3_primitive_desc,
            c3_src, c3_weights, c3_bias, c3_dst);
#else
    auto c3 = convolution(prop_kind::forward, convolution::direct,
            c3_src, c3_weights, c3_bias, c3_dst,
            strides, padding, padding_kind::zero);
#endif

    stream().submit({c3}).wait();

    return 0;
}

#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    int rc = doit(false);
    printf("eager: %s\n", rc ? "failed" : "passed");
    rc = doit(true);
    printf("lazy: %s\n", rc ? "failed" : "passed");
    return rc;
}
