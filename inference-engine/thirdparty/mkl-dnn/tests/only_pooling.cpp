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

typedef float real_t;

static void init_src(const mkldnn::tensor::dims &dim, real_t *x)
{
    int N = dim[0], C = dim[1], H = dim[2], W = dim[3];
#   pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n += 1)
    for (int c = 0; c < C; c += 1)
    for (int h = 2; h+2 <= H; h += 2)
    for (int w = 2; w+2 <= W; w += 2)
        x[w + W*h + c*W*H + n*W*H*C] = c*n;
}

static int check_dst(const mkldnn::tensor::dims &dim, const real_t *x)
{
    int n_errors = 0;
    int N = dim[0], C = dim[1], H = dim[2], W = dim[3];
#   pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w)
    {
        if (x[w + W*h + c*W*H + n*W*H*C] != c*n)
#           pragma omp atomic
            n_errors += 1;
    }
    return n_errors;
}

static int doit(bool lazy) {
    using namespace mkldnn;

    /* AlexNet: p1
     * {16, 96, 55, 55} -> {16, 96, 27, 27}
     * strides: {2, 2}
     * kernel : {3, 3}
     * padding: {0, 0}
     */

    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    auto p1_src_desc     = memory::desc({{16, 96, 55, 55}}, memory::precision::f32, memory::format::nchw);
    auto p1_indices_desc = memory::desc({{16, 96, 27, 27}}, memory::precision::u32, memory::format::nchw);
    auto p1_dst_desc     = memory::desc({{16, 96, 27, 27}}, memory::precision::f32, memory::format::nchw);

    std::vector<real_t> src(16*96*55*55);
    std::vector<real_t> dst(16*96*27*27);
    std::vector<int> indices(16*96*27*27);

    auto p1_src     = memory({p1_src_desc    , cpu_engine}, src.data()    );
    auto p1_indices = memory({p1_indices_desc, cpu_engine}, indices.data());
    auto p1_dst     = memory({p1_dst_desc    , cpu_engine}, dst.data()    );

    auto p1 = pooling(prop_kind::forward, pooling::max, p1_src, p1_indices, p1_dst,
        {2, 2}, {3, 3}, {0, 0}, padding_kind::zero);

    init_src({16, 96, 55, 55}, src.data());
    stream().submit({p1}).wait();
    int n_errors = check_dst({ 16, 96, 27, 27 }, dst.data());

    return n_errors;
}

#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    int rc = doit(false);
    printf("eager: %s\n", rc ? "failed" : "passed");
    rc = doit(true);
    printf("lazy: %s\n", rc ? "failed" : "passed");
    return rc;
}
