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

#include "mkldnn.h"

typedef float real_t;

#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

static size_t tensor_size(const mkldnn_tensor_desc_t *t)
{
    size_t size = 1;
    for (size_t i = 0; i < t->ndims; ++i)
        size *= t->dims[i];
    return size;
}

static void init_src(int dim[4], real_t *x)
{
    int N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (int n = 0; n < N; n += 1)
    for (int c = 0; c < C; c += 1)
    for (int h = 2; h+2 <= H; h += 2)
    for (int w = 2; w+2 <= W; w += 2)
        x[w + W*h + c*W*H + n*W*H*C] = c*n;
}

static int check_dst(int dim[4], const real_t *x)
{
    int n_errors = 0;
    int N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w)
    {
        if (x[w + W*h + c*W*H + n*W*H*C] != c*n) n_errors += 1;
    }
    return n_errors;
}

static int doit() {
    /* AlexNet: p1
     * {16, 96, 55, 55} -> {16, 96, 27, 27}
     * pad: {0, 0}
     * strides: {2, 2}
     * kernel: {3, 3}
     */

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
    * pooling w/ the specified format -- we do not want to do a reorder */
    int p1_src_sizes[4] = { 16, 96, 55, 55 };
    mkldnn_tensor_desc_t p1_src_tz;
    mkldnn_memory_desc_t p1_src_md;
    mkldnn_memory_primitive_desc_t p1_src_pd;
    mkldnn_primitive_t p1_src;
    CHECK(mkldnn_tensor_desc_init(&p1_src_tz, 4, p1_src_sizes));
    CHECK(mkldnn_memory_desc_init(&p1_src_md, &p1_src_tz, mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_primitive_desc_init(&p1_src_pd, &p1_src_md, engine));
    real_t *src = (real_t*)calloc(tensor_size(&p1_src_md.tensor_desc), sizeof(real_t));
    CHECK(mkldnn_memory_create(&p1_src, &p1_src_pd, src));

    int p1_dst_sizes[4] = { 16, 96, 27, 27 };
    mkldnn_tensor_desc_t p1_dst_tz;
    mkldnn_memory_desc_t p1_dst_md;
    mkldnn_memory_primitive_desc_t p1_dst_pd;
    mkldnn_primitive_t p1_dst;
    CHECK(mkldnn_tensor_desc_init(&p1_dst_tz, 4, p1_dst_sizes));
    CHECK(mkldnn_memory_desc_init(&p1_dst_md, &p1_dst_tz, mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_primitive_desc_init(&p1_dst_pd, &p1_dst_md, engine));
    real_t *dst = (real_t*)calloc(tensor_size(&p1_dst_md.tensor_desc), sizeof(real_t));
    CHECK(mkldnn_memory_create(&p1_dst, &p1_dst_pd, dst));

    int strides[] = { 2, 2 };
    int kernel [] = { 3, 3 };
    int32_t  padding[] = { 0, 0 };
    mkldnn_pooling_desc_t p1_desc;
    mkldnn_pooling_primitive_desc_t p1_pd;
    CHECK(mkldnn_pooling_desc_init(&p1_desc, mkldnn_forward, mkldnn_pooling_max,
        &p1_src_md, &p1_dst_md, strides, kernel, padding, mkldnn_padding_zero));
    CHECK(mkldnn_pooling_primitive_desc_init(&p1_pd, &p1_desc, engine));

    mkldnn_primitive_t p1_indices;
    CHECK(mkldnn_memory_create(&p1_indices, &p1_pd.indices_primitive_desc, NULL));

    /* create a pooling */
    mkldnn_primitive_t p1;
    mkldnn_primitive_at_t p1_srcs[] = {
        mkldnn_primitive_at(p1_src, 0),
        mkldnn_primitive_at(p1_indices, 0)
    };
    const_mkldnn_primitive_t p1_dsts[] = { p1_dst };

    CHECK(mkldnn_primitive_create(&p1, &p1_pd, p1_srcs, p1_dsts));

    assert(mkldnn_memory_primitive_desc_equal(&p1_pd.src_primitive_desc, &p1_src_pd));
    assert(mkldnn_memory_primitive_desc_equal(&p1_pd.dst_primitive_desc, &p1_dst_pd));

    init_src(p1_src_sizes, src);

    /* let us build a net */
    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream));
    CHECK(mkldnn_stream_submit(stream, 1, &p1, NULL));
    CHECK(mkldnn_stream_wait(stream, 1, NULL));

    /* clean-up */
    CHECK(mkldnn_stream_destroy(stream));
    mkldnn_primitive_destroy(p1);
    mkldnn_primitive_destroy(p1_src);
    mkldnn_primitive_destroy(p1_indices);
    mkldnn_primitive_destroy(p1_dst);
    mkldnn_engine_destroy(engine);

    int n_errors = check_dst(p1_dst_sizes, dst);

    free(src);
    free(dst);

    return n_errors;
}

int main(int argc, char **argv) {
    int rc = doit();
    return rc;
}
