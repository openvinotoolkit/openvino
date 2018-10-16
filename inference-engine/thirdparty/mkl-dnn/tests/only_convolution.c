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

static size_t product(int *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

int doit(int lazy) {
    /* AlexNet: c3
     * {2, 256, 13, 13} (x) {384, 256, 3, 3} -> {2, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    const int mb = 2;
    const int groups = 2;
    int c3_src_sizes[4] = {mb, 256, 13, 13};
    int c3_weights_sizes[] = {groups, 384/groups, 256/groups, 3, 3};
    int c3_bias_sizes[1] = {384};
    int strides[] = {1, 1};
    int32_t  padding[] = {0, 0}; // set proper values
    int c3_dst_sizes[4] = {mb, 384,
        (c3_src_sizes[2] + 2*padding[0] - c3_weights_sizes[3])/strides[0] + 1,
        (c3_src_sizes[3] + 2*padding[1] - c3_weights_sizes[4])/strides[1] + 1
    };

    real_t *src = (real_t*)calloc(product(c3_src_sizes, 4), sizeof(real_t));
    real_t *weights = (real_t*)calloc(product(c3_weights_sizes, 5), sizeof(real_t));
    real_t *bias = (real_t*)calloc(product(c3_bias_sizes, 1), sizeof(real_t));
    real_t *dst = (real_t*)calloc(product(c3_dst_sizes, 4), sizeof(real_t));
    if (src == NULL || weights == NULL || bias == NULL || dst == NULL) {
        free(src);
        free(weights);
        free(bias);
        free(dst);
        return -1;
    }

    for (int i = 0; i < c3_bias_sizes[0]; ++i) bias[i] = i;

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, lazy ? mkldnn_cpu_lazy : mkldnn_cpu, 0 /* idx */));

    /* first describe user data and create data descriptors for future
     * convolution w/ the specified format -- we do not want to do a reorder */
    mkldnn_tensor_desc_t c3_src_tz, c3_weights_tz, c3_bias_tz, c3_dst_tz;
    mkldnn_memory_desc_t c3_src_md, c3_weights_md, c3_bias_md, c3_dst_md;
    mkldnn_memory_primitive_desc_t c3_src_pd, c3_weights_pd, c3_bias_pd, c3_dst_pd;
    mkldnn_primitive_t c3_src, c3_weights, c3_bias, c3_dst;

    CHECK(mkldnn_tensor_desc_init(&c3_src_tz, 4, c3_src_sizes));
    CHECK(mkldnn_memory_desc_init(&c3_src_md, &c3_src_tz, mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_primitive_desc_init(&c3_src_pd, &c3_src_md, engine));
    CHECK(mkldnn_memory_create(&c3_src, &c3_src_pd, 0 ? NULL : src));

    if (groups == 1) {
        CHECK(mkldnn_tensor_desc_init(&c3_weights_tz, 4, c3_weights_sizes + 1));
        CHECK(mkldnn_memory_desc_init(&c3_weights_md, &c3_weights_tz, mkldnn_f32, mkldnn_oihw));
    } else {
        CHECK(mkldnn_tensor_desc_init(&c3_weights_tz, 5, c3_weights_sizes));
        CHECK(mkldnn_memory_desc_init(&c3_weights_md, &c3_weights_tz, mkldnn_f32, mkldnn_goihw));
    }
    CHECK(mkldnn_memory_primitive_desc_init(&c3_weights_pd, &c3_weights_md, engine));
    CHECK(mkldnn_memory_create(&c3_weights, &c3_weights_pd, weights));

    CHECK(mkldnn_tensor_desc_init(&c3_bias_tz, 1, c3_bias_sizes));
    CHECK(mkldnn_memory_desc_init(&c3_bias_md, &c3_bias_tz, mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_primitive_desc_init(&c3_bias_pd, &c3_bias_md, engine));
    CHECK(mkldnn_memory_create(&c3_bias, &c3_bias_pd, bias));

    CHECK(mkldnn_tensor_desc_init(&c3_dst_tz, 4, c3_dst_sizes));
    CHECK(mkldnn_memory_desc_init(&c3_dst_md, &c3_dst_tz, mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_primitive_desc_init(&c3_dst_pd, &c3_dst_md, engine));

    mkldnn_primitive_at_t c3_srcs[] = {
        mkldnn_primitive_at(c3_src, 0),
        mkldnn_primitive_at(c3_weights, 0),
        mkldnn_primitive_at(c3_bias, 0)
    };

    const_mkldnn_primitive_t c3_dsts[1];
	CHECK(mkldnn_memory_create(&c3_dst, &c3_dst_pd, dst));
	c3_dsts[0] = c3_dst;

    /* create a convolution */
    mkldnn_convolution_desc_t c3_desc;
    mkldnn_convolution_primitive_desc_t c3_pd;
    mkldnn_primitive_t c3;

    CHECK(mkldnn_convolution_desc_init(&c3_desc, mkldnn_forward, mkldnn_convolution_direct,
                &c3_src_md, &c3_weights_md, &c3_bias_md, &c3_dst_md,
                strides, padding, mkldnn_padding_zero));
    CHECK(mkldnn_convolution_primitive_desc_init(&c3_pd, &c3_desc, engine));
    CHECK(mkldnn_primitive_create(&c3, &c3_pd, c3_srcs, c3_dsts));

    assert(mkldnn_memory_primitive_desc_equal(&c3_pd.src_primitive_desc, &c3_src_pd));
    assert(mkldnn_memory_primitive_desc_equal(&c3_pd.weights_primitive_desc, &c3_weights_pd));
    assert(mkldnn_memory_primitive_desc_equal(&c3_pd.bias_primitive_desc, &c3_bias_pd));
    assert(mkldnn_memory_primitive_desc_equal(&c3_pd.dst_primitive_desc, &c3_dst_pd));

    /* let us build a net */
    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream));
    CHECK(mkldnn_stream_submit(stream, 1, &c3, NULL));
    CHECK(mkldnn_stream_wait(stream, 1, NULL));

    /* clean-up */
    mkldnn_stream_destroy(stream);
    mkldnn_primitive_destroy(c3);
    mkldnn_primitive_destroy(c3_src);
    mkldnn_primitive_destroy(c3_weights);
    mkldnn_primitive_destroy(c3_bias);
    mkldnn_primitive_destroy(c3_dst);
    mkldnn_engine_destroy(engine);

    int rc = 0;
    const int N = c3_dst_sizes[0], C = c3_dst_sizes[1],
          H = c3_dst_sizes[2], W = c3_dst_sizes[3];
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w)
    {
        size_t off = ((n*C + c)*H + h)*W + w;
        if (dst[off] != bias[c]) rc = 1;
    }

    free(src);
    free(weights);
    free(bias);
    free(dst);

    return rc;
}

int main(int argc, char **argv) {
    int rc = doit(0);
    printf("eager: %s\n", rc ? "failed" : "passed");
    rc = doit(1);
    printf("lazy:  %s\n", rc ? "failed" : "passed");
    return rc;
}
